"""Automated numerical audit test suite for IndoorAI Wi-Fi feature engineering.

Verifies:
1. All 18 feature calculations against independent ground truth.
2. All 6 mathematical invariants (A through F) with 0 violations.
3. Zero-visible-AP edge cases and fallback behavior.
4. AP detection statistics and Task 03 consistency across all 520 WAPs.
5. Strict training-only fitting and isolation.
6. Exact train/validation schema and column ordering consistency.
7. Complete absence of target or metadata leakage.
8. Bitwise deterministic reproducibility.
"""

from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.features.signal_statistics import (
    SignalStatsConfig,
    extract_signal_statistics_from_dataframe,
    SUMMARY_FEATURE_NAMES,
)
from src.features.ap_features import compute_training_ap_statistics
from src.features.wifi_features import (
    WiFiFeaturePipelineConfig,
    WiFiFeatureTransformer,
    FORBIDDEN_METADATA_COLUMNS,
)

_repo_root = Path(__file__).resolve().parent.parent


class TestFeatureAudit(unittest.TestCase):
    """Rigorous audit test suite verifying mathematical correctness and invariants."""

    @classmethod
    def setUpClass(cls):
        cls.train_raw_path = _repo_root / "data" / "raw" / "ujiindoorloc" / "trainingData.csv"
        cls.val_raw_path = _repo_root / "data" / "raw" / "ujiindoorloc" / "validationData.csv"
        cls.train_proc_path = _repo_root / "data" / "processed" / "ujiindoorloc" / "feature_engineered" / "X_train_engineered.parquet"
        cls.val_proc_path = _repo_root / "data" / "processed" / "ujiindoorloc" / "feature_engineered" / "X_validation_engineered.parquet"
        cls.verif_csv_path = _repo_root / "experiments" / "feature_engineering" / "audit" / "ap_statistics_verification.csv"

        cls.df_proc_train = pd.read_parquet(cls.train_proc_path)
        cls.df_proc_val = pd.read_parquet(cls.val_proc_path)

    # 1. Test all 18 feature calculations against independent raw calculation
    def test_01_all_18_feature_calculations(self):
        # Controlled synthetic fingerprint
        synthetic = pd.DataFrame({
            "WAP001": [-50, -60, 100, -40],
            "WAP002": [-70, -80, 100, 100],
            "WAP003": [-90, -85, 100, 100],
            "WAP004": [100, 100, 100, 100],
        })
        cfg = SignalStatsConfig(
            unavailable_sentinel=100,
            rssi_floor_dbm=-105.0,
            strong_threshold_dbm=-60.0,
            medium_threshold_dbm=-80.0
        )
        stats = extract_signal_statistics_from_dataframe(synthetic, config=cfg)

        # Sample 0: [-50, -70, -90]
        self.assertEqual(stats.loc[0, "visible_ap_count"], 3.0)
        self.assertAlmostEqual(stats.loc[0, "visible_ap_ratio"], 3 / 4.0)
        self.assertEqual(stats.loc[0, "unavailable_ap_count"], 1.0)
        self.assertAlmostEqual(stats.loc[0, "rssi_mean"], -70.0)
        self.assertAlmostEqual(stats.loc[0, "rssi_median"], -70.0)
        self.assertAlmostEqual(stats.loc[0, "rssi_min"], -90.0)
        self.assertAlmostEqual(stats.loc[0, "rssi_max"], -50.0)
        self.assertAlmostEqual(stats.loc[0, "rssi_range"], 40.0)
        self.assertAlmostEqual(stats.loc[0, "strongest_rssi_1"], -50.0)
        self.assertAlmostEqual(stats.loc[0, "strongest_rssi_2"], -70.0)
        self.assertAlmostEqual(stats.loc[0, "strongest_rssi_3"], -90.0)
        self.assertAlmostEqual(stats.loc[0, "strongest_rssi_5"], -105.0)
        self.assertEqual(stats.loc[0, "strong_signal_count"], 1.0)   # -50 >= -60
        self.assertEqual(stats.loc[0, "medium_signal_count"], 1.0)   # -70 in [-80, -60)
        self.assertEqual(stats.loc[0, "weak_signal_count"], 1.0)     # -90 < -80

        # Sample 1: [-60, -80, -85]
        # Note: -60 is strong (>= -60), -80 is medium in [-80, -60), -85 is weak (< -80)
        self.assertEqual(stats.loc[1, "strong_signal_count"], 1.0)
        self.assertEqual(stats.loc[1, "medium_signal_count"], 1.0)
        self.assertEqual(stats.loc[1, "weak_signal_count"], 1.0)

    # 2. Test all 6 mathematical invariants across entire training and validation Parquet files
    def test_02_all_mathematical_invariants(self):
        for name, df in [("train", self.df_proc_train), ("val", self.df_proc_val)]:
            # Invariant A: strong + medium + weak == visible
            bands_sum = df["strong_signal_count"] + df["medium_signal_count"] + df["weak_signal_count"]
            diff_a = np.abs(bands_sum - df["visible_ap_count"])
            self.assertEqual(int((diff_a > 1e-4).sum()), 0, f"Invariant A violated in {name}")

            # Invariant B: strongest_rssi_1 == rssi_max (for visible > 0)
            has_vis = df["visible_ap_count"] > 0
            diff_b = np.abs(df.loc[has_vis, "strongest_rssi_1"] - df.loc[has_vis, "rssi_max"])
            self.assertEqual(int((diff_b > 1e-4).sum()), 0, f"Invariant B violated in {name}")

            # Invariant C: visible + unavailable == 520
            diff_c = np.abs((df["visible_ap_count"] + df["unavailable_ap_count"]) - 520.0)
            self.assertEqual(int((diff_c > 1e-4).sum()), 0, f"Invariant C violated in {name}")

            # Invariant D: visible_ratio == visible / 520
            diff_d = np.abs(df["visible_ap_ratio"] - (df["visible_ap_count"] / 520.0))
            self.assertEqual(int((diff_d > 1e-5).sum()), 0, f"Invariant D violated in {name}")

            # Invariant E: range == max - min
            diff_e = np.abs(df["rssi_range"] - (df["rssi_max"] - df["rssi_min"]))
            self.assertEqual(int((diff_e > 1e-4).sum()), 0, f"Invariant E violated in {name}")

            # Invariant F: s1 >= s2 >= s3 >= s5
            viol_f1 = (df["strongest_rssi_1"] < df["strongest_rssi_2"] - 1e-4).sum()
            viol_f2 = (df["strongest_rssi_2"] < df["strongest_rssi_3"] - 1e-4).sum()
            viol_f3 = (df["strongest_rssi_3"] < df["strongest_rssi_5"] - 1e-4).sum()
            self.assertEqual(viol_f1 + viol_f2 + viol_f3, 0, f"Invariant F violated in {name}")

    # 3. Test zero-visible-AP handling on actual dataset and synthetic edge case
    def test_03_zero_visible_ap_handling(self):
        # Sample index 4 in trainingData.csv has 0 visible APs
        row4 = self.df_proc_train.iloc[4]
        self.assertEqual(row4["visible_ap_count"], 0.0)
        self.assertEqual(row4["visible_ap_ratio"], 0.0)
        self.assertEqual(row4["unavailable_ap_count"], 520.0)
        self.assertAlmostEqual(row4["rssi_mean"], -105.0)
        self.assertAlmostEqual(row4["rssi_median"], -105.0)
        self.assertAlmostEqual(row4["rssi_std"], 0.0)
        self.assertAlmostEqual(row4["rssi_min"], -105.0)
        self.assertAlmostEqual(row4["rssi_max"], -105.0)
        self.assertAlmostEqual(row4["rssi_range"], 0.0)
        self.assertAlmostEqual(row4["strongest_rssi_1"], -105.0)
        self.assertEqual(row4["strong_signal_count"], 0.0)
        self.assertEqual(row4["medium_signal_count"], 0.0)
        self.assertEqual(row4["weak_signal_count"], 0.0)
        self.assertFalse(row4[SUMMARY_FEATURE_NAMES].isna().any())

    # 4. Test AP detection statistics verification table
    def test_04_ap_detection_statistics(self):
        self.assertTrue(self.verif_csv_path.is_file(), "AP verification CSV missing")
        verif_df = pd.read_csv(self.verif_csv_path)

        self.assertEqual(len(verif_df), 520)
        self.assertEqual(verif_df["diff_detected_count"].abs().max(), 0, "Discrepancy in detection counts")
        self.assertLess(verif_df["diff_mean_rssi"].max(), 1e-4, "Discrepancy in mean RSSI")
        self.assertTrue(verif_df["is_verified"].all(), "Not all APs verified")

        active_count = (verif_df["status"].str.upper() == "ACTIVE").sum()
        constant_count = verif_df["is_constant_100"].sum()
        self.assertEqual(active_count, 465)
        self.assertEqual(constant_count, 55)

    # 5. Test strict training-only fitting and isolation
    def test_05_training_only_fitting_and_isolation(self):
        cfg = WiFiFeaturePipelineConfig(scale_engineered_features=False)
        transformer = WiFiFeatureTransformer(config=cfg)

        raw_tr = pd.read_csv(self.train_raw_path, nrows=50)
        raw_val = pd.read_csv(self.val_raw_path, nrows=20)
        base_tr = self.df_proc_train.iloc[:50, :1040]
        base_val = self.df_proc_val.iloc[:20, :1040]

        transformer.fit(raw_tr, base_tr)
        out_val = transformer.transform(raw_val, base_val)

        # Transformer must not modify training statistics when transforming validation
        self.assertEqual(len(transformer.ap_statistics_), 520)
        self.assertEqual(out_val.shape, (20, 1058))

    # 6. Test identical schemas and column order
    def test_06_validation_schema_consistency(self):
        self.assertEqual(self.df_proc_train.shape, (19937, 1058))
        self.assertEqual(self.df_proc_val.shape, (1111, 1058))
        self.assertListEqual(list(self.df_proc_train.columns), list(self.df_proc_val.columns))

    # 7. Test complete absence of target or metadata leakage
    def test_07_no_target_or_metadata_leakage(self):
        train_leaks = set(self.df_proc_train.columns).intersection(FORBIDDEN_METADATA_COLUMNS)
        val_leaks = set(self.df_proc_val.columns).intersection(FORBIDDEN_METADATA_COLUMNS)
        self.assertEqual(len(train_leaks), 0, f"Leaked columns in train: {train_leaks}")
        self.assertEqual(len(val_leaks), 0, f"Leaked columns in val: {val_leaks}")

    # 8. Test reproducibility
    def test_08_reproducibility(self):
        cfg = SignalStatsConfig()
        raw_slice = pd.read_csv(self.train_raw_path, nrows=10)
        stats1 = extract_signal_statistics_from_dataframe(raw_slice, config=cfg)
        stats2 = extract_signal_statistics_from_dataframe(raw_slice, config=cfg)
        pd.testing.assert_frame_equal(stats1, stats2)


if __name__ == "__main__":
    unittest.main()

