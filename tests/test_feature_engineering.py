"""Unit and integration tests for Wi-Fi feature engineering pipeline.

Tests verify the 12 explicit conditions specified in Prompt 05:
1. Per-sample visible AP count.
2. RSSI mean and median excluding unavailable APs.
3. Strongest signal extraction.
4. Signal strength grouping (strong, medium, weak).
5. Safe handling of zero-visible-AP records.
6. Relative signal calculations.
7. Training-only AP statistics fitting.
8. Training-only feature selection fitting.
9. Identical train/validation feature schemas.
10. No target or metadata leakage.
11. No NaN or infinite output features.
12. Reproducibility of feature generation.
"""

from pathlib import Path
import unittest
import numpy as np
import pandas as pd

from src.features.signal_statistics import (
    SignalStatsConfig,
    compute_sample_signal_statistics,
    extract_signal_statistics_from_dataframe,
    SUMMARY_FEATURE_NAMES,
)
from src.features.ap_features import (
    compute_training_ap_statistics,
    compute_relative_signal_features,
)
from src.features.feature_selection import (
    evaluate_feature_selection_thresholds,
    apply_feature_selection,
)
from src.features.wifi_features import (
    WiFiFeaturePipelineConfig,
    WiFiFeatureTransformer,
    FORBIDDEN_METADATA_COLUMNS,
)


class TestFeatureEngineering(unittest.TestCase):
    """Test suite covering the 12 feature engineering specifications."""

    def setUp(self):
        """Set up controlled synthetic Wi-Fi fingerprints."""
        # Synthetic AP readings for 4 WAPs:
        # Sample 0: [-50, -70, -90, 100] -> 3 visible (-50 strong, -70 med, -90 weak)
        # Sample 1: [-65, -75, 100, 100] -> 2 visible (0 strong, 2 med, 0 weak)
        # Sample 2: [100, 100, 100, 100] -> 0 visible (Edge case: completely blank)
        # Sample 3: [-45, 100, 100, 100] -> 1 visible (-45 strong)
        self.synthetic_waps = pd.DataFrame({
            "WAP001": [-50, -65, 100, -45],
            "WAP002": [-70, -75, 100, 100],
            "WAP003": [-90, 100, 100, 100],
            "WAP004": [100, 100, 100, 100],  # Constant 100 in train
        })
        self.config = SignalStatsConfig(
            unavailable_sentinel=100,
            rssi_floor_dbm=-105.0,
            strong_threshold_dbm=-60.0,
            medium_threshold_dbm=-80.0
        )

    # 1. Per-sample visible AP count
    def test_01_visible_ap_count(self):
        stats = extract_signal_statistics_from_dataframe(self.synthetic_waps, config=self.config)
        expected_visible = [3.0, 2.0, 0.0, 1.0]
        expected_unavailable = [1.0, 2.0, 4.0, 3.0]
        np.testing.assert_array_equal(stats["visible_ap_count"].values, expected_visible)
        np.testing.assert_array_equal(stats["unavailable_ap_count"].values, expected_unavailable)
        np.testing.assert_allclose(stats["visible_ap_ratio"].values, [3/4, 2/4, 0/4, 1/4])

    # 2. RSSI mean and median excluding unavailable APs
    def test_02_rssi_mean_and_median_excluding_unavailable(self):
        stats = extract_signal_statistics_from_dataframe(self.synthetic_waps, config=self.config)
        # Sample 0: [-50, -70, -90] -> mean = -70.0, median = -70.0
        self.assertAlmostEqual(stats.loc[0, "rssi_mean"], -70.0)
        self.assertAlmostEqual(stats.loc[0, "rssi_median"], -70.0)
        self.assertAlmostEqual(stats.loc[0, "rssi_min"], -90.0)
        self.assertAlmostEqual(stats.loc[0, "rssi_max"], -50.0)
        self.assertAlmostEqual(stats.loc[0, "rssi_range"], 40.0)

        # Sample 1: [-65, -75] -> mean = -70.0, median = -70.0
        self.assertAlmostEqual(stats.loc[1, "rssi_mean"], -70.0)
        self.assertAlmostEqual(stats.loc[1, "rssi_median"], -70.0)

    # 3. Strongest signal extraction
    def test_03_strongest_signal_extraction(self):
        stats = extract_signal_statistics_from_dataframe(self.synthetic_waps, config=self.config)
        # Sample 0: ranks are -50, -70, -90, then floor (-105)
        self.assertAlmostEqual(stats.loc[0, "strongest_rssi_1"], -50.0)
        self.assertAlmostEqual(stats.loc[0, "strongest_rssi_2"], -70.0)
        self.assertAlmostEqual(stats.loc[0, "strongest_rssi_3"], -90.0)
        self.assertAlmostEqual(stats.loc[0, "strongest_rssi_5"], -105.0)

        # Sample 1: strongest 1 is -65, strongest 2 is -75, strongest 3 is floor
        self.assertAlmostEqual(stats.loc[1, "strongest_rssi_1"], -65.0)
        self.assertAlmostEqual(stats.loc[1, "strongest_rssi_2"], -75.0)
        self.assertAlmostEqual(stats.loc[1, "strongest_rssi_3"], -105.0)

    # 4. Signal strength grouping (strong, medium, weak)
    def test_04_signal_strength_grouping(self):
        stats = extract_signal_statistics_from_dataframe(self.synthetic_waps, config=self.config)
        # Sample 0: -50 (strong), -70 (medium), -90 (weak)
        self.assertEqual(stats.loc[0, "strong_signal_count"], 1.0)
        self.assertEqual(stats.loc[0, "medium_signal_count"], 1.0)
        self.assertEqual(stats.loc[0, "weak_signal_count"], 1.0)

        # Sample 1: -65 (medium), -75 (medium)
        self.assertEqual(stats.loc[1, "strong_signal_count"], 0.0)
        self.assertEqual(stats.loc[1, "medium_signal_count"], 2.0)
        self.assertEqual(stats.loc[1, "weak_signal_count"], 0.0)

        # Sample 3: -45 (strong)
        self.assertEqual(stats.loc[3, "strong_signal_count"], 1.0)
        self.assertEqual(stats.loc[3, "medium_signal_count"], 0.0)
        self.assertEqual(stats.loc[3, "weak_signal_count"], 0.0)

    # 5. Safe handling of zero-visible-AP records
    def test_05_zero_visible_ap_handling(self):
        stats = extract_signal_statistics_from_dataframe(self.synthetic_waps, config=self.config)
        # Sample 2 has all 100s
        row2 = stats.loc[2]
        self.assertEqual(row2["visible_ap_count"], 0.0)
        self.assertEqual(row2["visible_ap_ratio"], 0.0)
        self.assertEqual(row2["unavailable_ap_count"], 4.0)
        self.assertAlmostEqual(row2["rssi_mean"], -105.0)
        self.assertAlmostEqual(row2["rssi_median"], -105.0)
        self.assertAlmostEqual(row2["rssi_std"], 0.0)
        self.assertAlmostEqual(row2["rssi_range"], 0.0)
        self.assertAlmostEqual(row2["strongest_rssi_1"], -105.0)
        self.assertEqual(row2["strong_signal_count"], 0.0)
        self.assertEqual(row2["medium_signal_count"], 0.0)
        self.assertEqual(row2["weak_signal_count"], 0.0)
        # Ensure no NaNs or infs
        self.assertFalse(row2.isna().any())

    # 6. Relative signal calculations
    def test_06_relative_signal_calculations(self):
        rel_df = compute_relative_signal_features(
            df=self.synthetic_waps,
            wap_prefix="WAP",
            sentinel=100,
            diff_to_max=True,
            diff_to_mean=True,
            default_unavailable_diff=0.0
        )
        # Check generated column names
        self.assertIn("WAP001_DIFF_TO_MAX", rel_df.columns)
        self.assertIn("WAP001_DIFF_TO_MEAN", rel_df.columns)

        # Sample 0: max = -50.
        # WAP001 is -50: diff_to_max = -50 - (-50) = 0.0
        # WAP002 is -70: diff_to_max = -70 - (-50) = -20.0
        # WAP003 is -90: diff_to_max = -90 - (-50) = -40.0
        # WAP004 is 100 (unavail): diff_to_max = default (0.0)
        self.assertAlmostEqual(rel_df.loc[0, "WAP001_DIFF_TO_MAX"], 0.0)
        self.assertAlmostEqual(rel_df.loc[0, "WAP002_DIFF_TO_MAX"], -20.0)
        self.assertAlmostEqual(rel_df.loc[0, "WAP003_DIFF_TO_MAX"], -40.0)
        self.assertAlmostEqual(rel_df.loc[0, "WAP004_DIFF_TO_MAX"], 0.0)

    # 7. Training-only AP statistics fitting
    def test_07_training_only_ap_statistics(self):
        ap_stats = compute_training_ap_statistics(self.synthetic_waps, wap_prefix="WAP", sentinel=100)
        self.assertEqual(len(ap_stats), 4)

        # WAP004 has 0 detections in training -> CONSTANT_UNAVAILABLE
        wap4_row = ap_stats[ap_stats["ap_name"] == "WAP004"].iloc[0]
        self.assertTrue(wap4_row["is_constant_100"])
        self.assertEqual(wap4_row["train_detection_count"], 0)
        self.assertEqual(wap4_row["status"], "CONSTANT_UNAVAILABLE")

        # WAP001 detected in 3 of 4 samples -> ACTIVE
        wap1_row = ap_stats[ap_stats["ap_name"] == "WAP001"].iloc[0]
        self.assertFalse(wap1_row["is_constant_100"])
        self.assertEqual(wap1_row["train_detection_count"], 3)
        self.assertEqual(wap1_row["status"], "ACTIVE")

    # 8. Training-only feature selection fitting
    def test_08_training_only_feature_selection(self):
        # Create baseline feature matrix (4 WAPs continuous + 4 visibility)
        X_base = pd.DataFrame({
            "WAP001": [0.5, 0.4, 0.0, 0.6],
            "WAP002": [0.3, 0.2, 0.0, 0.0],
            "WAP003": [0.1, 0.0, 0.0, 0.0],
            "WAP004": [0.0, 0.0, 0.0, 0.0],  # Constant 0
            "WAP001_VISIBLE": [1, 1, 0, 1],
            "WAP002_VISIBLE": [1, 1, 0, 0],
            "WAP003_VISIBLE": [1, 0, 0, 0],
            "WAP004_VISIBLE": [0, 0, 0, 0],  # Constant 0
        })
        ap_stats = compute_training_ap_statistics(self.synthetic_waps, wap_prefix="WAP", sentinel=100)
        eval_result = evaluate_feature_selection_thresholds(X_base, ap_stats)

        # 'remove_constant' should drop WAP004 and WAP004_VISIBLE
        rem_const = eval_result["strategies"]["remove_constant"]
        self.assertIn("WAP004", rem_const["dropped_features"])
        self.assertIn("WAP004_VISIBLE", rem_const["dropped_features"])
        self.assertEqual(rem_const["dropped_feature_count"], 2)
        self.assertEqual(rem_const["retained_feature_count"], 6)

    # 9. Identical train/validation feature schemas
    def test_09_identical_train_validation_feature_schemas(self):
        train_p = Path("data/processed/ujiindoorloc/feature_engineered/X_train_engineered.parquet")
        val_p = Path("data/processed/ujiindoorloc/feature_engineered/X_validation_engineered.parquet")
        self.assertTrue(train_p.is_file(), "Engineered train Parquet not found")
        self.assertTrue(val_p.is_file(), "Engineered validation Parquet not found")

        df_train = pd.read_parquet(train_p)
        df_val = pd.read_parquet(val_p)

        self.assertEqual(df_train.shape[1], df_val.shape[1])
        self.assertListEqual(list(df_train.columns), list(df_val.columns))
        # Ensure row counts match expectations
        self.assertEqual(len(df_train), 19937)
        self.assertEqual(len(df_val), 1111)

    # 10. No target or metadata leakage
    def test_10_no_target_or_metadata_leakage(self):
        train_p = Path("data/processed/ujiindoorloc/feature_engineered/X_train_engineered.parquet")
        df_train = pd.read_parquet(train_p)

        intersection = set(df_train.columns).intersection(FORBIDDEN_METADATA_COLUMNS)
        self.assertEqual(len(intersection), 0, f"Found leaked metadata/target columns: {intersection}")

        # Also verify transformer rejects forbidden columns
        pipe_cfg = WiFiFeaturePipelineConfig(strict_leakage_checks=True)
        transformer = WiFiFeatureTransformer(config=pipe_cfg)
        leaked_matrix = pd.DataFrame({"WAP001": [0.1], "BUILDINGID": [1]})
        with self.assertRaises(ValueError):
            transformer._validate_output_matrix(leaked_matrix)

    # 11. No NaN or infinite output features
    def test_11_no_nan_or_infinite_features(self):
        train_p = Path("data/processed/ujiindoorloc/feature_engineered/X_train_engineered.parquet")
        val_p = Path("data/processed/ujiindoorloc/feature_engineered/X_validation_engineered.parquet")
        df_train = pd.read_parquet(train_p)
        df_val = pd.read_parquet(val_p)

        self.assertEqual(df_train.isna().sum().sum(), 0, "X_train contains NaN values")
        self.assertEqual(df_val.isna().sum().sum(), 0, "X_val contains NaN values")
        self.assertFalse(np.isinf(df_train.values).any(), "X_train contains infinite values")
        self.assertFalse(np.isinf(df_val.values).any(), "X_val contains infinite values")

    # 12. Reproducibility of feature generation
    def test_12_reproducibility(self):
        # Generate summary stats twice on same synthetic slice
        stats1 = extract_signal_statistics_from_dataframe(self.synthetic_waps, config=self.config)
        stats2 = extract_signal_statistics_from_dataframe(self.synthetic_waps, config=self.config)
        pd.testing.assert_frame_equal(stats1, stats2)


if __name__ == "__main__":
    unittest.main()
