"""Unit and validation tests for UJIIndoorLoc dataset acquisition, loading, and inspection."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.core.config import get_settings
from src.data.download_ujiindoorloc import compute_sha256, safe_extract_zip
from src.data.inspect_ujiindoorloc import (
    UNAVAILABLE_WAP_VALUE,
    compute_categorical_statistics,
    compute_numerical_statistics,
    compute_wap_visibility_statistics,
    get_dataset_paths,
    identify_columns,
    load_dataset,
    run_quality_checks,
)


class TestUJIIndoorLocAcquisitionAndInspection(unittest.TestCase):
    """Test suite for dataset acquisition logic, schema validation, and statistical analysis."""

    def setUp(self) -> None:
        self.settings = get_settings()
        self.train_path, self.val_path, self.eda_dir = get_dataset_paths()

        # Create an isolated in-memory fixture for deterministic logic unit testing
        # Matches the exact schema of UJIIndoorLoc (3 WAPs and context fields)
        self.mock_train_data = pd.DataFrame({
            "WAP001": [-65, -80, 100, 100],
            "WAP002": [100, -75, -70, 100],
            "WAP003": [100, 100, 100, 100],  # Constant 100 (never observed)
            "LONGITUDE": [-7345.2, -7346.1, -7347.0, -7348.5],
            "LATITUDE": [4864800.1, 4864810.2, 4864820.3, 4864830.4],
            "FLOOR": [0, 1, 2, 2],
            "BUILDINGID": [0, 1, 2, 2],
            "SPACEID": [101, 102, 103, 104],
            "RELATIVEPOSITION": [1, 2, 1, 2],
            "USERID": [1, 2, 3, 4],
            "PHONEID": [10, 11, 12, 13],
            "TIMESTAMP": [1370000001, 1370000002, 1370000003, 1370000004],
        })

        self.mock_val_data = pd.DataFrame({
            "WAP001": [-68, 100],
            "WAP002": [100, -72],
            "WAP003": [100, 100],
            "LONGITUDE": [-7345.5, -7346.5],
            "LATITUDE": [4864805.0, 4864815.0],
            "FLOOR": [0, 1],
            "BUILDINGID": [0, 1],
            "SPACEID": [201, 202],
            "RELATIVEPOSITION": [1, 2],
            "USERID": [0, 0],
            "PHONEID": [14, 15],
            "TIMESTAMP": [1370001001, 1370001002],
        })

    def test_column_identification(self) -> None:
        """Verify partition of WAP columns and context columns."""
        wap_cols, context_cols = identify_columns(self.mock_train_data)
        self.assertEqual(wap_cols, ["WAP001", "WAP002", "WAP003"])
        self.assertIn("BUILDINGID", context_cols)
        self.assertIn("FLOOR", context_cols)
        self.assertIn("LONGITUDE", context_cols)
        self.assertIn("LATITUDE", context_cols)
        self.assertEqual(len(context_cols), 9)

    def test_unavailable_wap_handling_in_statistics(self) -> None:
        """Verify that value 100 is recognized as unavailable and excluded from true RSSI stats."""
        stats_df = compute_numerical_statistics(self.mock_train_data, "mock_train")
        wap1_stats = stats_df[stats_df["column"] == "WAP001"].iloc[0]
        
        # WAP001 has [-65, -80, 100, 100] -> 2 valid observations (-65 and -80), 2 unavailable (100)
        self.assertEqual(wap1_stats["valid_obs_count"], 2)
        self.assertEqual(wap1_stats["unavailable_100_count"], 2)
        self.assertEqual(wap1_stats["max"], -65.0)
        self.assertEqual(wap1_stats["min"], -80.0)
        self.assertEqual(wap1_stats["mean"], -72.5)

        # WAP003 has all 100s -> 0 valid observations (mean is NaN/None in pandas DataFrame)
        wap3_stats = stats_df[stats_df["column"] == "WAP003"].iloc[0]
        self.assertEqual(wap3_stats["valid_obs_count"], 0)
        self.assertEqual(wap3_stats["unavailable_100_count"], 4)
        self.assertTrue(pd.isna(wap3_stats["mean"]))

    def test_wap_visibility_statistics(self) -> None:
        """Verify visibility rates calculation."""
        vis_df = compute_wap_visibility_statistics(self.mock_train_data, self.mock_val_data)
        self.assertEqual(len(vis_df), 3)
        
        wap1_row = vis_df[vis_df["wap_id"] == "WAP001"].iloc[0]
        self.assertEqual(wap1_row["train_detected_count"], 2)
        self.assertEqual(wap1_row["val_detected_count"], 1)
        self.assertEqual(wap1_row["total_detected_count"], 3)
        self.assertFalse(wap1_row["is_constant_100_in_train"])

        wap3_row = vis_df[vis_df["wap_id"] == "WAP003"].iloc[0]
        self.assertEqual(wap3_row["total_detected_count"], 0)
        self.assertTrue(wap3_row["is_constant_in_both"])

    def test_quality_checks_on_fixture(self) -> None:
        """Verify duplicate detection, constant detection, and schema consistency."""
        qc = run_quality_checks(self.mock_train_data, self.mock_val_data)
        self.assertTrue(qc.schema_match)
        self.assertEqual(qc.train_duplicates_count, 0)
        self.assertEqual(qc.train_constant_waps_count, 1)  # WAP003
        self.assertIn("WAP003", qc.never_detected_waps_both_splits)

        # Introduce duplicate row in train fixture
        dup_train = pd.concat([self.mock_train_data, self.mock_train_data.iloc[[0]]], ignore_index=True)
        qc_dup = run_quality_checks(dup_train, self.mock_val_data)
        self.assertEqual(qc_dup.train_duplicates_count, 1)

    def test_sha256_computation(self) -> None:
        """Verify file checksum hashing."""
        with tempfile.NamedTemporaryFile("w+", delete=False) as tf:
            tf.write("IndoorAI UJIIndoorLoc Test Content")
            tpath = Path(tf.name)
        
        try:
            digest = compute_sha256(tpath)
            self.assertIsInstance(digest, str)
            self.assertEqual(len(digest), 64)
        finally:
            if tpath.exists():
                tpath.unlink()

    def test_actual_downloaded_dataset_integrity(self) -> None:
        """Verify properties of the real downloaded UJIIndoorLoc dataset if files exist on disk."""
        if not (self.train_path.exists() and self.val_path.exists()):
            self.skipTest("Raw UJIIndoorLoc dataset files not yet downloaded.")

        df_train = load_dataset(self.train_path)
        df_val = load_dataset(self.val_path)

        # Expected dimensions from official UCI release
        self.assertEqual(len(df_train), 19937, "Official trainingData.csv must contain 19,937 rows")
        self.assertEqual(len(df_train.columns), 529, "Official trainingData.csv must contain 529 columns")
        self.assertEqual(len(df_val), 1111, "Official validationData.csv must contain 1,111 rows")
        self.assertEqual(len(df_val.columns), 529, "Official validationData.csv must contain 529 columns")

        # 520 WAPs and 9 context attributes
        wap_cols, context_cols = identify_columns(df_train)
        self.assertEqual(len(wap_cols), 520)
        self.assertEqual(len(context_cols), 9)

        # Exact schema match between splits
        self.assertEqual(list(df_train.columns), list(df_val.columns))

        # Check sentinel value
        self.assertEqual(UNAVAILABLE_WAP_VALUE, 100)


if __name__ == "__main__":
    unittest.main()
