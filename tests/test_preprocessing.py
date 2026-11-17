"""Unit and integration tests for the IndoorAI data validation and preprocessing pipeline."""

import hashlib
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.core.config import get_settings
from src.preprocessing.cleaning import audit_ap_feature_quality, select_features
from src.preprocessing.duplicate_handler import (
    DuplicateStrategy,
    handle_duplicates,
)
from src.preprocessing.scaling import fit_and_transform_features, get_scaler_instance
from src.preprocessing.target_encoding import (
    extract_targets_and_metadata,
    fit_target_encoders,
    verify_no_leakage,
)
from src.preprocessing.validation import validate_datasets, validate_partition
from src.preprocessing.wifi_preprocessing import (
    create_visibility_mask,
    extract_wap_columns,
    generate_wifi_features,
    process_rssi_matrix,
)


def compute_file_sha256(filepath: Path) -> str:
    """Calculate the SHA-256 hex digest of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class TestPreprocessingPipeline(unittest.TestCase):
    """Test suite covering all 12 preprocessing and data leakage prevention requirements."""

    def setUp(self) -> None:
        """Create controlled mock fixtures for unit testing without relying on raw files."""
        self.settings = get_settings()

        # Controlled training fixture
        self.mock_train = pd.DataFrame({
            "WAP001": [-60, -70, 100, -80, -60],
            "WAP002": [100, -75, -65, -85, 100],
            "WAP003": [100, 100, 100, 100, 100],  # Constant 100 in training
            "LONGITUDE": [-7500.0, -7510.0, -7520.0, -7600.0, -7500.0],
            "LATITUDE": [4864800.0, 4864810.0, 4864820.0, 4864900.0, 4864800.0],
            "FLOOR": [0, 0, 1, 1, 0],
            "BUILDINGID": [0, 0, 1, 1, 0],
            "SPACEID": [101, 102, 201, 202, 101],
            "RELATIVEPOSITION": [1, 1, 2, 2, 1],
            "USERID": [1, 1, 2, 2, 1],
            "PHONEID": [10, 10, 11, 11, 10],
            "TIMESTAMP": [100, 101, 200, 201, 100],  # Row 0 and Row 4 are exact duplicates!
        })

        # Controlled validation fixture
        self.mock_val = pd.DataFrame({
            "WAP001": [-65, 100],
            "WAP002": [100, -72],
            "WAP003": [-90, 100],  # Active in validation even though constant in train!
            "LONGITUDE": [-7505.0, -7605.0],
            "LATITUDE": [4864805.0, 4864905.0],
            "FLOOR": [0, 1],
            "BUILDINGID": [0, 1],
            "SPACEID": [105, 205],
            "RELATIVEPOSITION": [1, 2],
            "USERID": [0, 0],
            "PHONEID": [15, 15],
            "TIMESTAMP": [500, 501],
        })

    def test_01_wap_identification(self) -> None:
        """Requirement 1: Correct WAP column extraction."""
        waps = extract_wap_columns(self.mock_train)
        self.assertEqual(waps, ["WAP001", "WAP002", "WAP003"])
        
        # Verify that existing _VISIBLE columns are not mistaken for raw WAP features
        df_with_masks = self.mock_train.copy()
        df_with_masks["WAP001_VISIBLE"] = 1
        waps_after = extract_wap_columns(df_with_masks)
        self.assertEqual(waps_after, ["WAP001", "WAP002", "WAP003"])

    def test_02_unavailable_rssi_handling(self) -> None:
        """Requirement 2: Replace unavailable sentinel 100 with configured noise floor."""
        floor_val = -105.0
        cleaned_df = process_rssi_matrix(self.mock_train, rssi_floor=floor_val)
        
        # WAP001 row 2 was 100 -> must now be -105.0
        self.assertEqual(cleaned_df.loc[2, "WAP001"], floor_val)
        # WAP001 row 0 was -60 -> must remain -60.0
        self.assertEqual(cleaned_df.loc[0, "WAP001"], -60.0)
        # WAP003 was all 100 -> must all be -105.0
        self.assertTrue((cleaned_df["WAP003"] == floor_val).all())

    def test_03_binary_visibility_mask_generation(self) -> None:
        """Requirement 3: Binary visibility mask (1=detected, 0=unavailable)."""
        mask_df = create_visibility_mask(self.mock_train)
        self.assertEqual(list(mask_df.columns), ["WAP001_VISIBLE", "WAP002_VISIBLE", "WAP003_VISIBLE"])
        # WAP001 signals: [-60, -70, 100, -80, -60] -> [1, 1, 0, 1, 1]
        np.testing.assert_array_equal(mask_df["WAP001_VISIBLE"].values, [1, 1, 0, 1, 1])
        # WAP003 signals: [100, 100, 100, 100, 100] -> [0, 0, 0, 0, 0]
        np.testing.assert_array_equal(mask_df["WAP003_VISIBLE"].values, [0, 0, 0, 0, 0])

    def test_04_rssi_plus_visibility_representation(self) -> None:
        """Requirement 4: Combined RSSI plus visibility representation (2x features)."""
        combined = generate_wifi_features(self.mock_train, representation="rssi_with_visibility", rssi_floor=-105.0)
        self.assertEqual(len(combined.columns), 6)  # 3 RSSI + 3 VISIBLE
        self.assertIn("WAP001", combined.columns)
        self.assertIn("WAP001_VISIBLE", combined.columns)
        self.assertEqual(len(combined), len(self.mock_train))

    def test_05_duplicate_strategies(self) -> None:
        """Requirement 5: Test duplicate handling strategies: keep_all, remove_exact, sample_weight."""
        # Row 0 and Row 4 are exact duplicates in mock_train (5 rows total, 1 redundant)
        
        # Strategy A: KEEP_ALL
        df_keep, rep_keep = handle_duplicates(self.mock_train, strategy="keep_all")
        self.assertEqual(len(df_keep), 5)
        self.assertEqual(rep_keep.rows_removed, 0)
        self.assertTrue((df_keep["sample_weight"] == 1.0).all())

        # Strategy B: REMOVE_EXACT
        df_rem, rep_rem = handle_duplicates(self.mock_train, strategy="remove_exact")
        self.assertEqual(len(df_rem), 4)
        self.assertEqual(rep_rem.rows_removed, 1)

        # Strategy C: SAMPLE_WEIGHT
        df_wt, rep_wt = handle_duplicates(self.mock_train, strategy="sample_weight")
        self.assertEqual(len(df_wt), 4)
        # Duplicate row (Row 0 / Row 4) should receive sample_weight=2.0
        dup_row = df_wt[(df_wt["LONGITUDE"] == -7500.0) & (df_wt["LATITUDE"] == 4864800.0)].iloc[0]
        self.assertEqual(dup_row["sample_weight"], 2.0)

    def test_06_training_only_scaler_fitting(self) -> None:
        """Requirement 6: Scaler parameters must be fitted exclusively on training data."""
        X_tr = pd.DataFrame({"WAP001": [-60.0, -100.0, -80.0]})
        X_vl_1 = pd.DataFrame({"WAP001": [-50.0, -110.0]})  # Min/Max in val exceed train!
        X_vl_2 = pd.DataFrame({"WAP001": [-10.0, 0.0]})

        # Fit on X_tr
        scaled_tr, scaled_vl_1, scaler_1 = fit_and_transform_features(X_tr, X_val=X_vl_1, method="minmax")
        # In X_tr: min = -100, max = -60. Range = 40.
        self.assertAlmostEqual(scaler_1.data_min_[0], -100.0)
        self.assertAlmostEqual(scaler_1.data_max_[0], -60.0)

        # Ensure validation features do not alter fitted parameters
        _, _, scaler_2 = fit_and_transform_features(X_tr, X_val=X_vl_2, method="minmax")
        self.assertEqual(scaler_1.data_min_[0], scaler_2.data_min_[0])
        self.assertEqual(scaler_1.data_max_[0], scaler_2.data_max_[0])

    def test_07_validation_transformation(self) -> None:
        """Requirement 7: Validation data is transformed using training scaler parameters."""
        X_tr = pd.DataFrame({"WAP001": [0.0, 100.0]})
        X_vl = pd.DataFrame({"WAP001": [50.0, 150.0]})
        _, scaled_vl, _ = fit_and_transform_features(X_tr, X_val=X_vl, method="minmax")
        # 50.0 relative to [0, 100] -> 0.5. 150.0 -> 1.5.
        self.assertAlmostEqual(scaled_vl.loc[0, "WAP001"], 0.5)
        self.assertAlmostEqual(scaled_vl.loc[1, "WAP001"], 1.5)

    def test_08_handling_of_constant_features(self) -> None:
        """Requirement 8: Constant features (std=0 or min=max) must scale gracefully without NaNs/Infs."""
        X_tr_const = pd.DataFrame({
            "WAP_VAR": [-60.0, -80.0, -100.0],
            "WAP_CONST": [-105.0, -105.0, -105.0]  # Constant zero-variance column
        })
        X_vl = pd.DataFrame({
            "WAP_VAR": [-70.0],
            "WAP_CONST": [-105.0]
        })

        for method in ["minmax", "standard", "robust"]:
            scaled_tr, scaled_vl, _ = fit_and_transform_features(X_tr_const, X_val=X_vl, method=method)
            self.assertFalse(scaled_tr.isna().any().any(), f"NaN in scaled_tr with method={method}")
            self.assertFalse(np.isinf(scaled_tr.values).any(), f"Inf in scaled_tr with method={method}")
            self.assertFalse(scaled_vl.isna().any().any(), f"NaN in scaled_vl with method={method}")

    def test_09_target_encoding(self) -> None:
        """Requirement 9: Target encoding maps categorical classes consistently."""
        enc_meta = fit_target_encoders(self.mock_train)
        y_df, meta_df = extract_targets_and_metadata(self.mock_train, enc_meta)
        
        self.assertIn("BUILDING_INDEX", y_df.columns)
        self.assertIn("FLOOR_INDEX", y_df.columns)
        self.assertIn("BUILDING_FLOOR_INDEX", y_df.columns)
        self.assertIn("LONGITUDE", y_df.columns)
        self.assertIn("LATITUDE", y_df.columns)
        
        # Verify bidirectional reversibility
        for bld_val, bld_idx in enc_meta.building_to_idx.items():
            self.assertEqual(enc_meta.idx_to_building[bld_idx], bld_val)

    def test_10_no_target_leakage(self) -> None:
        """Requirement 10: Assert programmatically that target or metadata columns never leak into X."""
        clean_X = pd.DataFrame({"WAP001": [0.1, 0.2], "WAP002": [0.5, 0.6]})
        y_dummy = pd.DataFrame({"BUILDINGID": [0, 1]})
        meta_dummy = pd.DataFrame({"USERID": [1, 2]})

        # Clean X should pass
        self.assertTrue(verify_no_leakage(clean_X, y_dummy, meta_dummy))

        # Injected target must raise AssertionError
        leaky_X = clean_X.copy()
        leaky_X["BUILDINGID"] = [0, 1]
        with self.assertRaises(AssertionError):
            verify_no_leakage(leaky_X, y_dummy, meta_dummy)

        # Injected metadata must raise AssertionError
        leaky_meta_X = clean_X.copy()
        leaky_meta_X["USERID"] = [1, 2]
        with self.assertRaises(AssertionError):
            verify_no_leakage(leaky_meta_X, y_dummy, meta_dummy)

    def test_11_consistent_train_val_feature_schemas(self) -> None:
        """Requirement 11: Training and validation features must share identical columns, count, and order."""
        features_tr = generate_wifi_features(self.mock_train, representation="rssi_with_visibility")
        features_vl = generate_wifi_features(self.mock_val, representation="rssi_with_visibility")
        self.assertEqual(list(features_tr.columns), list(features_vl.columns))

    def test_12_raw_dataset_immutability(self) -> None:
        """Requirement 12: Verify that raw dataset files were never modified or corrupted."""
        raw_train_path = self.settings.paths.raw_data_dir / "ujiindoorloc" / "trainingData.csv"
        raw_val_path = self.settings.paths.raw_data_dir / "ujiindoorloc" / "validationData.csv"

        if raw_train_path.exists() and raw_val_path.exists():
            # Check against original official SHA-256 checksums from Task 02
            expected_train_sha256 = "45ca0128bd12019c976bb4793e407c979c82995d7a5940ab7288620247905168"
            expected_val_sha256 = "5f90c536648cd657b2c516d20c4e0968d4003279ea6bd5d5d5322d3f1e8905c0"
            
            actual_train_sha256 = compute_file_sha256(raw_train_path)
            actual_val_sha256 = compute_file_sha256(raw_val_path)
            
            self.assertEqual(actual_train_sha256, expected_train_sha256, "Raw trainingData.csv was modified!")
            self.assertEqual(actual_val_sha256, expected_val_sha256, "Raw validationData.csv was modified!")


if __name__ == "__main__":
    unittest.main()

