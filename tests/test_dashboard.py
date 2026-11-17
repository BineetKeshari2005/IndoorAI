"""Unit tests for IndoorAI Web Dashboard and Prediction Service (Tasks 11 & 11.1).

Tests:
1. PredictionService model loading and resource caching.
2. Validation dataset retrieval and sample formatting.
3. Multi-model localization inference (RF, k-NN with k=7, Hybrid).
4. Raw CSV upload preprocessing and schema validation.
5. Evaluation metrics calculation (Euclidean error in dataset units).
6. Target isolation and zero data leakage.
7. Verification of audited benchmark metrics from Task 09 and Task 10.
8. Live inference verification on validation samples 0, 150, 450, 948.
"""

import io
import unittest
import numpy as np
import pandas as pd

from backend.app.services.prediction_service import (
    PredictionService,
    get_prediction_service,
    EXPECTED_WAP_COUNT,
    EXPECTED_ENGINEERED_FEATURE_COUNT,
)


class TestDashboardAndPredictionService(unittest.TestCase):
    """Test suite for IndoorAI dashboard backend service."""

    @classmethod
    def setUpClass(cls):
        """Initialize prediction service once for all tests."""
        cls.service = get_prediction_service()
        cls.service.load_all()

    def test_service_initialization(self):
        """Verify models and datasets are loaded."""
        self.assertIsNotNone(self.service.rf_model)
        self.assertIsNotNone(self.service.knn_model)
        self.assertIsNotNone(self.service.hybrid_model)
        self.assertIsNotNone(self.service.scaler)
        self.assertEqual(len(self.service.validation_features), 1111)
        self.assertEqual(len(self.service.validation_targets), 1111)

    def test_get_validation_sample(self):
        """Verify retrieval of a validation sample by index."""
        sample = self.service.get_validation_sample(0)
        self.assertEqual(sample["index"], 0)
        self.assertIn("features", sample)
        self.assertIn("ground_truth", sample)
        self.assertIn("metadata", sample)
        self.assertIn("detected_waps", sample)
        self.assertEqual(len(sample["features"].columns), EXPECTED_ENGINEERED_FEATURE_COUNT)
        self.assertIn("BUILDINGID", sample["ground_truth"])
        self.assertIn("FLOOR", sample["ground_truth"])
        self.assertIn("LONGITUDE", sample["ground_truth"])
        self.assertIn("LATITUDE", sample["ground_truth"])

    def test_predict_random_forest(self):
        """Verify Random Forest inference returns expected keys and reasonable values."""
        sample = self.service.get_validation_sample(10)
        pred = self.service.predict(sample["features"], model_name="Random Forest")

        self.assertEqual(pred["model_name"], "Random Forest")
        self.assertIn(pred["pred_building"], [0, 1, 2])
        self.assertIn(pred["pred_floor"], [0, 1, 2, 3, 4])
        self.assertIsInstance(pred["pred_longitude"], float)
        self.assertIsInstance(pred["pred_latitude"], float)
        self.assertGreaterEqual(pred["confidence"], 0.0)
        self.assertLessEqual(pred["confidence"], 1.0)
        self.assertFalse(pred["is_confidence_calibrated"])
        self.assertGreater(pred["spatial_uncertainty"], 0.0)

    def test_predict_knn_k7_configuration(self):
        """Verify k-Nearest Neighbors inference uses the audited Task 08 configuration (k=7)."""
        sample = self.service.get_validation_sample(25)
        pred = self.service.predict(sample["features"], model_name="k-Nearest Neighbors")

        self.assertIn("k=7", pred["model_name"])
        self.assertEqual(pred["details"]["k_neighbors"], 7)
        self.assertIn(pred["pred_building"], [0, 1, 2])
        self.assertIn(pred["pred_floor"], [0, 1, 2, 3, 4])
        self.assertIsInstance(pred["pred_longitude"], float)
        self.assertIsInstance(pred["pred_latitude"], float)
        self.assertGreaterEqual(pred["confidence"], 0.0)
        self.assertLessEqual(pred["confidence"], 1.0)
        self.assertFalse(pred["is_confidence_calibrated"])
        self.assertGreater(pred["spatial_uncertainty"], 0.0)

    def test_predict_hybrid(self):
        """Verify Hybrid model inference returns expected keys and routing details."""
        sample = self.service.get_validation_sample(50)
        pred = self.service.predict(sample["features"], model_name="Hybrid")

        self.assertEqual(pred["model_name"], "Hybrid (Two-Stage Hierarchical)")
        self.assertIn(pred["pred_building"], [0, 1, 2])
        self.assertIn(pred["pred_floor"], [0, 1, 2, 3, 4])
        self.assertIsInstance(pred["pred_longitude"], float)
        self.assertIsInstance(pred["pred_latitude"], float)
        self.assertGreaterEqual(pred["confidence"], 0.0)
        self.assertLessEqual(pred["confidence"], 1.0)
        self.assertFalse(pred["is_confidence_calibrated"])
        self.assertGreater(pred["spatial_uncertainty"], 0.0)
        self.assertIn("used_local_knn", pred["details"])

    def test_calculate_evaluation_metrics(self):
        """Verify spatial error metrics computation in dataset coordinate units."""
        pred = {
            "pred_building": 1,
            "pred_floor": 2,
            "pred_longitude": -7500.0,
            "pred_latitude": 4864900.0,
        }
        gt = {
            "BUILDINGID": 1,
            "FLOOR": 2,
            "LONGITUDE": -7503.0,
            "LATITUDE": 4864904.0,
        }
        metrics = self.service.calculate_evaluation_metrics(pred, gt)

        self.assertTrue(metrics["building_correct"])
        self.assertTrue(metrics["floor_correct"])
        self.assertTrue(metrics["building_floor_correct"])
        self.assertEqual(metrics["floor_difference"], 0)
        self.assertAlmostEqual(metrics["longitude_error"], 3.0, places=3)
        self.assertAlmostEqual(metrics["latitude_error"], -4.0, places=3)
        # Euclidean error = sqrt(3^2 + (-4)^2) = 5.0
        self.assertAlmostEqual(metrics["euclidean_error_2d"], 5.0, places=3)
        self.assertEqual(metrics["units"], "dataset coordinate units")

    def test_preprocess_raw_csv_valid(self):
        """Verify raw CSV preprocessing produces 1,058 engineered features."""
        raw_slice = self.service.raw_validation.iloc[:3].copy()
        csv_buffer = io.StringIO(raw_slice.to_csv(index=False))

        features_df, ground_truth, detected_waps = self.service.preprocess_raw_csv(csv_buffer)

        self.assertEqual(len(features_df), 3)
        self.assertEqual(len(features_df.columns), EXPECTED_ENGINEERED_FEATURE_COUNT)
        self.assertIsNotNone(ground_truth)
        self.assertEqual(len(ground_truth), 3)
        self.assertEqual(len(detected_waps), 3)

        # Confirm zero leakage: target columns must NOT be in features_df
        for col in ["BUILDINGID", "FLOOR", "LONGITUDE", "LATITUDE", "SPACEID"]:
            self.assertNotIn(col, features_df.columns)

    def test_preprocess_raw_csv_missing_columns(self):
        """Verify that CSV with missing WAP columns is rejected with helpful error."""
        invalid_df = pd.DataFrame({"WAP001": [-70], "WAP002": [-80]})
        csv_buffer = io.StringIO(invalid_df.to_csv(index=False))

        with self.assertRaises(ValueError) as ctx:
            self.service.preprocess_raw_csv(csv_buffer)

        self.assertIn("missing", str(ctx.exception).lower())
        self.assertIn("520", str(ctx.exception))

    def test_audited_benchmark_table(self):
        """Verify benchmark table contains the 2 audited baselines with exact Task 09 metrics."""
        bench = self.service.get_benchmark_table()
        self.assertEqual(len(bench), 2)

        rf_row = bench[bench["Model"].str.contains("Random Forest")].iloc[0]
        knn_row = bench[bench["Model"].str.contains("k-Nearest Neighbors")].iloc[0]

        # Verify exact Task 09 audited values
        self.assertEqual(rf_row["Building Acc (%)"], 99.82)
        self.assertEqual(rf_row["Floor Acc (%)"], 88.66)
        self.assertEqual(rf_row["Building-Floor Acc (%)"], 88.48)
        self.assertEqual(rf_row["Mean 2D Error (units)"], 11.53)
        self.assertEqual(rf_row["Median 2D Error (units)"], 8.65)
        self.assertEqual(rf_row["95th Pct Error (units)"], 32.74)

        self.assertEqual(knn_row["Building Acc (%)"], 99.91)
        self.assertEqual(knn_row["Floor Acc (%)"], 84.43)
        self.assertEqual(knn_row["Building-Floor Acc (%)"], 84.43)
        self.assertEqual(knn_row["Mean 2D Error (units)"], 10.05)
        self.assertEqual(knn_row["Median 2D Error (units)"], 7.33)
        self.assertEqual(knn_row["95th Pct Error (units)"], 29.06)

    def test_experimental_hybrid_table(self):
        """Verify experimental hybrid table contains Task 10 error analysis metrics."""
        hyb = self.service.get_experimental_hybrid_table()
        self.assertEqual(len(hyb), 1)
        row = hyb.iloc[0]
        self.assertEqual(row["Building Acc (%)"], 99.82)
        self.assertEqual(row["Floor Acc (%)"], 88.66)
        self.assertEqual(row["Building-Floor Acc (%)"], 88.48)
        self.assertEqual(row["Mean 2D Error (units)"], 9.98)
        self.assertEqual(row["Median 2D Error (units)"], 7.29)
        self.assertEqual(row["95th Pct Error (units)"], 29.26)
        self.assertIn("Experimental", row["Status"])

    def test_sample_live_inference_verification(self):
        """Verify live inference across required audit samples: 0, 150, 450, 948."""
        for idx in [0, 150, 450, 948]:
            sample = self.service.get_validation_sample(idx)
            for model_name in ["Random Forest", "k-Nearest Neighbors", "Hybrid"]:
                pred = self.service.predict(sample["features"], model_name=model_name)
                self.assertIn(pred["pred_building"], [0, 1, 2])
                self.assertIn(pred["pred_floor"], [0, 1, 2, 3, 4])
                self.assertIsInstance(pred["pred_longitude"], float)
                self.assertIsInstance(pred["pred_latitude"], float)


if __name__ == "__main__":
    unittest.main()
