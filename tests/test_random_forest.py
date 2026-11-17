"""Unit and integration tests for Random Forest baseline models.

Verifies:
1. Feature and target separation (no metadata or target in X).
2. Strict data leakage rejection.
3. Model training and prediction dimensions.
4. Building-floor label generation and schema consistency.
5. Classification metric calculations.
6. Regression metric calculations (MAE, RMSE, 2D Euclidean error).
7. Deterministic reproducibility with a fixed random seed.
8. Model serialization and reloading with joblib.
"""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.evaluation.random_forest_evaluation import (
    compute_classification_metrics,
    compute_hierarchical_breakdown,
    compute_regression_metrics,
)
from src.models.traditional_ml.random_forest import (
    FORBIDDEN_COLUMNS,
    IndoorAIRandomForest,
    RFHyperparameters,
)

_repo_root = Path(__file__).resolve().parent.parent


class TestRandomForestBaseline(unittest.TestCase):
    """Test suite covering the Random Forest baseline requirements."""

    def setUp(self):
        """Set up controlled synthetic dataset for fast deterministic unit tests."""
        np.random.seed(42)
        n_samples = 40
        n_features = 20

        self.feature_names = [f"WAP{i:03d}" for i in range(1, 11)] + \
                             [f"WAP{i:03d}_VISIBLE" for i in range(1, 11)]

        X_mat = np.random.uniform(0.0, 1.0, size=(n_samples, n_features))
        self.X_synthetic = pd.DataFrame(X_mat, columns=self.feature_names)

        self.y_synthetic = pd.DataFrame({
            "BUILDINGID": np.random.choice([0, 1, 2], size=n_samples),
            "FLOOR": np.random.choice([0, 1, 2, 3], size=n_samples),
            "BUILDING_FLOOR_INDEX": np.random.choice(range(12), size=n_samples),
            "LONGITUDE": np.random.uniform(-7500.0, -7300.0, size=n_samples),
            "LATITUDE": np.random.uniform(4864800.0, 4865000.0, size=n_samples),
        })

    # 1. Feature and target separation
    def test_01_feature_target_separation(self):
        train_p = _repo_root / "data" / "processed" / "ujiindoorloc" / "feature_engineered" / "X_train_engineered.parquet"
        y_train_p = _repo_root / "data" / "processed" / "ujiindoorloc" / "y_train.parquet"

        X_tr = pd.read_parquet(train_p)
        y_tr = pd.read_parquet(y_train_p)

        # Ensure no overlap between feature columns and target columns
        overlap = set(X_tr.columns).intersection(set(y_tr.columns))
        self.assertEqual(len(overlap), 0, f"Overlap between X and y: {overlap}")
        self.assertEqual(len(X_tr), len(y_tr))

    # 2. Strict leakage detection
    def test_02_leakage_detection_assertion(self):
        model = IndoorAIRandomForest(random_state=42)
        leaked_df = self.X_synthetic.copy()
        leaked_df["BUILDINGID"] = 0

        with self.assertRaises(ValueError) as ctx:
            model._verify_features(leaked_df)
        self.assertIn("DATA LEAKAGE DETECTED", str(ctx.exception))

    # 3. Model training and prediction dimensions
    def test_03_training_and_prediction_dimensions(self):
        hp = RFHyperparameters(n_estimators=10, random_state=42, n_jobs=1)
        model = IndoorAIRandomForest(classifier_params=hp, regressor_params=hp)
        model.fit(self.X_synthetic, self.y_synthetic)

        self.assertTrue(model.is_fitted_)
        preds = model.predict(self.X_synthetic)

        expected_keys = {
            "pred_building",
            "pred_floor",
            "pred_building_floor",
            "pred_longitude",
            "pred_latitude"
        }
        self.assertEqual(set(preds.keys()), expected_keys)
        for k, v in preds.items():
            self.assertEqual(len(v), len(self.X_synthetic), f"Dimension mismatch for {k}")

    # 4. Building-floor label generation and consistency
    def test_04_building_floor_label_consistency(self):
        y_tr = pd.read_parquet(_repo_root / "data" / "processed" / "ujiindoorloc" / "y_train.parquet")
        # Ensure that each unique (BUILDINGID, FLOOR) maps to exactly one BUILDING_FLOOR_INDEX
        grouped = y_tr.groupby(["BUILDINGID", "FLOOR"])["BUILDING_FLOOR_INDEX"].nunique()
        self.assertTrue((grouped == 1).all(), "Inconsistent building-floor mapping")

        # In UJIIndoorLoc, there are 13 unique building-floor combinations
        self.assertEqual(y_tr["BUILDING_FLOOR_INDEX"].nunique(), 13)

    # 5. Classification metrics calculation
    def test_05_classification_metrics(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 1, 2])  # 5 out of 6 correct

        res = compute_classification_metrics(y_true, y_pred, "test_target", class_labels=[0, 1, 2])
        self.assertAlmostEqual(res["accuracy"], 5 / 6, places=4)
        self.assertIn("precision_macro", res)
        self.assertIn("recall_macro", res)
        self.assertIn("f1_macro", res)
        self.assertEqual(len(res["confusion_matrix"]), 3)

    # 6. Regression metrics calculation (MAE, RMSE, 2D Euclidean error)
    def test_06_regression_metrics(self):
        # 3 points:
        # Point 0: True=(0, 0), Pred=(3, 4) -> error = 5
        # Point 1: True=(10, 10), Pred=(10, 10) -> error = 0
        # Point 2: True=(5, 5), Pred=(5, 5) -> error = 0
        t_lon = np.array([0.0, 10.0, 5.0])
        p_lon = np.array([3.0, 10.0, 5.0])
        t_lat = np.array([0.0, 10.0, 5.0])
        p_lat = np.array([4.0, 10.0, 5.0])

        res = compute_regression_metrics(t_lon, p_lon, t_lat, p_lat)

        # Longitude MAE: (|3-0| + 0 + 0) / 3 = 1.0
        self.assertAlmostEqual(res["longitude"]["mae"], 1.0)
        # Latitude MAE: (|4-0| + 0 + 0) / 3 = 1.3333
        self.assertAlmostEqual(res["latitude"]["mae"], 4 / 3, places=3)
        # 2D Euclidean errors: [5.0, 0.0, 0.0] -> mean = 5/3 = 1.6667, median = 0.0, max = 5.0
        self.assertAlmostEqual(res["euclidean_error"]["mean"], 5 / 3, places=3)
        self.assertAlmostEqual(res["euclidean_error"]["median"], 0.0)
        self.assertAlmostEqual(res["euclidean_error"]["max"], 5.0)

    # 7. Reproducibility with fixed random seed
    def test_07_reproducibility(self):
        hp1 = RFHyperparameters(n_estimators=10, random_state=42, n_jobs=1)
        hp2 = RFHyperparameters(n_estimators=10, random_state=42, n_jobs=1)

        m1 = IndoorAIRandomForest(classifier_params=hp1, regressor_params=hp1, random_state=42)
        m2 = IndoorAIRandomForest(classifier_params=hp2, regressor_params=hp2, random_state=42)

        m1.fit(self.X_synthetic, self.y_synthetic)
        m2.fit(self.X_synthetic, self.y_synthetic)

        p1 = m1.predict(self.X_synthetic)
        p2 = m2.predict(self.X_synthetic)

        for k in p1.keys():
            np.testing.assert_array_equal(p1[k], p2[k])

    # 8. Model serialization and loading with joblib
    def test_08_model_serialization(self):
        hp = RFHyperparameters(n_estimators=10, random_state=42, n_jobs=1)
        model = IndoorAIRandomForest(classifier_params=hp, regressor_params=hp)
        model.fit(self.X_synthetic, self.y_synthetic)

        preds_before = model.predict(self.X_synthetic)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save(tmp_dir)
            loaded_model = IndoorAIRandomForest.load(tmp_dir)
            preds_after = loaded_model.predict(self.X_synthetic)

            for k in preds_before.keys():
                np.testing.assert_array_equal(preds_before[k], preds_after[k])


if __name__ == "__main__":
    unittest.main()

