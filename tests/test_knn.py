"""Unit and integration tests for k-Nearest Neighbors (k-NN) baseline models.

Verifies:
1. Feature and target separation (no metadata or target in X).
2. Strict data leakage rejection.
3. Training-only scaler fitting.
4. Prediction dimensions and keys.
5. Classification metric calculations.
6. Coordinate regression metric calculations (MAE, RMSE, 2D Euclidean error).
7. Building-floor label consistency.
8. Model serialization and reloading with joblib.
9. Deterministic reproducibility.
10. Validation schema consistency.
"""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from src.evaluation.knn_evaluation import (
    compute_classification_metrics,
    compute_hierarchical_breakdown,
    compute_regression_metrics,
)
from src.models.traditional_ml.knn import (
    FORBIDDEN_COLUMNS,
    IndoorAIKNN,
    KNNHyperparameters,
    SummaryFeatureScaler,
)

_repo_root = Path(__file__).resolve().parent.parent


class TestKNNBaseline(unittest.TestCase):
    """Test suite covering the k-NN baseline requirements."""

    def setUp(self):
        """Set up controlled synthetic dataset for fast deterministic unit tests."""
        np.random.seed(42)
        n_samples = 50
        n_features = 20

        self.base_cols = [f"WAP{i:03d}" for i in range(1, 11)] + \
                         [f"WAP{i:03d}_VISIBLE" for i in range(1, 11)]
        self.summary_cols = ["visible_ap_count", "rssi_mean", "unavailable_ap_count"]

        all_cols = self.base_cols + self.summary_cols

        X_base = np.random.uniform(0.0, 1.0, size=(n_samples, len(self.base_cols)))
        X_summ = np.column_stack([
            np.random.randint(1, 25, size=n_samples),      # visible_ap_count
            np.random.uniform(-100.0, -40.0, size=n_samples), # rssi_mean
            np.random.randint(450, 520, size=n_samples),  # unavailable_ap_count
        ])
        X_mat = np.hstack([X_base, X_summ])
        self.X_synthetic = pd.DataFrame(X_mat, columns=all_cols)

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

        overlap = set(X_tr.columns).intersection(set(y_tr.columns))
        self.assertEqual(len(overlap), 0, f"Overlap between X and y: {overlap}")
        self.assertEqual(len(X_tr), len(y_tr))

    # 2. Strict leakage detection
    def test_02_leakage_detection_assertion(self):
        model = IndoorAIKNN()
        leaked_df = self.X_synthetic.copy()
        leaked_df["BUILDINGID"] = 0

        with self.assertRaises(ValueError) as ctx:
            model._verify_features(leaked_df)
        self.assertIn("DATA LEAKAGE DETECTED", str(ctx.exception))

    # 3. Training-only scaler fitting
    def test_03_training_only_scaler_fitting(self):
        scaler = SummaryFeatureScaler(scaler_type="minmax")
        X_tr_proc = scaler.fit_transform(self.X_synthetic, base_feature_count=len(self.base_cols))

        self.assertTrue(scaler.is_fitted)
        self.assertEqual(set(scaler.summary_columns), set(self.summary_cols))

        # Check that summary columns are scaled to [0, 1]
        for col in self.summary_cols:
            self.assertGreaterEqual(X_tr_proc[col].min(), -1e-6)
            self.assertLessEqual(X_tr_proc[col].max(), 1.0 + 1e-6)

        # Base features remain untouched
        for col in self.base_cols:
            np.testing.assert_array_almost_equal(X_tr_proc[col].values, self.X_synthetic[col].values)

    # 4. Prediction dimensions and keys
    def test_04_prediction_dimensions(self):
        params = KNNHyperparameters(n_neighbors=3, metric="euclidean", scale_summary=False)
        model = IndoorAIKNN(params=params)
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
        t_lon = np.array([0.0, 10.0, 5.0])
        p_lon = np.array([3.0, 10.0, 5.0])
        t_lat = np.array([0.0, 10.0, 5.0])
        p_lat = np.array([4.0, 10.0, 5.0])

        res = compute_regression_metrics(t_lon, p_lon, t_lat, p_lat)

        self.assertAlmostEqual(res["longitude"]["mae"], 1.0)
        self.assertAlmostEqual(res["latitude"]["mae"], 4 / 3, places=3)
        self.assertAlmostEqual(res["euclidean_error"]["mean"], 5 / 3, places=3)
        self.assertAlmostEqual(res["euclidean_error"]["median"], 0.0)
        self.assertAlmostEqual(res["euclidean_error"]["max"], 5.0)

    # 7. Building-floor label consistency
    def test_07_building_floor_label_consistency(self):
        y_tr = pd.read_parquet(_repo_root / "data" / "processed" / "ujiindoorloc" / "y_train.parquet")
        grouped = y_tr.groupby(["BUILDINGID", "FLOOR"])["BUILDING_FLOOR_INDEX"].nunique()
        self.assertTrue((grouped == 1).all(), "Inconsistent building-floor mapping")
        self.assertEqual(y_tr["BUILDING_FLOOR_INDEX"].nunique(), 13)

    # 8. Model serialization and reloading with joblib
    def test_08_model_serialization(self):
        params = KNNHyperparameters(n_neighbors=3, metric="euclidean", scale_summary=True)
        model = IndoorAIKNN(params=params)
        model.fit(self.X_synthetic, self.y_synthetic)

        preds_before = model.predict(self.X_synthetic)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save(tmp_dir)
            loaded_model = IndoorAIKNN.load(tmp_dir)
            preds_after = loaded_model.predict(self.X_synthetic)

            for k in preds_before.keys():
                np.testing.assert_array_equal(preds_before[k], preds_after[k])

    # 9. Deterministic reproducibility
    def test_09_reproducibility(self):
        p1 = KNNHyperparameters(n_neighbors=3, metric="cosine")
        p2 = KNNHyperparameters(n_neighbors=3, metric="cosine")

        m1 = IndoorAIKNN(params=p1)
        m2 = IndoorAIKNN(params=p2)

        m1.fit(self.X_synthetic, self.y_synthetic)
        m2.fit(self.X_synthetic, self.y_synthetic)

        preds1 = m1.predict(self.X_synthetic)
        preds2 = m2.predict(self.X_synthetic)

        for k in preds1.keys():
            np.testing.assert_array_equal(preds1[k], preds2[k])

    # 10. Validation schema consistency
    def test_10_validation_schema_consistency(self):
        train_p = _repo_root / "data" / "processed" / "ujiindoorloc" / "feature_engineered" / "X_train_engineered.parquet"
        val_p = _repo_root / "data" / "processed" / "ujiindoorloc" / "feature_engineered" / "X_validation_engineered.parquet"

        X_tr = pd.read_parquet(train_p)
        X_val = pd.read_parquet(val_p)

        self.assertEqual(list(X_tr.columns), list(X_val.columns))
        self.assertEqual(len(X_tr.columns), 1058)
        self.assertEqual(len(X_val.columns), 1058)


if __name__ == "__main__":
    unittest.main()

