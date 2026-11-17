"""
Unit Tests for Two-Stage Hierarchical Hybrid Localizer (Task 10).
"""

import json
import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd

from src.models.traditional_ml.random_forest import IndoorAIRandomForest
from src.models.traditional_ml.knn import IndoorAIKNN
from src.models.traditional_ml.hybrid_localizer import HybridLocalizer


class TestHybridLocalizer(unittest.TestCase):
    """Test suite for HybridLocalizer architecture and routing logic."""

    @classmethod
    def setUpClass(cls):
        cls.rf = IndoorAIRandomForest.load("experiments/random_forest/models")
        cls.knn = IndoorAIKNN.load("experiments/knn/models")

        # Load small slice of real training data for rapid testing
        cls.X_train = pd.read_parquet("data/processed/ujiindoorloc/feature_engineered/X_train_engineered.parquet")
        cls.y_train = pd.read_parquet("data/processed/ujiindoorloc/y_train.parquet")
        cls.X_val = pd.read_parquet("data/processed/ujiindoorloc/feature_engineered/X_validation_engineered.parquet").iloc[:25]
        cls.y_val = pd.read_parquet("data/processed/ujiindoorloc/y_validation.parquet").iloc[:25]

    def test_hybrid_initialization(self):
        """Verify initialization parameters and defaults."""
        hybrid = HybridLocalizer(
            rf_model=self.rf,
            knn_global_model=self.knn,
            confidence_threshold=0.70,
            min_subset_samples=10,
            local_k=7,
        )
        self.assertEqual(hybrid.confidence_threshold, 0.70)
        self.assertEqual(hybrid.min_subset_samples, 10)
        self.assertEqual(hybrid.local_k, 7)
        self.assertFalse(hybrid.is_fitted)

    def test_hybrid_fit_and_prediction_shapes(self):
        """Verify fitting creates 13 building-floor subsets and predict returns valid shapes."""
        hybrid = HybridLocalizer(
            rf_model=self.rf,
            knn_global_model=self.knn,
            confidence_threshold=0.70,
            min_subset_samples=10,
            local_k=7,
        )
        hybrid.fit(self.X_train, self.y_train)
        self.assertTrue(hybrid.is_fitted)
        self.assertEqual(len(hybrid.local_knn_models), 13)

        preds = hybrid.predict(self.X_val)
        n = len(self.X_val)

        self.assertIn("pred_building", preds)
        self.assertIn("pred_floor", preds)
        self.assertIn("pred_longitude", preds)
        self.assertIn("pred_latitude", preds)
        self.assertIn("rf_confidence", preds)
        self.assertIn("coord_uncertainty", preds)
        self.assertIn("used_local_knn", preds)

        self.assertEqual(len(preds["pred_building"]), n)
        self.assertEqual(len(preds["pred_floor"]), n)
        self.assertEqual(len(preds["pred_longitude"]), n)
        self.assertEqual(len(preds["pred_latitude"]), n)
        self.assertEqual(len(preds["rf_confidence"]), n)
        self.assertEqual(len(preds["coord_uncertainty"]), n)
        self.assertEqual(len(preds["used_local_knn"]), n)

    def test_fallback_on_insufficient_samples(self):
        """Verify that when a subset has fewer samples than min_subset_samples, it falls back."""
        hybrid = HybridLocalizer(
            rf_model=self.rf,
            knn_global_model=self.knn,
            confidence_threshold=0.70,
            min_subset_samples=2000,  # Deliberately high so several subsets have < 2000 samples
            local_k=7,
        )
        hybrid.fit(self.X_train, self.y_train)

        # Some subsets have < 2000 samples and must be skipped
        self.assertLess(len(hybrid.local_knn_models), 13)
        self.assertGreater(len(hybrid.local_knn_models), 0)

        # Predicting should still succeed seamlessly via global k-NN fallback
        preds = hybrid.predict(self.X_val)
        self.assertEqual(len(preds["pred_longitude"]), len(self.X_val))

    def test_confidence_gating_behavior(self):
        """Verify routing behavior at tau=0.0 (always local) vs tau=1.0 (always fallback)."""
        # tau=0.0: always routes to local k-NN
        h_local = HybridLocalizer(
            rf_model=self.rf,
            knn_global_model=self.knn,
            confidence_threshold=0.0,
            min_subset_samples=10,
        ).fit(self.X_train, self.y_train)
        preds_local = h_local.predict(self.X_val)
        self.assertTrue(np.all(preds_local["used_local_knn"]))

        # tau=1.0: always routes to fallback (rf_confidence is strictly < 1.0)
        h_fallback = HybridLocalizer(
            rf_model=self.rf,
            knn_global_model=self.knn,
            confidence_threshold=1.0001,
            min_subset_samples=10,
        ).fit(self.X_train, self.y_train)
        preds_fallback = h_fallback.predict(self.X_val)
        self.assertFalse(np.any(preds_fallback["used_local_knn"]))

    def test_serialization_and_reproducibility(self):
        """Verify joblib serialization and identical predictions upon loading."""
        hybrid = HybridLocalizer(
            rf_model=self.rf,
            knn_global_model=self.knn,
            confidence_threshold=0.70,
            min_subset_samples=10,
        ).fit(self.X_train, self.y_train)

        preds_orig = hybrid.predict(self.X_val)

        temp_dir = tempfile.mkdtemp()
        try:
            hybrid.save(temp_dir)
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "hybrid_metadata.json")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "local_knn_models.joblib")))

            loaded = HybridLocalizer.load(temp_dir, rf_model=self.rf, knn_global_model=self.knn)
            preds_loaded = loaded.predict(self.X_val)

            np.testing.assert_array_almost_equal(preds_orig["pred_longitude"], preds_loaded["pred_longitude"])
            np.testing.assert_array_almost_equal(preds_orig["pred_latitude"], preds_loaded["pred_latitude"])
            np.testing.assert_array_equal(preds_orig["used_local_knn"], preds_loaded["used_local_knn"])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

