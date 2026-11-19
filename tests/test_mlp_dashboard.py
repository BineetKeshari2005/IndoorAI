"""Unit and integration tests for Multitask MLP Dashboard Integration (Task 12 in Dashboard).

Verifies checkpoint loading, model architecture, 1,058-feature ordering,
prediction schema, uncalibrated confidence score, spatial uncertainty handling,
four-model PredictionService switching on the same fingerprint, explicit error
on missing checkpoint (no silent fallback), and three-way benchmark integrity.
"""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from backend.app.services.prediction_service import (
    EXPECTED_ENGINEERED_FEATURE_COUNT,
    PredictionService,
    get_prediction_service,
)
from src.models.deep_learning.multitask_mlp import (
    IndoorAIMultitaskMLP,
    MultitaskMLP,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestMLPDashboardIntegration(unittest.TestCase):
    """Test suite for Multitask MLP dashboard integration."""

    @classmethod
    def setUpClass(cls):
        cls.service = get_prediction_service(base_dir=REPO_ROOT)
        cls.service.load_all()
        cls.sample = cls.service.get_validation_sample(0)
        cls.features = cls.sample["features"]
        cls.ground_truth = cls.sample["ground_truth"]

    def test_mlp_checkpoint_exists(self):
        """1. Verify that multitask_mlp.pt exists and can be loaded via classmethod."""
        checkpoint_path = REPO_ROOT / "experiments" / "mlp" / "models" / "multitask_mlp.pt"
        self.assertTrue(checkpoint_path.exists(), f"Checkpoint not found at {checkpoint_path}")

        loaded = IndoorAIMultitaskMLP.load(checkpoint_path, device="cpu")
        self.assertIsNotNone(loaded)
        self.assertIsInstance(loaded.model, MultitaskMLP)

    def test_mlp_architecture_and_parameters(self):
        """2. Verify exact network architecture and parameter count (~709,258)."""
        model = self.service.mlp_model.model
        self.assertIsInstance(model, MultitaskMLP)

        total_params = sum(p.numel() for p in model.parameters())
        self.assertEqual(total_params, 709258, f"Expected 709,258 parameters, got {total_params}")

        # Check task head layer dimensions
        self.assertEqual(model.head_building.out_features, 3)
        self.assertEqual(model.head_floor.out_features, 5)
        self.assertEqual(model.head_longitude.out_features, 1)
        self.assertEqual(model.head_latitude.out_features, 1)

    def test_mlp_feature_ordering_and_dimension(self):
        """3. Verify requirement of exactly 1,058 engineered features matching training."""
        self.assertEqual(len(self.service.mlp_model.feature_names_in_), EXPECTED_ENGINEERED_FEATURE_COUNT)
        self.assertEqual(self.features.shape[1], EXPECTED_ENGINEERED_FEATURE_COUNT)

        # Invalid feature count must raise ValueError
        invalid_features = self.features.iloc[:, :500]
        with self.assertRaises(ValueError):
            self.service.predict(invalid_features, model_name="Multitask MLP")

    def test_mlp_prediction_shapes_and_types(self):
        """4. Verify prediction returns proper decoded types and valid bounds."""
        res = self.service.predict(self.features, model_name="Multitask MLP — Task 12 Deep Learning")

        self.assertIn("pred_building", res)
        self.assertIn("pred_floor", res)
        self.assertIn("pred_longitude", res)
        self.assertIn("pred_latitude", res)

        self.assertIsInstance(res["pred_building"], int)
        self.assertIsInstance(res["pred_floor"], int)
        self.assertIsInstance(res["pred_longitude"], float)
        self.assertIsInstance(res["pred_latitude"], float)

        self.assertIn(res["pred_building"], [0, 1, 2])
        self.assertIn(res["pred_floor"], [0, 1, 2, 3, 4])
        self.assertGreater(res["pred_latitude"], 4.8e6)
        self.assertLess(res["pred_latitude"], 5.0e6)
        self.assertLess(res["pred_longitude"], -7200.0)
        self.assertGreater(res["pred_longitude"], -7700.0)

    def test_mlp_confidence_semantics(self):
        """5. Verify confidence represents uncalibrated joint softmax score."""
        res = self.service.predict(self.features, model_name="Multitask MLP")

        self.assertIn("confidence", res)
        self.assertIn("confidence_pct", res)
        self.assertFalse(res["is_confidence_calibrated"])
        self.assertGreaterEqual(res["confidence"], 0.0)
        self.assertLessEqual(res["confidence"], 1.0)
        self.assertEqual(res["confidence_pct"], round(res["confidence"] * 100, 2))

        # Joint softmax probability = Pmax_bldg * Pmax_floor
        b_conf = res["details"]["building_confidence"]
        f_conf = res["details"]["floor_confidence"]
        expected_joint = b_conf * f_conf
        self.assertAlmostEqual(res["confidence"], expected_joint, places=4)

    def test_mlp_spatial_uncertainty_is_none(self):
        """6. Verify spatial uncertainty is None for MLP point regressor."""
        res = self.service.predict(self.features, model_name="Multitask MLP")

        self.assertIsNone(res["spatial_uncertainty"])
        self.assertIn("uncertainty_note", res["details"])
        self.assertIn("point regression", res["details"]["uncertainty_note"].lower())

    def test_prediction_service_four_models(self):
        """7. Verify switching between all four models on the exact same fingerprint."""
        model_names = [
            "Random Forest — Task 07 Baseline",
            "k-Nearest Neighbors — Task 08 Baseline",
            "Multitask MLP — Task 12 Deep Learning",
            "Hybrid RF + k-NN — Task 10 Experimental",
        ]

        predictions = {}
        for m in model_names:
            pred = self.service.predict(self.features, model_name=m)
            self.assertIsNotNone(pred)
            self.assertIn("pred_building", pred)
            self.assertIn("pred_floor", pred)
            self.assertIn("pred_longitude", pred)
            self.assertIn("pred_latitude", pred)
            predictions[m] = pred

        # RF, k-NN, and Hybrid provide float spatial uncertainty
        self.assertIsInstance(predictions["Random Forest — Task 07 Baseline"]["spatial_uncertainty"], float)
        self.assertIsInstance(predictions["k-Nearest Neighbors — Task 08 Baseline"]["spatial_uncertainty"], float)
        self.assertIsInstance(predictions["Hybrid RF + k-NN — Task 10 Experimental"]["spatial_uncertainty"], float)

        # MLP explicitly sets spatial_uncertainty to None
        self.assertIsNone(predictions["Multitask MLP — Task 12 Deep Learning"]["spatial_uncertainty"])

    def test_missing_mlp_checkpoint_raises_error(self):
        """8. Verify that if MLP is not loaded, prediction raises RuntimeError without silent fallback."""
        temp_service = PredictionService(base_dir=REPO_ROOT)
        temp_service.load_datasets()
        temp_service._rf_model = self.service.rf_model
        temp_service._knn_model = self.service.knn_model
        temp_service._hybrid_model = self.service.hybrid_model

        with patch.object(temp_service, "load_models"):
            temp_service.mlp_model = None  # simulate missing MLP

            with self.assertRaises(RuntimeError) as ctx:
                temp_service.predict(self.features, model_name="Multitask MLP")

            self.assertIn("Multitask MLP model checkpoint is not loaded", str(ctx.exception))
            self.assertIn("Cannot silently substitute", str(ctx.exception))

    def test_benchmark_table_contains_three_baselines(self):
        """9. Verify 3-way benchmark table contains RF, k-NN, and Multitask MLP with audited metrics."""
        bench_df = self.service.get_benchmark_table()
        self.assertEqual(len(bench_df), 3)

        models = list(bench_df["Model"])
        self.assertTrue(any("Random Forest" in m for m in models))
        self.assertTrue(any("k-Nearest Neighbors" in m for m in models))
        self.assertTrue(any("Multitask MLP" in m for m in models))

        mlp_row = bench_df[bench_df["Model"].str.contains("Multitask MLP")].iloc[0]
        self.assertAlmostEqual(mlp_row["Building Acc (%)"], 99.82, delta=0.5)
        self.assertAlmostEqual(mlp_row["Floor Acc (%)"], 90.37, delta=0.5)
        self.assertAlmostEqual(mlp_row["Building-Floor Acc (%)"], 90.37, delta=0.5)
        self.assertAlmostEqual(mlp_row["Mean 2D Error (units)"], 12.18, delta=0.5)
        self.assertAlmostEqual(mlp_row["Median 2D Error (units)"], 10.06, delta=0.5)
        self.assertEqual(mlp_row["Model Size"], "~2.75 MB")

        # Verify units in column headers denote native dataset units, not metres
        col_names = " ".join(bench_df.columns).lower()
        self.assertIn("units", col_names)
        self.assertNotIn("metre", col_names)
        self.assertNotIn("meter", col_names)

    def test_experimental_hybrid_table_separated(self):
        """10. Verify experimental hybrid model is kept in a separate table."""
        hybrid_df = self.service.get_experimental_hybrid_table()
        self.assertIsInstance(hybrid_df, pd.DataFrame)
        self.assertGreaterEqual(len(hybrid_df), 1)
        self.assertIn("Hybrid", hybrid_df.iloc[0]["Model"])


if __name__ == "__main__":
    unittest.main()
