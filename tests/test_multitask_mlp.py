"""Unit tests for Multitask Deep Learning (MLP) Architecture and Inference.

Tests:
1. `MultitaskMLP` PyTorch model architecture and tensor output shapes.
2. `CoordinateStandardizer` zero-leakage target scaling and exact inverse-transform roundtrip.
3. `SummaryFeatureScaler` feature scaling and serialization.
4. `MultitaskLoss` weighted loss computation and backpropagation gradient flow.
5. `IndoorAIMultitaskMLP` end-to-end fit, predict, predict_proba, and checkpoint persistence.
6. Zero-leakage enforcement rejecting forbidden metadata and target columns.
"""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from src.models.deep_learning.multitask_mlp import (
    BUILDING_FLOOR_MAP,
    CoordinateStandardizer,
    IndoorAIMultitaskMLP,
    MLPHyperparameters,
    MultitaskLoss,
    MultitaskMLP,
    SummaryFeatureScaler,
)


class TestMultitaskMLPArchitecture(unittest.TestCase):
    """Test PyTorch MultitaskMLP layer shapes, forward pass, and gradient flow."""

    def setUp(self):
        self.input_dim = 1058
        self.num_buildings = 3
        self.num_floors = 5
        self.batch_size = 16
        self.model = MultitaskMLP(
            input_dim=self.input_dim,
            hidden_dims=[512, 256, 128],
            dropouts=[0.30, 0.25],
            num_buildings=self.num_buildings,
            num_floors=self.num_floors,
        )

    def test_forward_pass_output_shapes(self):
        """Verify forward pass returns expected dictionary keys and tensor dimensions."""
        x = torch.randn(self.batch_size, self.input_dim)
        outputs = self.model(x)

        self.assertIn("building", outputs)
        self.assertIn("floor", outputs)
        self.assertIn("longitude", outputs)
        self.assertIn("latitude", outputs)

        self.assertEqual(outputs["building"].shape, (self.batch_size, self.num_buildings))
        self.assertEqual(outputs["floor"].shape, (self.batch_size, self.num_floors))
        self.assertEqual(outputs["longitude"].shape, (self.batch_size, 1))
        self.assertEqual(outputs["latitude"].shape, (self.batch_size, 1))

    def test_loss_and_backward_pass(self):
        """Verify MultitaskLoss calculates finite loss and backpropagates gradients cleanly."""
        x = torch.randn(self.batch_size, self.input_dim)
        preds = self.model(x)

        targets = {
            "building": torch.randint(0, self.num_buildings, (self.batch_size,)),
            "floor": torch.randint(0, self.num_floors, (self.batch_size,)),
            "longitude": torch.randn(self.batch_size, 1),
            "latitude": torch.randn(self.batch_size, 1),
        }

        criterion = MultitaskLoss(loss_type="smooth_l1")
        total_loss, loss_dict = criterion(preds, targets)

        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertFalse(torch.isnan(total_loss))
        self.assertFalse(torch.isinf(total_loss))
        self.assertGreater(total_loss.item(), 0.0)

        # Verify component dictionary
        for key in ["loss_total", "loss_building", "loss_floor", "loss_longitude", "loss_latitude"]:
            self.assertIn(key, loss_dict)
            self.assertGreaterEqual(loss_dict[key], 0.0)

        # Backpropagation
        total_loss.backward()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient")
                self.assertFalse(torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients")


class TestCoordinateStandardizer(unittest.TestCase):
    """Test CoordinateStandardizer target scaling, inverse-transform, and serialization."""

    def setUp(self):
        np.random.seed(42)
        # Coordinates resembling UJIIndoorLoc range
        self.raw_coords = np.column_stack([
            np.random.uniform(-7600.0, -7300.0, 500),
            np.random.uniform(4864700.0, 4865000.0, 500),
        ])

    def test_fit_transform_inverse_transform(self):
        """Verify standardized targets have mean ~ 0 and std ~ 1, and inverse is exact."""
        scaler = CoordinateStandardizer(scaler_type="standard")
        scaled = scaler.fit(self.raw_coords).transform(self.raw_coords)

        # Standardized check
        np.testing.assert_allclose(np.mean(scaled, axis=0), [0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(np.std(scaled, axis=0), [1.0, 1.0], atol=1e-5)

        # Inverse transform
        reconstructed = scaler.inverse_transform(scaled)
        np.testing.assert_allclose(reconstructed, self.raw_coords, atol=1e-4)

    def test_serialization_roundtrip(self):
        """Verify CoordinateStandardizer serializes and deserializes accurately."""
        scaler = CoordinateStandardizer(scaler_type="standard")
        scaler.fit(self.raw_coords)

        state = scaler.to_dict()
        restored = CoordinateStandardizer.from_dict(state)

        test_sample = np.array([[-7500.0, 4864850.0]])
        orig_scaled = scaler.transform(test_sample)
        restored_scaled = restored.transform(test_sample)

        np.testing.assert_allclose(orig_scaled, restored_scaled, atol=1e-6)


class TestSummaryFeatureScaler(unittest.TestCase):
    """Test SummaryFeatureScaler for Variant B engineered summary features."""

    def test_summary_feature_scaling(self):
        cols = [f"WAP_{i:03d}" for i in range(1040)] + ["visible_ap_count", "rssi_mean"]
        data = np.zeros((100, 1042))
        data[:, 1040] = np.random.uniform(0, 50, 100)      # visible_ap_count
        data[:, 1041] = np.random.uniform(-105, -30, 100)  # rssi_mean

        df = pd.DataFrame(data, columns=cols)
        scaler = SummaryFeatureScaler(scaler_type="minmax")
        df_scaled = scaler.fit_transform(df, base_feature_count=1040)

        # Basic WAPs untouched
        np.testing.assert_allclose(df_scaled[cols[:1040]].values, 0.0)

        # Summary columns scaled to [0, 1]
        self.assertAlmostEqual(df_scaled["visible_ap_count"].min(), 0.0, places=4)
        self.assertAlmostEqual(df_scaled["visible_ap_count"].max(), 1.0, places=4)
        self.assertAlmostEqual(df_scaled["rssi_mean"].min(), 0.0, places=4)
        self.assertAlmostEqual(df_scaled["rssi_mean"].max(), 1.0, places=4)


class TestIndoorAIMultitaskMLP(unittest.TestCase):
    """Test end-to-end Multitask MLP training, inference, and serialization."""

    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)

        # Create synthetic dataset with 1,058 features
        self.feature_cols = [f"WAP_{i:03d}" for i in range(520)] + \
                            [f"WAP_{i:03d}_vis" for i in range(520)] + \
                            [f"summary_{i}" for i in range(18)]

        n_samples = 120
        X_mat = np.random.uniform(0.0, 1.0, (n_samples, len(self.feature_cols)))
        self.df_X = pd.DataFrame(X_mat, columns=self.feature_cols)

        self.df_y = pd.DataFrame({
            "BUILDINGID": np.random.choice([0, 1, 2], n_samples),
            "FLOOR": np.random.choice([0, 1, 2, 3], n_samples),
            "LONGITUDE": np.random.uniform(-7600.0, -7400.0, n_samples),
            "LATITUDE": np.random.uniform(4864800.0, 4864950.0, n_samples),
        })

    def test_fit_predict_pipeline(self):
        """Test training pipeline on synthetic data and verify predictions."""
        params = MLPHyperparameters(
            input_dim=len(self.feature_cols),
            hidden_dims=[64, 32, 16],
            dropouts=[0.1, 0.1],
            batch_size=16,
            epochs=3,
            patience=2,
            device="cpu",
        )
        model = IndoorAIMultitaskMLP(params=params)
        model.fit(self.df_X, self.df_y, verbose=False)

        self.assertTrue(model.is_fitted_)

        # Predictions
        preds = model.predict(self.df_X)
        self.assertIn("building", preds)
        self.assertIn("floor", preds)
        self.assertIn("building_floor", preds)
        self.assertIn("longitude", preds)
        self.assertIn("latitude", preds)

        self.assertEqual(len(preds["building"]), len(self.df_X))
        self.assertEqual(len(preds["floor"]), len(self.df_X))
        self.assertEqual(len(preds["longitude"]), len(self.df_X))
        self.assertEqual(len(preds["latitude"]), len(self.df_X))

        # Check classes in valid range
        self.assertTrue(np.all(np.isin(preds["building"], [0, 1, 2])))
        self.assertTrue(np.all(np.isin(preds["floor"], [0, 1, 2, 3, 4])))

        # Check probabilities
        probs = model.predict_proba(self.df_X)
        self.assertEqual(probs["building"].shape, (len(self.df_X), 3))
        self.assertEqual(probs["floor"].shape, (len(self.df_X), 5))
        np.testing.assert_allclose(probs["building"].sum(axis=1), 1.0, atol=1e-5)
        np.testing.assert_allclose(probs["floor"].sum(axis=1), 1.0, atol=1e-5)

    def test_checkpoint_save_and_load(self):
        """Test checkpoint saving, loading, and prediction equivalence."""
        params = MLPHyperparameters(
            input_dim=len(self.feature_cols),
            hidden_dims=[64, 32, 16],
            dropouts=[0.1, 0.1],
            batch_size=16,
            epochs=2,
            device="cpu",
        )
        model = IndoorAIMultitaskMLP(params=params)
        model.fit(self.df_X, self.df_y, verbose=False)

        preds_orig = model.predict(self.df_X.iloc[:10])

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_mlp.pt"
            model.save(model_path)
            self.assertTrue(model_path.exists())

            loaded_model = IndoorAIMultitaskMLP.load(model_path, device="cpu")
            preds_loaded = loaded_model.predict(self.df_X.iloc[:10])

            np.testing.assert_array_equal(preds_orig["building"], preds_loaded["building"])
            np.testing.assert_array_equal(preds_orig["floor"], preds_loaded["floor"])
            np.testing.assert_allclose(preds_orig["longitude"], preds_loaded["longitude"], atol=1e-4)
            np.testing.assert_allclose(preds_orig["latitude"], preds_loaded["latitude"], atol=1e-4)

    def test_zero_leakage_enforcement(self):
        """Verify model raises ValueError when forbidden target or metadata columns enter input."""
        params = MLPHyperparameters(input_dim=len(self.feature_cols), epochs=1, device="cpu")
        model = IndoorAIMultitaskMLP(params=params)

        leaky_df = self.df_X.copy()
        leaky_df["BUILDINGID"] = 0

        with self.assertRaises(ValueError) as ctx:
            model.fit(leaky_df, self.df_y, verbose=False)
        self.assertIn("CRITICAL ZERO-LEAKAGE VIOLATION", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

