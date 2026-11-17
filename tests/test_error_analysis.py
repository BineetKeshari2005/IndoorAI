"""
Unit Tests for Localization Error Analysis (Task 10).
"""

import json
import os
import unittest
import numpy as np
import pandas as pd

from src.analysis.localization_error_analysis import (
    analyze_building_and_floor_errors,
    analyze_coordinate_errors,
    analyze_confidence_and_uncertainty,
    investigate_difficult_fingerprints,
)
from src.models.traditional_ml.knn import IndoorAIKNN


class TestErrorAnalysis(unittest.TestCase):
    """Test suite for localization error diagnostics and artifact verification."""

    @classmethod
    def setUpClass(cls):
        cls.rf_preds = pd.read_parquet("experiments/random_forest/predictions/val_predictions.parquet")
        cls.knn_preds = pd.read_parquet("experiments/knn/predictions/val_predictions.parquet")
        cls.X_val = pd.read_parquet("data/processed/ujiindoorloc/feature_engineered/X_validation_engineered.parquet")
        cls.y_val = pd.read_parquet("data/processed/ujiindoorloc/y_validation.parquet")
        cls.meta_val = pd.read_parquet("data/processed/ujiindoorloc/metadata_validation.parquet")
        cls.X_train = pd.read_parquet("data/processed/ujiindoorloc/feature_engineered/X_train_engineered.parquet")
        cls.knn_model = IndoorAIKNN.load("experiments/knn/models")

    def test_building_and_floor_error_grouping(self):
        """Verify structural classification error grouping and adjacent floor dominance."""
        res = analyze_building_and_floor_errors(self.rf_preds, self.knn_preds, output_dir="/tmp/test_err_metrics")

        # Building breakdown: 3 buildings
        self.assertEqual(len(res["building_error_summary"]), 3)

        # Building 1 should have highest floor error count
        b1_data = [b for b in res["building_error_summary"] if b["building_id"] == 1][0]
        b0_data = [b for b in res["building_error_summary"] if b["building_id"] == 0][0]
        self.assertGreater(b1_data["rf_floor_errors"], b0_data["rf_floor_errors"])

        # Floor confusion matrices: 5x5
        cm_rf = np.array(res["floor_confusion_matrix_rf"])
        cm_knn = np.array(res["floor_confusion_matrix_knn"])
        self.assertEqual(cm_rf.shape, (5, 5))
        self.assertEqual(cm_knn.shape, (5, 5))

        # Adjacent floor errors must represent >85% of all floor errors
        rf_adj_pct = res["floor_adjacency_breakdown"]["random_forest"]["adjacent_error_pct"]
        knn_adj_pct = res["floor_adjacency_breakdown"]["knn"]["adjacent_error_pct"]
        self.assertGreater(rf_adj_pct, 85.0)
        self.assertGreater(knn_adj_pct, 85.0)

    def test_coordinate_error_calculations_and_top50(self):
        """Verify coordinate error calculations and Top 50 high error CSV generation."""
        res = analyze_coordinate_errors(
            self.rf_preds, self.knn_preds, self.X_val, self.meta_val, self.y_val, output_dir="/tmp/test_err_metrics"
        )

        csv_path = "/tmp/test_err_metrics/high_error_samples.csv"
        self.assertTrue(os.path.exists(csv_path))

        df_high = pd.read_csv(csv_path)
        # 50 samples for RF + 50 samples for k-NN = 100 rows
        self.assertEqual(len(df_high), 100)
        self.assertEqual(len(df_high[df_high["model"] == "Random Forest"]), 50)
        self.assertEqual(len(df_high[df_high["model"] == "k-Nearest Neighbors"]), 50)

        # Check Euclidean error calculation accuracy for first row
        first_row = df_high.iloc[0]
        expected_err = np.sqrt(
            (first_row["true_longitude"] - first_row["pred_longitude"]) ** 2 +
            (first_row["true_latitude"] - first_row["pred_latitude"]) ** 2
        )
        self.assertAlmostEqual(first_row["euclidean_error"], expected_err, places=3)

        # Strongest signal negative correlation with error
        rf_corr = res["signal_feature_correlations"]["random_forest"]["strongest_rssi_1"]
        knn_corr = res["signal_feature_correlations"]["knn"]["strongest_rssi_1"]
        self.assertLess(rf_corr, 0.0)
        self.assertLess(knn_corr, 0.0)

    def test_difficult_fingerprint_characteristics(self):
        """Verify that high-error fingerprints share fewer APs with training data than low-error ones."""
        knn_eucl = np.sqrt(
            (self.y_val["LONGITUDE"].values - self.knn_preds["pred_longitude"]) ** 2 +
            (self.y_val["LATITUDE"].values - self.knn_preds["pred_latitude"]) ** 2
        ).values

        diff = investigate_difficult_fingerprints(self.X_val, self.X_train, self.knn_model, knn_eucl)

        high_jaccard = diff["jaccard_overlap_with_training_neighbors"]["high_error_mean"]
        low_jaccard = diff["jaccard_overlap_with_training_neighbors"]["low_error_mean"]

        self.assertLess(high_jaccard, low_jaccard)
        self.assertLess(diff["jaccard_overlap_with_training_neighbors"]["correlation_with_error"], 0.0)

    def test_confidence_and_uncertainty_calibration(self):
        """Verify calibration of classification confidence and neighbor spatial dispersion."""
        with open("experiments/error_analysis/metrics/error_analysis.json") as f:
            data = json.load(f)

        conf = data["confidence_and_uncertainty"]
        calib = conf["classification_confidence_calibration"]

        # Higher confidence should correspond to higher empirical accuracy
        acc_03 = [b["actual_accuracy"] for b in calib if b["confidence_threshold"] == 0.3][0]
        acc_09 = [b["actual_accuracy"] for b in calib if b["confidence_threshold"] == 0.9][0]
        self.assertGreater(acc_09, acc_03)

        # Spatial dispersion correlation with coordinate error must be strongly positive (> 0.40)
        disp_corr = conf["spatial_dispersion_analysis"]["correlation_with_coordinate_error"]
        self.assertGreater(disp_corr, 0.40)

    def test_artifacts_presence(self):
        """Verify that all Task 10 artifacts exist and are non-empty."""
        expected_files = [
            "experiments/error_analysis/metrics/error_analysis.json",
            "experiments/error_analysis/metrics/high_error_samples.csv",
            "experiments/error_analysis/metrics/floor_confusion_analysis.csv",
            "experiments/error_analysis/metrics/building_error_analysis.csv",
            "experiments/error_analysis/metrics/hybrid_model_results.json",
            "experiments/error_analysis/predictions/val_predictions.parquet",
            "experiments/error_analysis/predictions/val_predictions.csv",
            "experiments/error_analysis/reports/experiment_report.md",
            "experiments/error_analysis/plots/floor_error_heatmap.png",
            "experiments/error_analysis/plots/building_coordinate_error_comparison.png",
            "experiments/error_analysis/plots/coordinate_error_distribution.png",
            "experiments/error_analysis/plots/actual_vs_predicted_positions.png",
            "experiments/error_analysis/plots/visible_aps_vs_coordinate_error.png",
            "experiments/error_analysis/plots/rf_vs_knn_error_comparison.png",
            "experiments/error_analysis/plots/hybrid_vs_standalone_comparison.png",
        ]
        for fpath in expected_files:
            self.assertTrue(os.path.exists(fpath), f"Missing artifact: {fpath}")
            self.assertGreater(os.path.getsize(fpath), 0, f"Empty artifact: {fpath}")


if __name__ == "__main__":
    unittest.main()

