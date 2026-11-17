"""
Unit Tests for IndoorAI Model Benchmarking and Selection (Task 09).

Tests cover:
1. Validation sample alignment verification
2. Raw dataset distinct phone and user counts
3. k-NN configuration selection criteria and float-precision audit
4. Independent metric recalculation from saved predictions
5. Hierarchical breakdown integrity across buildings and floors
6. Error diagnostics and mutual failure mode calculations
7. Unified comparison table generation
8. Evidence-based model selection recommendations
9. Benchmark artifact presence and formatting
"""

import json
import os
import unittest
import numpy as np
import pandas as pd

from src.evaluation.model_benchmark import (
    audit_dataset_phone_and_user_counts,
    build_unified_comparison_table,
    compute_error_overlap_and_diagnostics,
    compute_hierarchical_comparison,
    formulate_model_recommendations,
    investigate_knn_configuration_selection,
    recalculate_metrics,
    verify_validation_sample_alignment,
)


class TestModelBenchmark(unittest.TestCase):
    """Test suite for IndoorAI model benchmarking and verification."""

    @classmethod
    def setUpClass(cls):
        cls.rf_preds_path = "experiments/random_forest/predictions/val_predictions.parquet"
        cls.knn_preds_path = "experiments/knn/predictions/val_predictions.parquet"
        cls.rf_results_path = "experiments/random_forest/metrics/random_forest_results.json"
        cls.knn_results_path = "experiments/knn/metrics/knn_results.json"
        cls.cv_grid_path = "experiments/knn/metrics/cv_grid_results.csv"

        cls.rf_preds = pd.read_parquet(cls.rf_preds_path)
        cls.knn_preds = pd.read_parquet(cls.knn_preds_path)

        with open(cls.rf_results_path) as f:
            cls.rf_json = json.load(f)
        with open(cls.knn_results_path) as f:
            cls.knn_json = json.load(f)

    def test_validation_sample_alignment(self):
        """Verify that RF and k-NN predictions are perfectly aligned on the same 1,111 samples."""
        alignment = verify_validation_sample_alignment(self.rf_preds, self.knn_preds)
        self.assertTrue(alignment["is_aligned"])
        self.assertEqual(alignment["sample_count"], 1111)
        self.assertEqual(alignment["building_mismatches"], 0)
        self.assertEqual(alignment["floor_mismatches"], 0)
        self.assertEqual(alignment["max_longitude_diff"], 0.0)
        self.assertEqual(alignment["max_latitude_diff"], 0.0)

    def test_phone_and_user_counts_audit(self):
        """Verify accurate distinct PHONEID and USERID counts in raw datasets."""
        audit = audit_dataset_phone_and_user_counts()

        # Training assertions
        self.assertEqual(audit["training"]["sample_count"], 19937)
        self.assertEqual(audit["training"]["distinct_user_count"], 18)
        self.assertEqual(audit["training"]["distinct_phone_count"], 16)
        self.assertEqual(audit["training"]["max_phone_id"], 24)
        self.assertEqual(audit["training"]["min_phone_id"], 1)

        # Validation assertions
        self.assertEqual(audit["validation"]["sample_count"], 1111)
        self.assertEqual(audit["validation"]["distinct_user_count"], 1)
        self.assertEqual(audit["validation"]["user_ids"], [0])
        self.assertEqual(audit["validation"]["distinct_phone_count"], 11)

        # Dataset union assertions
        self.assertEqual(audit["comparison"]["total_distinct_phones_combined"], 25)
        self.assertEqual(audit["comparison"]["shared_phone_count"], 2)
        self.assertEqual(audit["comparison"]["shared_phone_ids"], [13, 14])
        self.assertEqual(audit["comparison"]["val_only_phone_count"], 9)
        self.assertEqual(audit["comparison"]["train_only_phone_count"], 14)

    def test_knn_selection_investigation(self):
        """Verify the mathematical basis for selecting k=7 in Task 08 cross-validation."""
        investigation = investigate_knn_configuration_selection(self.cv_grid_path)
        k7 = investigation["k7_metrics"]
        k11 = investigation["k11_metrics"]
        k5 = investigation["k5_metrics"]

        # Primary objective: k=7 achieves the global minimum mean Euclidean error
        self.assertLess(k7["mean_euclidean_error"], k11["mean_euclidean_error"])
        self.assertLess(k7["mean_euclidean_error"], k5["mean_euclidean_error"])
        self.assertAlmostEqual(k7["mean_euclidean_error"], 10.039661, places=5)

        # Statistical significance: accuracy difference (83.03% vs 83.19%) is within standard deviation (6.14%)
        acc_diff = abs(k5["mean_bf_acc"] - k7["mean_bf_acc"])
        self.assertLess(acc_diff, k7["std_bf_acc"])

    def test_recalculate_metrics_accuracy(self):
        """Verify that recalculated metrics match the saved validation performance JSONs."""
        rf_recalc = recalculate_metrics(self.rf_preds, model_name="Random Forest", is_rf=True)
        knn_recalc = recalculate_metrics(self.knn_preds, model_name="k-NN", is_rf=False)

        # Check Random Forest metrics
        rf_val_json = self.rf_json["validation_performance"]["variant_b"]
        self.assertAlmostEqual(rf_recalc["building"]["accuracy_pct"], rf_val_json["building"]["accuracy_pct"], places=1)
        self.assertAlmostEqual(rf_recalc["floor"]["accuracy_pct"], rf_val_json["floor"]["accuracy_pct"], places=1)
        self.assertAlmostEqual(rf_recalc["building_floor"]["accuracy_pct"], rf_val_json["building_floor"]["accuracy_pct"], places=1)
        self.assertAlmostEqual(rf_recalc["coordinates"]["euclidean_error"]["mean"], rf_val_json["coordinates"]["euclidean_error"]["mean"], places=2)
        self.assertAlmostEqual(rf_recalc["coordinates"]["euclidean_error"]["median"], rf_val_json["coordinates"]["euclidean_error"]["median"], places=2)

        # Check k-NN metrics
        knn_val_json = self.knn_json["validation_performance"]
        self.assertAlmostEqual(knn_recalc["building"]["accuracy_pct"], knn_val_json["building"]["accuracy_pct"], places=1)
        self.assertAlmostEqual(knn_recalc["floor"]["accuracy_pct"], knn_val_json["floor"]["accuracy_pct"], places=1)
        self.assertAlmostEqual(knn_recalc["building_floor"]["accuracy_pct"], knn_val_json["building_floor"]["accuracy_pct"], places=1)
        self.assertAlmostEqual(knn_recalc["coordinates"]["euclidean_error"]["mean"], knn_val_json["coordinates"]["euclidean_error"]["mean"], places=2)
        self.assertAlmostEqual(knn_recalc["coordinates"]["euclidean_error"]["median"], knn_val_json["coordinates"]["euclidean_error"]["median"], places=2)

    def test_hierarchical_breakdown_integrity(self):
        """Verify hierarchical breakdowns partition all 1,111 validation samples across buildings and floors."""
        hier = compute_hierarchical_comparison(self.rf_preds, self.knn_preds)

        # Building breakdown
        b_counts = [b["sample_count"] for b in hier["by_building"].values()]
        self.assertEqual(sum(b_counts), 1111)
        self.assertEqual(len(hier["by_building"]), 3)

        # In all 3 buildings, k-NN achieves lower mean Euclidean error
        for b_name, b_data in hier["by_building"].items():
            self.assertLess(
                b_data["knn"]["euclidean_mean"],
                b_data["rf"]["euclidean_mean"],
                f"k-NN did not achieve lower mean Euclidean error in {b_name}"
            )

        # In all 3 buildings, RF achieves higher floor accuracy
        for b_name, b_data in hier["by_building"].items():
            self.assertGreater(
                b_data["rf"]["floor_acc"],
                b_data["knn"]["floor_acc"],
                f"RF did not achieve higher floor accuracy in {b_name}"
            )

        # Floor breakdown
        f_counts = [f["sample_count"] for f in hier["by_floor"].values()]
        self.assertEqual(sum(f_counts), 1111)
        self.assertEqual(len(hier["by_floor"]), 5)

    def test_error_overlap_diagnostics(self):
        """Verify error overlaps and failure mode partitions sum exactly to total sample count."""
        diag = compute_error_overlap_and_diagnostics(self.rf_preds, self.knn_preds)
        overlap = diag["floor_overlap"]

        sum_parts = (
            overlap["both_correct_count"] +
            overlap["both_wrong_count"] +
            overlap["rf_only_correct_count"] +
            overlap["knn_only_correct_count"]
        )
        self.assertEqual(sum_parts, 1111)
        self.assertGreater(overlap["potential_oracle_ensemble_acc"], 90.0)

    def test_unified_comparison_table_completeness(self):
        """Verify that build_unified_comparison_table produces a complete 12-row benchmark table."""
        rf_recalc = recalculate_metrics(self.rf_preds, model_name="Random Forest", is_rf=True)
        knn_recalc = recalculate_metrics(self.knn_preds, model_name="k-NN", is_rf=False)

        table = build_unified_comparison_table(
            rf_recalc, knn_recalc, {}, {}, rf_size_mb=561.06, knn_size_mb=17.68
        )
        self.assertEqual(len(table), 12)
        self.assertEqual(table.shape[1], 6)
        self.assertFalse(table.isnull().values.any())

    def test_model_recommendations(self):
        """Verify evidence-based model recommendations for each localization task."""
        recs = formulate_model_recommendations()

        self.assertIn("Random Forest", recs["floor_identification"]["recommended_primary"])
        self.assertIn("k-NN", recs["coordinate_prediction"]["recommended_primary"])
        self.assertIn("Two-Stage Hierarchical Hybrid", recs["hybrid_architecture_proposal"]["concept"])

    def test_benchmark_artifacts_exist(self):
        """Verify that all required benchmark artifacts exist and are non-empty."""
        expected_files = [
            "experiments/model_benchmark/metrics/benchmark_results.json",
            "experiments/model_benchmark/metrics/benchmark_comparison.csv",
            "experiments/model_benchmark/plots/model_accuracy_comparison.png",
            "experiments/model_benchmark/plots/coordinate_error_comparison.png",
            "experiments/model_benchmark/plots/building_floor_comparison.png",
            "experiments/model_benchmark/reports/model_benchmark_report.md",
        ]
        for path in expected_files:
            self.assertTrue(os.path.exists(path), f"Artifact missing: {path}")
            self.assertGreater(os.path.getsize(path), 0, f"Artifact empty: {path}")


if __name__ == "__main__":
    unittest.main()

