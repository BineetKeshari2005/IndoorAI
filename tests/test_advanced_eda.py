"""Unit and validation tests for advanced Wi-Fi fingerprint exploratory data analysis."""

from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.data.advanced_eda import (
    UNAVAILABLE_WAP_VALUE,
    analyze_access_points,
    analyze_ap_cooccurrence,
    analyze_building_wifi,
    analyze_duplicates,
    analyze_fingerprint_distinguishability,
    analyze_floor_wifi,
    analyze_spatial_distribution,
    get_wap_and_context_cols,
)


class TestAdvancedEDA(unittest.TestCase):
    """Test suite ensuring accuracy of Wi-Fi signal analysis, co-occurrence, and spatial metrics."""

    def setUp(self) -> None:
        """Create controlled synthetic fixtures with known ground truth properties."""
        self.mock_df = pd.DataFrame({
            "WAP001": [-60, -70, 100, 100, -80, 100],
            "WAP002": [100, -75, -65, 100, -85, 100],
            "WAP003": [-50, -55, -60, 100, 100, 100],
            "WAP004": [100, 100, 100, 100, 100, 100],  # Constant 100
            "LONGITUDE": [-7500.0, -7510.0, -7520.0, -7600.0, -7610.0, -7600.0],
            "LATITUDE": [4864800.0, 4864810.0, 4864820.0, 4864900.0, 4864910.0, 4864900.0],
            "FLOOR": [0, 0, 1, 0, 1, 0],
            "BUILDINGID": [0, 0, 0, 1, 1, 1],
            "SPACEID": [101, 102, 103, 201, 202, 201],
            "RELATIVEPOSITION": [1, 1, 2, 1, 2, 1],
            "USERID": [1, 1, 2, 3, 3, 3],
            "PHONEID": [10, 10, 11, 12, 12, 12],
            "TIMESTAMP": [1000, 1001, 1002, 2000, 2001, 2000],  # Row 3 and Row 5 are exact duplicates!
        })

    def test_column_identification(self) -> None:
        """Verify partition of WAP columns and context columns."""
        waps, context = get_wap_and_context_cols(self.mock_df)
        self.assertEqual(waps, ["WAP001", "WAP002", "WAP003", "WAP004"])
        self.assertEqual(len(context), 9)
        self.assertIn("BUILDINGID", context)
        self.assertIn("FLOOR", context)

    def test_access_point_analysis(self) -> None:
        """Verify per-WAP detection counts, detection %, RSSI stats, and distinct locations."""
        ap_res = analyze_access_points(self.mock_df)
        self.assertEqual(len(ap_res), 4)

        # WAP001: visible in rows 0, 1, 4 (3 detections, values: -60, -70, -80)
        wap1 = ap_res[ap_res["wap_id"] == "WAP001"].iloc[0]
        self.assertEqual(wap1["detected_count"], 3)
        self.assertAlmostEqual(wap1["detection_pct"], 50.0)
        self.assertEqual(wap1["min_rssi"], -80.0)
        self.assertEqual(wap1["max_rssi"], -60.0)
        self.assertEqual(wap1["mean_rssi"], -70.0)
        self.assertEqual(wap1["median_rssi"], -70.0)
        self.assertFalse(wap1["is_constant_100"])

        # WAP004: constant 100 (never observed)
        wap4 = ap_res[ap_res["wap_id"] == "WAP004"].iloc[0]
        self.assertEqual(wap4["detected_count"], 0)
        self.assertTrue(wap4["is_constant_100"])
        self.assertTrue(pd.isna(wap4["mean_rssi"]))

    def test_building_and_floor_analysis(self) -> None:
        """Verify building-wise and floor-wise aggregation."""
        bld_df = analyze_building_wifi(self.mock_df)
        self.assertEqual(len(bld_df), 2)  # Buildings 0 and 1
        
        bld0 = bld_df[bld_df["building_id"] == 0].iloc[0]
        self.assertEqual(bld0["sample_count"], 3)

        flr_df = analyze_floor_wifi(self.mock_df)
        # Groups: (0, 0), (0, 1), (1, 0), (1, 1) -> 4 groups
        self.assertEqual(len(flr_df), 4)

    def test_ap_cooccurrence_and_jaccard(self) -> None:
        """Verify pairwise co-occurrence counts and Jaccard similarity."""
        # Top 3 WAPs (excluding WAP004 which has 0 detections)
        cooc, jaccard = analyze_ap_cooccurrence(self.mock_df, top_n=3)
        
        # WAP001 detected in {0, 1, 4}
        # WAP003 detected in {0, 1, 2}
        # Co-occurrence(WAP001, WAP003) = {0, 1} -> count 2
        # Union = {0, 1, 2, 4} -> count 4
        # Jaccard = 2/4 = 0.5
        self.assertEqual(cooc.loc["WAP001", "WAP003"], 2)
        self.assertAlmostEqual(jaccard.loc["WAP001", "WAP003"], 0.5)

    def test_fingerprint_distinguishability(self) -> None:
        """Verify intra-building similarity exceeds inter-building similarity."""
        dist = analyze_fingerprint_distinguishability(self.mock_df, sample_size_per_bld=2, random_seed=42)
        self.assertIn("visibility_jaccard_similarity", dist)
        self.assertIn("signal_cosine_similarity", dist)
        
        intra_j = dist["visibility_jaccard_similarity"]["intra_building_mean"]
        inter_j = dist["visibility_jaccard_similarity"]["inter_building_mean"]
        self.assertGreater(intra_j, inter_j)

    def test_duplicate_analysis(self) -> None:
        """Verify duplicate detection identifies exact repeated records."""
        # In mock_df: Row 3 and Row 5 are identical in all fields
        dup_res = analyze_duplicates(self.mock_df)
        self.assertEqual(dup_res["redundant_duplicate_rows_count"], 1)
        self.assertEqual(dup_res["total_rows_involved_in_duplicate_groups"], 2)
        self.assertEqual(dup_res["user_concentration"]["3"], 1)

    def test_spatial_analysis_unverified_crs(self) -> None:
        """Verify spatial summary documents CRS uncertainty without converting coordinates."""
        spatial = analyze_spatial_distribution(self.mock_df)
        self.assertIn("global_coordinate_bounds", spatial)
        self.assertIn("coordinate_reference_system_investigation", spatial)
        crs_status = spatial["coordinate_reference_system_investigation"]["status"]
        self.assertEqual(crs_status, "UNVERIFIED_PRESERVED_METRIC_COORDINATES")
        
        # Original values preserved
        self.assertEqual(spatial["global_coordinate_bounds"]["longitude_min"], -7610.0)
        self.assertEqual(spatial["global_coordinate_bounds"]["longitude_max"], -7500.0)


if __name__ == "__main__":
    unittest.main()
