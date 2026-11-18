"""Leakage-Resistant Session-Level Wi-Fi Localization Evaluator.

Enforces session-based train/test partitioning to prevent temporal leakage
from repeated stationary scans, computing building, floor, location, and coordinate errors.
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.core.logging import setup_logger
from src.wifi_localization.knn_localizer import KnnLocalizer
from src.wifi_localization.models import (
    EvaluationMetrics,
    RadioMap,
    ReferenceLocation,
    SurveySweep,
)
from src.wifi_localization.probabilistic_localizer import ProbabilisticLocalizer
from src.wifi_localization.radio_map_builder import RadioMapBuilder

logger = setup_logger("WifiLocalizationEvaluator")


class WifiLocalizationEvaluator:
    """Evaluates Wi-Fi localization accuracy under strict session-level train/test separation."""

    def __init__(
        self,
        algorithm: str = "weighted_knn",
        k: int = 3,
        distance_metric: str = "euclidean",
    ) -> None:
        """Initialize evaluator.

        Args:
            algorithm: 'weighted_knn' or 'probabilistic'.
            k: Top-k nearest neighbors / candidates.
            distance_metric: Distance metric for k-NN ('euclidean' or 'manhattan').
        """
        self.algorithm = algorithm.lower()
        self.k = k
        self.distance_metric = distance_metric

    def evaluate_session_split(
        self,
        sweeps: List[SurveySweep],
        locations: Dict[str, ReferenceLocation],
        train_session_ids: List[str],
        test_session_ids: List[str],
        site_id: str = "SITE-EVAL",
    ) -> EvaluationMetrics:
        """Run evaluation with explicit train and test session partitions.

        Args:
            sweeps: All available survey sweeps.
            locations: Reference location dictionary.
            train_session_ids: Sessions used exclusively to build the training radio map.
            test_session_ids: Sessions used exclusively for evaluation queries.
            site_id: Site identifier for radio map building.

        Returns:
            EvaluationMetrics summarizing localization performance.
        """
        # Verify strict session separation (zero intersection)
        overlap = set(train_session_ids) & set(test_session_ids)
        if overlap:
            raise ValueError(
                f"Data leakage detected! Sessions {overlap} appear in both train and test partitions."
            )

        train_sweeps = [s for s in sweeps if s.session_id in train_session_ids]
        test_sweeps = [s for s in sweeps if s.session_id in test_session_ids]

        logger.info(
            f"Evaluating Wi-Fi localization: {len(train_sweeps)} train sweeps "
            f"({len(train_session_ids)} sessions) vs {len(test_sweeps)} test sweeps "
            f"({len(test_session_ids)} sessions)."
        )

        if not train_sweeps:
            raise ValueError("Train partition contains zero survey sweeps.")
        if not test_sweeps:
            raise ValueError("Test partition contains zero survey sweeps.")

        # 1. Build training radio map
        builder = RadioMapBuilder(site_id=site_id, version="eval-train")
        train_radio_map = builder.build_from_sweeps(
            train_sweeps, locations, description="Evaluation training radio map"
        )

        # 2. Instantiate localizer
        if self.algorithm == "probabilistic":
            localizer: Union[KnnLocalizer, ProbabilisticLocalizer] = ProbabilisticLocalizer(
                radio_map=train_radio_map, k=self.k
            )
        else:
            localizer = KnnLocalizer(
                radio_map=train_radio_map, k=self.k, distance_metric=self.distance_metric
            )

        # 3. Evaluate each test sweep
        return self.evaluate_test_sweeps(
            test_sweeps=test_sweeps,
            localizer=localizer,
            locations=locations,
            train_session_ids=train_session_ids,
            test_session_ids=test_session_ids,
        )

    def evaluate_test_sweeps(
        self,
        test_sweeps: List[SurveySweep],
        localizer: Union[KnnLocalizer, ProbabilisticLocalizer],
        locations: Dict[str, ReferenceLocation],
        train_session_ids: Optional[List[str]] = None,
        test_session_ids: Optional[List[str]] = None,
    ) -> EvaluationMetrics:
        """Evaluate a localizer on a set of test sweeps with known ground truth locations."""
        total_samples = 0
        correct_building = 0
        correct_floor = 0
        correct_building_floor = 0
        correct_loc_id = 0
        correct_top_3_loc = 0
        correct_top_5_loc = 0
        low_confidence_count = 0

        coordinate_errors: List[float] = []
        confusion_floors: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        errors_by_loc: Dict[str, List[float]] = defaultdict(list)

        for sweep in test_sweeps:
            if sweep.location_id not in locations:
                continue

            true_loc = locations[sweep.location_id]
            query_rssi = sweep.to_rssi_dict()

            pred = localizer.localize(query_rssi)

            total_samples += 1
            if pred.is_low_confidence:
                low_confidence_count += 1

            # Building & Floor accuracy
            b_match = str(pred.predicted_building_id) == str(true_loc.building_id)
            f_match = int(pred.predicted_floor_id) == int(true_loc.floor_id)

            if b_match:
                correct_building += 1
            if f_match:
                correct_floor += 1
            if b_match and f_match:
                correct_building_floor += 1

            # Floor confusion matrix
            true_f_str = f"Floor {true_loc.floor_id}"
            pred_f_str = f"Floor {pred.predicted_floor_id}"
            confusion_floors[true_f_str][pred_f_str] += 1

            # Candidate match accuracy
            candidate_ids = [c.location_id for c in pred.candidate_matches]
            if candidate_ids and candidate_ids[0] == true_loc.location_id:
                correct_loc_id += 1
            if true_loc.location_id in candidate_ids[:3]:
                correct_top_3_loc += 1
            if true_loc.location_id in candidate_ids[:5]:
                correct_top_5_loc += 1

            # 2D Euclidean coordinate error
            dx = pred.predicted_x_m - true_loc.x_m
            dy = pred.predicted_y_m - true_loc.y_m
            err_m = math.sqrt(dx * dx + dy * dy)
            coordinate_errors.append(err_m)
            errors_by_loc[true_loc.location_id].append(err_m)

        if total_samples == 0:
            return EvaluationMetrics(
                total_test_samples=0,
                building_accuracy=0.0,
                floor_accuracy=0.0,
                building_floor_accuracy=0.0,
                location_id_accuracy=0.0,
                top_3_location_accuracy=0.0,
                top_5_location_accuracy=0.0,
                mean_coordinate_error_m=0.0,
                median_coordinate_error_m=0.0,
                p95_coordinate_error_m=0.0,
                max_coordinate_error_m=0.0,
                low_confidence_rate=0.0,
                train_session_ids=train_session_ids or [],
                test_session_ids=test_session_ids or [],
                confusion_matrix_floors={},
                per_location_errors=[],
            )

        err_arr = np.array(coordinate_errors, dtype=np.float64)

        # Per-location summaries
        per_loc_list: List[Dict[str, Any]] = []
        for loc_id, errs in errors_by_loc.items():
            loc = locations[loc_id]
            per_loc_list.append(
                {
                    "location_id": loc_id,
                    "building_id": loc.building_id,
                    "floor_id": loc.floor_id,
                    "x_m": loc.x_m,
                    "y_m": loc.y_m,
                    "sample_count": len(errs),
                    "mean_error_m": round(float(np.mean(errs)), 2),
                    "median_error_m": round(float(np.median(errs)), 2),
                }
            )

        return EvaluationMetrics(
            total_test_samples=total_samples,
            building_accuracy=round(correct_building / total_samples, 4),
            floor_accuracy=round(correct_floor / total_samples, 4),
            building_floor_accuracy=round(correct_building_floor / total_samples, 4),
            location_id_accuracy=round(correct_loc_id / total_samples, 4),
            top_3_location_accuracy=round(correct_top_3_loc / total_samples, 4),
            top_5_location_accuracy=round(correct_top_5_loc / total_samples, 4),
            mean_coordinate_error_m=round(float(np.mean(err_arr)), 2),
            median_coordinate_error_m=round(float(np.median(err_arr)), 2),
            p95_coordinate_error_m=round(float(np.percentile(err_arr, 95)), 2),
            max_coordinate_error_m=round(float(np.max(err_arr)), 2),
            low_confidence_rate=round(low_confidence_count / total_samples, 4),
            train_session_ids=train_session_ids or [],
            test_session_ids=test_session_ids or [],
            confusion_matrix_floors={k: dict(v) for k, v in confusion_floors.items()},
            per_location_errors=sorted(per_loc_list, key=lambda x: x["location_id"]),
        )
