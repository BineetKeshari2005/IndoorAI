"""Step detection accuracy and classification metrics evaluator (Task 22)."""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import pandas as pd

from src.core.logging import setup_logger
from src.pdr.calibration.models import StepDetectionMetrics

logger = setup_logger("StepDetectionEvaluator")


class StepDetectionEvaluator:
    """Evaluates human step detection performance against ground-truth counts and timestamps."""

    def __init__(self, timing_tolerance_ms: float = 250.0):
        self.timing_tolerance_ms = timing_tolerance_ms

    def evaluate(
        self,
        detected_steps_df: pd.DataFrame,
        ground_truth_count: Optional[int] = None,
        ground_truth_timestamps_ms: Optional[List[int]] = None
    ) -> StepDetectionMetrics:
        """Computes TP, FP, FN, Precision, Recall, F1-score, and count error."""
        total_detected = len(detected_steps_df) if not detected_steps_df.empty else 0

        # Case 1: Exact ground-truth step timestamps available
        if ground_truth_timestamps_ms and len(ground_truth_timestamps_ms) > 0:
            gt_steps = len(ground_truth_timestamps_ms)
            if total_detected == 0:
                return StepDetectionMetrics(
                    total_detected_steps=0,
                    ground_truth_steps=gt_steps,
                    true_positives=0,
                    false_positives=0,
                    false_negatives=gt_steps,
                    precision=0.0,
                    recall=0.0,
                    f1_score=0.0,
                    count_error=-gt_steps,
                    relative_count_error=1.0,
                    timing_mean_error_ms=None
                )

            det_times = detected_steps_df["timestamp_ms"].values.astype(float)
            matched_det = set()
            timing_errors: List[float] = []
            tp = 0

            for gt_t in ground_truth_timestamps_ms:
                # Find closest unmatched detected step within tolerance
                diffs = np.abs(det_times - gt_t)
                closest_idx = int(np.argmin(diffs))
                if diffs[closest_idx] <= self.timing_tolerance_ms and closest_idx not in matched_det:
                    matched_det.add(closest_idx)
                    timing_errors.append(float(diffs[closest_idx]))
                    tp += 1

            fn = gt_steps - tp
            fp = total_detected - tp

            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            mean_time_err = float(np.mean(timing_errors)) if timing_errors else None

            return StepDetectionMetrics(
                total_detected_steps=total_detected,
                ground_truth_steps=gt_steps,
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                precision=round(precision, 4),
                recall=round(recall, 4),
                f1_score=round(f1, 4),
                count_error=total_detected - gt_steps,
                relative_count_error=round(abs(total_detected - gt_steps) / gt_steps, 4),
                timing_mean_error_ms=round(mean_time_err, 2) if mean_time_err is not None else None
            )

        # Case 2: Manual total step count available
        gt_count = ground_truth_count if ground_truth_count is not None else total_detected
        if gt_count <= 0:
            gt_count = 1  # Avoid division by zero

        if total_detected >= gt_count:
            tp = gt_count
            fp = total_detected - gt_count
            fn = 0
        else:
            tp = total_detected
            fp = 0
            fn = gt_count - total_detected

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return StepDetectionMetrics(
            total_detected_steps=total_detected,
            ground_truth_steps=gt_count,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1_score=round(f1, 4),
            count_error=total_detected - gt_count,
            relative_count_error=round(abs(total_detected - gt_count) / gt_count, 4),
            timing_mean_error_ms=None
        )
