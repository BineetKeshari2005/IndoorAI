"""Heading and direction evaluation engine with circular statistics and wraparound handling (Task 22)."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.logging import setup_logger
from src.pdr.calibration.models import HeadingMetrics

logger = setup_logger("HeadingEvaluator")


def circular_difference_deg(angle_est_deg: float, angle_ref_deg: float) -> float:
    """Calculates signed circular angular difference in [-180, 180] degrees.

    Handles wraparound seamlessly (e.g. 1 deg - 359 deg = +2 deg).
    """
    rad_est = math.radians(angle_est_deg)
    rad_ref = math.radians(angle_ref_deg)
    diff_rad = math.atan2(math.sin(rad_est - rad_ref), math.cos(rad_est - rad_ref))
    return math.degrees(diff_rad)


class HeadingEvaluator:
    """Evaluates estimated trajectory heading against manual or geometric reference headings."""

    def evaluate(
        self,
        trajectory_df: pd.DataFrame,
        reference_start_heading_deg: Optional[float] = None,
        reference_end_heading_deg: Optional[float] = None,
        reference_segments: Optional[List[Tuple[float, float, float]]] = None  # (start_s, end_s, ref_heading_deg)
    ) -> HeadingMetrics:
        """Computes absolute heading errors, circular statistics, and drift rate over time."""
        if trajectory_df.empty or "heading_deg" not in trajectory_df.columns:
            return HeadingMetrics(
                mean_absolute_error_deg=0.0,
                median_absolute_error_deg=0.0,
                p95_absolute_error_deg=0.0,
                circular_mean_error_deg=0.0,
                circular_std_deg=0.0,
                drift_rate_deg_per_sec=0.0,
                drift_rate_deg_per_min=0.0,
                num_heading_samples=0
            )

        df = trajectory_df.sort_values("timestamp_ms").reset_index(drop=True)
        headings = df["heading_deg"].values.astype(float)
        timestamps_s = (df["timestamp_ms"].values.astype(float) - df["timestamp_ms"].iloc[0]) / 1000.0
        n_samples = len(headings)

        abs_errors: List[float] = []
        signed_errors: List[float] = []

        # 1. Compare against piecewise segments or start/end headings
        if reference_segments and len(reference_segments) > 0:
            for t_s, h_est in zip(timestamps_s, headings):
                matched_ref = None
                for seg_start, seg_end, ref_h in reference_segments:
                    if seg_start <= t_s <= seg_end:
                        matched_ref = ref_h
                        break
                if matched_ref is not None:
                    diff = circular_difference_deg(h_est, matched_ref)
                    signed_errors.append(diff)
                    abs_errors.append(abs(diff))
        elif reference_start_heading_deg is not None:
            # Single reference direction (e.g. straight line trial along known corridor)
            ref_h = reference_start_heading_deg
            for h_est in headings:
                diff = circular_difference_deg(h_est, ref_h)
                signed_errors.append(diff)
                abs_errors.append(abs(diff))

        # Default fallback if no explicit segment: evaluate start vs end drift
        if len(abs_errors) == 0:
            # Drift calculation relative to initial trajectory heading
            h_initial = headings[0]
            for h_est in headings:
                diff = circular_difference_deg(h_est, h_initial)
                signed_errors.append(diff)
                abs_errors.append(abs(diff))

        # 2. Compute circular statistics
        rad_signed = np.radians(signed_errors)
        sin_sum = np.sum(np.sin(rad_signed))
        cos_sum = np.sum(np.cos(rad_signed))
        circ_mean_rad = math.atan2(sin_sum, cos_sum)
        r_bar = math.sqrt(sin_sum**2 + cos_sum**2) / len(signed_errors) if len(signed_errors) > 0 else 1.0
        circ_std_rad = math.sqrt(max(0.0, -2.0 * math.log(max(1e-6, r_bar))))

        # 3. Compute drift rate
        total_time_s = timestamps_s[-1] if len(timestamps_s) > 1 else 1.0
        if reference_start_heading_deg is not None and reference_end_heading_deg is not None:
            err_start = abs(circular_difference_deg(headings[0], reference_start_heading_deg))
            err_end = abs(circular_difference_deg(headings[-1], reference_end_heading_deg))
            total_drift_deg = abs(err_end - err_start)
        else:
            total_drift_deg = abs(circular_difference_deg(headings[-1], headings[0]))

        drift_per_sec = total_drift_deg / max(1.0, total_time_s)
        drift_per_min = drift_per_sec * 60.0

        # Magnetic disturbance ratio
        mag_ratio = 0.0
        if "heading_source" in df.columns:
            disturbed = (df["heading_source"] == "MAGNETOMETER_DISTURBED").sum()
            mag_ratio = round(float(disturbed / n_samples), 3)

        return HeadingMetrics(
            mean_absolute_error_deg=round(float(np.mean(abs_errors)), 2),
            median_absolute_error_deg=round(float(np.median(abs_errors)), 2),
            p95_absolute_error_deg=round(float(np.percentile(abs_errors, 95)), 2),
            circular_mean_error_deg=round(math.degrees(circ_mean_rad), 2),
            circular_std_deg=round(math.degrees(circ_std_rad), 2),
            drift_rate_deg_per_sec=round(drift_per_sec, 4),
            drift_rate_deg_per_min=round(drift_per_min, 2),
            num_heading_samples=n_samples,
            magnetic_disturbance_sample_ratio=mag_ratio
        )
