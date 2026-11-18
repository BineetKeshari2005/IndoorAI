"""Trajectory reconstruction accuracy and dead-reckoning drift evaluation engine (Task 22)."""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import pandas as pd

from src.core.logging import setup_logger
from src.pdr.calibration.models import ReferenceWaypoint, TrajectoryDriftMetrics

logger = setup_logger("TrajectoryDriftEvaluator")


def point_to_segment_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """Computes minimum Euclidean distance from 2D point (px, py) to line segment (x1, y1)-(x2, y2)."""
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx**2 + dy**2
    if seg_len_sq == 0.0:
        return math.sqrt((px - x1)**2 + (py - y1)**2)

    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.sqrt((px - proj_x)**2 + (py - proj_y)**2)


class TrajectoryDriftEvaluator:
    """Evaluates 2D PDR trajectory fidelity, endpoint accuracy, and accumulated dead-reckoning drift."""

    def evaluate(
        self,
        trajectory_df: pd.DataFrame,
        ground_truth_distance_m: float,
        ground_truth_endpoint: Optional[Tuple[float, float]] = None,
        reference_waypoints: Optional[List[ReferenceWaypoint]] = None,
        align_initial_heading: bool = False,
        reference_initial_heading_deg: Optional[float] = None
    ) -> TrajectoryDriftMetrics:
        """Computes distance error, endpoint error, drift per meter, and cross-track deviations."""
        if trajectory_df.empty or "rel_x_m" not in trajectory_df.columns:
            return TrajectoryDriftMetrics(
                estimated_distance_m=0.0,
                ground_truth_distance_m=ground_truth_distance_m,
                absolute_distance_error_m=ground_truth_distance_m,
                relative_distance_error=1.0,
                endpoint_error_m=ground_truth_distance_m,
                relative_endpoint_error=1.0,
                net_displacement_error_m=ground_truth_distance_m,
                duration_s=0.0
            )

        df = trajectory_df.sort_values("timestamp_ms").reset_index(drop=True)
        t_start = df["timestamp_ms"].iloc[0]
        t_end = df["timestamp_ms"].iloc[-1]
        duration_s = max(1.0, (t_end - t_start) / 1000.0)
        duration_min = duration_s / 60.0

        est_dist = float(df["cum_distance_m"].iloc[-1]) if "cum_distance_m" in df.columns else 0.0
        gt_dist = max(0.01, ground_truth_distance_m)

        # Handle optional initial rotation alignment if starting direction is specified
        xs = df["rel_x_m"].values.astype(float)
        ys = df["rel_y_m"].values.astype(float)

        if align_initial_heading and reference_initial_heading_deg is not None and len(df) > 1:
            initial_est_deg = float(df["heading_deg"].iloc[0]) if "heading_deg" in df.columns else 0.0
            rot_deg = reference_initial_heading_deg - initial_est_deg
            rot_rad = math.radians(rot_deg)
            cos_r = math.cos(rot_rad)
            sin_r = math.sin(rot_rad)
            # Rotate points around origin
            new_xs = xs * cos_r - ys * sin_r
            new_ys = xs * sin_r + ys * cos_r
            xs = new_xs
            ys = new_ys

        est_endpoint = (float(xs[-1]), float(ys[-1]))

        # Net displacement calculations
        est_net_disp = math.sqrt(est_endpoint[0]**2 + est_endpoint[1]**2)

        # Ground truth endpoint determination
        if ground_truth_endpoint is not None:
            gt_end = ground_truth_endpoint
            gt_net_disp = math.sqrt(gt_end[0]**2 + gt_end[1]**2)
        elif reference_waypoints and len(reference_waypoints) > 0:
            gt_end = (reference_waypoints[-1].x_m, reference_waypoints[-1].y_m)
            gt_net_disp = math.sqrt(gt_end[0]**2 + gt_end[1]**2)
        else:
            # Assume straight-line along initial direction if no multi-turn endpoint given
            gt_end = (0.0, gt_dist)
            gt_net_disp = gt_dist

        # Endpoint Euclidean error
        endpoint_error = math.sqrt((est_endpoint[0] - gt_end[0])**2 + (est_endpoint[1] - gt_end[1])**2)

        # Cross-track error against piecewise reference waypoints
        mean_cross_track = None
        if reference_waypoints and len(reference_waypoints) >= 2:
            ct_errors = []
            for px, py in zip(xs, ys):
                min_seg_dist = float("inf")
                for w_idx in range(len(reference_waypoints) - 1):
                    w1 = reference_waypoints[w_idx]
                    w2 = reference_waypoints[w_idx + 1]
                    d_seg = point_to_segment_distance(px, py, w1.x_m, w1.y_m, w2.x_m, w2.y_m)
                    if d_seg < min_seg_dist:
                        min_seg_dist = d_seg
                ct_errors.append(min_seg_dist)
            mean_cross_track = round(float(np.mean(ct_errors)), 3)

        abs_dist_err = abs(est_dist - gt_dist)
        rel_dist_err = abs_dist_err / gt_dist
        rel_endpoint_err = endpoint_error / gt_dist
        disp_err = abs(est_net_disp - gt_net_disp)

        drift_per_m = endpoint_error / gt_dist
        drift_per_min = endpoint_error / duration_min

        return TrajectoryDriftMetrics(
            estimated_distance_m=round(est_dist, 3),
            ground_truth_distance_m=round(gt_dist, 3),
            absolute_distance_error_m=round(abs_dist_err, 3),
            relative_distance_error=round(rel_dist_err, 4),
            endpoint_error_m=round(endpoint_error, 3),
            relative_endpoint_error=round(rel_endpoint_err, 4),
            net_displacement_error_m=round(disp_err, 3),
            mean_cross_track_error_m=mean_cross_track,
            drift_per_meter=round(drift_per_m, 4),
            drift_per_minute_m=round(drift_per_min, 3),
            duration_s=round(duration_s, 1)
        )
