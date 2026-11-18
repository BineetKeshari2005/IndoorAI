"""Comparative trajectory evaluation framework for Wi-Fi + PDR sensor fusion (Task 24).

Compares:
1. PDR-only dead reckoning trajectory.
2. Wi-Fi-only position observations.
3. Wi-Fi + PDR Extended Kalman Filter fusion.

Computes mean, median, 95th percentile, max, endpoint, and drift metrics against ground truth.
Includes controlled synthetic benchmark generators for mathematical verification.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.core.logging import setup_logger
from src.sensor_fusion.models import (
    CHI2_2DOF_95,
    FusedState,
    FusionConfig,
    FusionStatus,
    PdrStepEvent,
    WifiMeasurementEvent,
)
from src.sensor_fusion.pipeline import FusionPipeline

logger = setup_logger("FusionEvaluator")


@dataclass
class TrajectoryEvaluationMetrics:
    """Metrics quantifying trajectory fidelity against a ground-truth reference path."""
    mean_error_m: float
    median_error_m: float
    p95_error_m: float
    max_error_m: float
    endpoint_error_m: float
    drift_per_meter: float
    drift_per_minute_m_min: float
    uncertainty_95_coverage: Optional[float] = None  # Fraction of GT points within 95% ellipse

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class FusionComparisonReport:
    """Complete comparative evaluation report across PDR-only, Wi-Fi-only, and Fused modes."""
    evaluation_id: str
    is_synthetic_fixture: bool
    duration_s: float
    total_distance_m: float
    pdr_only_metrics: TrajectoryEvaluationMetrics
    wifi_only_metrics: TrajectoryEvaluationMetrics
    fused_metrics: TrajectoryEvaluationMetrics
    wifi_scans_total: int
    wifi_scans_accepted: int
    wifi_scans_rejected: int
    acceptance_rate: float
    outlier_rejection_rate: float
    mean_innovation_m: float
    mean_mahalanobis_dist_sq: float
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "is_synthetic_fixture": self.is_synthetic_fixture,
            "duration_s": round(self.duration_s, 2),
            "total_distance_m": round(self.total_distance_m, 2),
            "pdr_only": self.pdr_only_metrics.to_dict(),
            "wifi_only": self.wifi_only_metrics.to_dict(),
            "fused": self.fused_metrics.to_dict(),
            "wifi_scans_total": self.wifi_scans_total,
            "wifi_scans_accepted": self.wifi_scans_accepted,
            "wifi_scans_rejected": self.wifi_scans_rejected,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "outlier_rejection_rate": round(self.outlier_rejection_rate, 4),
            "mean_innovation_m": round(self.mean_innovation_m, 3),
            "mean_mahalanobis_dist_sq": round(self.mean_mahalanobis_dist_sq, 3),
            "notes": self.notes,
        }


class FusionEvaluator:
    """Evaluates and benchmarks sensor-fusion trajectories against reference ground truth."""

    def __init__(self, config: Optional[FusionConfig] = None) -> None:
        self.config = config or FusionConfig()

    def evaluate_against_ground_truth(
        self,
        fused_states: List[FusedState],
        ground_truth: List[Tuple[int, float, float]],  # List of (timestamp_ms, gt_x, gt_y)
        pdr_states: Optional[List[Tuple[int, float, float]]] = None,
        wifi_measurements: Optional[List[WifiMeasurementEvent]] = None,
        evaluation_id: str = "FUSION-EVAL-01",
        is_synthetic_fixture: bool = False,
        notes: str = "",
    ) -> FusionComparisonReport:
        """Evaluate fused trajectory and compare against PDR-only and Wi-Fi-only baselines."""
        if not fused_states or not ground_truth:
            raise ValueError("Evaluation requires non-empty fused states and ground truth points.")

        gt_dict = {item[0]: (item[1], item[2]) for item in ground_truth}

        # 1. Evaluate Fused Trajectory
        fused_eval_points = []
        gt_fused_points = []
        ellipses = []
        timestamps = []

        innovations = []
        mahalanobis_vals = []
        wifi_accepted = 0
        wifi_rejected = 0

        for s in fused_states:
            ts = s.timestamp_epoch_ms
            # Find nearest or exact ground truth point
            gt_pt = self._interpolate_ground_truth(ts, ground_truth)
            if gt_pt is not None:
                fused_eval_points.append((s.x_m, s.y_m))
                gt_fused_points.append(gt_pt)
                ellipses.append((s.state_covariance[0:2, 0:2], s.x_m, s.y_m))
                timestamps.append(ts)

            if s.status == FusionStatus.WIFI_CORRECTED:
                wifi_accepted += 1
                if s.innovation:
                    innovations.append(math.hypot(s.innovation[0], s.innovation[1]))
                if s.mahalanobis_dist_sq is not None:
                    mahalanobis_vals.append(s.mahalanobis_dist_sq)
            elif s.status == FusionStatus.WIFI_REJECTED:
                wifi_rejected += 1
                if s.innovation:
                    innovations.append(math.hypot(s.innovation[0], s.innovation[1]))
                if s.mahalanobis_dist_sq is not None:
                    mahalanobis_vals.append(s.mahalanobis_dist_sq)

        fused_metrics = self._calculate_metrics(fused_eval_points, gt_fused_points, timestamps, ellipses)

        # 2. Evaluate PDR-only Trajectory
        if pdr_states:
            pdr_eval_points = []
            gt_pdr_points = []
            pdr_ts = []
            for item in pdr_states:
                gt_pt = self._interpolate_ground_truth(item[0], ground_truth)
                if gt_pt is not None:
                    pdr_eval_points.append((item[1], item[2]))
                    gt_pdr_points.append(gt_pt)
                    pdr_ts.append(item[0])
            pdr_metrics = self._calculate_metrics(pdr_eval_points, gt_pdr_points, pdr_ts)
        else:
            # Fallback to step integration without Wi-Fi updates
            pdr_metrics = TrajectoryEvaluationMetrics(0, 0, 0, 0, 0, 0, 0)

        # 3. Evaluate Wi-Fi-only Points
        if wifi_measurements:
            wifi_eval_points = []
            gt_wifi_points = []
            wifi_ts = []
            for w in wifi_measurements:
                gt_pt = self._interpolate_ground_truth(w.timestamp_epoch_ms, ground_truth)
                if gt_pt is not None:
                    wifi_eval_points.append((w.x_m, w.y_m))
                    gt_wifi_points.append(gt_pt)
                    wifi_ts.append(w.timestamp_epoch_ms)
            wifi_metrics = self._calculate_metrics(wifi_eval_points, gt_wifi_points, wifi_ts)
        else:
            wifi_metrics = TrajectoryEvaluationMetrics(0, 0, 0, 0, 0, 0, 0)

        # Totals
        wifi_total = (wifi_accepted + wifi_rejected)
        acc_rate = (wifi_accepted / wifi_total) if wifi_total > 0 else 0.0
        rej_rate = (wifi_rejected / wifi_total) if wifi_total > 0 else 0.0
        mean_innov = float(np.mean(innovations)) if innovations else 0.0
        mean_maha = float(np.mean(mahalanobis_vals)) if mahalanobis_vals else 0.0

        # Duration & Distance
        duration_s = max(1.0, (timestamps[-1] - timestamps[0]) / 1000.0) if len(timestamps) > 1 else 1.0
        total_dist = 0.0
        for i in range(1, len(gt_fused_points)):
            total_dist += math.hypot(
                gt_fused_points[i][0] - gt_fused_points[i-1][0],
                gt_fused_points[i][1] - gt_fused_points[i-1][1]
            )

        return FusionComparisonReport(
            evaluation_id=evaluation_id,
            is_synthetic_fixture=is_synthetic_fixture,
            duration_s=duration_s,
            total_distance_m=total_dist,
            pdr_only_metrics=pdr_metrics,
            wifi_only_metrics=wifi_metrics,
            fused_metrics=fused_metrics,
            wifi_scans_total=wifi_total,
            wifi_scans_accepted=wifi_accepted,
            wifi_scans_rejected=wifi_rejected,
            acceptance_rate=acc_rate,
            outlier_rejection_rate=rej_rate,
            mean_innovation_m=mean_innov,
            mean_mahalanobis_dist_sq=mean_maha,
            notes=notes,
        )

    def _calculate_metrics(
        self,
        est_points: List[Tuple[float, float]],
        gt_points: List[Tuple[float, float]],
        timestamps: List[int],
        ellipses: Optional[List[Tuple[np.ndarray, float, float]]] = None,
    ) -> TrajectoryEvaluationMetrics:
        """Calculate quantitative positioning error statistics."""
        if not est_points or not gt_points or len(est_points) != len(gt_points):
            return TrajectoryEvaluationMetrics(0, 0, 0, 0, 0, 0, 0)

        errors = []
        coverage_hits = 0

        for idx, ((ex, ey), (gx, gy)) in enumerate(zip(est_points, gt_points)):
            err = math.hypot(ex - gx, ey - gy)
            errors.append(err)

            # Check 95% ellipse coverage via Mahalanobis distance
            if ellipses and idx < len(ellipses):
                cov_2x2, cx, cy = ellipses[idx]
                diff = np.array([gx - cx, gy - cy], dtype=np.float64)
                try:
                    inv_cov = np.linalg.inv(cov_2x2)
                    d2 = float(diff.T @ inv_cov @ diff)
                    if d2 <= CHI2_2DOF_95:
                        coverage_hits += 1
                except np.linalg.LinAlgError:
                    pass

        err_arr = np.array(errors, dtype=np.float64)
        endpoint_err = errors[-1] if errors else 0.0

        # Trajectory path length
        path_len = 0.0
        for i in range(1, len(gt_points)):
            path_len += math.hypot(gt_points[i][0] - gt_points[i-1][0], gt_points[i][1] - gt_points[i-1][1])
        drift_per_m = (endpoint_err / path_len) if path_len > 0.1 else 0.0

        duration_min = max(0.01, (timestamps[-1] - timestamps[0]) / 60000.0) if len(timestamps) > 1 else 0.01
        drift_per_min = endpoint_err / duration_min

        coverage_frac = (coverage_hits / len(est_points)) if ellipses else None

        return TrajectoryEvaluationMetrics(
            mean_error_m=round(float(np.mean(err_arr)), 2),
            median_error_m=round(float(np.median(err_arr)), 2),
            p95_error_m=round(float(np.percentile(err_arr, 95)), 2),
            max_error_m=round(float(np.max(err_arr)), 2),
            endpoint_error_m=round(float(endpoint_err), 2),
            drift_per_meter=round(float(drift_per_m), 4),
            drift_per_minute_m_min=round(float(drift_per_min), 2),
            uncertainty_95_coverage=round(coverage_frac, 3) if coverage_frac is not None else None,
        )

    def _interpolate_ground_truth(
        self,
        query_ts: int,
        gt_points: List[Tuple[int, float, float]],
    ) -> Optional[Tuple[float, float]]:
        """Linear time interpolation of reference ground-truth coordinates."""
        if not gt_points:
            return None
        if query_ts <= gt_points[0][0]:
            return (gt_points[0][1], gt_points[0][2])
        if query_ts >= gt_points[-1][0]:
            return (gt_points[-1][1], gt_points[-1][2])

        # Binary search / linear scan for surrounding interval
        for i in range(len(gt_points) - 1):
            t0, x0, y0 = gt_points[i]
            t1, x1, y1 = gt_points[i+1]
            if t0 <= query_ts <= t1:
                alpha = (query_ts - t0) / float(t1 - t0) if t1 > t0 else 0.0
                return (x0 + alpha * (x1 - x0), y0 + alpha * (y1 - y0))

        return None

    @staticmethod
    def generate_controlled_synthetic_trajectory(
        duration_s: float = 60.0,
        step_freq_hz: float = 1.8,
        true_step_length_m: float = 0.75,
        pdr_drift_rate_deg_min: float = 15.0,
        wifi_interval_s: float = 5.0,
        wifi_noise_sigma_m: float = 2.5,
        inject_outlier: bool = True,
        seed: int = 42,
    ) -> Tuple[List[Tuple[int, float, float]], List[PdrStepEvent], List[WifiMeasurementEvent]]:
        """Generates a mathematically controlled synthetic ground truth fixture.

        Used for strictly labeled unit testing and mathematical validation.
        """
        np.random.seed(seed)
        n_steps = int(duration_s * step_freq_hz)
        dt_step_ms = int(1000.0 / step_freq_hz)

        gt_points: List[Tuple[int, float, float]] = []
        pdr_steps: List[PdrStepEvent] = []
        wifi_events: List[WifiMeasurementEvent] = []

        # True path: L-shaped corridor walking (East for 30s, North for 30s)
        curr_gt_x = 0.0
        curr_gt_y = 0.0
        accum_drift_rad = 0.0
        drift_per_step_rad = math.radians(pdr_drift_rate_deg_min / (60.0 * step_freq_hz))

        start_ts = 1700000000000
        gt_points.append((start_ts, 0.0, 0.0))

        last_wifi_ts = start_ts
        scan_idx = 1
        last_pdr_heading = 0.0

        for k in range(1, n_steps + 1):
            t_ms = start_ts + k * dt_step_ms
            # True heading: East (0 rad) for first half, North (pi/2) for second half
            true_theta = 0.0 if (k <= n_steps // 2) else (math.pi / 2.0)

            curr_gt_x += true_step_length_m * math.cos(true_theta)
            curr_gt_y += true_step_length_m * math.sin(true_theta)
            gt_points.append((t_ms, round(curr_gt_x, 3), round(curr_gt_y, 3)))

            # PDR step has systematic heading drift + random walk
            accum_drift_rad += drift_per_step_rad + float(np.random.normal(0, 0.02))
            noisy_step_len = true_step_length_m + float(np.random.normal(0, 0.05))
            pdr_heading = true_theta + accum_drift_rad
            from src.sensor_fusion.coordinates import wrap_angle_rad
            delta_h = wrap_angle_rad(pdr_heading - last_pdr_heading)
            last_pdr_heading = pdr_heading

            pdr_steps.append(
                PdrStepEvent(
                    step_index=k,
                    timestamp_epoch_ms=t_ms,
                    step_length_m=round(noisy_step_len, 3),
                    heading_rad=round(pdr_heading, 4),
                    heading_delta_rad=round(delta_h, 4),
                    sigma_length_m=0.08,
                    sigma_heading_rad=0.06,
                )
            )

            # Periodic Wi-Fi scans
            if (t_ms - last_wifi_ts) >= int(wifi_interval_s * 1000):
                last_wifi_ts = t_ms
                # Add Gaussian measurement noise
                wx = curr_gt_x + float(np.random.normal(0, wifi_noise_sigma_m))
                wy = curr_gt_y + float(np.random.normal(0, wifi_noise_sigma_m))

                # Inject severe outlier at halfway mark if requested
                if inject_outlier and scan_idx == 4:
                    wx += 25.0  # +25m jump (should be rejected by chi2 gate)
                    wy -= 20.0

                cov = np.eye(2, dtype=np.float64) * (wifi_noise_sigma_m ** 2)
                wifi_events.append(
                    WifiMeasurementEvent(
                        scan_index=scan_idx,
                        timestamp_epoch_ms=t_ms + 50,  # slight asynchronous offset
                        x_m=round(wx, 2),
                        y_m=round(wy, 2),
                        building_id="BLDG-1",
                        floor_id=1,
                        covariance=cov,
                        confidence_score=0.85,
                        matched_ap_count=8,
                        total_query_aps=10,
                        is_fresh=True,
                    )
                )
                scan_idx += 1

        return gt_points, pdr_steps, wifi_events
