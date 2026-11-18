"""End-to-end execution pipeline for Wi-Fi + PDR Extended Kalman Filter fusion (Task 24).

Merges asynchronous PDR step events and Wi-Fi absolute position measurements
into a chronologically ordered sequence, applies outlier gating and Joseph updates,
and serializes fused trajectories.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.core.logging import setup_logger
from src.sensor_fusion.coordinates import Pose2D
from src.sensor_fusion.ekf import IndoorEKF
from src.sensor_fusion.models import (
    FusedState,
    FusionConfig,
    FusionStatus,
    FusionTrajectorySummary,
    PdrStepEvent,
    WifiMeasurementEvent,
)
from src.sensor_fusion.uncertainty import WifiUncertaintyEstimator

logger = setup_logger("FusionPipeline")


class FusionPipeline:
    """Orchestrates chronologically ordered sensor fusion over preprocessed survey datasets."""

    def __init__(
        self,
        config: Optional[FusionConfig] = None,
        uncertainty_estimator: Optional[WifiUncertaintyEstimator] = None,
    ) -> None:
        self.config = config or FusionConfig()
        self.uncertainty_estimator = uncertainty_estimator or WifiUncertaintyEstimator(config=self.config)

    def run(
        self,
        pdr_steps: List[PdrStepEvent],
        wifi_measurements: List[WifiMeasurementEvent],
        initial_pose: Optional[Pose2D] = None,
        candidate_matches_by_scan: Optional[Dict[int, List[Dict[str, Any]]]] = None,
        site_id: Optional[str] = None,
    ) -> Tuple[List[FusedState], FusionTrajectorySummary]:
        """Execute chronological EKF fusion combining PDR step events and Wi-Fi measurements.

        Args:
            pdr_steps: List of PdrStepEvent instances.
            wifi_measurements: List of WifiMeasurementEvent instances.
            initial_pose: Optional starting Pose2D (defaults to first Wi-Fi or origin).
            candidate_matches_by_scan: Optional map from scan_index to candidate matches.
            site_id: Physical site identifier for empirical noise lookup.

        Returns:
            Tuple of (List of FusedState objects, FusionTrajectorySummary).
        """
        candidate_matches_by_scan = candidate_matches_by_scan or {}

        # 1. Merge and chronologically sort events
        # Events: list of tuples (timestamp, event_type, event_obj)
        tagged_events: List[Tuple[int, str, Union[PdrStepEvent, WifiMeasurementEvent]]] = []
        for step in pdr_steps:
            tagged_events.append((step.timestamp_epoch_ms, "PDR", step))
        for wifi in wifi_measurements:
            tagged_events.append((wifi.timestamp_epoch_ms, "WIFI", wifi))

        # Sort strictly chronologically
        tagged_events.sort(key=lambda item: item[0])

        if not tagged_events:
            logger.warning("Fusion pipeline called with zero input events.")
            return [], FusionTrajectorySummary()

        # 2. Determine initial pose
        first_ts = tagged_events[0][0]
        if initial_pose is None:
            # Try to initialize from first Wi-Fi measurement if present
            first_wifi = next((w for w in wifi_measurements if w.x_m is not None and w.y_m is not None), None)
            if first_wifi:
                initial_pose = Pose2D(
                    x_m=first_wifi.x_m,
                    y_m=first_wifi.y_m,
                    theta_rad=0.0,
                    building_id=first_wifi.building_id,
                    floor_id=first_wifi.floor_id,
                )
            else:
                initial_pose = Pose2D(x_m=0.0, y_m=0.0, theta_rad=0.0)

        # 3. Instantiate and initialize filter
        ekf = IndoorEKF(config=self.config)
        init_state = ekf.initialize(
            pose=initial_pose,
            timestamp_ms=first_ts,
            initial_pos_uncertainty_m=self.config.initial_pos_uncertainty_m,
            initial_heading_uncertainty_rad=self.config.initial_heading_uncertainty_rad,
        )

        fused_states: List[FusedState] = [init_state]

        # Tracking counters
        pdr_steps_count = 0
        wifi_scans_total = len(wifi_measurements)
        wifi_scans_accepted = 0
        wifi_scans_rejected = 0
        rejection_breakdown: Dict[str, int] = {}
        total_dist_m = 0.0

        last_x = initial_pose.x_m
        last_y = initial_pose.y_m

        # 4. Sequential Chronological Fusion Loop
        for ts, event_type, event in tagged_events:
            if event_type == "PDR":
                step_evt: PdrStepEvent = event  # type: ignore
                pdr_steps_count += 1
                state = ekf.predict_pdr_step(step_evt)
                fused_states.append(state)

                # Track distance
                dx = state.x_m - last_x
                dy = state.y_m - last_y
                total_dist_m += math.hypot(dx, dy)
                last_x = state.x_m
                last_y = state.y_m

            elif event_type == "WIFI":
                wifi_evt: WifiMeasurementEvent = event  # type: ignore

                # Quality pre-validation (freshness, latency, AP overlap, confidence)
                is_valid, quality_reason = self.uncertainty_estimator.validate_measurement_quality(
                    wifi_event=wifi_evt,
                    current_timestamp_ms=ts,
                )

                if not is_valid:
                    wifi_scans_rejected += 1
                    reason_key = quality_reason.split(":")[0] if quality_reason else "QUALITY_FILTER_FAILED"
                    rejection_breakdown[reason_key] = rejection_breakdown.get(reason_key, 0) + 1

                    # Append a rejected state without modifying the filter
                    pos_cov = ekf.P[0:2, 0:2].copy()
                    from src.sensor_fusion.models import ConfidenceEllipse2D
                    ellipse = ConfidenceEllipse2D.from_covariance(pos_cov, chi2_val=self.config.chi2_threshold)

                    fused_states.append(
                        FusedState(
                            timestamp_epoch_ms=ts,
                            building_id=ekf.building_id,
                            floor_id=ekf.floor_id,
                            x_m=float(ekf.x[0]),
                            y_m=float(ekf.x[1]),
                            theta_rad=float(ekf.x[2]),
                            state_covariance=ekf.P.copy(),
                            status=FusionStatus.WIFI_REJECTED,
                            ellipse=ellipse,
                            wifi_event=wifi_evt.to_dict(),
                            rejection_reason=quality_reason,
                        )
                    )
                    continue

                # Estimate 2x2 measurement covariance R
                cands = candidate_matches_by_scan.get(wifi_evt.scan_index)
                R_cov, unc_type = self.uncertainty_estimator.estimate_covariance(
                    wifi_event=wifi_evt,
                    candidate_matches=cands,
                    site_id=site_id,
                )
                wifi_evt.covariance = R_cov
                wifi_evt.uncertainty_type = unc_type

                # Execute EKF measurement update
                state = ekf.update_wifi(wifi=wifi_evt, R_cov=R_cov)
                fused_states.append(state)

                if state.status == FusionStatus.WIFI_CORRECTED:
                    wifi_scans_accepted += 1
                    dx = state.x_m - last_x
                    dy = state.y_m - last_y
                    total_dist_m += math.hypot(dx, dy)
                    last_x = state.x_m
                    last_y = state.y_m
                else:
                    wifi_scans_rejected += 1
                    reason_key = state.rejection_reason or "REJECTED"
                    rejection_breakdown[reason_key] = rejection_breakdown.get(reason_key, 0) + 1

        # 5. Compute summary metrics
        duration_s = max(0.0, (tagged_events[-1][0] - first_ts) / 1000.0)
        mean_radius = float(np.mean([s.ellipse.conservative_95_radius_m for s in fused_states]))
        acceptance_rate = (wifi_scans_accepted / wifi_scans_total) if wifi_scans_total > 0 else 0.0

        summary = FusionTrajectorySummary(
            total_points=len(fused_states),
            pdr_steps_count=pdr_steps_count,
            wifi_scans_total=wifi_scans_total,
            wifi_scans_accepted=wifi_scans_accepted,
            wifi_scans_rejected=wifi_scans_rejected,
            acceptance_rate=round(acceptance_rate, 4),
            total_distance_m=round(total_dist_m, 2),
            duration_s=round(duration_s, 2),
            mean_uncertainty_radius_m=round(mean_radius, 3),
            final_x_m=round(float(ekf.x[0]), 2),
            final_y_m=round(float(ekf.x[1]), 2),
            rejection_breakdown=rejection_breakdown,
        )

        logger.info(
            f"Fusion execution complete: {len(fused_states)} states ({pdr_steps_count} steps, "
            f"{wifi_scans_accepted}/{wifi_scans_total} Wi-Fi accepted, "
            f"{wifi_scans_rejected} rejected, dist={total_dist_m:.1f}m)"
        )

        return fused_states, summary

    def save_artifacts(
        self,
        fused_states: List[FusedState],
        summary: FusionTrajectorySummary,
        output_dir: Path,
    ) -> Dict[str, Path]:
        """Serialize and save fusion trajectory to Parquet and JSON artifacts."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Trajectory DataFrame
        rows = []
        for s in fused_states:
            rows.append({
                "timestamp_ms": s.timestamp_epoch_ms,
                "building_id": s.building_id,
                "floor_id": s.floor_id,
                "x_m": s.x_m,
                "y_m": s.y_m,
                "theta_rad": s.theta_rad,
                "theta_deg": s.theta_deg,
                "sigma_x_m": s.sigma_x_m,
                "sigma_y_m": s.sigma_y_m,
                "sigma_theta_deg": round(float(np.degrees(s.sigma_theta_rad)), 2),
                "status": s.status.value,
                "rejection_reason": s.rejection_reason or "",
                "semi_major_axis_m": s.ellipse.semi_major_axis_m,
                "semi_minor_axis_m": s.ellipse.semi_minor_axis_m,
                "ellipse_angle_deg": s.ellipse.ellipse_angle_deg,
                "conservative_95_radius_m": s.ellipse.conservative_95_radius_m,
                "mahalanobis_dist_sq": s.mahalanobis_dist_sq if s.mahalanobis_dist_sq is not None else np.nan,
            })
        df_traj = pd.DataFrame(rows)
        parquet_path = output_dir / "fused_trajectory.parquet"
        df_traj.to_parquet(parquet_path, index=False)

        # 2. Summary JSON
        summary_path = output_dir / "fusion_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2)

        # 3. Full states JSON
        states_path = output_dir / "fusion_states.json"
        with open(states_path, "w", encoding="utf-8") as f:
            json.dump([s.to_dict() for s in fused_states], f, indent=2)

        return {
            "trajectory_parquet": parquet_path,
            "summary_json": summary_path,
            "states_json": states_path,
        }
