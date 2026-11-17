"""2D relative trajectory reconstruction and path accumulation (Task 21)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.logging import setup_logger
from src.pdr.config import PDRConfig
from src.pdr.heading import HeadingEstimator, StepHeading
from src.pdr.uncertainty import PDRUncertaintyPropagator

logger = setup_logger("PDRTrajectory")


@dataclass
class TrajectorySummary:
    """Summary metrics of the reconstructed 2D relative trajectory."""

    total_steps: int = 0
    total_distance_m: float = 0.0
    net_displacement_m: float = 0.0
    final_uncertainty_radius_m: float = 0.0
    mean_step_length_m: float = 0.0
    duration_s: float = 0.0


class TrajectoryReconstructor:
    """Reconstructs 2D relative pedestrian trajectory from step lengths and heading estimates."""

    def __init__(self, config: Optional[PDRConfig] = None):
        self.config = config or PDRConfig()

    def process(
        self,
        steps_df: pd.DataFrame,
        orientation_df: pd.DataFrame,
        session_id: str = "UNKNOWN"
    ) -> Tuple[pd.DataFrame, TrajectorySummary]:
        """Integrates steps into a sequential 2D relative path."""
        summary = TrajectorySummary()
        if steps_df.empty:
            logger.warning("Empty steps DataFrame provided to TrajectoryReconstructor.")
            origin_point = {
                "point_id": "PDR-00000",
                "session_id": session_id,
                "step_id": "ORIGIN",
                "timestamp_ms": 0,
                "rel_x_m": self.config.origin_x_m,
                "rel_y_m": self.config.origin_y_m,
                "heading_rad": round(self.config.initial_heading_rad, 4),
                "heading_deg": round(float(np.degrees(self.config.initial_heading_rad)), 1),
                "step_length_m": 0.0,
                "cum_distance_m": 0.0,
                "pos_uncertainty_radius_m": round(self.config.initial_pos_uncertainty_m, 3),
                "heading_source": "INITIAL_ORIGIN",
                "confidence": 1.0,
                "is_reliable": True,
            }
            return pd.DataFrame([origin_point]), summary

        df_steps = steps_df.sort_values("timestamp_ms").reset_index(drop=True).copy()
        heading_est = HeadingEstimator(self.config)
        uncertainty_prop = PDRUncertaintyPropagator(self.config.initial_pos_uncertainty_m)

        session_start_ms = int(df_steps["timestamp_ms"].iloc[0])
        session_end_ms = int(df_steps["timestamp_ms"].iloc[-1])

        curr_x = self.config.origin_x_m
        curr_y = self.config.origin_y_m
        cum_dist = 0.0

        traj_points: List[Dict[str, Any]] = []

        # Origin point at step 0
        traj_points.append({
            "point_id": "PDR-00000",
            "session_id": session_id,
            "step_id": "ORIGIN",
            "timestamp_ms": session_start_ms,
            "rel_x_m": round(curr_x, 3),
            "rel_y_m": round(curr_y, 3),
            "heading_rad": round(self.config.initial_heading_rad, 4),
            "heading_deg": round(float(np.degrees(self.config.initial_heading_rad)), 1),
            "step_length_m": 0.0,
            "cum_distance_m": 0.0,
            "pos_uncertainty_radius_m": round(self.config.initial_pos_uncertainty_m, 3),
            "heading_source": "INITIAL_ORIGIN",
            "confidence": 1.0,
            "is_reliable": True,
        })

        for i in range(len(df_steps)):
            row = df_steps.iloc[i]
            t_step = int(row["timestamp_ms"])
            length = float(row.get("step_length_m", 0.7))
            sigma_l = float(row.get("step_length_uncertainty_m", 0.1))
            step_id = str(row.get("step_id", f"STEP-{i+1:05d}"))
            step_conf = float(row.get("confidence", 0.8))

            # Retrieve heading at step instant
            step_heading: StepHeading = heading_est.get_heading_at_timestamp(
                timestamp_ms=t_step,
                orientation_df=orientation_df,
                session_start_ms=session_start_ms
            )

            theta = step_heading.heading_rad

            # Compute relative displacement
            # Convention: dx = L * cos(theta), dy = L * sin(theta)
            dx = length * math.cos(theta)
            dy = length * math.sin(theta)

            curr_x += dx
            curr_y += dy
            cum_dist += length

            # Propagate spatial covariance
            cov_result = uncertainty_prop.update(
                step_length_m=length,
                heading_rad=theta,
                sigma_length_m=sigma_l,
                sigma_heading_rad=step_heading.heading_uncertainty_rad,
                has_sensor_gap=False,
                is_magnetic_disturbance=False
            )

            # Combined confidence
            point_conf = min(step_conf, 1.0 if step_heading.is_reliable else 0.5)

            traj_points.append({
                "point_id": f"PDR-{i+1:05d}",
                "session_id": session_id,
                "step_id": step_id,
                "timestamp_ms": t_step,
                "rel_x_m": round(curr_x, 3),
                "rel_y_m": round(curr_y, 3),
                "heading_rad": round(theta, 4),
                "heading_deg": round(step_heading.heading_deg, 1),
                "step_length_m": round(length, 3),
                "cum_distance_m": round(cum_dist, 3),
                "pos_uncertainty_radius_m": round(cov_result.uncertainty_radius_m, 3),
                "heading_source": step_heading.heading_source,
                "confidence": round(point_conf, 2),
                "is_reliable": step_heading.is_reliable and (step_conf >= 0.7),
            })

        traj_df = pd.DataFrame(traj_points)

        # Populate summary
        net_disp = math.sqrt((curr_x - self.config.origin_x_m)**2 + (curr_y - self.config.origin_y_m)**2)
        summary.total_steps = len(df_steps)
        summary.total_distance_m = round(cum_dist, 2)
        summary.net_displacement_m = round(net_disp, 2)
        summary.final_uncertainty_radius_m = round(traj_points[-1]["pos_uncertainty_radius_m"], 2)
        summary.mean_step_length_m = round(cum_dist / len(df_steps), 2) if len(df_steps) > 0 else 0.0
        summary.duration_s = round((session_end_ms - session_start_ms) / 1000.0, 1)

        logger.info(
            "PDR Trajectory for %s: %d steps, %.2f m distance, %.2f m net displacement (Final uncertainty: +/-%.2f m).",
            session_id, summary.total_steps, summary.total_distance_m, summary.net_displacement_m, summary.final_uncertainty_radius_m
        )
        return traj_df, summary
