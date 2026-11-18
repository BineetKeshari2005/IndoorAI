"""Master evaluation engine orchestrating step, heading, trajectory, and uncertainty evaluation (Task 22)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.logging import setup_logger
from src.pdr.calibration.heading_evaluator import HeadingEvaluator
from src.pdr.calibration.models import (
    PDREvaluationReport,
    WalkingTrial,
)
from src.pdr.calibration.step_evaluator import StepDetectionEvaluator
from src.pdr.calibration.trajectory_evaluator import TrajectoryDriftEvaluator
from src.pdr.calibration.uncertainty_evaluator import UncertaintyAuditor

logger = setup_logger("PDREvaluator")


class PDREvaluator:
    """Orchestrates comprehensive scientific validation of PDR outputs against ground-truth walking trials."""

    def __init__(self):
        self.step_evaluator = StepDetectionEvaluator()
        self.trajectory_evaluator = TrajectoryDriftEvaluator()
        self.heading_evaluator = HeadingEvaluator()
        self.uncertainty_auditor = UncertaintyAuditor()

    def evaluate_trial(
        self,
        trial: WalkingTrial,
        trajectory_df: pd.DataFrame,
        steps_df: pd.DataFrame,
        calibration_profile_id: str = "PROF-DEFAULT",
        notes: str = ""
    ) -> PDREvaluationReport:
        """Executes full evaluation suite for a single ground-truth trial against reconstructed trajectory and steps."""
        now_iso = datetime.now(timezone.utc).isoformat()
        report_id = f"EVAL-{trial.trial_id}-{abs(hash(now_iso)) % 1000000:06d}"

        # 1. Step detection evaluation
        step_metrics = self.step_evaluator.evaluate(
            detected_steps_df=steps_df,
            ground_truth_count=trial.ground_truth.ground_truth_step_count
        )

        # 2. Trajectory & drift evaluation
        gt_dist = trial.ground_truth.measured_distance_m
        waypoints = trial.ground_truth.waypoints
        start_heading = trial.ground_truth.measured_start_heading_deg
        end_heading = trial.ground_truth.measured_end_heading_deg

        traj_metrics = self.trajectory_evaluator.evaluate(
            trajectory_df=trajectory_df,
            ground_truth_distance_m=gt_dist,
            reference_waypoints=waypoints if len(waypoints) > 0 else None,
            align_initial_heading=True if start_heading is not None else False,
            reference_initial_heading_deg=start_heading
        )

        # 3. Heading & direction evaluation
        heading_metrics = None
        if start_heading is not None or "heading_deg" in trajectory_df.columns:
            heading_metrics = self.heading_evaluator.evaluate(
                trajectory_df=trajectory_df,
                reference_start_heading_deg=start_heading,
                reference_end_heading_deg=end_heading
            )

        # 4. Uncertainty audit
        uncertainty_metrics = None
        if not trajectory_df.empty:
            final_row = trajectory_df.iloc[-1]
            est_endpoint = (float(final_row.get("rel_x_m", 0.0)), float(final_row.get("rel_y_m", 0.0)))
            if waypoints and len(waypoints) > 0:
                gt_endpoint = (waypoints[-1].x_m, waypoints[-1].y_m)
            else:
                gt_endpoint = (0.0, gt_dist)

            # Approximate final covariance from radius or variance columns if available
            cov = np.eye(2, dtype=np.float64)
            if "semi_major_axis_m" in final_row and "semi_minor_axis_m" in final_row:
                a = float(final_row["semi_major_axis_m"])
                b = float(final_row["semi_minor_axis_m"])
                chi2 = 5.99146
                cov[0, 0] = (a ** 2) / chi2
                cov[1, 1] = (b ** 2) / chi2
            elif "pos_uncertainty_radius_m" in final_row:
                r = float(final_row["pos_uncertainty_radius_m"])
                cov[0, 0] = (r / 2.0) ** 2
                cov[1, 1] = (r / 2.0) ** 2

            uncertainty_metrics = self.uncertainty_auditor.audit_trajectory_uncertainty(
                final_covariance=cov,
                estimated_endpoint=est_endpoint,
                ground_truth_endpoint=gt_endpoint
            )

        report = PDREvaluationReport(
            report_id=report_id,
            dataset_id=trial.dataset_id,
            session_id=trial.session_id,
            generated_at_iso=now_iso,
            is_synthetic_fixture=trial.is_synthetic,
            calibration_profile_id=calibration_profile_id,
            trial_ids_evaluated=[trial.trial_id],
            step_metrics=step_metrics,
            trajectory_metrics=traj_metrics,
            heading_metrics=heading_metrics,
            uncertainty_metrics=uncertainty_metrics,
            notes=notes or f"Evaluation of trial {trial.trial_id} ({trial.trial_type.value}, {trial.phone_placement.value})."
        )

        logger.info(
            "PDR Evaluation Report %s: Distance Error: %.2fm (%.1f%%), Endpoint Error: %.2fm (Drift: %.3fm/m), F1: %.3f.",
            report_id, traj_metrics.absolute_distance_error_m, traj_metrics.relative_distance_error * 100.0,
            traj_metrics.endpoint_error_m, traj_metrics.drift_per_meter, step_metrics.f1_score
        )
        return report
