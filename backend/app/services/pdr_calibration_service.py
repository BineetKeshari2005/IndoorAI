"""Service layer managing PDR ground-truth trials, Weinberg calibration, and evaluation workflows (Task 22)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from backend.app.core.dataset_config import DatasetConfig, default_dataset_config
from backend.app.schemas.pdr_calibration import CalibrateRequest, EvaluateRequest, TrialRegisterRequest
from backend.app.services.dataset_registry import DatasetRegistry
from backend.app.services.pdr_service import PDRService
from src.core.logging import setup_logger
from src.pdr.calibration.evaluator import PDREvaluator
from src.pdr.calibration.models import (
    CalibrationProfile,
    CalibrationStatus,
    GroundTruthMeasurement,
    PDREvaluationReport,
    PhonePlacement,
    ProfileType,
    ReferenceWaypoint,
    TrialType,
    WalkingCondition,
    WalkingTrial,
)
from src.pdr.calibration.step_length_calibrator import WeinbergCalibrator
from src.pdr.calibration.storage import CalibrationStorage
from src.pdr.config import PDRConfig
from src.pdr.pipeline import PDRPipeline

logger = setup_logger("PDRCalibrationService")


class PDRCalibrationService:
    """Coordinates walking trial registration, parameter calibration, and evaluation against ground truth."""

    def __init__(
        self,
        config: Optional[DatasetConfig] = None,
        registry: Optional[DatasetRegistry] = None,
        storage: Optional[CalibrationStorage] = None,
        pdr_service: Optional[PDRService] = None
    ):
        self.config = config or default_dataset_config
        self.registry = registry or DatasetRegistry(self.config)
        self.storage = storage or CalibrationStorage()
        self.pdr_service = pdr_service or PDRService(self.config, self.registry)
        self.calibrator = WeinbergCalibrator()
        self.evaluator = PDREvaluator()

    # -------------------------------------------------------------------------
    # Trial Operations
    # -------------------------------------------------------------------------

    def register_trial(self, dataset_id: str, request: TrialRegisterRequest) -> WalkingTrial:
        """Registers a measured walking trial with manual ground-truth metrics."""
        dataset = self.registry.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset '{dataset_id}' not found in registry.")

        now_iso = datetime.now(timezone.utc).isoformat()
        trial_id = request.trial_id or f"TRIAL-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{abs(hash(request.session_id + now_iso)) % 10000:04d}"

        # Map waypoints
        waypoints: List[ReferenceWaypoint] = []
        for wp in request.ground_truth.waypoints:
            waypoints.append(ReferenceWaypoint(
                waypoint_id=wp.waypoint_id,
                x_m=wp.x_m,
                y_m=wp.y_m,
                timestamp_ms=wp.timestamp_ms,
                uncertainty_m=wp.uncertainty_m,
                description=wp.description
            ))

        gt = GroundTruthMeasurement(
            measured_distance_m=request.ground_truth.measured_distance_m,
            measurement_method=request.ground_truth.measurement_method,
            measurement_uncertainty_m=request.ground_truth.measurement_uncertainty_m,
            measured_start_heading_deg=request.ground_truth.measured_start_heading_deg,
            measured_end_heading_deg=request.ground_truth.measured_end_heading_deg,
            heading_measurement_uncertainty_deg=request.ground_truth.heading_measurement_uncertainty_deg,
            ground_truth_step_count=request.ground_truth.ground_truth_step_count,
            step_count_uncertainty=request.ground_truth.step_count_uncertainty,
            num_turns=request.ground_truth.num_turns,
            waypoints=waypoints
        )

        trial = WalkingTrial(
            trial_id=trial_id,
            dataset_id=dataset_id,
            session_id=request.session_id,
            participant_id=request.participant_id,
            device_model=request.device_model,
            phone_placement=PhonePlacement(request.phone_placement.upper()),
            walking_condition=WalkingCondition(request.walking_condition.upper()),
            trial_type=TrialType(request.trial_type.upper()),
            start_timestamp_ms=request.start_timestamp_ms,
            end_timestamp_ms=request.end_timestamp_ms,
            ground_truth=gt,
            trial_notes=request.trial_notes,
            is_synthetic=request.is_synthetic,
            created_at_iso=now_iso
        )

        self.storage.save_trial(trial)
        return trial

    def get_trial(self, trial_id: str) -> Optional[WalkingTrial]:
        """Retrieves a walking trial by trial ID."""
        return self.storage.get_trial(trial_id)

    def list_trials(self, dataset_id: Optional[str] = None) -> List[WalkingTrial]:
        """Lists registered walking trials."""
        return self.storage.list_trials(dataset_id=dataset_id)

    # -------------------------------------------------------------------------
    # Calibration Operations
    # -------------------------------------------------------------------------

    def _get_trial_steps_slice(self, trial: WalkingTrial) -> pd.DataFrame:
        """Retrieves detected steps for a trial within its active time window."""
        dataset_pdr_dir = self.config.base_dir.parent / "processed" / "multimodal" / trial.dataset_id / "pdr"
        steps_file = dataset_pdr_dir / "step_features.parquet"
        if not steps_file.is_file():
            steps_file = dataset_pdr_dir / "detected_steps.parquet"

        if not steps_file.is_file():
            # Automatically run PDR if not yet computed
            logger.info("PDR outputs missing for %s; running PDR pipeline...", trial.dataset_id)
            self.pdr_service.run_pdr(trial.dataset_id)

        if not steps_file.is_file():
            return pd.DataFrame()

        df_steps = pd.read_parquet(steps_file)
        if df_steps.empty:
            return df_steps

        # Filter steps in timestamp window [start, end]
        mask = (df_steps["timestamp_ms"] >= trial.start_timestamp_ms) & (df_steps["timestamp_ms"] <= trial.end_timestamp_ms)
        matched_df = df_steps[mask].copy()

        # If window is inclusive of entire session or steps fall close to window
        if matched_df.empty and len(df_steps) > 0:
            logger.warning("No steps directly in [%d, %d]; using all session steps for trial %s", trial.start_timestamp_ms, trial.end_timestamp_ms, trial.trial_id)
            matched_df = df_steps.copy()

        return matched_df

    def calibrate(self, dataset_id: str, request: CalibrateRequest) -> CalibrationProfile:
        """Executes Weinberg stride calibration using registered training and validation trials."""
        train_pairs: List[Tuple[WalkingTrial, pd.DataFrame]] = []
        for tid in request.training_trial_ids:
            trial = self.storage.get_trial(tid)
            if not trial:
                raise ValueError(f"Training trial '{tid}' not found in registry.")
            steps_df = self._get_trial_steps_slice(trial)
            train_pairs.append((trial, steps_df))

        val_pairs: List[Tuple[WalkingTrial, pd.DataFrame]] = []
        if request.held_out_trial_ids:
            for vid in request.held_out_trial_ids:
                trial = self.storage.get_trial(vid)
                if trial:
                    steps_df = self._get_trial_steps_slice(trial)
                    val_pairs.append((trial, steps_df))

        profile_type = ProfileType(request.profile_type.upper()) if request.profile_type else ProfileType.PHONE_PLACEMENT_SPECIFIC
        placement = PhonePlacement(request.phone_placement.upper()) if request.phone_placement else None

        calibrated_prof = self.calibrator.calibrate(
            training_trials=train_pairs,
            held_out_trials=val_pairs if len(val_pairs) > 0 else None,
            profile_name=request.profile_name,
            profile_type=profile_type,
            participant_id=request.participant_id,
            phone_placement=placement,
            device_model=request.device_model
        )

        self.storage.save_profile(calibrated_prof)
        return calibrated_prof

    def list_profiles(self) -> List[CalibrationProfile]:
        """Returns all calibration profiles."""
        return self.storage.list_profiles()

    def get_profile(self, profile_id: str) -> Optional[CalibrationProfile]:
        """Retrieves a profile by ID."""
        return self.storage.get_profile(profile_id)

    # -------------------------------------------------------------------------
    # Evaluation Operations
    # -------------------------------------------------------------------------

    def evaluate(self, dataset_id: str, request: EvaluateRequest) -> List[PDREvaluationReport]:
        """Evaluates PDR trajectories against specified ground-truth walking trials."""
        reports: List[PDREvaluationReport] = []

        prof_id = request.calibration_profile_id or "PROF-DEFAULT"
        profile = self.storage.get_profile(prof_id)
        if not profile:
            profile = self.storage.get_profile("PROF-DEFAULT")

        k_val = profile.weinberg_k if profile else 0.42

        for tid in request.trial_ids:
            trial = self.storage.get_trial(tid)
            if not trial:
                raise ValueError(f"Trial '{tid}' not found.")

            dataset_pdr_dir = self.config.base_dir.parent / "processed" / "multimodal" / trial.dataset_id / "pdr"
            traj_file = dataset_pdr_dir / "relative_trajectory.parquet"
            steps_file = dataset_pdr_dir / "step_features.parquet"

            if not traj_file.is_file() or not steps_file.is_file():
                self.pdr_service.run_pdr(trial.dataset_id)

            if not traj_file.is_file() or not steps_file.is_file():
                logger.error("Unable to produce PDR trajectory files for trial %s", tid)
                continue

            traj_df = pd.read_parquet(traj_file)
            steps_df = pd.read_parquet(steps_file)

            # Filter by trial window
            mask_t = (traj_df["timestamp_ms"] >= trial.start_timestamp_ms) & (traj_df["timestamp_ms"] <= trial.end_timestamp_ms)
            trial_traj = traj_df[mask_t].copy() if mask_t.sum() > 1 else traj_df.copy()

            mask_s = (steps_df["timestamp_ms"] >= trial.start_timestamp_ms) & (steps_df["timestamp_ms"] <= trial.end_timestamp_ms)
            trial_steps = steps_df[mask_s].copy() if mask_s.sum() > 0 else steps_df.copy()

            # If a calibrated K is selected that differs from 0.42, rescale distances and points
            if abs(k_val - 0.42) > 1e-4 and "accel_range_mps2" in trial_steps.columns:
                calibrated_lengths = self.calibrator.estimate_trial_distance(k_val, trial_steps["accel_range_mps2"].values)
                # Rescale trajectory cumulative distance and positions proportionally
                scale = calibrated_lengths / max(0.01, float(trial_traj["cum_distance_m"].iloc[-1])) if len(trial_traj) > 0 else 1.0
                trial_traj["rel_x_m"] = trial_traj["rel_x_m"] * scale
                trial_traj["rel_y_m"] = trial_traj["rel_y_m"] * scale
                trial_traj["cum_distance_m"] = trial_traj["cum_distance_m"] * scale

            report = self.evaluator.evaluate_trial(
                trial=trial,
                trajectory_df=trial_traj,
                steps_df=trial_steps,
                calibration_profile_id=prof_id
            )

            # Save report to calibration_results/
            res_path = self.storage.results_dir / f"{report.report_id}.json"
            with open(res_path, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2)

            reports.append(report)

        return reports

    def get_evaluation_reports(self, dataset_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves stored evaluation reports."""
        res_list: List[Dict[str, Any]] = []
        if not self.storage.results_dir.is_dir():
            return res_list

        for f_path in sorted(self.storage.results_dir.glob("EVAL-*.json"), reverse=True):
            try:
                with open(f_path, "r", encoding="utf-8") as f:
                    rep_dict = json.load(f)
                    if dataset_id is None or rep_dict.get("dataset_id") == dataset_id:
                        res_list.append(rep_dict)
            except Exception as e:
                logger.warning("Failed to read report %s: %s", f_path, e)
        return res_list
