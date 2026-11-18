"""Weinberg step-length parameter estimation and calibration engine (Task 22)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from src.core.logging import setup_logger
from src.pdr.calibration.models import (
    CalibrationProfile,
    CalibrationStatus,
    PhonePlacement,
    ProfileType,
    WalkingTrial,
)

logger = setup_logger("WeinbergCalibrator")


class WeinbergCalibrator:
    """Estimates optimal Weinberg stride parameter K from ground-truth walking trials."""

    def __init__(
        self,
        min_k: float = 0.20,
        max_k: float = 0.80,
        default_k: float = 0.42,
        step_bounds: Tuple[float, float] = (0.30, 1.20)
    ):
        self.min_k = min_k
        self.max_k = max_k
        self.default_k = default_k
        self.min_step_len = step_bounds[0]
        self.max_step_len = step_bounds[1]

    def estimate_trial_distance(
        self,
        k: float,
        accel_ranges: np.ndarray
    ) -> float:
        """Computes total estimated distance for a sequence of step bounce ranges at given K."""
        if len(accel_ranges) == 0:
            return 0.0
        # L = K * (delta_a)^0.25 clamped to [min_step, max_step]
        raw_lengths = k * (np.maximum(0.01, accel_ranges) ** 0.25)
        clamped_lengths = np.clip(raw_lengths, self.min_step_len, self.max_step_len)
        return float(np.sum(clamped_lengths))

    def calibrate(
        self,
        training_trials: List[Tuple[WalkingTrial, pd.DataFrame]],
        held_out_trials: Optional[List[Tuple[WalkingTrial, pd.DataFrame]]] = None,
        profile_name: str = "Calibrated Weinberg Profile",
        profile_type: ProfileType = ProfileType.DEFAULT_UNCALIBRATED,
        participant_id: Optional[str] = None,
        phone_placement: Optional[PhonePlacement] = None,
        device_model: Optional[str] = None
    ) -> CalibrationProfile:
        """Executes bounded optimization to determine optimal K over training trials and evaluates on held-out trials."""
        from datetime import datetime, timezone

        now_iso = datetime.now(timezone.utc).isoformat()
        profile_id = f"PROF-{abs(hash(profile_name + now_iso)) % 100000000:08d}"

        # 1. Check for sufficient data
        valid_train_data: List[Tuple[float, np.ndarray, float]] = []
        source_trial_ids: List[str] = []

        for trial, steps_df in training_trials:
            if steps_df.empty or "accel_range_mps2" not in steps_df.columns:
                continue
            measured_dist = trial.ground_truth.measured_distance_m
            if measured_dist <= 0:
                continue
            ranges = steps_df["accel_range_mps2"].values.astype(float)
            uncertainty = max(0.01, trial.ground_truth.measurement_uncertainty_m)
            valid_train_data.append((measured_dist, ranges, uncertainty))
            source_trial_ids.append(trial.trial_id)

        if len(valid_train_data) < 1:
            logger.warning("Insufficient valid trials for calibration (%d provided).", len(valid_train_data))
            return CalibrationProfile(
                profile_id=profile_id,
                profile_name=profile_name,
                profile_type=profile_type,
                calibration_method="INSUFFICIENT_DATA_FALLBACK",
                weinberg_k=self.default_k,
                step_length_bounds=(self.min_step_len, self.max_step_len),
                device_model=device_model,
                phone_placement=phone_placement,
                participant_id=participant_id,
                source_trial_ids=source_trial_ids,
                calibration_date_iso=now_iso,
                calibration_status=CalibrationStatus.INSUFFICIENT_DATA,
                notes="Calibration aborted due to zero valid walking trials. Default K=0.42 retained."
            )

        # 2. Objective Function: Weighted relative squared distance error
        def objective_func(k_val: float) -> float:
            total_loss = 0.0
            for d_true, ranges, sigma in valid_train_data:
                d_est = self.estimate_trial_distance(k_val, ranges)
                weight = 1.0 / (sigma ** 2)
                rel_err = (d_est - d_true) / d_true
                total_loss += weight * (rel_err ** 2)
            return total_loss

        # 3. Optimize K within [min_k, max_k]
        res = minimize_scalar(
            objective_func,
            bounds=(self.min_k, self.max_k),
            method="bounded",
            options={"xatol": 1e-4}
        )
        fitted_k = round(float(res.x), 4) if res.success else self.default_k

        # 4. Evaluate Training Set Metrics
        train_errors = []
        train_rel_errors = []
        for d_true, ranges, _ in valid_train_data:
            d_est = self.estimate_trial_distance(fitted_k, ranges)
            err = abs(d_est - d_true)
            train_errors.append(err)
            train_rel_errors.append(err / d_true)

        training_metrics = {
            "num_trials": len(valid_train_data),
            "mae_m": round(float(np.mean(train_errors)), 3),
            "rmse_m": round(float(np.sqrt(np.mean(np.array(train_errors)**2))), 3),
            "mean_relative_error_pct": round(float(np.mean(train_rel_errors) * 100.0), 2),
            "max_relative_error_pct": round(float(np.max(train_rel_errors) * 100.0), 2),
            "fitted_k": fitted_k,
            "default_k_mae_m": round(float(np.mean([abs(self.estimate_trial_distance(self.default_k, r) - d) for d, r, _ in valid_train_data])), 3)
        }

        # 5. Evaluate Held-out Validation Metrics if available
        validation_metrics: Dict[str, Any] = {}
        status = CalibrationStatus.CALIBRATED

        if held_out_trials and len(held_out_trials) > 0:
            val_errors = []
            val_rel_errors = []
            for trial, steps_df in held_out_trials:
                if steps_df.empty or "accel_range_mps2" not in steps_df.columns:
                    continue
                d_true = trial.ground_truth.measured_distance_m
                if d_true <= 0:
                    continue
                ranges = steps_df["accel_range_mps2"].values.astype(float)
                d_est = self.estimate_trial_distance(fitted_k, ranges)
                err = abs(d_est - d_true)
                val_errors.append(err)
                val_rel_errors.append(err / d_true)

            if len(val_errors) > 0:
                validation_metrics = {
                    "num_held_out_trials": len(val_errors),
                    "mae_m": round(float(np.mean(val_errors)), 3),
                    "rmse_m": round(float(np.sqrt(np.mean(np.array(val_errors)**2))), 3),
                    "mean_relative_error_pct": round(float(np.mean(val_rel_errors) * 100.0), 2),
                    "max_relative_error_pct": round(float(np.max(val_rel_errors) * 100.0), 2),
                }
                status = CalibrationStatus.VALIDATED

        logger.info(
            "Weinberg Calibration succeeded: K=%.4f (Default=%.2f) on %d trials. Status: %s. Training MAE: %.2fm (%.1f%%).",
            fitted_k, self.default_k, len(valid_train_data), status.value, training_metrics["mae_m"], training_metrics["mean_relative_error_pct"]
        )

        return CalibrationProfile(
            profile_id=profile_id,
            profile_name=profile_name,
            profile_type=profile_type,
            calibration_method="WEINBERG_BOUNDED_OPT",
            weinberg_k=fitted_k,
            step_length_bounds=(self.min_step_len, self.max_step_len),
            device_model=device_model,
            phone_placement=phone_placement,
            participant_id=participant_id,
            source_trial_ids=source_trial_ids,
            calibration_date_iso=now_iso,
            calibration_status=status,
            training_metrics=training_metrics,
            validation_metrics=validation_metrics,
            notes=f"Calibrated via bounded relative squared error minimization. Fitted K={fitted_k}."
        )
