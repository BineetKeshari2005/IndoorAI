"""Extended Kalman Filter (EKF) engine for 2D Wi-Fi + PDR indoor trajectory fusion (Task 24).

Implements:
1. 3-state state vector: x = [x, y, theta]^T.
2. Motion model prediction from PDR step length and heading change.
3. Linear 2D position observation model: z = [x_wifi, y_wifi]^T.
4. Statistical Chi-Square Mahalanobis outlier gating (2-DOF 95% threshold = 5.991).
5. Numerically stable Joseph form covariance update:
   P = (I - K*H) * P * (I - K*H)^T + K * R * K^T.
6. Covariance symmetrization, eigenvalue bounding, and angular wrapping.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.core.logging import setup_logger
from src.sensor_fusion.coordinates import Pose2D, wrap_angle_rad
from src.sensor_fusion.models import (
    ConfidenceEllipse2D,
    FusedState,
    FusionConfig,
    FusionStatus,
    PdrStepEvent,
    WifiMeasurementEvent,
)

logger = setup_logger("EKFEngine")


def enforce_psd_covariance(cov: np.ndarray, min_eigenval: float = 1e-6) -> np.ndarray:
    """Enforce symmetry and positive semi-definiteness on a covariance matrix."""
    sym = 0.5 * (cov + cov.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    if np.any(eigvals < min_eigenval):
        eigvals_clamped = np.clip(eigvals, min_eigenval, None)
        sym = eigvecs @ np.diag(eigvals_clamped) @ eigvecs.T
        sym = 0.5 * (sym + sym.T)
    return sym


class IndoorEKF:
    """Extended Kalman Filter fusing PDR dead reckoning with absolute Wi-Fi corrections."""

    def __init__(self, config: Optional[FusionConfig] = None) -> None:
        self.config = config or FusionConfig()
        # State: [x, y, theta]^T
        self.x = np.zeros(3, dtype=np.float64)
        # Covariance: 3x3
        self.P = np.eye(3, dtype=np.float64)
        self.building_id: str = "DEFAULT_BUILDING"
        self.floor_id: int = 0
        self.is_initialized: bool = False
        self.last_timestamp_ms: int = 0

    def initialize(
        self,
        pose: Pose2D,
        timestamp_ms: int,
        initial_pos_uncertainty_m: Optional[float] = None,
        initial_heading_uncertainty_rad: Optional[float] = None,
    ) -> FusedState:
        """Initialize filter state, building/floor context, and initial covariance.

        Args:
            pose: Starting Pose2D (x, y, theta, building_id, floor_id).
            timestamp_ms: Initial epoch timestamp.
            initial_pos_uncertainty_m: Initial 1-sigma position standard deviation (m).
            initial_heading_uncertainty_rad: Initial 1-sigma heading standard deviation (rad).

        Returns:
            Initial FusedState.
        """
        pos_sigma = initial_pos_uncertainty_m or self.config.initial_pos_uncertainty_m
        head_sigma = initial_heading_uncertainty_rad or self.config.initial_heading_uncertainty_rad

        self.x = pose.to_array()
        self.P = np.zeros((3, 3), dtype=np.float64)
        self.P[0, 0] = pos_sigma ** 2
        self.P[1, 1] = pos_sigma ** 2
        self.P[2, 2] = head_sigma ** 2
        self.P = enforce_psd_covariance(self.P)

        self.building_id = pose.building_id
        self.floor_id = pose.floor_id
        self.is_initialized = True
        self.last_timestamp_ms = timestamp_ms

        ellipse = ConfidenceEllipse2D.from_covariance(self.P[0:2, 0:2], chi2_val=self.config.chi2_threshold)

        logger.info(
            f"Initialized EKF at ({pose.x_m:.2f}, {pose.y_m:.2f}, theta={pose.theta_deg:.1f}deg) "
            f"in {pose.building_id} Floor {pose.floor_id} (sigma_pos={pos_sigma}m)"
        )

        return FusedState(
            timestamp_epoch_ms=timestamp_ms,
            building_id=self.building_id,
            floor_id=self.floor_id,
            x_m=float(self.x[0]),
            y_m=float(self.x[1]),
            theta_rad=float(self.x[2]),
            state_covariance=self.P.copy(),
            status=FusionStatus.INITIALIZING,
            ellipse=ellipse,
            metadata={"origin_pose": {"x_m": pose.x_m, "y_m": pose.y_m, "theta_rad": pose.theta_rad}},
        )

    def predict_pdr_step(self, step: PdrStepEvent) -> FusedState:
        """Execute EKF state and covariance prediction from a PDR step displacement.

        Motion model:
            x_k = x_{k-1} + L_k * cos(theta_{k-1} + delta_theta_k)
            y_k = y_{k-1} + L_k * sin(theta_{k-1} + delta_theta_k)
            theta_k = wrap(theta_{k-1} + delta_theta_k)

        Args:
            step: PdrStepEvent containing step length, heading delta, and noise sigmas.

        Returns:
            Propagated FusedState with status PDR_ONLY.
        """
        if not self.is_initialized:
            raise RuntimeError("Cannot execute EKF prediction before initialize() is called.")

        L = float(step.step_length_m)
        delta_theta = float(step.heading_delta_rad)
        curr_theta = float(self.x[2])
        new_theta = wrap_angle_rad(curr_theta + delta_theta)

        cos_t = math.cos(new_theta)
        sin_t = math.sin(new_theta)

        # 1. State vector propagation
        self.x[0] += L * cos_t
        self.x[1] += L * sin_t
        self.x[2] = new_theta

        # 2. State transition Jacobian F_k (3x3)
        F = np.array([
            [1.0, 0.0, -L * sin_t],
            [0.0, 1.0,  L * cos_t],
            [0.0, 0.0,  1.0],
        ], dtype=np.float64)

        # 3. Control noise Jacobian G_k (3x2) for u = [L, delta_theta]^T
        G = np.array([
            [cos_t, -L * sin_t],
            [sin_t,  L * cos_t],
            [0.0,    1.0],
        ], dtype=np.float64)

        # 4. Motion noise covariance Q_k
        sigma_l = step.sigma_length_m * (1.5 if step.has_sensor_gap else 1.0)
        sigma_h = step.sigma_heading_rad * (2.0 if step.is_magnetic_disturbance else 1.0)
        Q = np.diag([sigma_l ** 2, sigma_h ** 2]).astype(np.float64)

        # 5. Covariance propagation
        self.P = F @ self.P @ F.T + G @ Q @ G.T
        self.P = enforce_psd_covariance(self.P)
        self.last_timestamp_ms = step.timestamp_epoch_ms

        ellipse = ConfidenceEllipse2D.from_covariance(self.P[0:2, 0:2], chi2_val=self.config.chi2_threshold)

        return FusedState(
            timestamp_epoch_ms=step.timestamp_epoch_ms,
            building_id=self.building_id,
            floor_id=self.floor_id,
            x_m=float(self.x[0]),
            y_m=float(self.x[1]),
            theta_rad=float(self.x[2]),
            state_covariance=self.P.copy(),
            status=FusionStatus.PDR_ONLY,
            ellipse=ellipse,
            step_event=step.to_dict(),
        )

    def update_wifi(
        self,
        wifi: WifiMeasurementEvent,
        R_cov: np.ndarray,
    ) -> FusedState:
        """Execute EKF measurement update from an absolute Wi-Fi position observation.

        Observation model:
            z = H * x + v,  where H = [[1, 0, 0], [0, 1, 0]]
            Innovation: r = z - H * x
            Innovation Covariance: S = H * P * H^T + R
            Mahalanobis Distance: d^2 = r^T * S^-1 * r

        Applies Chi-Square statistical outlier gating. If gated out, state is preserved
        and status is WIFI_REJECTED.
        If accepted, covariance is updated via the numerically stable Joseph form:
            P = (I - K*H) * P * (I - K*H)^T + K * R * K^T

        Args:
            wifi: WifiMeasurementEvent containing observed x, y coordinates and metadata.
            R_cov: 2x2 measurement noise covariance matrix.

        Returns:
            Updated FusedState with status WIFI_CORRECTED or WIFI_REJECTED.
        """
        if not self.is_initialized:
            raise RuntimeError("Cannot execute EKF update before initialize() is called.")

        # Floor and building consistency check
        if wifi.building_id != self.building_id or wifi.floor_id != self.floor_id:
            logger.warning(
                f"Wi-Fi observation ({wifi.building_id} F{wifi.floor_id}) mismatches "
                f"filter state ({self.building_id} F{self.floor_id}); rejecting cross-floor scan."
            )
            ellipse = ConfidenceEllipse2D.from_covariance(self.P[0:2, 0:2], chi2_val=self.config.chi2_threshold)
            return FusedState(
                timestamp_epoch_ms=wifi.timestamp_epoch_ms,
                building_id=self.building_id,
                floor_id=self.floor_id,
                x_m=float(self.x[0]),
                y_m=float(self.x[1]),
                theta_rad=float(self.x[2]),
                state_covariance=self.P.copy(),
                status=FusionStatus.WIFI_REJECTED,
                ellipse=ellipse,
                wifi_event=wifi.to_dict(),
                rejection_reason="BUILDING_OR_FLOOR_MISMATCH",
            )

        # Measurement model
        z = np.array([wifi.x_m, wifi.y_m], dtype=np.float64)
        H = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float64)

        # 1. Innovation vector r (2x1)
        r = z - (H @ self.x)

        # 2. Innovation covariance S (2x2)
        R = enforce_psd_covariance(R_cov, min_eigenval=self.config.min_wifi_sigma_m ** 2)
        S = H @ self.P @ H.T + R
        S = 0.5 * (S + S.T)

        # Invert S with conditioning
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        # 3. Mahalanobis distance squared d^2
        mahalanobis_sq = float(r.T @ S_inv @ r)

        # 4. Statistical Chi-Square Outlier Gating (2-DOF)
        if self.config.enable_outlier_gating and mahalanobis_sq > self.config.chi2_threshold:
            logger.warning(
                f"Wi-Fi observation rejected by Chi-Square gate: d^2={mahalanobis_sq:.2f} > {self.config.chi2_threshold:.2f} "
                f"(innovation=[{r[0]:.2f}, {r[1]:.2f}]m)"
            )
            ellipse = ConfidenceEllipse2D.from_covariance(self.P[0:2, 0:2], chi2_val=self.config.chi2_threshold)
            return FusedState(
                timestamp_epoch_ms=wifi.timestamp_epoch_ms,
                building_id=self.building_id,
                floor_id=self.floor_id,
                x_m=float(self.x[0]),
                y_m=float(self.x[1]),
                theta_rad=float(self.x[2]),
                state_covariance=self.P.copy(),
                status=FusionStatus.WIFI_REJECTED,
                ellipse=ellipse,
                wifi_event=wifi.to_dict(),
                innovation=[float(r[0]), float(r[1])],
                mahalanobis_dist_sq=mahalanobis_sq,
                rejection_reason="STATISTICAL_GATE_EXCEEDED",
            )

        # 5. Kalman Gain K (3x2)
        K = self.P @ H.T @ S_inv

        # 6. State update
        self.x = self.x + K @ r
        self.x[2] = wrap_angle_rad(float(self.x[2]))

        # 7. Numerically stable Joseph form covariance update
        I3 = np.eye(3, dtype=np.float64)
        I_KH = I3 - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = enforce_psd_covariance(self.P)
        self.last_timestamp_ms = wifi.timestamp_epoch_ms

        ellipse = ConfidenceEllipse2D.from_covariance(self.P[0:2, 0:2], chi2_val=self.config.chi2_threshold)

        return FusedState(
            timestamp_epoch_ms=wifi.timestamp_epoch_ms,
            building_id=self.building_id,
            floor_id=self.floor_id,
            x_m=float(self.x[0]),
            y_m=float(self.x[1]),
            theta_rad=float(self.x[2]),
            state_covariance=self.P.copy(),
            status=FusionStatus.WIFI_CORRECTED,
            ellipse=ellipse,
            wifi_event=wifi.to_dict(),
            innovation=[float(r[0]), float(r[1])],
            mahalanobis_dist_sq=mahalanobis_sq,
        )
