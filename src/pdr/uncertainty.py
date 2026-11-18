"""Spatial covariance and uncertainty propagation for Pedestrian Dead Reckoning (Tasks 21 & 22)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

# Chi-Square 95% Quantile for 2 Degrees of Freedom: -2 * ln(1 - 0.95) ≈ 5.99146
CHI2_2DOF_95 = 5.9914645471


@dataclass
class StepCovariance:
    """Propagated 2D spatial uncertainty for a single step update."""

    sigma_x_m: float
    sigma_y_m: float
    uncertainty_radius_m: float
    cov_matrix: np.ndarray
    semi_major_axis_m: float = 0.0
    semi_minor_axis_m: float = 0.0
    ellipse_angle_deg: float = 0.0
    conservative_95_radius_m: float = 0.0


class PDRUncertaintyPropagator:
    """Propagates positional uncertainty using first-order Taylor expansion over step length and heading errors.

    Supports:
    1. Standard 2-state position covariance update (white-noise heading assumption):
       P_k = P_{k-1} + J_k Q_k J_k^T
    2. Augmented 3-state [x, y, theta]^T covariance update (correlated heading random-walk):
       P_{3, k} = F_k P_{3, k-1} F_k^T + G_k Q_k G_k^T
    3. Rigorous 2D Gaussian 95% confidence ellipse calculation using chi2(2, 0.95) = 5.991.
    """

    def __init__(self, initial_uncertainty_m: float = 0.5, initial_heading_uncertainty_rad: float = 0.087):
        # Initial 2x2 position covariance matrix
        self.P = np.eye(2, dtype=np.float64) * (initial_uncertainty_m ** 2)

        # Initial 3x3 augmented state covariance matrix [x, y, theta]
        self.P3 = np.zeros((3, 3), dtype=np.float64)
        self.P3[0, 0] = initial_uncertainty_m ** 2
        self.P3[1, 1] = initial_uncertainty_m ** 2
        self.P3[2, 2] = initial_heading_uncertainty_rad ** 2

    def update(
        self,
        step_length_m: float,
        heading_rad: float,
        sigma_length_m: float,
        sigma_heading_rad: float,
        has_sensor_gap: bool = False,
        is_magnetic_disturbance: bool = False
    ) -> StepCovariance:
        """Propagates 2D position covariance with step displacement noise.

        Delta x = L * cos(theta)
        Delta y = L * sin(theta)
        """
        cos_t = math.cos(heading_rad)
        sin_t = math.sin(heading_rad)

        # Jacobian J = d[dx, dy] / d[L, theta]
        J = np.array([
            [cos_t, -step_length_m * sin_t],
            [sin_t,  step_length_m * cos_t],
        ], dtype=np.float64)

        # Noise covariance Q = diag(sigma_L^2, sigma_theta^2)
        eff_sigma_l = sigma_length_m * (1.5 if has_sensor_gap else 1.0)
        eff_sigma_h = sigma_heading_rad * (2.0 if is_magnetic_disturbance else 1.0)

        Q = np.array([
            [eff_sigma_l ** 2, 0.0],
            [0.0, eff_sigma_h ** 2],
        ], dtype=np.float64)

        # Displacement covariance Sigma_step = J * Q * J^T
        sigma_step = J @ Q @ J.T

        # Update accumulated covariance
        self.P += sigma_step

        var_x = float(self.P[0, 0])
        var_y = float(self.P[1, 1])
        var_xy = float(self.P[0, 1])

        # Backward-compatible heuristic radius (sqrt(var_x + var_y))
        radius = math.sqrt(max(0.0, var_x + var_y))

        # Rigorous 95% Chi-Square Confidence Ellipse (2 DOF)
        mean_diag = (var_x + var_y) / 2.0
        half_diff = (var_x - var_y) / 2.0
        delta = math.sqrt(max(0.0, half_diff**2 + var_xy**2))
        lambda_1 = max(0.0, mean_diag + delta)
        lambda_2 = max(0.0, mean_diag - delta)

        semi_major = math.sqrt(CHI2_2DOF_95 * lambda_1)
        semi_minor = math.sqrt(CHI2_2DOF_95 * lambda_2)
        angle_deg = math.degrees(0.5 * math.atan2(2.0 * var_xy, var_x - var_y))

        return StepCovariance(
            sigma_x_m=math.sqrt(max(0.0, var_x)),
            sigma_y_m=math.sqrt(max(0.0, var_y)),
            uncertainty_radius_m=radius,
            cov_matrix=self.P.copy(),
            semi_major_axis_m=round(semi_major, 3),
            semi_minor_axis_m=round(semi_minor, 3),
            ellipse_angle_deg=round(angle_deg, 2),
            conservative_95_radius_m=round(semi_major, 3)
        )

    def update_with_heading_drift(
        self,
        step_length_m: float,
        heading_rad: float,
        sigma_length_m: float,
        sigma_heading_drift_rad: float,
        has_sensor_gap: bool = False,
        is_magnetic_disturbance: bool = False
    ) -> StepCovariance:
        """Propagates 3-state [x, y, theta] covariance accounting for accumulated heading drift autocorrelation.

        State transition:
            x_k = x_{k-1} + L_k * cos(theta_{k-1})
            y_k = y_{k-1} + L_k * sin(theta_{k-1})
            theta_k = theta_{k-1} + delta_theta_k
        """
        cos_t = math.cos(heading_rad)
        sin_t = math.sin(heading_rad)

        # State transition Jacobian F_k
        F = np.array([
            [1.0, 0.0, -step_length_m * sin_t],
            [0.0, 1.0,  step_length_m * cos_t],
            [0.0, 0.0,  1.0],
        ], dtype=np.float64)

        # Control noise Jacobian G_k
        G = np.array([
            [cos_t, 0.0],
            [sin_t, 0.0],
            [0.0,   1.0],
        ], dtype=np.float64)

        eff_sigma_l = sigma_length_m * (1.5 if has_sensor_gap else 1.0)
        eff_sigma_h = sigma_heading_drift_rad * (2.0 if is_magnetic_disturbance else 1.0)

        Q = np.array([
            [eff_sigma_l ** 2, 0.0],
            [0.0, eff_sigma_h ** 2],
        ], dtype=np.float64)

        self.P3 = F @ self.P3 @ F.T + G @ Q @ G.T

        # Extract 2D position block
        self.P = self.P3[0:2, 0:2].copy()

        var_x = float(self.P[0, 0])
        var_y = float(self.P[1, 1])
        var_xy = float(self.P[0, 1])

        radius = math.sqrt(max(0.0, var_x + var_y))
        mean_diag = (var_x + var_y) / 2.0
        half_diff = (var_x - var_y) / 2.0
        delta = math.sqrt(max(0.0, half_diff**2 + var_xy**2))
        lambda_1 = max(0.0, mean_diag + delta)
        lambda_2 = max(0.0, mean_diag - delta)

        semi_major = math.sqrt(CHI2_2DOF_95 * lambda_1)
        semi_minor = math.sqrt(CHI2_2DOF_95 * lambda_2)
        angle_deg = math.degrees(0.5 * math.atan2(2.0 * var_xy, var_x - var_y))

        return StepCovariance(
            sigma_x_m=math.sqrt(max(0.0, var_x)),
            sigma_y_m=math.sqrt(max(0.0, var_y)),
            uncertainty_radius_m=radius,
            cov_matrix=self.P.copy(),
            semi_major_axis_m=round(semi_major, 3),
            semi_minor_axis_m=round(semi_minor, 3),
            ellipse_angle_deg=round(angle_deg, 2),
            conservative_95_radius_m=round(semi_major, 3)
        )
