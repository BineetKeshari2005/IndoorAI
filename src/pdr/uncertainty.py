"""Spatial covariance and uncertainty propagation for Pedestrian Dead Reckoning (Task 21)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class StepCovariance:
    """Propagated 2D spatial uncertainty for a single step update."""

    sigma_x_m: float
    sigma_y_m: float
    uncertainty_radius_m: float
    cov_matrix: np.ndarray


class PDRUncertaintyPropagator:
    """Propagates positional uncertainty using first-order Taylor expansion over step length and heading errors."""

    def __init__(self, initial_uncertainty_m: float = 0.5):
        # Initial 2x2 covariance matrix
        self.P = np.eye(2, dtype=np.float64) * (initial_uncertainty_m ** 2)

    def update(
        self,
        step_length_m: float,
        heading_rad: float,
        sigma_length_m: float,
        sigma_heading_rad: float,
        has_sensor_gap: bool = False,
        is_magnetic_disturbance: bool = False
    ) -> StepCovariance:
        """Propagates position covariance with step displacement noise.

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
        # Inflate variance if gaps or magnetic disturbances occurred
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
        radius = math.sqrt(max(0.0, var_x + var_y))

        return StepCovariance(
            sigma_x_m=math.sqrt(max(0.0, var_x)),
            sigma_y_m=math.sqrt(max(0.0, var_y)),
            uncertainty_radius_m=radius,
            cov_matrix=self.P.copy()
        )
