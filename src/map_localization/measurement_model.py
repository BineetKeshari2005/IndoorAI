"""Wi-Fi measurement observation model updating particle weights using spatial likelihoods."""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from src.map_localization.models import ParticleFilterConfig
from src.map_localization.particle import Particle
from src.sensor_fusion.models import WifiMeasurementEvent


class WifiMeasurementModel:
    """Computes Gaussian observation likelihoods for Wi-Fi position fixes with floor gating."""

    def __init__(self, config: Optional[ParticleFilterConfig] = None):
        self.config = config or ParticleFilterConfig()

    def update_weights(
        self,
        particles: List[Particle],
        wifi_event: WifiMeasurementEvent,
    ) -> List[Particle]:
        """Update particle unnormalized log weights based on Wi-Fi observation likelihood.

        ln p(z | s_i) = -0.5 * (z - [x_i, y_i])^T * R^{-1} * (z - [x_i, y_i]) - 0.5 * ln|2*pi*R|
        """
        zx = float(wifi_event.x_m)
        zy = float(wifi_event.y_m)
        obs_floor = int(wifi_event.floor_id)

        # 1. Resolve 2x2 measurement covariance
        if wifi_event.covariance is not None and isinstance(wifi_event.covariance, np.ndarray):
            cov = np.array(wifi_event.covariance, dtype=float)
            cov = 0.5 * (cov + cov.T) + np.eye(2) * 1e-4
        else:
            std = float(self.config.wifi_position_std_fallback)
            cov = np.array([[std**2, 0.0], [0.0, std**2]], dtype=float)

        try:
            inv_cov = np.linalg.inv(cov)
            det_cov = np.linalg.det(cov)
            norm_const = -0.5 * math.log(max(1e-12, (2.0 * math.pi)**2 * det_cov))
        except np.linalg.LinAlgError:
            std = float(self.config.wifi_position_std_fallback)
            inv_cov = np.eye(2) / (std**2)
            norm_const = -math.log(2.0 * math.pi * (std**2))

        # Soft floor to prevent particle collapse from anomalous single scans
        min_log_lik = -30.0
        floor_penalty = self.config.floor_mismatch_log_penalty

        updated_particles: List[Particle] = []
        for p in particles:
            if not p.is_alive:
                updated_particles.append(p)
                continue

            dx = p.x - zx
            dy = p.y - zy
            diff = np.array([dx, dy])

            mahalanobis_sq = float(diff.T @ inv_cov @ diff)
            log_lik = norm_const - 0.5 * mahalanobis_sq

            # Floor mismatch penalty
            if p.floor_id != obs_floor:
                log_lik += floor_penalty

            log_lik = max(min_log_lik, log_lik)

            p_new = p.copy()
            p_new.log_weight += log_lik
            updated_particles.append(p_new)

        return updated_particles
