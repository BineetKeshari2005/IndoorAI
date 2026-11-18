"""PDR motion propagation model adding stochastic noise to step events."""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from src.map_localization.models import ParticleFilterConfig
from src.map_localization.particle import Particle, wrap_heading_rad
from src.sensor_fusion.models import PdrStepEvent


class PdrMotionModel:
    """Propagates particles along kinematic PDR step displacements with calibrated noise."""

    def __init__(self, config: Optional[ParticleFilterConfig] = None):
        self.config = config or ParticleFilterConfig()
        self.rng = np.random.default_rng(self.config.random_seed)

    def set_seed(self, seed: Optional[int]) -> None:
        """Sets random number generator seed for reproducibility."""
        self.rng = np.random.default_rng(seed)

    def predict_step(
        self,
        particles: List[Particle],
        step_event: PdrStepEvent,
    ) -> List[Particle]:
        """Propagate each particle forward according to a detected PDR step event.

        x'_i = x_i + L_i * cos(theta'_i)
        y'_i = y_i + L_i * sin(theta'_i)
        theta'_i = wrap(theta_i + Delta_theta_i)

        where L_i ~ N(step_length, sigma_L^2) and Delta_theta_i ~ N(heading_delta, sigma_theta^2).
        """
        nominal_length = max(0.0, float(step_event.step_length_m))
        heading_delta = float(step_event.heading_delta_rad)

        sigma_l = float(step_event.sigma_length_m) if step_event.sigma_length_m > 0 else self.config.sigma_step_length_m
        sigma_theta = float(step_event.sigma_heading_rad) if step_event.sigma_heading_rad > 0 else self.config.sigma_heading_rad

        # If sensor gap or magnetic disturbance reported, scale noise up
        if step_event.has_sensor_gap:
            sigma_l *= 1.8
            sigma_theta *= 2.0
        if step_event.is_magnetic_disturbance:
            sigma_theta *= 2.5

        N = len(particles)
        # Sample stochastic perturbations vectorized
        length_noise = self.rng.normal(0.0, sigma_l, N)
        heading_noise = self.rng.normal(0.0, sigma_theta, N)

        updated_particles: List[Particle] = []
        for i, p in enumerate(particles):
            if not p.is_alive:
                updated_particles.append(p)
                continue

            # Compute perturbed step length and heading
            l_i = max(0.0, nominal_length + length_noise[i])
            delta_theta_i = heading_delta + heading_noise[i]

            new_theta = wrap_heading_rad(p.theta + delta_theta_i)
            new_x = p.x + l_i * math.cos(new_theta)
            new_y = p.y + l_i * math.sin(new_theta)

            p_new = p.copy()
            p_new.x = new_x
            p_new.y = new_y
            p_new.theta = new_theta
            updated_particles.append(p_new)

        return updated_particles

    def predict_stationary(
        self,
        particles: List[Particle],
    ) -> List[Particle]:
        """Diffuse heading slightly during detected stationary intervals without spatial displacement."""
        N = len(particles)
        heading_noise = self.rng.normal(0.0, self.config.stationary_sigma_heading_rad, N)

        updated: List[Particle] = []
        for i, p in enumerate(particles):
            if not p.is_alive:
                updated.append(p)
                continue
            p_new = p.copy()
            p_new.theta = wrap_heading_rad(p.theta + heading_noise[i])
            updated.append(p_new)
        return updated

    def apply_stationary_diffusion(
        self,
        particles: List[Particle],
        dt_s: float = 1.0,
    ) -> List[Particle]:
        """Applies stationary diffusion over duration dt_s."""
        return self.predict_stationary(particles)

