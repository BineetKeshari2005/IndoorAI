"""Weight normalization, effective sample size, and systematic resampling algorithms."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from src.map_localization.models import ParticleFilterConfig, ResampleAlgorithm
from src.map_localization.particle import Particle, wrap_heading_rad


class Resampler:
    """Manages weight normalization, degeneracy detection, and low-variance systematic resampling."""

    def __init__(self, config: Optional[ParticleFilterConfig] = None):
        self.config = config or ParticleFilterConfig()
        self.rng = np.random.default_rng(self.config.random_seed)

    def set_seed(self, seed: Optional[int]) -> None:
        """Sets random number generator seed for reproducibility."""
        self.rng = np.random.default_rng(seed)

    def normalize_weights(self, particles: List[Particle]) -> Tuple[List[Particle], float]:
        """Normalizes particle weights using numerically stable log-sum-exp.

        Returns:
            Tuple of (updated particles with normalized weights, effective sample size Neff).
        """
        N = len(particles)
        if N == 0:
            return [], 0.0

        log_weights = np.array([p.log_weight for p in particles], dtype=float)
        alive_mask = np.array([p.is_alive for p in particles], dtype=bool)

        # If all particles died (e.g. all penetrated walls), restore uniform weights
        if not np.any(alive_mask) or np.all(np.isneginf(log_weights)):
            uniform_w = 1.0 / N
            for p in particles:
                p.weight = uniform_w
                p.log_weight = math.log(uniform_w)
                p.is_alive = True
            return particles, float(N)

        max_log = float(np.max(log_weights[alive_mask]))
        shifted_weights = np.zeros(N, dtype=float)
        shifted_weights[alive_mask] = np.exp(log_weights[alive_mask] - max_log)

        total_weight = float(np.sum(shifted_weights))
        if total_weight <= 1e-15:
            uniform_w = 1.0 / N
            for p in particles:
                p.weight = uniform_w
                p.log_weight = math.log(uniform_w)
                p.is_alive = True
            return particles, float(N)

        normalized_weights = shifted_weights / total_weight

        # Effective sample size: Neff = 1 / sum(w_i^2)
        sum_sq = float(np.sum(normalized_weights**2))
        n_eff = (1.0 / sum_sq) if sum_sq > 0 else 1.0

        for i, p in enumerate(particles):
            p.weight = float(normalized_weights[i])
            # Reset log_weight to ln(w_i) for next cycle
            p.log_weight = math.log(max(1e-15, p.weight))

        return particles, n_eff

    def should_resample(self, n_eff: float, particle_count: int) -> bool:
        """Determines whether Neff has fallen below the resampling threshold."""
        threshold = particle_count * self.config.resample_threshold_ratio
        return n_eff < threshold

    def systematic_resample(
        self,
        particles: List[Particle],
        jitter_sigma_m: float = 0.05,
        n_particles: Optional[int] = None,
    ) -> Tuple[List[Particle], float]:
        """Performs $O(N)$ low-variance systematic resampling on normalized particles.

        Returns:
            Tuple of (new resampled particles with uniform weights 1/N, diversity ratio).
        """
        N = n_particles if n_particles is not None else len(particles)
        if N == 0 or len(particles) == 0:
            return [], 0.0

        weights = np.array([p.weight for p in particles], dtype=float)
        cumsum = np.cumsum(weights)
        cumsum[-1] = 1.0  # Guarantee exact upper bound

        # Single random draw for systematic comb
        u0 = float(self.rng.uniform(0.0, 1.0 / N))
        step = 1.0 / N

        resampled: List[Particle] = []
        ancestor_indices: List[int] = []

        idx = 0
        for k in range(N):
            u_k = u0 + k * step
            while idx < N - 1 and u_k > cumsum[idx]:
                idx += 1
            ancestor_indices.append(idx)

            p_parent = particles[idx]
            p_child = p_parent.copy()
            p_child.weight = 1.0 / N
            p_child.log_weight = math.log(1.0 / N)
            p_child.is_alive = True

            # Slight stochastic roughening jitter to prevent particle collapse
            if jitter_sigma_m > 0:
                p_child.x += float(self.rng.normal(0.0, jitter_sigma_m))
                p_child.y += float(self.rng.normal(0.0, jitter_sigma_m))
                p_child.theta = wrap_heading_rad(p_child.theta + float(self.rng.normal(0.0, 0.02)))

            resampled.append(p_child)

        # Particle diversity metric: proportion of unique ancestor particles
        unique_ancestors = len(set(ancestor_indices))
        diversity_ratio = unique_ancestors / float(N)

        return resampled, diversity_ratio
