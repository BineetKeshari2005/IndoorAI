"""Visual landmark measurement model applying spatial proximity constraints."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from src.computer_vision.models import LandmarkRecord, VerificationStatus
from src.map_localization.models import ParticleFilterConfig
from src.map_localization.particle import Particle


class LandmarkMeasurementModel:
    """Updates particle weights based on visual recognition of verified indoor landmarks.

    A detected landmark constrains the camera to a viewing proximity zone rather than
    implying exact 2D point coincidence.
    """

    def __init__(self, config: Optional[ParticleFilterConfig] = None):
        self.config = config or ParticleFilterConfig()

    def update_weights(
        self,
        particles: List[Particle],
        landmark: LandmarkRecord,
        observed_distance_m: Optional[float] = None,
    ) -> List[Particle]:
        """Updates particle likelihoods given an observation of a verified landmark."""
        # 1. Verification and coordinates check
        if landmark.verification_status != VerificationStatus.MANUALLY_VERIFIED:
            # Only manually verified landmarks provide valid spatial constraints
            return particles

        if not landmark.verified_coordinates_m:
            return particles

        lm_x, lm_y = landmark.verified_coordinates_m
        lm_floor = landmark.floor_id
        r_max = self.config.landmark_visibility_radius_m
        sigma_lm = r_max / 2.0  # 95% of visibility within r_max

        min_log_lik = -25.0
        floor_penalty = self.config.floor_mismatch_log_penalty

        updated_particles: List[Particle] = []
        for p in particles:
            if not p.is_alive:
                updated_particles.append(p)
                continue

            dist = p.distance_to(lm_x, lm_y)

            # Gaussian proximity likelihood
            log_lik = -0.5 * (dist / sigma_lm)**2

            # Beyond viewing range, apply steep decay
            if dist > r_max:
                excess = dist - r_max
                log_lik -= excess * 2.0

            # Floor mismatch penalty
            if p.floor_id != lm_floor:
                log_lik += floor_penalty

            log_lik = max(min_log_lik, log_lik)

            p_new = p.copy()
            p_new.log_weight += log_lik
            updated_particles.append(p_new)

        return updated_particles
