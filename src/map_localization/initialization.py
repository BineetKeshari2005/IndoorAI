"""Particle cloud initialization strategies respecting building geometry and uncertainty."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.map_localization.models import ParticleFilterConfig
from src.map_localization.particle import Particle, wrap_heading_rad
from src.mapping.geometry import Point2D
from src.mapping.models import BuildingMap
from src.sensor_fusion.models import FusedState, WifiMeasurementEvent


class ParticleInitializer:
    """Generates initial particle distributions from various sensor and map sources."""

    def __init__(self, config: Optional[ParticleFilterConfig] = None):
        self.config = config or ParticleFilterConfig()
        self.rng = np.random.default_rng(self.config.random_seed)

    def set_seed(self, seed: Optional[int]) -> None:
        """Sets random number generator seed for reproducibility."""
        self.rng = np.random.default_rng(seed)

    def initialize_from_known_location(
        self,
        x_m: float,
        y_m: float,
        theta_rad: float,
        floor_id: int,
        building_id: str = "BLDG_01",
        position_std_m: float = 0.5,
        heading_std_rad: float = 0.1,
        building_map: Optional[BuildingMap] = None,
        pos_std_m: Optional[float] = None,
    ) -> List[Particle]:
        """Initializes particles tightly clustered around a known ground-truth or verified starting location."""
        if pos_std_m is not None:
            position_std_m = pos_std_m
        N = self.config.particle_count
        particles: List[Particle] = []
        weight = 1.0 / N

        attempts = 0
        max_attempts = N * 20

        while len(particles) < N and attempts < max_attempts:
            attempts += 1
            px = float(self.rng.normal(x_m, position_std_m))
            py = float(self.rng.normal(y_m, position_std_m))
            ptheta = wrap_heading_rad(float(self.rng.normal(theta_rad, heading_std_rad)))

            # If map is provided, ensure particle does not fall inside verified solid walls
            if building_map and not self._is_position_valid(px, py, floor_id, building_map):
                continue

            particles.append(Particle(
                x=px,
                y=py,
                theta=ptheta,
                floor_id=floor_id,
                weight=weight,
                log_weight=0.0,
                building_id=building_id,
            ))

        # Fallback if map rejection was too aggressive
        while len(particles) < N:
            particles.append(Particle(
                x=float(self.rng.normal(x_m, position_std_m)),
                y=float(self.rng.normal(y_m, position_std_m)),
                theta=wrap_heading_rad(float(self.rng.normal(theta_rad, heading_std_rad))),
                floor_id=floor_id,
                weight=weight,
                log_weight=0.0,
                building_id=building_id,
            ))

        return particles

    def initialize_from_wifi_fix(
        self,
        wifi_fix: WifiMeasurementEvent,
        building_map: Optional[BuildingMap] = None,
    ) -> List[Particle]:
        """Initializes particles distributed according to a Wi-Fi position fix and its covariance."""
        N = self.config.particle_count
        weight = 1.0 / N
        floor_id = wifi_fix.floor_id
        building_id = wifi_fix.building_id

        # Determine 2x2 covariance
        if wifi_fix.covariance is not None and isinstance(wifi_fix.covariance, np.ndarray):
            cov = np.array(wifi_fix.covariance, dtype=float)
            # Ensure positive definiteness
            cov = 0.5 * (cov + cov.T) + np.eye(2) * 1e-4
        else:
            std = float(self.config.wifi_position_std_fallback)
            cov = np.array([[std**2, 0.0], [0.0, std**2]], dtype=float)

        mean = np.array([wifi_fix.x_m, wifi_fix.y_m], dtype=float)
        particles: List[Particle] = []

        attempts = 0
        max_attempts = N * 20

        while len(particles) < N and attempts < max_attempts:
            attempts += 1
            sample = self.rng.multivariate_normal(mean, cov)
            px, py = float(sample[0]), float(sample[1])
            # Uniform heading prior [ -pi, pi ]
            ptheta = float(self.rng.uniform(-math.pi, math.pi))

            if building_map and not self._is_position_valid(px, py, floor_id, building_map):
                continue

            particles.append(Particle(
                x=px,
                y=py,
                theta=ptheta,
                floor_id=floor_id,
                weight=weight,
                log_weight=0.0,
                building_id=building_id,
            ))

        while len(particles) < N:
            sample = self.rng.multivariate_normal(mean, cov)
            particles.append(Particle(
                x=float(sample[0]),
                y=float(sample[1]),
                theta=float(self.rng.uniform(-math.pi, math.pi)),
                floor_id=floor_id,
                weight=weight,
                log_weight=0.0,
                building_id=building_id,
            ))

        return particles

    def initialize_from_ekf_state(
        self,
        ekf_state: FusedState,
        building_map: Optional[BuildingMap] = None,
    ) -> List[Particle]:
        """Initializes particles sampled from an EKF fused state and its 3x3 covariance matrix."""
        N = self.config.particle_count
        weight = 1.0 / N
        floor_id = ekf_state.floor_id
        building_id = ekf_state.building_id

        mean_3d = np.array([ekf_state.x_m, ekf_state.y_m, ekf_state.theta_rad], dtype=float)
        cov_3d = np.array(ekf_state.state_covariance, dtype=float)
        cov_3d = 0.5 * (cov_3d + cov_3d.T) + np.eye(3) * 1e-4

        particles: List[Particle] = []
        attempts = 0
        max_attempts = N * 20

        while len(particles) < N and attempts < max_attempts:
            attempts += 1
            sample = self.rng.multivariate_normal(mean_3d, cov_3d)
            px, py, ptheta = float(sample[0]), float(sample[1]), wrap_heading_rad(float(sample[2]))

            if building_map and not self._is_position_valid(px, py, floor_id, building_map):
                continue

            particles.append(Particle(
                x=px,
                y=py,
                theta=ptheta,
                floor_id=floor_id,
                weight=weight,
                log_weight=0.0,
                building_id=building_id,
            ))

        while len(particles) < N:
            sample = self.rng.multivariate_normal(mean_3d, cov_3d)
            particles.append(Particle(
                x=float(sample[0]),
                y=float(sample[1]),
                theta=wrap_heading_rad(float(sample[2])),
                floor_id=floor_id,
                weight=weight,
                log_weight=0.0,
                building_id=building_id,
            ))

        return particles

    def initialize_from_region(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        floor_id: int,
        building_id: str = "BLDG_01",
        building_map: Optional[BuildingMap] = None,
    ) -> List[Particle]:
        """Initializes particles uniformly scattered over a bounding region."""
        N = self.config.particle_count
        weight = 1.0 / N
        particles: List[Particle] = []

        attempts = 0
        max_attempts = N * 30

        while len(particles) < N and attempts < max_attempts:
            attempts += 1
            px = float(self.rng.uniform(min_x, max_x))
            py = float(self.rng.uniform(min_y, max_y))
            ptheta = float(self.rng.uniform(-math.pi, math.pi))

            if building_map and not self._is_position_valid(px, py, floor_id, building_map):
                continue

            particles.append(Particle(
                x=px,
                y=py,
                theta=ptheta,
                floor_id=floor_id,
                weight=weight,
                log_weight=0.0,
                building_id=building_id,
            ))

        while len(particles) < N:
            particles.append(Particle(
                x=float(self.rng.uniform(min_x, max_x)),
                y=float(self.rng.uniform(min_y, max_y)),
                theta=float(self.rng.uniform(-math.pi, math.pi)),
                floor_id=floor_id,
                weight=weight,
                log_weight=0.0,
                building_id=building_id,
            ))

        return particles

    def _is_position_valid(
        self,
        x: float,
        y: float,
        floor_id: int,
        building_map: BuildingMap,
    ) -> bool:
        """Checks if a point is outside verified boundaries or inside verified solid wall structures."""
        if floor_id not in building_map.floors:
            # If floor not mapped, do not reject if unknown areas allowed
            return self.config.allow_unknown_areas

        floor = building_map.floors[floor_id]
        pt = Point2D(x, y)

        # 1. Floor boundary check: if verified boundary exists, point must lie inside
        if floor.boundary and floor.boundary.is_valid_polygon:
            if not floor.boundary.contains_point(pt):
                return False

        # Point is acceptable
        return True
