"""Core Map-Constrained Particle Filter engine incorporating geometry, PDR, Wi-Fi, and visual landmarks."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.computer_vision.models import LandmarkRecord
from src.core.logging import setup_logger
from src.map_localization.initialization import ParticleInitializer
from src.map_localization.landmark_observation import LandmarkMeasurementModel
from src.map_localization.map_constraints import MapConstraintEvaluator
from src.map_localization.measurement_model import WifiMeasurementModel
from src.map_localization.models import (
    FilterStatus,
    ObservationType,
    ParticleFilterConfig,
    ParticleFilterState,
)
from src.map_localization.motion_model import PdrMotionModel
from src.map_localization.particle import Particle, wrap_heading_rad
from src.map_localization.resampling import Resampler
from src.mapping.models import BuildingMap
from src.sensor_fusion.models import FusedState, PdrStepEvent, WifiMeasurementEvent

logger = setup_logger("MapConstrainedPF")


class MapConstrainedParticleFilter:
    """Sequential Monte Carlo estimator enforcing verified architectural boundaries and topology."""

    def __init__(
        self,
        config: Optional[ParticleFilterConfig] = None,
        building_map: Optional[BuildingMap] = None,
    ):
        self.config = config or ParticleFilterConfig()
        self.building_map = building_map
        self.particles: List[Particle] = []
        self.history: List[ParticleFilterState] = []
        self.current_state: Optional[ParticleFilterState] = None

        # Submodules
        self.initializer = ParticleInitializer(config=self.config)
        self.motion_model = PdrMotionModel(config=self.config)
        self.map_constraints = MapConstraintEvaluator(config=self.config)
        self.wifi_model = WifiMeasurementModel(config=self.config)
        self.landmark_model = LandmarkMeasurementModel(config=self.config)
        self.resampler = Resampler(config=self.config)

    def set_building_map(self, building_map: BuildingMap) -> None:
        """Assigns or updates active building map geometry and topology."""
        self.building_map = building_map
        self.map_constraints.clear_cache()

    def initialize(
        self,
        particles: Optional[List[Particle]] = None,
        known_location: Optional[Tuple[float, float, float, int, str]] = None,  # (x, y, theta, floor, bldg)
        wifi_fix: Optional[WifiMeasurementEvent] = None,
        ekf_state: Optional[FusedState] = None,
        timestamp_epoch_ms: int = 0,
    ) -> ParticleFilterState:
        """Initializes particle distribution and establishes the initial filtered state."""
        if particles:
            self.particles = [p.copy() for p in particles]
        elif known_location:
            x, y, theta, floor, bldg = known_location
            self.particles = self.initializer.initialize_from_known_location(
                x_m=x, y_m=y, theta_rad=theta, floor_id=floor, building_id=bldg,
                building_map=self.building_map,
            )
        elif wifi_fix:
            self.particles = self.initializer.initialize_from_wifi_fix(
                wifi_fix=wifi_fix, building_map=self.building_map,
            )
        elif ekf_state:
            self.particles = self.initializer.initialize_from_ekf_state(
                ekf_state=ekf_state, building_map=self.building_map,
            )
        else:
            # Default zero fallback
            self.particles = self.initializer.initialize_from_region(
                min_x=0.0, min_y=0.0, max_x=10.0, max_y=10.0, floor_id=0, building_id="BLDG_01",
                building_map=self.building_map,
            )

        self.particles, n_eff = self.resampler.normalize_weights(self.particles)
        state = self._compute_state(
            timestamp_ms=timestamp_epoch_ms,
            obs_type=ObservationType.INITIALIZATION,
            n_eff=n_eff,
            wall_rejected=0,
            resampled=False,
        )
        self.current_state = state
        self.history = [state]
        return state

    def predict_step(
        self,
        step_event: PdrStepEvent,
        ekf_reference_state: Optional[FusedState] = None,
    ) -> ParticleFilterState:
        """Propagate particles forward with kinematic PDR step and evaluate building wall constraints."""
        if not self.particles:
            raise RuntimeError("Particle filter must be initialized before predicting steps.")

        # 1. Motion Propagation
        old_particles = [p.copy() for p in self.particles]
        new_particles = self.motion_model.predict_step(self.particles, step_event)

        # 2. Apply Map Spatial Constraints (reject wall penetrations)
        constrained_particles, wall_rejected = self.map_constraints.apply_constraints(
            old_particles=old_particles,
            new_particles=new_particles,
            building_map=self.building_map,
        )

        # 3. Normalize Weights
        normalized_particles, n_eff = self.resampler.normalize_weights(constrained_particles)
        self.particles = normalized_particles

        # 4. Systematic Resampling check
        resampled = False
        if self.resampler.should_resample(n_eff, len(self.particles)):
            self.particles, _ = self.resampler.systematic_resample(self.particles)
            n_eff = float(len(self.particles))
            resampled = True

        # 5. Compile Estimated State
        state = self._compute_state(
            timestamp_ms=step_event.timestamp_epoch_ms,
            obs_type=ObservationType.PDR_STEP,
            n_eff=n_eff,
            wall_rejected=wall_rejected,
            resampled=resampled,
            ekf_reference_state=ekf_reference_state,
        )
        self.current_state = state
        self.history.append(state)
        return state

    def update_wifi(
        self,
        wifi_event: WifiMeasurementEvent,
        ekf_reference_state: Optional[FusedState] = None,
    ) -> ParticleFilterState:
        """Update particle weights using Wi-Fi position likelihoods and floor transition constraints."""
        if not self.particles:
            raise RuntimeError("Particle filter must be initialized before updating Wi-Fi.")

        # 1. Floor transition check: particles cannot change floors unless near stairs/elevators
        target_floor = wifi_event.floor_id
        for p in self.particles:
            if p.floor_id != target_floor:
                if self.map_constraints.can_transition_floor(p, target_floor, self.building_map):
                    # Particle is in a stairwell/elevator zone: allow it to transition floors!
                    p.floor_id = target_floor

        # 2. Wi-Fi Measurement Model Likelihood Update
        updated_particles = self.wifi_model.update_weights(self.particles, wifi_event)

        # 3. Normalize Weights
        normalized_particles, n_eff = self.resampler.normalize_weights(updated_particles)
        self.particles = normalized_particles

        # 4. Resample check
        resampled = False
        if self.resampler.should_resample(n_eff, len(self.particles)):
            self.particles, _ = self.resampler.systematic_resample(self.particles)
            n_eff = float(len(self.particles))
            resampled = True

        # 5. Compile State
        state = self._compute_state(
            timestamp_ms=wifi_event.timestamp_epoch_ms,
            obs_type=ObservationType.WIFI_FIX,
            n_eff=n_eff,
            wall_rejected=0,
            resampled=resampled,
            ekf_reference_state=ekf_reference_state,
        )
        self.current_state = state
        self.history.append(state)
        return state

    def update_landmark(
        self,
        landmark_record: LandmarkRecord,
        timestamp_epoch_ms: int,
        observed_distance_m: Optional[float] = None,
        ekf_reference_state: Optional[FusedState] = None,
    ) -> ParticleFilterState:
        """Update particle weights using verified visual landmark proximity observations."""
        if not self.particles:
            raise RuntimeError("Particle filter must be initialized before updating landmarks.")

        # 1. Landmark Proximity Likelihood Update
        updated_particles = self.landmark_model.update_weights(
            particles=self.particles,
            landmark=landmark_record,
            observed_distance_m=observed_distance_m,
        )

        # 2. Normalize Weights
        normalized_particles, n_eff = self.resampler.normalize_weights(updated_particles)
        self.particles = normalized_particles

        # 3. Resample check
        resampled = False
        if self.resampler.should_resample(n_eff, len(self.particles)):
            self.particles, _ = self.resampler.systematic_resample(self.particles)
            n_eff = float(len(self.particles))
            resampled = True

        # 4. Compile State
        state = self._compute_state(
            timestamp_ms=timestamp_epoch_ms,
            obs_type=ObservationType.VISUAL_LANDMARK,
            n_eff=n_eff,
            wall_rejected=0,
            resampled=resampled,
            ekf_reference_state=ekf_reference_state,
        )
        self.current_state = state
        self.history.append(state)
        return state

    def _compute_state(
        self,
        timestamp_ms: int,
        obs_type: ObservationType,
        n_eff: float,
        wall_rejected: int,
        resampled: bool,
        ekf_reference_state: Optional[FusedState] = None,
    ) -> ParticleFilterState:
        """Computes weighted point estimate, heading, spread, and operational status."""
        N = len(self.particles)
        weights = np.array([p.weight for p in self.particles], dtype=float)
        xs = np.array([p.x for p in self.particles], dtype=float)
        ys = np.array([p.y for p in self.particles], dtype=float)
        thetas = np.array([p.theta for p in self.particles], dtype=float)

        # 1. Weighted Mean Position
        mean_x = float(np.sum(weights * xs))
        mean_y = float(np.sum(weights * ys))

        # 2. Weighted Circular Mean Heading
        sin_sum = float(np.sum(weights * np.sin(thetas)))
        cos_sum = float(np.sum(weights * np.cos(thetas)))
        mean_theta = math.atan2(sin_sum, cos_sum)

        # 3. Floor Vote (floor with highest accumulated weight)
        floor_weights: Dict[int, float] = {}
        for p in self.particles:
            floor_weights[p.floor_id] = floor_weights.get(p.floor_id, 0.0) + p.weight
        best_floor = max(floor_weights.items(), key=lambda item: item[1])[0] if floor_weights else 0

        # 4. Particle Spread (weighted standard deviation in meters)
        var_x = float(np.sum(weights * (xs - mean_x)**2))
        var_y = float(np.sum(weights * (ys - mean_y)**2))
        spread_m = math.sqrt(max(0.0, var_x + var_y))
        # 95% circular confidence radius for 2D Rayleigh/Gaussian: ~2.447 * sigma
        uncert_m = 2.447 * max(0.5, spread_m)

        # 5. Status Determination
        status = FilterStatus.TRACKING
        if obs_type == ObservationType.INITIALIZATION:
            status = FilterStatus.INITIALIZING
        elif wall_rejected > 0:
            status = FilterStatus.MAP_CONSTRAINED
        elif n_eff < N * 0.15 or spread_m > 12.0:
            status = FilterStatus.LOW_CONFIDENCE
        elif n_eff < 3.0:
            status = FilterStatus.LOST

        bldg_id = self.particles[0].building_id if self.particles else "BLDG_01"
        map_ver = self.building_map.map_version if self.building_map else "1.0.0"

        # Innovation against EKF reference baseline if supplied
        ekf_x = ekf_reference_state.x_m if ekf_reference_state else None
        ekf_y = ekf_reference_state.y_m if ekf_reference_state else None
        innov_x = (mean_x - ekf_x) if ekf_x is not None else None
        innov_y = (mean_y - ekf_y) if ekf_y is not None else None

        return ParticleFilterState(
            timestamp_epoch_ms=timestamp_ms,
            building_id=bldg_id,
            floor_id=best_floor,
            x_m=mean_x,
            y_m=mean_y,
            theta_rad=mean_theta,
            status=status,
            particle_count=N,
            effective_sample_size=n_eff,
            particle_spread_m=spread_m,
            position_uncertainty_m=uncert_m,
            wall_crossings_rejected_count=wall_rejected,
            resampled_this_step=resampled,
            observation_type=obs_type,
            map_version=map_ver,
            ekf_reference_x_m=ekf_x,
            ekf_reference_y_m=ekf_y,
            innovation_x_m=innov_x,
            innovation_y_m=innov_y,
        )
