"""End-to-end Map-Constrained Multimodal Localization Pipeline (Task 27).

Orchestrates chronological execution of PDR propagation, map boundary and wall constraints,
Wi-Fi likelihood updates, and visual landmark observations alongside an optional EKF baseline.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.computer_vision.models import LandmarkRecord
from src.core.logging import setup_logger
from src.map_localization.models import (
    FilterStatus,
    ObservationType,
    ParticleFilterConfig,
    ParticleFilterState,
    ParticleFilterSummary,
)
from src.map_localization.particle_filter import MapConstrainedParticleFilter
from src.mapping.models import BuildingMap
from src.sensor_fusion.coordinates import Pose2D
from src.sensor_fusion.ekf import IndoorEKF
from src.sensor_fusion.models import FusedState, PdrStepEvent, WifiMeasurementEvent

logger = setup_logger("MapLocalizationPipeline")


class MapLocalizationPipeline:
    """Pipeline for running map-constrained particle filtering on multimodal survey sessions."""

    def __init__(
        self,
        config: Optional[ParticleFilterConfig] = None,
        building_map: Optional[BuildingMap] = None,
    ) -> None:
        self.config = config or ParticleFilterConfig()
        self.building_map = building_map
        self.filter = MapConstrainedParticleFilter(config=self.config, building_map=self.building_map)

    def run(
        self,
        pdr_steps: List[PdrStepEvent],
        wifi_measurements: List[WifiMeasurementEvent],
        landmark_observations: Optional[List[Tuple[int, LandmarkRecord, Optional[float]]]] = None,
        initial_pose: Optional[Pose2D] = None,
        building_map: Optional[BuildingMap] = None,
        run_ekf_baseline: bool = True,
    ) -> Tuple[List[ParticleFilterState], ParticleFilterSummary]:
        """Execute chronological particle filtering combining PDR, Wi-Fi, landmarks, and map constraints.

        Args:
            pdr_steps: List of PdrStepEvent objects from PDR engine.
            wifi_measurements: List of WifiMeasurementEvent objects.
            landmark_observations: Optional list of (timestamp_ms, LandmarkRecord, observed_distance_m).
            initial_pose: Optional starting Pose2D (falls back to first Wi-Fi or origin).
            building_map: Optional BuildingMap to override or assign to filter.
            run_ekf_baseline: Whether to execute EKF concurrently as a baseline comparator.

        Returns:
            Tuple of (list of ParticleFilterState estimates, ParticleFilterSummary).
        """
        if building_map is not None:
            self.building_map = building_map
            self.filter.set_building_map(building_map)

        landmark_observations = landmark_observations or []

        # 1. Merge and chronologically tag events
        # Tagged items: (timestamp_ms, event_type, payload)
        tagged_events: List[Tuple[int, str, Any]] = []
        for step in pdr_steps:
            tagged_events.append((step.timestamp_epoch_ms, "PDR", step))
        for wifi in wifi_measurements:
            tagged_events.append((wifi.timestamp_epoch_ms, "WIFI", wifi))
        for ts_lm, lm_rec, dist_m in landmark_observations:
            tagged_events.append((ts_lm, "LANDMARK", (lm_rec, dist_m)))

        tagged_events.sort(key=lambda item: item[0])

        if not tagged_events:
            logger.warning("Map localization pipeline called with zero input events.")
            return [], ParticleFilterSummary()

        first_ts = tagged_events[0][0]

        # 2. Setup initial pose
        if initial_pose is None:
            first_wifi = next((w for w in wifi_measurements if w.x_m is not None and w.y_m is not None), None)
            if first_wifi:
                initial_pose = Pose2D(
                    x_m=first_wifi.x_m,
                    y_m=first_wifi.y_m,
                    theta_rad=0.0,
                    building_id=first_wifi.building_id,
                    floor_id=first_wifi.floor_id,
                )
            else:
                bldg_id = self.building_map.building_id if self.building_map else "BLDG_01"
                initial_pose = Pose2D(x_m=0.0, y_m=0.0, theta_rad=0.0, building_id=bldg_id, floor_id=0)

        # 3. Setup EKF baseline if requested
        ekf: Optional[IndoorEKF] = None
        current_ekf_state: Optional[FusedState] = None
        if run_ekf_baseline:
            try:
                ekf = IndoorEKF()
                current_ekf_state = ekf.initialize(
                    pose=initial_pose,
                    timestamp_ms=first_ts,
                    initial_pos_uncertainty_m=self.config.wifi_position_std_fallback,
                    initial_heading_uncertainty_rad=self.config.sigma_heading_rad * 2.0,
                )
            except Exception as e:
                logger.warning(f"Could not initialize EKF baseline: {e}")
                ekf = None
                current_ekf_state = None

        # 4. Initialize Particle Filter
        init_state = self.filter.initialize(
            known_location=(
                initial_pose.x_m,
                initial_pose.y_m,
                initial_pose.theta_rad,
                initial_pose.floor_id,
                initial_pose.building_id,
            ),
            timestamp_epoch_ms=first_ts,
        )

        states: List[ParticleFilterState] = [init_state]

        # Tracking metrics
        pdr_steps_count = 0
        wifi_updates_count = 0
        landmark_updates_count = 0
        total_wall_crossings_rejected = 0
        resampling_events_count = 0
        deviations_from_ekf: List[float] = []
        total_dist_m = 0.0
        last_x = init_state.x_m
        last_y = init_state.y_m

        # 5. Chronological Filter Loop
        for ts, event_type, payload in tagged_events:
            # Advance EKF baseline first if active
            if ekf is not None:
                try:
                    if event_type == "PDR":
                        current_ekf_state = ekf.predict_pdr_step(payload)
                    elif event_type == "WIFI":
                        current_ekf_state = ekf.update_wifi(payload)
                except Exception as e:
                    logger.debug(f"EKF baseline step warning at {ts}: {e}")

            # Advance Particle Filter
            pf_state: Optional[ParticleFilterState] = None
            if event_type == "PDR":
                step_evt: PdrStepEvent = payload
                pdr_steps_count += 1
                pf_state = self.filter.predict_step(step_evt, ekf_reference_state=current_ekf_state)
                total_wall_crossings_rejected += pf_state.wall_crossings_rejected_count

            elif event_type == "WIFI":
                wifi_evt: WifiMeasurementEvent = payload
                wifi_updates_count += 1
                pf_state = self.filter.update_wifi(wifi_evt, ekf_reference_state=current_ekf_state)

            elif event_type == "LANDMARK":
                lm_rec, dist_m = payload
                landmark_updates_count += 1
                pf_state = self.filter.update_landmark(
                    landmark_record=lm_rec,
                    timestamp_epoch_ms=ts,
                    observed_distance_m=dist_m,
                    ekf_reference_state=current_ekf_state,
                )

            if pf_state is not None:
                states.append(pf_state)
                if pf_state.resampled_this_step:
                    resampling_events_count += 1

                # Distance calculation
                dx = pf_state.x_m - last_x
                dy = pf_state.y_m - last_y
                step_len = math.hypot(dx, dy)
                if event_type == "PDR":
                    total_dist_m += step_len
                last_x = pf_state.x_m
                last_y = pf_state.y_m

                # EKF deviation tracking
                if pf_state.innovation_x_m is not None and pf_state.innovation_y_m is not None:
                    dev = math.hypot(pf_state.innovation_x_m, pf_state.innovation_y_m)
                    deviations_from_ekf.append(dev)

        # 6. Aggregate Summary
        duration_s = max(0.0, (tagged_events[-1][0] - first_ts) / 1000.0)
        mean_neff = float(np.mean([s.effective_sample_size for s in states])) if states else 0.0
        mean_spread = float(np.mean([s.particle_spread_m for s in states])) if states else 0.0
        mean_unc = float(np.mean([s.position_uncertainty_m for s in states])) if states else 0.0

        mean_ekf_dev = float(np.mean(deviations_from_ekf)) if deviations_from_ekf else None
        max_ekf_dev = float(np.max(deviations_from_ekf)) if deviations_from_ekf else None

        summary = ParticleFilterSummary(
            total_states=len(states),
            pdr_steps_count=pdr_steps_count,
            wifi_updates_count=wifi_updates_count,
            landmark_updates_count=landmark_updates_count,
            total_wall_crossings_rejected=total_wall_crossings_rejected,
            resampling_events_count=resampling_events_count,
            mean_effective_sample_size=mean_neff,
            mean_particle_spread_m=mean_spread,
            mean_position_uncertainty_m=mean_unc,
            total_distance_traveled_m=total_dist_m,
            mean_deviation_from_ekf_m=mean_ekf_dev,
            max_deviation_from_ekf_m=max_ekf_dev,
            duration_seconds=duration_s,
            status=states[-1].status if states else FilterStatus.INITIALIZING,
        )

        return states, summary
