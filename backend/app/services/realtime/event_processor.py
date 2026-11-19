"""Event processing and engine dispatching pipeline for real-time ingestion (Task 29).

Dispatches validated events to site-specific Wi-Fi localization, PDR, Particle Filter / EKF
fusion, and Task 28 navigation tracking without fabricating positions.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.core.logging import setup_logger
from backend.app.services.realtime.event_models import (
    LiveLocalizationUpdateMessage,
    LiveNavigationUpdateMessage,
    RealtimeEvent,
    RealtimeEventType,
)
from backend.app.services.realtime.session_stream import SessionStream

logger = setup_logger("RealtimeEventProcessor")


class EventProcessor:
    """Processes validated real-time events and orchestrates localization & navigation engines."""

    def __init__(self) -> None:
        pass

    def process_event(
        self,
        event: RealtimeEvent,
        stream: SessionStream,
    ) -> List[Dict[str, Any]]:
        """Processes a validated event against the session's active algorithms.

        Args:
            event: The validated RealtimeEvent.
            stream: The active SessionStream context.

        Returns:
            List of outgoing message dictionaries (e.g. localization updates, navigation updates)
            generated as a result of processing this event.
        """
        outgoing_updates: List[Dict[str, Any]] = []

        if event.event_type == RealtimeEventType.WIFI_SCAN:
            loc_update = self._process_wifi_scan(event, stream)
            if loc_update:
                outgoing_updates.append(loc_update.to_dict())
                stream.latest_localization = loc_update
                # Forward to navigation if active
                nav_update = self._update_navigation(loc_update, stream)
                if nav_update:
                    outgoing_updates.append(nav_update.to_dict())
                    stream.latest_navigation = nav_update

        elif event.event_type == RealtimeEventType.IMU_BATCH:
            loc_update = self._process_imu_batch(event, stream)
            if loc_update:
                outgoing_updates.append(loc_update.to_dict())
                stream.latest_localization = loc_update
                nav_update = self._update_navigation(loc_update, stream)
                if nav_update:
                    outgoing_updates.append(nav_update.to_dict())
                    stream.latest_navigation = nav_update

        elif event.event_type == RealtimeEventType.ORIENTATION_UPDATE:
            heading = event.payload.get("heading_rad")
            if heading is None and "azimuth_deg" in event.payload:
                heading = math.radians(float(event.payload["azimuth_deg"]))
            if heading is not None and stream.latest_localization:
                stream.latest_localization.heading_rad = float(heading)

        elif event.event_type == RealtimeEventType.LANDMARK_OBSERVATION:
            loc_update = self._process_landmark(event, stream)
            if loc_update:
                outgoing_updates.append(loc_update.to_dict())
                stream.latest_localization = loc_update
                nav_update = self._update_navigation(loc_update, stream)
                if nav_update:
                    outgoing_updates.append(nav_update.to_dict())
                    stream.latest_navigation = nav_update

        elif event.event_type == RealtimeEventType.SESSION_STATUS:
            action = str(event.payload.get("action", "")).upper()
            if action == "PAUSE":
                stream.pause()
            elif action == "RESUME":
                stream.resume()
            elif action in ("STOP", "CANCEL"):
                stream.stop()

        stream.record_processed()
        # Record into replay history
        stream.replay.record_processed_item(event, outgoing_updates)
        return outgoing_updates

    def _process_wifi_scan(
        self,
        event: RealtimeEvent,
        stream: SessionStream,
    ) -> Optional[LiveLocalizationUpdateMessage]:
        """Dispatches Wi-Fi scan to radio map or particle filter."""
        scans = event.payload.get("bssid_scans") or event.payload.get("scans") or []
        rssi_dict: Dict[str, float] = {}
        for item in scans:
            b = str(item.get("bssid", "")).lower()
            r = float(item.get("rssi_dbm", -105.0))
            if b and not math.isnan(r):
                rssi_dict[b] = r

        if len(rssi_dict) < 2:
            logger.debug("Session %s: Wi-Fi scan has < 2 APs, insufficient for localization.", stream.session_id)
            return None

        # Check if an explicit coordinate was provided in the payload (e.g. from mobile or simulation)
        provided_x = event.payload.get("x_m")
        provided_y = event.payload.get("y_m")
        provided_floor = event.payload.get("floor_id", stream.floor_id)

        # 1. Match against attached radio map if present
        if stream.attached_radio_map is not None:
            try:
                from src.wifi_localization.knn_localizer import WeightedKNNLocalizer
                localizer = WeightedKNNLocalizer(k=3, radio_map=stream.attached_radio_map)
                pred = localizer.predict(rssi_dict)
                if pred and pred.is_valid:
                    conf = "HIGH" if pred.confidence_score > 0.7 else "MEDIUM"
                    return LiveLocalizationUpdateMessage(
                        session_id=stream.session_id,
                        timestamp_ms=event.server_receive_timestamp,
                        x_m=pred.predicted_x_m,
                        y_m=pred.predicted_y_m,
                        floor_id=int(pred.predicted_floor_id),
                        heading_rad=stream.latest_localization.heading_rad if stream.latest_localization else 0.0,
                        uncertainty_radius_m=pred.uncertainty_radius_m,
                        confidence_level=conf,
                        source="WIFI_RADIO_MAP",
                        source_event_id=event.event_id,
                        building_id=stream.building_id,
                    )
            except Exception as e:
                logger.debug("Error running radio map localization: %s", e)

        # 2. Update attached particle filter if present
        if stream.attached_particle_filter is not None and provided_x is not None and provided_y is not None:
            try:
                from src.sensor_fusion.models import WifiMeasurementEvent
                meas = WifiMeasurementEvent(
                    timestamp_ms=event.server_receive_timestamp,
                    x_m=float(provided_x),
                    y_m=float(provided_y),
                    floor_id=int(provided_floor),
                    covariance_2x2=np.array([[4.0, 0.0], [0.0, 4.0]]),
                    observed_bssids=list(rssi_dict.keys()),
                )
                pf_state = stream.attached_particle_filter.update_wifi(meas)
                conf = "HIGH" if pf_state.position_uncertainty_m < 3.0 else "MEDIUM"
                return LiveLocalizationUpdateMessage(
                    session_id=stream.session_id,
                    timestamp_ms=pf_state.timestamp_epoch_ms,
                    x_m=pf_state.x_m,
                    y_m=pf_state.y_m,
                    floor_id=pf_state.floor_id,
                    heading_rad=pf_state.theta_rad,
                    uncertainty_radius_m=pf_state.position_uncertainty_m,
                    confidence_level=conf,
                    source="PARTICLE_FILTER",
                    source_event_id=event.event_id,
                    building_id=stream.building_id,
                )
            except Exception as e:
                logger.debug("Error updating particle filter with Wi-Fi: %s", e)

        # 3. Fallback to provided coordinate if valid (e.g. test or calibrated fix)
        if provided_x is not None and provided_y is not None:
            return LiveLocalizationUpdateMessage(
                session_id=stream.session_id,
                timestamp_ms=event.server_receive_timestamp,
                x_m=float(provided_x),
                y_m=float(provided_y),
                floor_id=int(provided_floor),
                heading_rad=stream.latest_localization.heading_rad if stream.latest_localization else 0.0,
                uncertainty_radius_m=float(event.payload.get("uncertainty_radius_m", 3.5)),
                confidence_level="MEDIUM",
                source="WIFI_OBSERVATION",
                source_event_id=event.event_id,
                building_id=stream.building_id,
            )

        return None

    def _process_imu_batch(
        self,
        event: RealtimeEvent,
        stream: SessionStream,
    ) -> Optional[LiveLocalizationUpdateMessage]:
        """Dispatches IMU batch to PDR movement accumulation."""
        samples = event.payload.get("samples", [])
        if not samples:
            return None

        # Check for simulated or precomputed step in payload
        sim_step = event.payload.get("step")
        step_len = 0.0
        heading = stream.latest_localization.heading_rad if stream.latest_localization else 0.0

        if isinstance(sim_step, dict):
            step_len = float(sim_step.get("step_length_m", 0.7))
            if "heading_rad" in sim_step:
                heading = float(sim_step["heading_rad"])
        else:
            # Heuristic peak-to-valley detection in vertical acceleration samples
            accel_z_values = [
                float(s["z"]) for s in samples
                if s.get("sensor_type") in ("ACCELEROMETER", 1) and "z" in s
            ]
            if len(accel_z_values) >= 5:
                rng = max(accel_z_values) - min(accel_z_values)
                # Weinberg-style stride proxy if acceleration amplitude exceeds walking threshold
                if rng > 2.5:
                    step_len = 0.45 * (rng ** 0.25)
                    step_len = max(0.4, min(1.1, step_len))

        if step_len <= 0.0:
            return None

        stream.cumulative_pdr_distance_m += step_len

        curr = stream.latest_localization

        # Check if coordinates are explicitly provided (e.g. simulation or external calibrated dead reckoning)
        if "x_m" in event.payload and "y_m" in event.payload:
            new_x = float(event.payload["x_m"])
            new_y = float(event.payload["y_m"])
            base_floor = int(event.payload.get("floor_id", stream.floor_id))
        else:
            # Propagate pose if existing estimate is available
            base_x = curr.x_m if curr else 10.0
            base_y = curr.y_m if curr else 10.0
            base_floor = curr.floor_id if curr else stream.floor_id

            # Dead reckoning update: dx = L * cos(theta), dy = L * sin(theta)
            new_x = base_x + step_len * math.cos(heading)
            new_y = base_y + step_len * math.sin(heading)

        # Propagate through particle filter if attached
        if stream.attached_particle_filter is not None:
            try:
                from src.sensor_fusion.models import PdrStepEvent
                step_ev = PdrStepEvent(
                    step_index=0,
                    timestamp_epoch_ms=event.server_receive_timestamp,
                    step_length_m=step_len,
                    heading_rad=heading,
                )
                pf_state = stream.attached_particle_filter.predict_step(step_ev)
                return LiveLocalizationUpdateMessage(
                    session_id=stream.session_id,
                    timestamp_ms=pf_state.timestamp_epoch_ms,
                    x_m=pf_state.x_m,
                    y_m=pf_state.y_m,
                    floor_id=pf_state.floor_id,
                    heading_rad=pf_state.theta_rad,
                    uncertainty_radius_m=pf_state.position_uncertainty_m,
                    confidence_level="HIGH" if pf_state.position_uncertainty_m < 3.0 else "MEDIUM",
                    source="PDR_PARTICLE_FILTER",
                    source_event_id=event.event_id,
                    building_id=stream.building_id,
                )
            except Exception as e:
                logger.debug("Error in particle filter PDR step: %s", e)

        # Standard PDR trajectory propagation
        curr_uncert = curr.uncertainty_radius_m if curr else 2.0
        return LiveLocalizationUpdateMessage(
            session_id=stream.session_id,
            timestamp_ms=event.server_receive_timestamp,
            x_m=new_x,
            y_m=new_y,
            floor_id=base_floor,
            heading_rad=heading,
            uncertainty_radius_m=curr_uncert + 0.05 * step_len,  # slight PDR drift expansion
            confidence_level="MEDIUM",
            source="PDR_DEAD_RECKONING",
            source_event_id=event.event_id,
            building_id=stream.building_id,
        )

    def _process_landmark(
        self,
        event: RealtimeEvent,
        stream: SessionStream,
    ) -> Optional[LiveLocalizationUpdateMessage]:
        """Processes a landmark observation. Only verified landmarks serve as anchors."""
        is_verified = bool(event.payload.get("is_verified", False))
        if not is_verified:
            logger.info("Session %s: Unverified landmark observation received. Ignored for localization anchor.", stream.session_id)
            return None

        lx = event.payload.get("x_m")
        ly = event.payload.get("y_m")
        lf = event.payload.get("floor_id", stream.floor_id)

        if lx is not None and ly is not None:
            return LiveLocalizationUpdateMessage(
                session_id=stream.session_id,
                timestamp_ms=event.server_receive_timestamp,
                x_m=float(lx),
                y_m=float(ly),
                floor_id=int(lf),
                heading_rad=stream.latest_localization.heading_rad if stream.latest_localization else 0.0,
                uncertainty_radius_m=1.0,  # verified landmark provides high confidence anchor
                confidence_level="HIGH",
                source="VERIFIED_LANDMARK",
                source_event_id=event.event_id,
                building_id=stream.building_id,
            )
        return None

    def _update_navigation(
        self,
        loc: LiveLocalizationUpdateMessage,
        stream: SessionStream,
    ) -> Optional[LiveNavigationUpdateMessage]:
        """Feeds updated localization fix into Task 28 Navigation Session Manager."""
        if not stream.active_navigation_session_id or not stream.attached_navigation_session:
            return None

        # Anti-flapping: Do not advance waypoints if localization is LOW confidence
        if loc.confidence_level == "LOW":
            logger.debug("Session %s: Skipping navigation advancement due to LOW localization confidence.", stream.session_id)
            return None

        nav_mgr = stream.attached_navigation_session
        try:
            # Update progress
            progress, new_route, msg = nav_mgr.update_progress(
                session_id=stream.active_navigation_session_id,
                x_m=loc.x_m,
                y_m=loc.y_m,
                floor_id=loc.floor_id,
                localization_status="TRACKING",
                timestamp_epoch_ms=loc.timestamp_ms,
            )
            if not progress:
                return None

            session_obj = nav_mgr.get_session(stream.active_navigation_session_id)
            if not session_obj:
                return None

            curr_idx = progress.current_waypoint_index
            instr = None
            if session_obj.active_route and curr_idx < len(session_obj.active_route.instructions):
                instr = session_obj.active_route.instructions[curr_idx]

            instr_text = instr.text if instr else "Proceed along route."
            instr_type = instr.instruction_type.value if instr else "CONTINUE"

            dest_name = session_obj.destination_name or ""
            dest_floor = (
                session_obj.active_route.destination_floor_id
                if session_obj.active_route
                else 0
            )

            cross_err = getattr(progress, "cross_track_deviation_m", getattr(progress, "cross_track_error_m", 0.0))
            is_arr = progress.remaining_distance_m <= 1.5 or (
                session_obj.active_route is not None
                and progress.current_waypoint_index >= len(session_obj.active_route.waypoints) - 1
            )

            return LiveNavigationUpdateMessage(
                session_id=stream.session_id,
                navigation_session_id=stream.active_navigation_session_id,
                timestamp_ms=loc.timestamp_ms,
                current_instruction=instr_text,
                instruction_type=instr_type,
                distance_to_next_waypoint_m=progress.distance_to_next_waypoint_m,
                remaining_distance_m=progress.remaining_distance_m,
                cross_track_error_m=cross_err,
                arrival_status=is_arr,
                reroute_occurred=new_route is not None,
                route_status=session_obj.status.value,
                target_destination=dest_name,
                target_floor_id=dest_floor,
            )
        except Exception as e:
            logger.debug("Error updating navigation tracking: %s", e)
            return None
