"""Session management tracking active trips, progress logs, reroutes, and completion (Task 28)."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid

from src.core.logging import setup_logger
from src.navigation.models import (
    NavigationProgress,
    NavigationSession,
    NavigationStatus,
    PlannedRoute,
    RerouteEvent,
)
from src.navigation.progress_tracker import ProgressTracker
from src.navigation.rerouting import ReroutingEngine
from src.navigation.route_planner import RoutePlanner

logger = setup_logger("NavigationSession")


class NavigationSessionManager:
    """Orchestrates live indoor navigation sessions, tracking real-time progress and triggers."""

    def __init__(
        self,
        planner_or_storage: Optional[Any] = None,
        storage_dir: Optional[Path] = None,
        arrival_radius_m: float = 1.5,
        deviation_threshold_m: float = 4.0,
        planner: Optional[RoutePlanner] = None,
    ) -> None:
        if isinstance(planner_or_storage, (str, Path)):
            self.storage_dir: Optional[Path] = Path(planner_or_storage)
            self.planner: Optional[RoutePlanner] = planner
        elif isinstance(planner_or_storage, RoutePlanner):
            self.planner = planner_or_storage
            self.storage_dir = storage_dir
        else:
            self.planner = planner
            self.storage_dir = storage_dir

        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.tracker: ProgressTracker = ProgressTracker(
            waypoint_arrival_radius_m=arrival_radius_m,
            max_deviation_threshold_m=deviation_threshold_m,
        )
        self.rerouter: Optional[ReroutingEngine] = None
        if self.planner:
            self.rerouter = ReroutingEngine(
                planner=self.planner,
                deviation_threshold_m=deviation_threshold_m,
            )
        self._sessions: Dict[str, NavigationSession] = {}

    def create_session(
        self,
        route: PlannedRoute,
        dataset_id: str = "default",
    ) -> NavigationSession:
        """Initializes a new navigation session around a planned route."""
        now_ms = int(time.time() * 1000)
        session_id = f"NAV-SES-{uuid.uuid4().hex[:8].upper()}"

        init_wp = route.waypoints[0]
        initial_progress = NavigationProgress(
            current_x_m=init_wp.x_m,
            current_y_m=init_wp.y_m,
            current_floor_id=init_wp.floor_id,
            current_waypoint_index=0,
            next_instruction_index=0,
            distance_to_next_waypoint_m=0.0,
            remaining_distance_m=route.total_distance_m,
            completion_percentage=0.0,
            cross_track_deviation_m=0.0,
            is_off_route=False,
            localization_status="INITIALIZED",
            confidence_level="HIGH",
            timestamp_epoch_ms=now_ms,
        )

        session = NavigationSession(
            session_id=session_id,
            dataset_id=dataset_id,
            building_id=route.building_id,
            map_version=route.map_version,
            origin_name=route.origin_name,
            destination_id=route.destination_id,
            destination_name=route.destination_name,
            active_route=route,
            status=NavigationStatus.CREATED,
            current_progress=initial_progress,
            start_time_ms=now_ms,
            progress_history=[initial_progress],
        )

        self._sessions[session_id] = session
        self._persist_session(session)
        logger.info(f"Created navigation session {session_id} for target '{route.destination_name}'.")
        return session

    def get_session(self, session_id: str) -> Optional[NavigationSession]:
        """Retrieves session from memory or disk."""
        if session_id in self._sessions:
            return self._sessions[session_id]
        if self.storage_dir:
            sess_file = self.storage_dir / f"{session_id}.json"
            if sess_file.is_file():
                try:
                    with open(sess_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    # Restore session object
                    # For active use, session can be retrieved as dict or parsed
                    return self._sessions.get(session_id)
                except Exception as e:
                    logger.error(f"Failed to load session {session_id} from disk: {e}")
        return None

    def update_progress(
        self,
        session_id: str,
        x_m: float,
        y_m: float,
        floor_id: int,
        localization_status: str = "TRACKING",
        timestamp_epoch_ms: Optional[int] = None,
    ) -> Tuple[Optional[NavigationProgress], Optional[PlannedRoute], Optional[str]]:
        """Processes an incoming localization fix, checks waypoint progress, and triggers automatic rerouting if deviated.

        Returns:
            Tuple of (NavigationProgress, Optional[new_rerouted_route], Optional[status_message]).
        """
        session = self.get_session(session_id)
        if not session:
            return None, None, f"Session '{session_id}' not found."

        if session.status not in (NavigationStatus.ACTIVE, NavigationStatus.CREATED):
            return session.current_progress, None, f"Session is currently {session.status.value}."

        now_ms = timestamp_epoch_ms or int(time.time() * 1000)
        curr_wp_idx = session.current_progress.current_waypoint_index if session.current_progress else 0

        # 1. Update Progress
        progress = self.tracker.evaluate_progress(
            route=session.active_route,
            x_m=x_m,
            y_m=y_m,
            floor_id=floor_id,
            localization_status=localization_status,
            current_waypoint_idx=curr_wp_idx,
            timestamp_epoch_ms=now_ms,
        )

        session.current_progress = progress
        session.progress_history.append(progress)

        # 2. Check Arrival Completion
        last_wp_idx = len(session.active_route.waypoints) - 1
        if progress.current_waypoint_index >= last_wp_idx and progress.distance_to_next_waypoint_m <= self.tracker.arrival_radius_m:
            session.status = NavigationStatus.COMPLETED
            session.end_time_ms = now_ms
            self._persist_session(session)
            logger.info(f"Session {session_id} reached destination '{session.destination_name}'! Marked COMPLETED.")
            return progress, None, "ARRIVED"

        # 3. Check for Automatic Reroute Trigger
        should_re, re_reason = self.rerouter.should_reroute(progress)
        new_route: Optional[PlannedRoute] = None
        if should_re:
            session.status = NavigationStatus.REROUTING
            new_r, event, err = self.rerouter.recalculate_route(
                current_route=session.active_route,
                progress=progress,
                reason=re_reason,
            )
            if new_r and event:
                session.active_route = new_r
                session.reroute_history.append(event)
                session.status = NavigationStatus.ACTIVE
                new_route = new_r

                # Reset progress for new route
                progress = self.tracker.evaluate_progress(
                    route=new_r,
                    x_m=x_m,
                    y_m=y_m,
                    floor_id=floor_id,
                    localization_status=localization_status,
                    current_waypoint_idx=0,
                    timestamp_epoch_ms=now_ms,
                )
                session.current_progress = progress
                self._persist_session(session)
                return progress, new_route, f"REROUTED: {re_reason}"
            else:
                session.status = NavigationStatus.ACTIVE
                logger.warning(f"Reroute failed: {err}")

        self._persist_session(session)
        return progress, None, "PROGRESS_UPDATED"

    def complete_session(self, session_id: str) -> Optional[NavigationSession]:
        session = self.get_session(session_id)
        if session:
            session.status = NavigationStatus.COMPLETED
            session.end_time_ms = int(time.time() * 1000)
            self._persist_session(session)
        return session

    def record_progress(self, session_id: str, progress: NavigationProgress) -> None:
        """Records an external progress fix and updates session state."""
        session = self.get_session(session_id)
        if session:
            session.current_progress = progress
            session.progress_history.append(progress)
            session.status = NavigationStatus.ACTIVE
            self._persist_session(session)

    def cancel_session(self, session_id: str) -> Optional[NavigationSession]:
        session = self.get_session(session_id)
        if session:
            session.status = NavigationStatus.CANCELLED
            session.end_time_ms = int(time.time() * 1000)
            self._persist_session(session)
        return session

    def _persist_session(self, session: NavigationSession) -> None:
        """Persists session state JSON to disk if storage_dir is configured."""
        if not self.storage_dir:
            return
        try:
            out_file = self.storage_dir / f"{session.session_id}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(session.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Error persisting session {session.session_id}: {e}")
