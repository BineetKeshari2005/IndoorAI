"""Dynamic rerouting engine executing automatic path recalculations upon deviation (Task 28)."""

from __future__ import annotations

import time
from typing import Optional, Tuple

from src.core.logging import setup_logger
from src.navigation.models import NavigationProgress, PlannedRoute, RerouteEvent
from src.navigation.path_geometry import PathGeometryBuilder
from src.navigation.route_planner import RoutePlanner

logger = setup_logger("ReroutingEngine")


class ReroutingEngine:
    """Manages automatic rerouting triggers, deviation debounce, and path recalculation."""

    def __init__(
        self,
        planner: RoutePlanner,
        deviation_threshold_m: float = 4.0,
        min_reroute_interval_s: float = 5.0,
        reroute_cooldown_s: Optional[float] = None,
    ) -> None:
        self.planner: RoutePlanner = planner
        self.deviation_threshold_m: float = deviation_threshold_m
        self.min_reroute_interval_s: float = reroute_cooldown_s if reroute_cooldown_s is not None else min_reroute_interval_s
        self.last_reroute_time_s: float = 0.0

    def is_in_cooldown(self, timestamp_epoch_ms: Optional[int] = None) -> bool:
        """Checks if rerouting is currently suppressed by debounce cooldown."""
        now_s = (timestamp_epoch_ms / 1000.0) if timestamp_epoch_ms is not None else time.time()
        return (now_s - self.last_reroute_time_s) < self.min_reroute_interval_s

    def check_deviation(
        self,
        x_m: float,
        y_m: float,
        floor_id: int,
        route: PlannedRoute,
        current_wp_idx: int = 0,
    ) -> Tuple[bool, str]:
        """Direct check if an (x, y, floor) position fix deviates from active route."""
        waypoints = route.waypoints
        if not waypoints:
            return False, "Empty route."
        target_idx = min(current_wp_idx + 1, len(waypoints) - 1) if len(waypoints) > 1 else 0
        target_wp = waypoints[target_idx]
        prev_wp = waypoints[max(0, target_idx - 1)]

        if floor_id != target_wp.floor_id:
            return True, f"Floor deviation: on Floor {floor_id}, expected Floor {target_wp.floor_id}"

        cross_track = PathGeometryBuilder.compute_cross_track_distance_m(
            px=x_m, py=y_m, p1=prev_wp.position, p2=target_wp.position
        )
        if cross_track > self.deviation_threshold_m:
            return True, f"CROSS_TRACK_DEVIATION: {cross_track:.1f} m exceeds {self.deviation_threshold_m:.1f} m"
        return False, "OK"

    def trigger_reroute(
        self,
        current_x_m: float,
        current_y_m: float,
        current_floor_id: int,
        active_route: PlannedRoute,
        reason: str = "User deviated from active path",
        timestamp_epoch_ms: Optional[int] = None,
    ) -> Tuple[Optional[PlannedRoute], Optional[RerouteEvent]]:
        """Directly triggers reroute from specified coordinates."""
        now_ms = timestamp_epoch_ms or int(time.time() * 1000)
        self.last_reroute_time_s = now_ms / 1000.0
        prog = NavigationProgress(
            current_x_m=current_x_m,
            current_y_m=current_y_m,
            current_floor_id=current_floor_id,
            current_waypoint_index=0,
            next_instruction_index=0,
            distance_to_next_waypoint_m=0.0,
            remaining_distance_m=0.0,
            completion_percentage=0.0,
            cross_track_deviation_m=0.0,
            is_off_route=True,
            localization_status="TRACKING",
            timestamp_epoch_ms=now_ms,
        )
        new_route, event, _ = self.recalculate_route(active_route, prog, reason=reason)
        return new_route, event

    def should_reroute(self, progress: NavigationProgress) -> Tuple[bool, str]:
        """Determines whether current pedestrian deviation warrants triggering an automatic reroute."""
        # 1. Unreliable localization: do NOT trigger false reroutes during transient dropouts!
        if progress.confidence_level == "LOW" or progress.localization_status in ("LOST", "UNRELIABLE"):
            return False, "Localization confidence low; suppressing reroute."

        # 2. Debounce cooldown to prevent flapping
        now_s = time.time()
        if now_s - self.last_reroute_time_s < self.min_reroute_interval_s:
            return False, "Reroute cooldown active."

        # 3. Off-route detection
        if progress.is_off_route:
            if progress.cross_track_deviation_m > self.deviation_threshold_m:
                reason = f"Cross-track deviation ({progress.cross_track_deviation_m:.1f} m) exceeds threshold ({self.deviation_threshold_m:.1f} m)."
                return True, reason

        return False, "User on path."

    def recalculate_route(
        self,
        current_route: PlannedRoute,
        progress: NavigationProgress,
        reason: str = "User deviated from active path",
    ) -> Tuple[Optional[PlannedRoute], Optional[RerouteEvent], Optional[str]]:
        """Recalculates an updated route from current pedestrian coordinate while preserving destination and preferences.

        Returns:
            Tuple of (new_planned_route, reroute_event, error_reason).
        """
        logger.info(f"Triggering automatic reroute for route {current_route.route_id}: {reason}")
        now_epoch_ms = int(time.time() * 1000)
        self.last_reroute_time_s = time.time()

        # Execute route planner from current estimated coordinate
        new_route, err = self.planner.plan_route(
            origin_coords=(progress.current_x_m, progress.current_y_m, progress.current_floor_id),
            destination_id=current_route.destination_id,
            algorithm=current_route.algorithm_used,
            preference=current_route.preference,
        )

        if not new_route:
            logger.warning(f"Rerouting failed: {err}")
            return None, None, err

        event = RerouteEvent(
            timestamp_epoch_ms=now_epoch_ms,
            reason=reason,
            deviation_distance_m=progress.cross_track_deviation_m,
            old_route_id=current_route.route_id,
            new_route_id=new_route.route_id,
            divergence_point=(progress.current_x_m, progress.current_y_m, progress.current_floor_id),
        )

        logger.info(f"Reroute successful. New route {new_route.route_id} generated ({new_route.total_distance_m:.1f} m).")
        return new_route, event, None
