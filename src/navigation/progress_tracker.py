"""Navigation progress tracker monitoring waypoint advancement and cross-track deviations (Task 28)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from src.core.logging import setup_logger
from src.mapping.geometry import Point2D
from src.navigation.models import NavigationInstruction, NavigationProgress, NavigationWaypoint, PlannedRoute
from src.navigation.path_geometry import PathGeometryBuilder

logger = setup_logger("ProgressTracker")


class ProgressTracker:
    """Tracks pedestrian progress along an active planned route using incoming localization fixes."""

    def __init__(
        self,
        route_or_arrival_radius: Optional[Any] = None,
        max_deviation_threshold_m: float = 4.00,
        waypoint_arrival_radius_m: Optional[float] = None,
    ) -> None:
        if isinstance(route_or_arrival_radius, PlannedRoute):
            self.route: Optional[PlannedRoute] = route_or_arrival_radius
            self.arrival_radius_m: float = waypoint_arrival_radius_m or 1.50
        elif isinstance(route_or_arrival_radius, (int, float)):
            self.route = None
            self.arrival_radius_m = float(route_or_arrival_radius)
        else:
            self.route = None
            self.arrival_radius_m = waypoint_arrival_radius_m or 1.50

        self.deviation_threshold_m: float = max_deviation_threshold_m
        self.current_waypoint_idx: int = 0
        self.last_progress: Optional[NavigationProgress] = None

    @property
    def is_arrived(self) -> bool:
        """Indicates whether user has arrived at the final destination waypoint."""
        if not self.route:
            return False
        if self.current_waypoint_idx >= len(self.route.waypoints) - 1:
            return True
        if self.last_progress and self.last_progress.remaining_distance_m <= self.arrival_radius_m:
            return True
        return False

    def update_position(
        self,
        x_m: float,
        y_m: float,
        floor_id: int,
        localization_status: str = "TRACKING",
        timestamp_epoch_ms: int = 0,
    ) -> NavigationProgress:
        """Convenience method when ProgressTracker was initialized with a route."""
        if not self.route:
            raise ValueError("No route assigned to ProgressTracker.")
        prog = self.evaluate_progress(
            route=self.route,
            x_m=x_m,
            y_m=y_m,
            floor_id=floor_id,
            localization_status=localization_status,
            current_waypoint_idx=self.current_waypoint_idx,
            timestamp_epoch_ms=timestamp_epoch_ms,
        )
        self.current_waypoint_idx = prog.current_waypoint_index
        self.last_progress = prog
        return prog

    def evaluate_progress(
        self,
        route: PlannedRoute,
        x_m: float,
        y_m: float,
        floor_id: int,
        localization_status: str = "TRACKING",
        current_waypoint_idx: int = 0,
        timestamp_epoch_ms: int = 0,
    ) -> NavigationProgress:
        """Evaluates user position against route, updates waypoint indices, and computes remaining distance."""
        waypoints = route.waypoints
        if not waypoints:
            return NavigationProgress(
                current_x_m=x_m,
                current_y_m=y_m,
                current_floor_id=floor_id,
                current_waypoint_index=0,
                next_instruction_index=0,
                distance_to_next_waypoint_m=0.0,
                remaining_distance_m=0.0,
                completion_percentage=100.0,
                cross_track_deviation_m=0.0,
                is_off_route=False,
                localization_status=localization_status,
                confidence_level="LOST",
                timestamp_epoch_ms=timestamp_epoch_ms,
            )

        # 1. Assess Localization Confidence
        confidence_level = "HIGH"
        is_reliable = True
        status_upper = (localization_status or "").upper()
        if status_upper in ("LOST", "UNINITIALIZED", "FAILED"):
            confidence_level = "LOST"
            is_reliable = False
        elif status_upper in ("LOW_CONFIDENCE", "DEGRADED"):
            confidence_level = "LOW"
            is_reliable = False
        elif status_upper in ("CONVERGING", "MEDIUM"):
            confidence_level = "MEDIUM"

        # 2. Check Waypoint Advancement
        wp_idx = min(current_waypoint_idx, len(waypoints) - 1)
        next_target_idx = min(wp_idx + 1, len(waypoints) - 1) if len(waypoints) > 1 else 0
        target_wp = waypoints[next_target_idx]

        # Only advance waypoint if localization is reliable and floor matches
        if is_reliable and target_wp.floor_id == floor_id:
            dist_to_next = math.hypot(x_m - target_wp.x_m, y_m - target_wp.y_m)
            # If within arrival radius of next target waypoint, advance!
            if dist_to_next <= self.arrival_radius_m and wp_idx < len(waypoints) - 1:
                wp_idx = next_target_idx
                next_target_idx = min(wp_idx + 1, len(waypoints) - 1)
                target_wp = waypoints[next_target_idx]

        # 3. Compute Distance to Target Waypoint
        dist_to_target = math.hypot(x_m - target_wp.x_m, y_m - target_wp.y_m)

        # 4. Compute Remaining Distance along subsequent waypoints
        final_wp = waypoints[-1]
        dist_to_final = math.hypot(x_m - final_wp.x_m, y_m - final_wp.y_m) if floor_id == final_wp.floor_id else float("inf")

        if wp_idx == len(waypoints) - 1 or dist_to_final <= self.arrival_radius_m:
            remaining_dist = min(dist_to_final, self.arrival_radius_m) if floor_id == final_wp.floor_id else dist_to_target
            completion_pct = 100.0 if dist_to_final <= self.arrival_radius_m else min(100.0, max(0.0, 100.0 - (remaining_dist / max(0.1, route.total_distance_m)) * 100.0))
        elif wp_idx == 0 and dist_to_target >= (route.total_distance_m * 0.9):
            remaining_dist = route.total_distance_m
            completion_pct = 0.0
        else:
            remaining_dist = dist_to_target
            for i in range(next_target_idx, len(waypoints) - 1):
                w_a = waypoints[i]
                w_b = waypoints[i + 1]
                if w_a.floor_id == w_b.floor_id:
                    remaining_dist += math.hypot(w_b.x_m - w_a.x_m, w_b.y_m - w_a.y_m)
                else:
                    remaining_dist += 4.5  # Nominal stair/elevator vertical displacement

            total_len = max(0.1, route.total_distance_m)
            completed_dist = max(0.0, total_len - remaining_dist)
            completion_pct = min(100.0, (completed_dist / total_len) * 100.0)

        # 6. Cross-Track Deviation Calculation
        cross_track_m = 0.0
        floor_mismatch = (floor_id != target_wp.floor_id)

        if len(waypoints) >= 2:
            prev_wp = waypoints[max(0, next_target_idx - 1)]
            if prev_wp.floor_id == floor_id and target_wp.floor_id == floor_id:
                cross_track_m = PathGeometryBuilder.compute_cross_track_distance_m(
                    px=x_m,
                    py=y_m,
                    p1=prev_wp.position,
                    p2=target_wp.position,
                )
            else:
                cross_track_m = dist_to_target

        # Determine if off-route
        is_off_route = False
        if is_reliable:
            if floor_mismatch or cross_track_m > self.deviation_threshold_m:
                is_off_route = True

        next_inst_idx = min(wp_idx, len(route.instructions) - 1)

        return NavigationProgress(
            current_x_m=x_m,
            current_y_m=y_m,
            current_floor_id=floor_id,
            current_waypoint_index=wp_idx,
            next_instruction_index=next_inst_idx,
            distance_to_next_waypoint_m=round(dist_to_target, 2),
            remaining_distance_m=round(remaining_dist, 2),
            completion_percentage=round(completion_pct, 1),
            cross_track_deviation_m=round(cross_track_m, 2),
            is_off_route=is_off_route,
            localization_status=localization_status,
            confidence_level=confidence_level,
            timestamp_epoch_ms=timestamp_epoch_ms,
        )
