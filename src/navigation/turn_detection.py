"""Turn detection engine evaluating angular deviations and spatial transitions between route segments (Task 28)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.core.logging import setup_logger
from src.mapping.models import ConnectionType
from src.navigation.models import InstructionType, NavigationWaypoint
from src.navigation.path_geometry import PathGeometryBuilder

logger = setup_logger("TurnDetection")


@dataclass
class TurnDetectionThresholds:
    """Configurable angular thresholds for classifying walking turn types (in degrees)."""
    straight_max_deg: float = 20.0     # Under 20 degrees: straight
    slight_turn_max_deg: float = 45.0  # 20 to 45 degrees: slight turn
    turn_max_deg: float = 135.0        # 45 to 135 degrees: normal 90-degree turn
    sharp_turn_max_deg: float = 170.0  # 135 to 170 degrees: sharp turn
    # Over 170 degrees: U-turn / reversal


class TurnDetector:
    """Evaluates geometric bearings and classifies turn types and movement transitions."""

    def __init__(self, thresholds: Optional[TurnDetectionThresholds] = None):
        self.thresholds: TurnDetectionThresholds = thresholds or TurnDetectionThresholds()

    def classify_turn(
        self,
        bearing_in_deg: float,
        bearing_out_deg: float,
    ) -> Tuple[InstructionType, float]:
        """Classifies the heading change from the inbound segment to outbound segment.

        Returns:
            Tuple of (InstructionType, delta_angle_degrees where positive is right/clockwise, negative is left).
        """
        # Delta angle = bearing_out - bearing_in
        # Local Cartesian: +X = East (0 deg), +Y = North (90 deg). Clockwise turning is negative in Cartesian,
        # but in pedestrian navigation standard: +deg = Turn Right, -deg = Turn Left.
        diff = bearing_out_deg - bearing_in_deg
        # Wrap to [-180, 180]
        wrapped_diff = ((diff + 180.0) % 360.0) - 180.0

        # Note: in Cartesian ENU:
        # Facing North (90 deg), turning East (0 deg) -> diff = -90 deg -> this is a RIGHT turn!
        # Facing North (90 deg), turning West (180 deg) -> diff = +90 deg -> this is a LEFT turn!
        # Therefore, turning direction: delta_turn_deg = -wrapped_diff.
        delta_turn_deg = -wrapped_diff
        abs_delta = abs(delta_turn_deg)

        if abs_delta <= self.thresholds.straight_max_deg:
            return InstructionType.CONTINUE_STRAIGHT, delta_turn_deg

        if delta_turn_deg > 0:
            # Right turn
            if abs_delta <= self.thresholds.slight_turn_max_deg:
                return InstructionType.SLIGHT_RIGHT, delta_turn_deg
            elif abs_delta <= self.thresholds.turn_max_deg:
                return InstructionType.TURN_RIGHT, delta_turn_deg
            elif abs_delta <= self.thresholds.sharp_turn_max_deg:
                return InstructionType.SHARP_RIGHT, delta_turn_deg
            else:
                return InstructionType.UTURN, delta_turn_deg
        else:
            # Left turn
            if abs_delta <= self.thresholds.slight_turn_max_deg:
                return InstructionType.SLIGHT_LEFT, delta_turn_deg
            elif abs_delta <= self.thresholds.turn_max_deg:
                return InstructionType.TURN_LEFT, delta_turn_deg
            elif abs_delta <= self.thresholds.sharp_turn_max_deg:
                return InstructionType.SHARP_LEFT, delta_turn_deg
            else:
                return InstructionType.UTURN, delta_turn_deg

    def detect_transitions(
        self,
        waypoints: List[NavigationWaypoint],
    ) -> List[Tuple[int, InstructionType, Optional[float], float]]:
        """Analyzes a full waypoint list and identifies turn and transition instructions.

        Returns:
            List of tuples: (waypoint_index, instruction_type, target_bearing_deg, distance_to_next_m)
        """
        if not waypoints:
            return []

        if len(waypoints) == 1:
            return [(0, InstructionType.ARRIVE_DESTINATION, None, 0.0)]

        results: List[Tuple[int, InstructionType, Optional[float], float]] = []

        # 1. Initial departure instruction
        first_dist = math.hypot(waypoints[1].x_m - waypoints[0].x_m, waypoints[1].y_m - waypoints[0].y_m)
        first_bearing = PathGeometryBuilder.calculate_segment_bearing_deg(waypoints[0], waypoints[1])
        results.append((0, InstructionType.START, first_bearing, first_dist))

        # 2. Intermediate Waypoints
        for i in range(1, len(waypoints) - 1):
            prev_wp = waypoints[i - 1]
            curr_wp = waypoints[i]
            next_wp = waypoints[i + 1]

            dist_next = math.hypot(next_wp.x_m - curr_wp.x_m, next_wp.y_m - curr_wp.y_m)
            out_bearing = PathGeometryBuilder.calculate_segment_bearing_deg(curr_wp, next_wp)

            # Special Transition Checks
            if curr_wp.floor_id != next_wp.floor_id:
                # Vertical transition occurred!
                if curr_wp.transition_type == ConnectionType.ELEVATOR or "elev" in curr_wp.label.lower():
                    results.append((i, InstructionType.USE_ELEVATOR, out_bearing, dist_next))
                else:
                    results.append((i, InstructionType.TAKE_STAIRS, out_bearing, dist_next))
                continue

            if curr_wp.is_portal:
                if "room" in next_wp.label.lower() or "office" in next_wp.label.lower():
                    results.append((i, InstructionType.ENTER_ROOM, out_bearing, dist_next))
                else:
                    results.append((i, InstructionType.PASS_DOORWAY, out_bearing, dist_next))
                continue

            # Standard Turn detection based on bearings
            in_bearing = PathGeometryBuilder.calculate_segment_bearing_deg(prev_wp, curr_wp)
            turn_type, _ = self.classify_turn(in_bearing, out_bearing)

            results.append((i, turn_type, out_bearing, dist_next))

        # 3. Final Destination arrival
        last_idx = len(waypoints) - 1
        results.append((last_idx, InstructionType.ARRIVE_DESTINATION, None, 0.0))

        return results
