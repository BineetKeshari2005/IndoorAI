"""Geometry construction, waypoint sequencing, and metric path smoothing (Task 28)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from src.core.logging import setup_logger
from src.mapping.geometry import LineSegment2D, Point2D
from src.mapping.models import BuildingMap, ConnectionType
from src.mapping.topology import IndoorTopologyGraph, TopologyEdge, TopologyNode, TopologyNodeType
from src.navigation.models import NavigationWaypoint

logger = setup_logger("PathGeometry")


class PathGeometryBuilder:
    """Transforms topological node sequences into metric walking waypoints and continuous path polylines."""

    @staticmethod
    def build_waypoints(
        node_path: List[str],
        topology: IndoorTopologyGraph,
    ) -> List[NavigationWaypoint]:
        """Constructs ordered NavigationWaypoint objects from a list of topological node IDs."""
        if not node_path:
            return []

        waypoints: List[NavigationWaypoint] = []
        for idx, node_id in enumerate(node_path):
            if node_id not in topology.nodes:
                continue

            node = topology.nodes[node_id]
            is_portal = node.node_type in (
                TopologyNodeType.DOORWAY,
                TopologyNodeType.BUILDING_ENTRANCE,
                TopologyNodeType.BUILDING_EXIT,
            )
            is_vt = node.node_type in (
                TopologyNodeType.STAIR_LANDING,
                TopologyNodeType.ELEVATOR_LANDING,
            )

            # Determine transition type if vertical
            t_type = None
            if is_vt:
                t_type = ConnectionType.ELEVATOR if node.node_type == TopologyNodeType.ELEVATOR_LANDING else ConnectionType.STAIRCASE

            waypoints.append(
                NavigationWaypoint(
                    waypoint_id=f"WP-{idx:03d}-{node_id}",
                    node_id=node_id,
                    floor_id=node.floor_id,
                    x_m=node.position.x,
                    y_m=node.position.y,
                    label=node.label if node.label else f"Waypoint {idx + 1}",
                    is_portal=is_portal,
                    is_floor_transition=is_vt,
                    entity_id=node.entity_id,
                    transition_type=t_type,
                )
            )

        return waypoints

    @staticmethod
    def create_waypoint(
        node_id: str,
        floor_id: int,
        x_m: float,
        y_m: float,
        label: str = "",
        is_portal: bool = False,
        is_floor_transition: bool = False,
        transition_type: Optional[ConnectionType] = None,
    ) -> NavigationWaypoint:
        """Utility factory to construct a standalone NavigationWaypoint."""
        return NavigationWaypoint(
            waypoint_id=f"WP-{node_id}",
            node_id=node_id,
            floor_id=floor_id,
            x_m=x_m,
            y_m=y_m,
            label=label,
            is_portal=is_portal,
            is_floor_transition=is_floor_transition,
            transition_type=transition_type,
        )

    @staticmethod
    def calculate_path_length_m(waypoints: List[NavigationWaypoint]) -> float:
        """Computes total 2D metric walking distance along waypoints, ignoring vertical jump distances."""
        if len(waypoints) < 2:
            return 0.0

        total_dist = 0.0
        for i in range(len(waypoints) - 1):
            w1 = waypoints[i]
            w2 = waypoints[i + 1]
            if w1.floor_id == w2.floor_id:
                total_dist += math.hypot(w2.x_m - w1.x_m, w2.y_m - w1.y_m)
            else:
                # Vertical transition floor change: approximate equivalent walking distance
                total_dist += 4.5  # Standard nominal stairs/elevator vertical distance

        return round(total_dist, 2)

    @staticmethod
    def calculate_segment_bearing_deg(w1: NavigationWaypoint, w2: NavigationWaypoint) -> float:
        """Calculates geographic/local Cartesian heading angle in degrees from w1 to w2 (-180 to +180).

        0 degrees = East (+X), 90 degrees = North (+Y).
        """
        dx = w2.x_m - w1.x_m
        dy = w2.y_m - w1.y_m
        rad = math.atan2(dy, dx)
        deg = math.degrees(rad)
        return ((deg + 180.0) % 360.0) - 180.0

    @staticmethod
    def snap_coordinate_to_nearest_corridor(
        x: float,
        y: float,
        floor_id: int,
        topology: IndoorTopologyGraph,
        max_snap_distance_m: float = 10.0,
    ) -> Optional[str]:
        """Snaps an arbitrary coordinate (e.g. from localization PF) to the nearest corridor waypoint or door node."""
        target_pt = Point2D(x, y)
        best_node_id: Optional[str] = None
        best_dist = float("inf")

        # Prefer corridor waypoints and door portals
        preferred_types = (
            TopologyNodeType.CORRIDOR_WAYPOINT,
            TopologyNodeType.CORRIDOR_INTERSECTION,
            TopologyNodeType.DOORWAY,
            TopologyNodeType.ROOM_CENTROID,
        )

        for node_id, node in topology.nodes.items():
            if node.floor_id != floor_id:
                continue

            dist = node.position.distance_to(target_pt)
            if dist < best_dist and dist <= max_snap_distance_m:
                if node.node_type in preferred_types:
                    best_dist = dist
                    best_node_id = node_id

        # Fallback to any node on that floor if no preferred type found within radius
        if not best_node_id:
            for node_id, node in topology.nodes.items():
                if node.floor_id == floor_id:
                    dist = node.position.distance_to(target_pt)
                    if dist < best_dist:
                        best_dist = dist
                        best_node_id = node_id

        return best_node_id

    @staticmethod
    def compute_cross_track_distance_m(
        px: float,
        py: float,
        p1: Point2D,
        p2: Point2D,
    ) -> float:
        """Computes perpendicular Euclidean distance from point (px, py) to finite line segment p1-p2."""
        seg = LineSegment2D(p1, p2)
        return seg.distance_to_point(Point2D(px, py))
