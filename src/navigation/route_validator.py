"""Strict route validation auditing building consistency, wall non-penetration, and connectivity (Task 28)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.core.logging import setup_logger
from src.mapping.geometry import LineSegment2D, Point2D
from src.mapping.models import BuildingMap, MapVerificationStatus
from src.mapping.topology import IndoorTopologyGraph
from src.navigation.accessibility import AccessibilityAuditor, AccessibilityAuditStatus
from src.navigation.models import NavigationWaypoint, PlannedRoute, RoutePreference

logger = setup_logger("RouteValidator")


class RouteValidationResult:
    """Outcome of a comprehensive route verification check."""

    def __init__(
        self,
        is_valid: bool,
        error_code: Optional[str] = None,
        reason: str = "",
        details: Optional[Dict[str, Any]] = None,
    ):
        self.is_valid: bool = is_valid
        self.error_code: Optional[str] = error_code
        self.reason: str = reason
        self.details: Dict[str, Any] = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "error_code": self.error_code,
            "reason": self.reason,
            "details": self.details,
        }


class RouteValidator:
    """Audits planned routes for topological continuity, wall non-penetration, and accessibility compliance."""

    def __init__(self, building_map: Optional[BuildingMap] = None) -> None:
        self.building_map: Optional[BuildingMap] = building_map

    def validate_path_geometry(
        self,
        waypoints: List[NavigationWaypoint],
        floor_id: Optional[int] = None,
        building_map: Optional[BuildingMap] = None,
    ) -> RouteValidationResult:
        """Audits route waypoints against room walls on the specified floor."""
        bmap = building_map or self.building_map
        if not bmap:
            return RouteValidationResult(is_valid=True)
        wall_violation = RouteValidator._check_wall_crossings(waypoints, bmap)
        if wall_violation:
            return RouteValidationResult(
                is_valid=False,
                error_code="WALL_COLLISION_DETECTED",
                reason=wall_violation,
            )
        return RouteValidationResult(is_valid=True)

    @staticmethod
    def validate_route(
        waypoints: List[NavigationWaypoint],
        topology: IndoorTopologyGraph,
        building_map: Optional[BuildingMap] = None,
        preference: RoutePreference = RoutePreference.SHORTEST,
        expected_map_version: Optional[str] = None,
    ) -> RouteValidationResult:
        """Validates all architectural and geometric invariants of a candidate route."""
        # 1. Non-empty check
        if not waypoints:
            return RouteValidationResult(
                is_valid=False,
                error_code="NO_VALID_ROUTE",
                reason="Candidate route has zero waypoints.",
            )

        # 2. Map Version Consistency
        if building_map and expected_map_version:
            bmap_ver = building_map.metadata.get("version", "1.0.0")
            if bmap_ver != expected_map_version:
                return RouteValidationResult(
                    is_valid=False,
                    error_code="MAP_VERSION_MISMATCH",
                    reason=f"Route planned on map version '{expected_map_version}' but active building map is '{bmap_ver}'.",
                )

        # 3. Node & Floor Continuity
        for i in range(len(waypoints) - 1):
            w1 = waypoints[i]
            w2 = waypoints[i + 1]

            # Check that w1 and w2 exist in topology
            if w1.node_id not in topology.nodes or w2.node_id not in topology.nodes:
                return RouteValidationResult(
                    is_valid=False,
                    error_code="INVALID_NODE_REFERENCE",
                    reason=f"Waypoint references non-existent node '{w1.node_id}' or '{w2.node_id}'.",
                )

            # Check floor change validity
            if w1.floor_id != w2.floor_id:
                # Floor change must occur at a verified vertical transition!
                if not (w1.is_floor_transition or w2.is_floor_transition):
                    return RouteValidationResult(
                        is_valid=False,
                        error_code="IMPOSSIBLE_FLOOR_HOP",
                        reason=f"Route changes from Floor {w1.floor_id} to Floor {w2.floor_id} without a staircase or elevator transition at {w1.label}.",
                    )

        # 4. Solid Wall Penetration Check
        if building_map:
            wall_violation = RouteValidator._check_wall_crossings(waypoints, building_map)
            if wall_violation:
                return RouteValidationResult(
                    is_valid=False,
                    error_code="WALL_COLLISION_DETECTED",
                    reason=wall_violation,
                )

        # 5. Accessibility Audit
        status, acc_reason, acc_details = AccessibilityAuditor.audit_route(
            waypoints=waypoints,
            topology=topology,
            preference=preference,
        )
        if status == AccessibilityAuditStatus.NOT_ACCESSIBLE and preference == RoutePreference.WHEELCHAIR_ACCESSIBLE:
            return RouteValidationResult(
                is_valid=False,
                error_code="ACCESSIBILITY_VIOLATION",
                reason=acc_reason,
                details=acc_details,
            )

        return RouteValidationResult(
            is_valid=True,
            error_code=None,
            reason="Route passes all topological, geometric, and accessibility audits.",
            details=acc_details,
        )

    @staticmethod
    def _check_wall_crossings(
        waypoints: List[NavigationWaypoint],
        building_map: BuildingMap,
    ) -> Optional[str]:
        """Audits route line segments against verified room walls to prevent wall penetration."""
        for i in range(len(waypoints) - 1):
            w1 = waypoints[i]
            w2 = waypoints[i + 1]

            if w1.floor_id != w2.floor_id:
                continue

            floor_id = w1.floor_id
            if floor_id not in building_map.floors:
                continue

            floor = building_map.floors[floor_id]
            seg = LineSegment2D(Point2D(w1.x_m, w1.y_m), Point2D(w2.x_m, w2.y_m))

            # Door positions on this floor
            door_positions = [d.position for d in floor.doors.values()]

            for room_id, room in floor.rooms.items():
                if room.verification_status == MapVerificationStatus.VERIFIED:
                    for edge in room.polygon.edges():
                        if seg.intersects(edge):
                            # Ensure intersection happened at a door opening!
                            doors_near = [dp for dp in door_positions if edge.distance_to_point(dp) <= 1.2]
                            if not doors_near:
                                return (
                                    f"Route segment from ({w1.x_m:.1f}, {w1.y_m:.1f}) to ({w2.x_m:.1f}, {w2.y_m:.1f}) "
                                    f"intersects solid wall of '{room.name}' ({room_id}) with no doorway opening."
                                )

        return None
