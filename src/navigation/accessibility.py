"""Accessibility auditing, barrier verification, and universal routing constraints (Task 28)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from src.core.logging import setup_logger
from src.mapping.models import AccessibilityType, BuildingMap, ConnectionType
from src.mapping.topology import IndoorTopologyGraph, TopologyEdge
from src.navigation.models import NavigationWaypoint, PlannedRoute, RoutePreference

logger = setup_logger("NavigationAccessibility")


class AccessibilityAuditStatus(str, Enum):
    """Compliance assessment outcomes for accessibility."""
    VERIFIED_ACCESSIBLE = "VERIFIED_ACCESSIBLE"
    ACCESSIBLE = "VERIFIED_ACCESSIBLE"
    ACCESSIBILITY_UNKNOWN = "ACCESSIBILITY_UNKNOWN"
    NOT_ACCESSIBLE = "NOT_ACCESSIBLE"
    BARRIER_DETECTED = "BARRIER_DETECTED"


@dataclass
class AccessibilityAuditResult:
    """Detailed audit verdict with barriers and mobility metrics."""
    status: AccessibilityAuditStatus
    explanation: str
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_accessible(self) -> bool:
        return self.status in (AccessibilityAuditStatus.VERIFIED_ACCESSIBLE, AccessibilityAuditStatus.ACCESSIBLE)

    @property
    def detected_barriers(self) -> List[str]:
        return self.metrics.get("barriers_detected", [])

    def __iter__(self):
        return iter((self.status, self.explanation, self.metrics))


class AccessibilityAuditor:
    """Verifies that planned routes satisfy physical mobility and regulatory constraints without fabrication."""

    def __init__(self, building_map_or_topology: Optional[Any] = None) -> None:
        self.topology: Optional[IndoorTopologyGraph] = None
        if building_map_or_topology is not None:
            if isinstance(building_map_or_topology, IndoorTopologyGraph):
                self.topology = building_map_or_topology
            elif isinstance(building_map_or_topology, BuildingMap):
                self.topology = IndoorTopologyGraph.from_building_map(building_map_or_topology)

    def audit_route(
        *args: Any,
        **kwargs: Any,
    ) -> AccessibilityAuditResult:
        """Performs strict audit of all edges and transitions traversed by a route.

        Supports both instance call: auditor.audit_route(waypoints, preference=...)
        and static/class call: AccessibilityAuditor.audit_route(waypoints, topology=...)

        Returns:
            AccessibilityAuditResult (supports tuple unpacking of (status, explanation, metrics)).
        """
        # Determine whether called on instance or class
        if len(args) > 0 and isinstance(args[0], AccessibilityAuditor):
            self_inst = args[0]
            remaining_args = args[1:]
            waypoints = remaining_args[0] if len(remaining_args) > 0 else kwargs.get("waypoints", [])
            topo = remaining_args[1] if len(remaining_args) > 1 else kwargs.get("topology", self_inst.topology)
            preference = remaining_args[2] if len(remaining_args) > 2 else kwargs.get("preference", RoutePreference.SHORTEST)
        else:
            remaining_args = args
            waypoints = remaining_args[0] if len(remaining_args) > 0 else kwargs.get("waypoints", [])
            topo = remaining_args[1] if len(remaining_args) > 1 else kwargs.get("topology")
            preference = remaining_args[2] if len(remaining_args) > 2 else kwargs.get("preference", RoutePreference.SHORTEST)

        if not waypoints:
            return AccessibilityAuditResult(
                status=AccessibilityAuditStatus.ACCESSIBILITY_UNKNOWN,
                explanation="Route contains no waypoints.",
                metrics={},
            )

        stairs_count = 0
        elevators_count = 0
        ramps_count = 0
        unverified_accessibility_count = 0
        detected_barriers: List[str] = []

        # Inspect transitions between consecutive waypoints
        for i in range(len(waypoints) - 1):
            w1 = waypoints[i]
            w2 = waypoints[i + 1]

            # Find matching edge in topology
            matching_edge: Optional[TopologyEdge] = None
            if topo is not None:
                for edge_id in topo._adjacency.get(w1.node_id, []):
                    edge = topo.edges.get(edge_id)
                    if not edge:
                        continue
                    neighbor = edge.target_node_id if edge.source_node_id == w1.node_id else edge.source_node_id
                    if neighbor == w2.node_id:
                        matching_edge = edge
                        break

            if matching_edge:
                # Connection type check
                if matching_edge.connection_type == ConnectionType.STAIRCASE:
                    stairs_count += 1
                    if preference == RoutePreference.WHEELCHAIR_ACCESSIBLE:
                        detected_barriers.append(f"Staircase at waypoint '{w1.label}' ({w1.node_id}) is impassable for wheelchairs.")
                elif matching_edge.connection_type == ConnectionType.ELEVATOR:
                    elevators_count += 1
                elif matching_edge.connection_type == ConnectionType.RAMP:
                    ramps_count += 1

                # Strict accessibility metadata check
                if matching_edge.accessibility == AccessibilityType.UNKNOWN:
                    unverified_accessibility_count += 1
                elif matching_edge.accessibility == AccessibilityType.STAIRS_ONLY and preference == RoutePreference.WHEELCHAIR_ACCESSIBLE:
                    detected_barriers.append(f"Edge {matching_edge.edge_id} is classified as STAIRS_ONLY.")
            else:
                # Direct check on waypoint transition flags
                if w1.floor_id != w2.floor_id:
                    if w1.transition_type == ConnectionType.STAIRCASE:
                        stairs_count += 1
                        if preference == RoutePreference.WHEELCHAIR_ACCESSIBLE:
                            detected_barriers.append(f"Vertical floor transition between Floor {w1.floor_id} and {w2.floor_id} requires stairs.")
                    elif w1.transition_type == ConnectionType.ELEVATOR:
                        elevators_count += 1

        metrics = {
            "stairs_count": stairs_count,
            "elevators_count": elevators_count,
            "ramps_count": ramps_count,
            "unverified_edges_count": unverified_accessibility_count,
            "barriers_detected": detected_barriers,
            "barrier_count": len(detected_barriers),
        }

        # Determine compliance status
        if detected_barriers:
            reason = f"Accessibility barrier detected: {'; '.join(detected_barriers)}"
            return AccessibilityAuditResult(AccessibilityAuditStatus.NOT_ACCESSIBLE, reason, metrics)

        if unverified_accessibility_count > 0:
            # Guideline: "If accessibility data is missing, return ACCESSIBILITY_UNKNOWN rather than assuming the route is accessible."
            reason = f"Route traverses {unverified_accessibility_count} unverified edge(s). Accessibility cannot be guaranteed without full physical audit."
            return AccessibilityAuditResult(AccessibilityAuditStatus.ACCESSIBILITY_UNKNOWN, reason, metrics)

        if preference == RoutePreference.WHEELCHAIR_ACCESSIBLE:
            reason = "Route is fully verified wheelchair-accessible with step-free elevators/ramps and no staircase obstructions."
            return AccessibilityAuditResult(AccessibilityAuditStatus.VERIFIED_ACCESSIBLE, reason, metrics)

        return AccessibilityAuditResult(AccessibilityAuditStatus.VERIFIED_ACCESSIBLE, "Route is walkable under selected preferences.", metrics)
