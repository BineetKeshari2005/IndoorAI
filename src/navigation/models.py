"""Domain models, enums, and data contracts for Indoor Navigation & Route Planning (Task 28)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple

from src.mapping.geometry import Point2D
from src.mapping.models import AccessibilityType, ConnectionType, MapVerificationStatus


class NavigationStatus(str, Enum):
    """Lifecycle states of an active indoor navigation session."""
    CREATED = "CREATED"
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    LOST = "LOST"
    REROUTING = "REROUTING"


class InstructionType(str, Enum):
    """Categorical turn and spatial traversal instruction types."""
    START = "START"
    CONTINUE_STRAIGHT = "CONTINUE_STRAIGHT"
    SLIGHT_LEFT = "SLIGHT_LEFT"
    TURN_LEFT = "TURN_LEFT"
    SHARP_LEFT = "SHARP_LEFT"
    SLIGHT_RIGHT = "SLIGHT_RIGHT"
    TURN_RIGHT = "TURN_RIGHT"
    SHARP_RIGHT = "SHARP_RIGHT"
    UTURN = "UTURN"
    ENTER_ROOM = "ENTER_ROOM"
    PASS_DOORWAY = "PASS_DOORWAY"
    TAKE_STAIRS = "TAKE_STAIRS"
    USE_ELEVATOR = "USE_ELEVATOR"
    USE_RAMP = "USE_RAMP"
    ARRIVE_DESTINATION = "ARRIVE_DESTINATION"


class RoutePreference(str, Enum):
    """Routing optimization objective and constraint preferences."""
    SHORTEST = "SHORTEST"
    FASTEST = "FASTEST"
    WHEELCHAIR_ACCESSIBLE = "WHEELCHAIR_ACCESSIBLE"
    ACCESSIBLE = "WHEELCHAIR_ACCESSIBLE"
    AVOID_STAIRS = "AVOID_STAIRS"
    PREFER_ELEVATORS = "PREFER_ELEVATORS"
    EMERGENCY_EXIT = "EMERGENCY_EXIT"


class RoutingAlgorithm(str, Enum):
    """Shortest-path graph search algorithms."""
    DIJKSTRA = "DIJKSTRA"
    ASTAR = "ASTAR"


class DestinationCategory(str, Enum):
    """Functional categorization of navigable indoor destinations."""
    ROOM = "ROOM"
    CLASSROOM = "CLASSROOM"
    LABORATORY = "LABORATORY"
    OFFICE = "OFFICE"
    RESTROOM = "RESTROOM"
    ELEVATOR = "ELEVATOR"
    STAIRCASE = "STAIRCASE"
    VERTICAL_TRANSITION = "VERTICAL_TRANSITION"
    ENTRANCE = "ENTRANCE"
    EXIT = "EXIT"
    DOORWAY_EXIT = "EXIT"
    LANDMARK = "LANDMARK"
    CONFERENCE = "CONFERENCE"
    OTHER = "OTHER"

    @classmethod
    def from_string(cls, val: str) -> "DestinationCategory":
        clean = (val or "").strip().upper().replace(" ", "_")
        for m in cls:
            if m.value == clean or m.name == clean:
                return m
        return cls.OTHER


@dataclass
class RouteCostWeights:
    """Configurable weight coefficients for the composite route cost function:
    C = w_d * D + w_t * T + w_s * S + w_a * A
    """
    weight_distance: float = 1.0       # w_d
    weight_time: float = 0.5           # w_t
    weight_stairs: float = 15.0        # w_s (penalty equivalent in meters)
    weight_accessibility: float = 50.0 # w_a (penalty if suboptimal accessibility)
    walking_speed_mps: float = 1.2     # Normal walking speed: 1.2 m/s
    elevator_wait_time_s: float = 25.0 # Average elevator wait/board duration

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RouteCostWeights":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Destination:
    """A verified indoor target destination accessible via navigation."""
    destination_id: str
    building_id: str
    floor_id: int
    display_name: str
    category: DestinationCategory
    entity_id: str  # Associated Room ID, Door ID, VerticalTransition ID, Landmark ID
    verified_coordinates_m: Tuple[float, float]
    verification_status: MapVerificationStatus = MapVerificationStatus.VERIFIED
    room_number: Optional[str] = None
    is_accessible: bool = True
    map_version: str = "1.0.0"
    aliases: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def x_m(self) -> float:
        return self.verified_coordinates_m[0]

    @property
    def y_m(self) -> float:
        return self.verified_coordinates_m[1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "destination_id": self.destination_id,
            "building_id": self.building_id,
            "floor_id": self.floor_id,
            "display_name": self.display_name,
            "category": self.category.value,
            "entity_id": self.entity_id,
            "verified_coordinates_m": list(self.verified_coordinates_m),
            "verification_status": self.verification_status.value,
            "room_number": self.room_number,
            "is_accessible": self.is_accessible,
            "map_version": self.map_version,
            "aliases": self.aliases,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Destination":
        return cls(
            destination_id=data["destination_id"],
            building_id=data["building_id"],
            floor_id=int(data["floor_id"]),
            display_name=data["display_name"],
            category=DestinationCategory.from_string(data.get("category", "OTHER")),
            entity_id=data["entity_id"],
            verified_coordinates_m=(float(data["verified_coordinates_m"][0]), float(data["verified_coordinates_m"][1])),
            verification_status=MapVerificationStatus(data.get("verification_status", "VERIFIED")),
            room_number=data.get("room_number"),
            is_accessible=bool(data.get("is_accessible", True)),
            map_version=data.get("map_version", "1.0.0"),
            aliases=list(data.get("aliases", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class NavigationWaypoint:
    """A discrete geometric node along an indoor route."""
    waypoint_id: str
    node_id: str
    floor_id: int
    x_m: float
    y_m: float
    label: str = ""
    is_portal: bool = False
    is_floor_transition: bool = False
    entity_id: str = ""
    transition_type: Optional[ConnectionType] = None

    @property
    def position(self) -> Point2D:
        return Point2D(self.x_m, self.y_m)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "waypoint_id": self.waypoint_id,
            "node_id": self.node_id,
            "floor_id": self.floor_id,
            "x_m": round(self.x_m, 3),
            "y_m": round(self.y_m, 3),
            "label": self.label,
            "is_portal": self.is_portal,
            "is_floor_transition": self.is_floor_transition,
            "entity_id": self.entity_id,
            "transition_type": self.transition_type.value if self.transition_type else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NavigationWaypoint":
        t_type = ConnectionType.from_string(data["transition_type"]) if data.get("transition_type") else None
        return cls(
            waypoint_id=data["waypoint_id"],
            node_id=data["node_id"],
            floor_id=int(data["floor_id"]),
            x_m=float(data["x_m"]),
            y_m=float(data["y_m"]),
            label=data.get("label", ""),
            is_portal=bool(data.get("is_portal", False)),
            is_floor_transition=bool(data.get("is_floor_transition", False)),
            entity_id=data.get("entity_id", ""),
            transition_type=t_type,
        )


@dataclass
class NavigationInstruction:
    """A human-readable turn-by-turn guidance directive."""
    instruction_id: str
    sequence_number: int
    instruction_type: InstructionType
    text: str
    waypoint: NavigationWaypoint
    floor_id: int
    target_heading_deg: Optional[float] = None
    distance_to_next_m: float = 0.0
    cumulative_distance_m: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instruction_id": self.instruction_id,
            "sequence_number": self.sequence_number,
            "instruction_type": self.instruction_type.value,
            "text": self.text,
            "waypoint": self.waypoint.to_dict(),
            "floor_id": self.floor_id,
            "target_heading_deg": round(self.target_heading_deg, 1) if self.target_heading_deg is not None else None,
            "distance_to_next_m": round(self.distance_to_next_m, 2),
            "cumulative_distance_m": round(self.cumulative_distance_m, 2),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NavigationInstruction":
        return cls(
            instruction_id=data["instruction_id"],
            sequence_number=int(data["sequence_number"]),
            instruction_type=InstructionType(data["instruction_type"]),
            text=data["text"],
            waypoint=NavigationWaypoint.from_dict(data["waypoint"]),
            floor_id=int(data["floor_id"]),
            target_heading_deg=float(data["target_heading_deg"]) if data.get("target_heading_deg") is not None else None,
            distance_to_next_m=float(data.get("distance_to_next_m", 0.0)),
            cumulative_distance_m=float(data.get("cumulative_distance_m", 0.0)),
        )


@dataclass
class PlannedRoute:
    """A fully planned, validated geometric route through an indoor facility."""
    route_id: str
    building_id: str
    origin_name: str
    destination_name: str
    origin_floor_id: int
    destination_floor_id: int
    destination_id: str
    total_distance_m: float
    estimated_time_s: float
    waypoints: List[NavigationWaypoint]
    instructions: List[NavigationInstruction]
    algorithm_used: RoutingAlgorithm = RoutingAlgorithm.ASTAR
    preference: RoutePreference = RoutePreference.SHORTEST
    is_accessible: bool = True
    floors_traversed: List[int] = field(default_factory=list)
    map_version: str = "1.0.0"
    creation_timestamp_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def waypoint_count(self) -> int:
        return len(self.waypoints)

    @property
    def instruction_count(self) -> int:
        return len(self.instructions)

    @property
    def is_multi_floor(self) -> bool:
        return len(self.floors_traversed) > 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "route_id": self.route_id,
            "building_id": self.building_id,
            "origin_name": self.origin_name,
            "destination_name": self.destination_name,
            "origin_floor_id": self.origin_floor_id,
            "destination_floor_id": self.destination_floor_id,
            "destination_id": self.destination_id,
            "total_distance_m": round(self.total_distance_m, 2),
            "estimated_time_s": round(self.estimated_time_s, 1),
            "waypoints": [w.to_dict() for w in self.waypoints],
            "instructions": [i.to_dict() for i in self.instructions],
            "algorithm_used": self.algorithm_used.value,
            "preference": self.preference.value,
            "is_accessible": self.is_accessible,
            "floors_traversed": self.floors_traversed,
            "map_version": self.map_version,
            "creation_timestamp_ms": self.creation_timestamp_ms,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlannedRoute":
        return cls(
            route_id=data["route_id"],
            building_id=data["building_id"],
            origin_name=data.get("origin_name", ""),
            destination_name=data.get("destination_name", ""),
            origin_floor_id=int(data.get("origin_floor_id", 0)),
            destination_floor_id=int(data.get("destination_floor_id", 0)),
            destination_id=data["destination_id"],
            total_distance_m=float(data["total_distance_m"]),
            estimated_time_s=float(data["estimated_time_s"]),
            waypoints=[NavigationWaypoint.from_dict(w) for w in data.get("waypoints", [])],
            instructions=[NavigationInstruction.from_dict(i) for i in data.get("instructions", [])],
            algorithm_used=RoutingAlgorithm(data.get("algorithm_used", "ASTAR")),
            preference=RoutePreference(data.get("preference", "SHORTEST")),
            is_accessible=bool(data.get("is_accessible", True)),
            floors_traversed=list(data.get("floors_traversed", [])),
            map_version=data.get("map_version", "1.0.0"),
            creation_timestamp_ms=int(data.get("creation_timestamp_ms", 0)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class NavigationProgress:
    """Current live tracking status of a user progressing along a route."""
    current_x_m: float
    current_y_m: float
    current_floor_id: int
    current_waypoint_index: int
    next_instruction_index: int
    distance_to_next_waypoint_m: float
    remaining_distance_m: float
    completion_percentage: float
    cross_track_deviation_m: float
    is_off_route: bool
    localization_status: str = "TRACKING"
    confidence_level: str = "HIGH"  # HIGH, MEDIUM, LOW, LOST
    timestamp_epoch_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_x_m": round(self.current_x_m, 3),
            "current_y_m": round(self.current_y_m, 3),
            "current_floor_id": self.current_floor_id,
            "current_waypoint_index": self.current_waypoint_index,
            "next_instruction_index": self.next_instruction_index,
            "distance_to_next_waypoint_m": round(self.distance_to_next_waypoint_m, 2),
            "remaining_distance_m": round(self.remaining_distance_m, 2),
            "completion_percentage": round(self.completion_percentage, 1),
            "cross_track_deviation_m": round(self.cross_track_deviation_m, 2),
            "is_off_route": self.is_off_route,
            "localization_status": self.localization_status,
            "confidence_level": self.confidence_level,
            "timestamp_epoch_ms": self.timestamp_epoch_ms,
        }


@dataclass
class RerouteEvent:
    """Record of an automatic reroute triggered by deviation or obstruction."""
    timestamp_epoch_ms: int
    reason: str
    deviation_distance_m: float
    old_route_id: str
    new_route_id: str
    divergence_point: Tuple[float, float, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_epoch_ms": self.timestamp_epoch_ms,
            "reason": self.reason,
            "deviation_distance_m": round(self.deviation_distance_m, 2),
            "old_route_id": self.old_route_id,
            "new_route_id": self.new_route_id,
            "divergence_point": list(self.divergence_point),
        }


@dataclass
class NavigationSession:
    """Full operational session representing an active or completed trip."""
    session_id: str
    dataset_id: str
    building_id: str
    map_version: str
    origin_name: str
    destination_id: str
    destination_name: str
    active_route: PlannedRoute
    status: NavigationStatus = NavigationStatus.CREATED
    current_progress: Optional[NavigationProgress] = None
    start_time_ms: int = 0
    end_time_ms: Optional[int] = None
    reroute_history: List[RerouteEvent] = field(default_factory=list)
    progress_history: List[NavigationProgress] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "dataset_id": self.dataset_id,
            "building_id": self.building_id,
            "map_version": self.map_version,
            "origin_name": self.origin_name,
            "destination_id": self.destination_id,
            "destination_name": self.destination_name,
            "status": self.status.value,
            "active_route": self.active_route.to_dict(),
            "current_progress": self.current_progress.to_dict() if self.current_progress else None,
            "start_time_ms": self.start_time_ms,
            "end_time_ms": self.end_time_ms,
            "reroute_count": len(self.reroute_history),
            "reroute_history": [r.to_dict() for r in self.reroute_history],
            "metadata": self.metadata,
        }
