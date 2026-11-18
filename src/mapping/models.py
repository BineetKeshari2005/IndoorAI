"""Domain data models and schemas for indoor spatial mapping and navigation topologies."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from src.mapping.geometry import Point2D, Polygon2D
from src.sensor_fusion.coordinates import CoordinateFrameMetadata


class MapVerificationStatus(str, Enum):
    """Survey and verification status of indoor spatial elements."""
    UNVERIFIED = "UNVERIFIED"
    PARTIALLY_SURVEYED = "PARTIALLY_SURVEYED"
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_string(cls, val: str) -> "MapVerificationStatus":
        clean = (val or "").strip().upper()
        for m in cls:
            if m.value == clean or m.name == clean:
                return m
        return cls.UNKNOWN


class RoomCategory(str, Enum):
    """Functional classification of indoor room spaces."""
    OFFICE = "OFFICE"
    LAB = "LAB"
    CLASSROOM = "CLASSROOM"
    RESTROOM = "RESTROOM"
    CONFERENCE = "CONFERENCE"
    UTILITY = "UTILITY"
    LOBBY = "LOBBY"
    KITCHEN = "KITCHEN"
    STORAGE = "STORAGE"
    STAIRWELL = "STAIRWELL"
    ELEVATOR_SHAFT = "ELEVATOR_SHAFT"
    OTHER = "OTHER"

    @classmethod
    def from_string(cls, val: str) -> "RoomCategory":
        clean = (val or "").strip().upper().replace(" ", "_")
        for m in cls:
            if m.value == clean or m.name == clean:
                return m
        return cls.OTHER


class ConnectionType(str, Enum):
    """Categorical type of spatial connection between indoor spaces."""
    DOOR = "DOOR"
    CORRIDOR = "CORRIDOR"
    STAIRCASE = "STAIRCASE"
    ELEVATOR = "ELEVATOR"
    RAMP = "RAMP"
    ENTRANCE = "ENTRANCE"
    EXIT = "EXIT"

    @classmethod
    def from_string(cls, val: str) -> "ConnectionType":
        clean = (val or "").strip().upper()
        for m in cls:
            if m.value == clean or m.name == clean:
                return m
        return cls.CORRIDOR


class AccessibilityType(str, Enum):
    """Physical accessibility status for mobility-impaired routing."""
    WHEELCHAIR_ACCESSIBLE = "WHEELCHAIR_ACCESSIBLE"
    STAIRS_ONLY = "STAIRS_ONLY"
    RESTRICTED = "RESTRICTED"
    UNKNOWN = "UNKNOWN"


class DoorType(str, Enum):
    """Mechanical and physical door classification."""
    SINGLE_SWING = "SINGLE_SWING"
    DOUBLE_SWING = "DOUBLE_SWING"
    SLIDING = "SLIDING"
    REVOLVING = "REVOLVING"
    AUTOMATIC = "AUTOMATIC"
    OPEN_PASSAGE = "OPEN_PASSAGE"
    UNKNOWN = "UNKNOWN"


@dataclass
class Room:
    """A bounded indoor room polygon with verified attributes and connectivity."""
    room_id: str
    building_id: str
    floor_id: int
    name: str
    polygon: Polygon2D
    category: RoomCategory = RoomCategory.OTHER
    room_number: Optional[str] = None
    entrance_door_ids: List[str] = field(default_factory=list)
    verification_status: MapVerificationStatus = MapVerificationStatus.UNVERIFIED
    wifi_reference_ids: List[str] = field(default_factory=list)
    landmark_ids: List[str] = field(default_factory=list)
    survey_session_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def area_sq_m(self) -> float:
        return self.polygon.area

    @property
    def centroid(self) -> Point2D:
        return self.polygon.centroid

    def to_dict(self) -> Dict[str, Any]:
        return {
            "room_id": self.room_id,
            "building_id": self.building_id,
            "floor_id": self.floor_id,
            "name": self.name,
            "room_number": self.room_number,
            "category": self.category.value,
            "polygon": self.polygon.to_dict(),
            "area_sq_m": round(self.area_sq_m, 2),
            "centroid": self.centroid.to_dict(),
            "entrance_door_ids": self.entrance_door_ids,
            "verification_status": self.verification_status.value,
            "wifi_reference_ids": self.wifi_reference_ids,
            "landmark_ids": self.landmark_ids,
            "survey_session_ids": self.survey_session_ids,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Room":
        return cls(
            room_id=data["room_id"],
            building_id=data["building_id"],
            floor_id=int(data["floor_id"]),
            name=data["name"],
            room_number=data.get("room_number"),
            category=RoomCategory.from_string(data.get("category", "OTHER")),
            polygon=Polygon2D.from_dict(data["polygon"]),
            entrance_door_ids=list(data.get("entrance_door_ids", [])),
            verification_status=MapVerificationStatus.from_string(data.get("verification_status", "UNVERIFIED")),
            wifi_reference_ids=list(data.get("wifi_reference_ids", [])),
            landmark_ids=list(data.get("landmark_ids", [])),
            survey_session_ids=list(data.get("survey_session_ids", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class Corridor:
    """A navigable indoor corridor defined by a centerline trajectory and optional boundary polygon."""
    corridor_id: str
    building_id: str
    floor_id: int
    name: str
    centerline: List[Point2D]
    polygon: Optional[Polygon2D] = None
    width_m: float = 2.0
    connected_room_ids: List[str] = field(default_factory=list)
    connected_door_ids: List[str] = field(default_factory=list)
    intersection_node_ids: List[str] = field(default_factory=list)
    verification_status: MapVerificationStatus = MapVerificationStatus.UNVERIFIED
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def length_m(self) -> float:
        if len(self.centerline) < 2:
            return 0.0
        total = 0.0
        for i in range(len(self.centerline) - 1):
            total += self.centerline[i].distance_to(self.centerline[i + 1])
        return total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "corridor_id": self.corridor_id,
            "building_id": self.building_id,
            "floor_id": self.floor_id,
            "name": self.name,
            "centerline": [p.to_dict() for p in self.centerline],
            "polygon": self.polygon.to_dict() if self.polygon else None,
            "width_m": round(self.width_m, 2),
            "length_m": round(self.length_m, 2),
            "connected_room_ids": self.connected_room_ids,
            "connected_door_ids": self.connected_door_ids,
            "intersection_node_ids": self.intersection_node_ids,
            "verification_status": self.verification_status.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Corridor":
        poly = Polygon2D.from_dict(data["polygon"]) if data.get("polygon") else None
        return cls(
            corridor_id=data["corridor_id"],
            building_id=data["building_id"],
            floor_id=int(data["floor_id"]),
            name=data["name"],
            centerline=[Point2D.from_dict(p) for p in data.get("centerline", [])],
            polygon=poly,
            width_m=float(data.get("width_m", 2.0)),
            connected_room_ids=list(data.get("connected_room_ids", [])),
            connected_door_ids=list(data.get("connected_door_ids", [])),
            intersection_node_ids=list(data.get("intersection_node_ids", [])),
            verification_status=MapVerificationStatus.from_string(data.get("verification_status", "UNVERIFIED")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class Door:
    """A door or entrance waypoint linking two distinct spatial entities."""
    door_id: str
    building_id: str
    floor_id: int
    name: str
    position: Point2D
    connects_from_id: str  # Room ID or Corridor ID
    connects_to_id: str    # Room ID or Corridor ID
    door_type: DoorType = DoorType.SINGLE_SWING
    width_m: float = 0.90
    is_bidirectional: bool = True
    is_emergency_exit: bool = False
    verification_status: MapVerificationStatus = MapVerificationStatus.UNVERIFIED
    associated_landmark_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "door_id": self.door_id,
            "building_id": self.building_id,
            "floor_id": self.floor_id,
            "name": self.name,
            "position": self.position.to_dict(),
            "connects_from_id": self.connects_from_id,
            "connects_to_id": self.connects_to_id,
            "door_type": self.door_type.value,
            "width_m": round(self.width_m, 2),
            "is_bidirectional": self.is_bidirectional,
            "is_emergency_exit": self.is_emergency_exit,
            "verification_status": self.verification_status.value,
            "associated_landmark_id": self.associated_landmark_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Door":
        return cls(
            door_id=data["door_id"],
            building_id=data["building_id"],
            floor_id=int(data["floor_id"]),
            name=data["name"],
            position=Point2D.from_dict(data["position"]),
            connects_from_id=data["connects_from_id"],
            connects_to_id=data["connects_to_id"],
            door_type=DoorType(data.get("door_type", "SINGLE_SWING")),
            width_m=float(data.get("width_m", 0.90)),
            is_bidirectional=bool(data.get("is_bidirectional", True)),
            is_emergency_exit=bool(data.get("is_emergency_exit", False)),
            verification_status=MapVerificationStatus.from_string(data.get("verification_status", "UNVERIFIED")),
            associated_landmark_id=data.get("associated_landmark_id"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class VerticalTransition:
    """A vertical transit mechanism (stairwell or elevator shaft) connecting multiple floors."""
    transition_id: str
    building_id: str
    transition_type: ConnectionType  # STAIRCASE or ELEVATOR
    name: str
    floor_positions: Dict[int, Point2D]  # floor_id -> landing position
    is_accessible: bool = False
    verification_status: MapVerificationStatus = MapVerificationStatus.UNVERIFIED
    associated_landmark_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def connected_floors(self) -> List[int]:
        return sorted(self.floor_positions.keys())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "building_id": self.building_id,
            "transition_type": self.transition_type.value,
            "name": self.name,
            "connected_floors": self.connected_floors,
            "floor_positions": {str(k): v.to_dict() for k, v in self.floor_positions.items()},
            "is_accessible": self.is_accessible,
            "verification_status": self.verification_status.value,
            "associated_landmark_ids": self.associated_landmark_ids,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerticalTransition":
        raw_pos = data.get("floor_positions", {})
        pos_dict = {int(k): Point2D.from_dict(v) for k, v in raw_pos.items()}
        return cls(
            transition_id=data["transition_id"],
            building_id=data["building_id"],
            transition_type=ConnectionType.from_string(data["transition_type"]),
            name=data["name"],
            floor_positions=pos_dict,
            is_accessible=bool(data.get("is_accessible", False)),
            verification_status=MapVerificationStatus.from_string(data.get("verification_status", "UNVERIFIED")),
            associated_landmark_ids=list(data.get("associated_landmark_ids", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class SpatialConnection:
    """Topological edge linking two spatial entities in the navigation graph."""
    connection_id: str
    building_id: str
    source_id: str
    destination_id: str
    source_floor_id: int
    destination_floor_id: int
    connection_type: ConnectionType
    distance_m: float
    is_bidirectional: bool = True
    accessibility: AccessibilityType = AccessibilityType.WHEELCHAIR_ACCESSIBLE
    verification_status: MapVerificationStatus = MapVerificationStatus.UNVERIFIED
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_vertical(self) -> bool:
        return self.source_floor_id != self.destination_floor_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "connection_id": self.connection_id,
            "building_id": self.building_id,
            "source_id": self.source_id,
            "destination_id": self.destination_id,
            "source_floor_id": self.source_floor_id,
            "destination_floor_id": self.destination_floor_id,
            "connection_type": self.connection_type.value,
            "distance_m": round(self.distance_m, 2),
            "is_bidirectional": self.is_bidirectional,
            "accessibility": self.accessibility.value,
            "is_vertical": self.is_vertical,
            "verification_status": self.verification_status.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpatialConnection":
        return cls(
            connection_id=data["connection_id"],
            building_id=data["building_id"],
            source_id=data["source_id"],
            destination_id=data["destination_id"],
            source_floor_id=int(data["source_floor_id"]),
            destination_floor_id=int(data["destination_floor_id"]),
            connection_type=ConnectionType.from_string(data["connection_type"]),
            distance_m=float(data["distance_m"]),
            is_bidirectional=bool(data.get("is_bidirectional", True)),
            accessibility=AccessibilityType(data.get("accessibility", "WHEELCHAIR_ACCESSIBLE")),
            verification_status=MapVerificationStatus.from_string(data.get("verification_status", "UNVERIFIED")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class FloorMap:
    """Comprehensive spatial representation of a single building level."""
    floor_id: int
    building_id: str
    floor_number: int
    label: str = ""
    elevation_m: float = 0.0
    boundary: Optional[Polygon2D] = None
    rooms: Dict[str, Room] = field(default_factory=dict)
    corridors: Dict[str, Corridor] = field(default_factory=dict)
    doors: Dict[str, Door] = field(default_factory=dict)
    wifi_reference_locations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    landmarks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    verification_status: MapVerificationStatus = MapVerificationStatus.PARTIALLY_SURVEYED
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_floor_area_sq_m(self) -> float:
        return self.boundary.area if self.boundary else sum(r.area_sq_m for r in self.rooms.values())

    @property
    def surveyed_room_area_sq_m(self) -> float:
        return sum(r.area_sq_m for r in self.rooms.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "floor_id": self.floor_id,
            "building_id": self.building_id,
            "floor_number": self.floor_number,
            "label": self.label,
            "elevation_m": round(self.elevation_m, 2),
            "total_floor_area_sq_m": round(self.total_floor_area_sq_m, 2),
            "surveyed_room_area_sq_m": round(self.surveyed_room_area_sq_m, 2),
            "boundary": self.boundary.to_dict() if self.boundary else None,
            "rooms": {k: v.to_dict() for k, v in self.rooms.items()},
            "corridors": {k: v.to_dict() for k, v in self.corridors.items()},
            "doors": {k: v.to_dict() for k, v in self.doors.items()},
            "wifi_reference_locations": self.wifi_reference_locations,
            "landmarks": self.landmarks,
            "verification_status": self.verification_status.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FloorMap":
        bound = Polygon2D.from_dict(data["boundary"]) if data.get("boundary") else None
        rooms = {k: Room.from_dict(v) for k, v in data.get("rooms", {}).items()}
        corridors = {k: Corridor.from_dict(v) for k, v in data.get("corridors", {}).items()}
        doors = {k: Door.from_dict(v) for k, v in data.get("doors", {}).items()}
        return cls(
            floor_id=int(data["floor_id"]),
            building_id=data["building_id"],
            floor_number=int(data.get("floor_number", data["floor_id"])),
            label=data["label"],
            elevation_m=float(data.get("elevation_m", 0.0)),
            boundary=bound,
            rooms=rooms,
            corridors=corridors,
            doors=doors,
            wifi_reference_locations=dict(data.get("wifi_reference_locations", {})),
            landmarks=dict(data.get("landmarks", {})),
            verification_status=MapVerificationStatus.from_string(data.get("verification_status", "PARTIALLY_SURVEYED")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class BuildingMap:
    """Hierarchical root map representing an entire multi-floor indoor structure."""
    building_id: str
    name: str
    coordinate_frame: CoordinateFrameMetadata = field(default_factory=CoordinateFrameMetadata)
    floors: Dict[int, FloorMap] = field(default_factory=dict)
    vertical_transitions: Dict[str, VerticalTransition] = field(default_factory=dict)
    connections: Dict[str, SpatialConnection] = field(default_factory=dict)
    map_version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    survey_provenance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def floor_count(self) -> int:
        return len(self.floors)

    @property
    def total_rooms_count(self) -> int:
        return sum(len(f.rooms) for f in self.floors.values())

    @property
    def total_doors_count(self) -> int:
        return sum(len(f.doors) for f in self.floors.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "building_id": self.building_id,
            "name": self.name,
            "map_version": self.map_version,
            "coordinate_frame": {
                "frame_id": self.coordinate_frame.frame_id,
                "units": self.coordinate_frame.units,
                "x_axis_description": self.coordinate_frame.x_axis_description,
                "y_axis_description": self.coordinate_frame.y_axis_description,
            },
            "floor_count": self.floor_count,
            "total_rooms_count": self.total_rooms_count,
            "total_doors_count": self.total_doors_count,
            "floors": {str(k): v.to_dict() for k, v in self.floors.items()},
            "vertical_transitions": {k: v.to_dict() for k, v in self.vertical_transitions.items()},
            "connections": {k: v.to_dict() for k, v in self.connections.items()},
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "survey_provenance": self.survey_provenance,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BuildingMap":
        coord_meta = CoordinateFrameMetadata()
        if "coordinate_frame" in data:
            cf = data["coordinate_frame"]
            coord_meta = CoordinateFrameMetadata(
                frame_id=cf.get("frame_id", "BUILDING_LOCAL_CARTESIAN_ENU"),
                units=cf.get("units", "meters"),
                x_axis_description=cf.get("x_axis_description", "Building Local East (+X)"),
                y_axis_description=cf.get("y_axis_description", "Building Local North (+Y)"),
            )

        floors = {int(k): FloorMap.from_dict(v) for k, v in data.get("floors", {}).items()}
        transitions = {k: VerticalTransition.from_dict(v) for k, v in data.get("vertical_transitions", {}).items()}
        connections = {k: SpatialConnection.from_dict(v) for k, v in data.get("connections", {}).items()}

        return cls(
            building_id=data["building_id"],
            name=data["name"],
            coordinate_frame=coord_meta,
            floors=floors,
            vertical_transitions=transitions,
            connections=connections,
            map_version=data.get("map_version", "1.0.0"),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=data.get("updated_at", datetime.now(timezone.utc).isoformat()),
            survey_provenance=dict(data.get("survey_provenance", {})),
            metadata=dict(data.get("metadata", {})),
        )
