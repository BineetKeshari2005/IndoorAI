"""Incremental builder API for constructing and integrating indoor building maps."""

from __future__ import annotations

import math
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from src.mapping.geometry import Point2D, Polygon2D
from src.mapping.map_validator import MapValidator, MapValidationReport
from src.mapping.models import (
    AccessibilityType,
    BuildingMap,
    ConnectionType,
    Corridor,
    Door,
    DoorType,
    FloorMap,
    MapVerificationStatus,
    Room,
    RoomCategory,
    SpatialConnection,
    VerticalTransition,
)
from src.sensor_fusion.coordinates import CoordinateFrameMetadata


class MapBuilder:
    """Builder for assembling validated, multi-floor indoor maps."""

    def __init__(self, building_id: str, name: str, coordinate_frame: Optional[CoordinateFrameMetadata] = None):
        self.building_id = building_id
        self.name = name
        self.coordinate_frame = coordinate_frame or CoordinateFrameMetadata()
        self.floors: Dict[int, FloorMap] = {}
        self.vertical_transitions: Dict[str, VerticalTransition] = {}
        self.connections: Dict[str, SpatialConnection] = {}
        self.survey_provenance: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}

    def add_floor(
        self,
        floor_id: int,
        label: str,
        floor_number: Optional[int] = None,
        elevation_m: float = 0.0,
        boundary: Optional[Union[Polygon2D, List[Tuple[float, float]], List[Dict[str, float]]]] = None,
        verification_status: MapVerificationStatus = MapVerificationStatus.PARTIALLY_SURVEYED,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "MapBuilder":
        """Add or update a floor in the building map."""
        bound_poly: Optional[Polygon2D] = None
        if boundary is not None:
            if isinstance(boundary, Polygon2D):
                bound_poly = boundary
            elif isinstance(boundary, list) and boundary:
                if isinstance(boundary[0], (tuple, list)):
                    bound_poly = Polygon2D([Point2D(float(p[0]), float(p[1])) for p in boundary])
                elif isinstance(boundary[0], dict):
                    bound_poly = Polygon2D([Point2D(float(p["x"]), float(p["y"])) for p in boundary])

        fnum = floor_number if floor_number is not None else floor_id
        self.floors[floor_id] = FloorMap(
            floor_id=floor_id,
            building_id=self.building_id,
            floor_number=fnum,
            label=label,
            elevation_m=elevation_m,
            boundary=bound_poly,
            verification_status=verification_status,
            metadata=metadata or {},
        )
        return self

    def _ensure_floor_exists(self, floor_id: int) -> FloorMap:
        if floor_id not in self.floors:
            self.add_floor(floor_id=floor_id, label=f"Floor {floor_id}", floor_number=floor_id)
        return self.floors[floor_id]

    def add_room(
        self,
        room_id: str,
        floor_id: int,
        name: str,
        polygon: Union[Polygon2D, List[Tuple[float, float]], List[Dict[str, float]]],
        category: Union[RoomCategory, str] = RoomCategory.OTHER,
        room_number: Optional[str] = None,
        entrance_door_ids: Optional[List[str]] = None,
        verification_status: Union[MapVerificationStatus, str] = MapVerificationStatus.UNVERIFIED,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "MapBuilder":
        """Add a room polygon to a specific floor."""
        floor = self._ensure_floor_exists(floor_id)

        poly: Polygon2D
        if isinstance(polygon, Polygon2D):
            poly = polygon
        elif isinstance(polygon, list) and polygon:
            if isinstance(polygon[0], (tuple, list)):
                poly = Polygon2D([Point2D(float(p[0]), float(p[1])) for p in polygon])
            elif isinstance(polygon[0], dict):
                poly = Polygon2D([Point2D(float(p["x"]), float(p["y"])) for p in polygon])
            else:
                raise ValueError("Invalid polygon vertex format")
        else:
            raise ValueError("Polygon vertices cannot be empty")

        cat = category if isinstance(category, RoomCategory) else RoomCategory.from_string(category)
        status = (
            verification_status
            if isinstance(verification_status, MapVerificationStatus)
            else MapVerificationStatus.from_string(verification_status)
        )

        room = Room(
            room_id=room_id,
            building_id=self.building_id,
            floor_id=floor_id,
            name=name,
            polygon=poly,
            category=cat,
            room_number=room_number,
            entrance_door_ids=list(entrance_door_ids or []),
            verification_status=status,
            metadata=metadata or {},
        )
        floor.rooms[room_id] = room
        return self

    def add_corridor(
        self,
        corridor_id: str,
        floor_id: int,
        name: str,
        centerline: Union[List[Point2D], List[Tuple[float, float]], List[Dict[str, float]]],
        width_m: float = 2.0,
        polygon: Optional[Union[Polygon2D, List[Tuple[float, float]], List[Dict[str, float]]]] = None,
        connected_room_ids: Optional[List[str]] = None,
        connected_door_ids: Optional[List[str]] = None,
        verification_status: Union[MapVerificationStatus, str] = MapVerificationStatus.UNVERIFIED,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "MapBuilder":
        """Add a navigable corridor centerline and optional polygon footprint."""
        floor = self._ensure_floor_exists(floor_id)

        points: List[Point2D] = []
        for p in centerline:
            if isinstance(p, Point2D):
                points.append(p)
            elif isinstance(p, (tuple, list)):
                points.append(Point2D(float(p[0]), float(p[1])))
            elif isinstance(p, dict):
                points.append(Point2D(float(p["x"]), float(p["y"])))

        poly: Optional[Polygon2D] = None
        if polygon is not None:
            if isinstance(polygon, Polygon2D):
                poly = polygon
            elif isinstance(polygon, list) and polygon:
                if isinstance(polygon[0], (tuple, list)):
                    poly = Polygon2D([Point2D(float(pt[0]), float(pt[1])) for pt in polygon])
                elif isinstance(polygon[0], dict):
                    poly = Polygon2D([Point2D(float(pt["x"]), float(pt["y"])) for pt in polygon])

        status = (
            verification_status
            if isinstance(verification_status, MapVerificationStatus)
            else MapVerificationStatus.from_string(verification_status)
        )

        corridor = Corridor(
            corridor_id=corridor_id,
            building_id=self.building_id,
            floor_id=floor_id,
            name=name,
            centerline=points,
            polygon=poly,
            width_m=width_m,
            connected_room_ids=list(connected_room_ids or []),
            connected_door_ids=list(connected_door_ids or []),
            verification_status=status,
            metadata=metadata or {},
        )
        floor.corridors[corridor_id] = corridor
        return self

    def add_door(
        self,
        door_id: str,
        floor_id: int,
        name: str,
        position: Union[Point2D, Tuple[float, float], Dict[str, float]],
        connects_from_id: str,
        connects_to_id: str,
        door_type: Union[DoorType, str] = DoorType.SINGLE_SWING,
        width_m: float = 0.90,
        is_bidirectional: bool = True,
        is_emergency_exit: bool = False,
        verification_status: Union[MapVerificationStatus, str] = MapVerificationStatus.UNVERIFIED,
        associated_landmark_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "MapBuilder":
        """Add a door waypoint linking two spaces."""
        floor = self._ensure_floor_exists(floor_id)

        pos: Point2D
        if isinstance(position, Point2D):
            pos = position
        elif isinstance(position, (tuple, list)):
            pos = Point2D(float(position[0]), float(position[1]))
        elif isinstance(position, dict):
            pos = Point2D(float(position["x"]), float(position["y"]))
        else:
            raise ValueError("Invalid position format for door")

        dtype = door_type if isinstance(door_type, DoorType) else DoorType(door_type)
        status = (
            verification_status
            if isinstance(verification_status, MapVerificationStatus)
            else MapVerificationStatus.from_string(verification_status)
        )

        door = Door(
            door_id=door_id,
            building_id=self.building_id,
            floor_id=floor_id,
            name=name,
            position=pos,
            connects_from_id=connects_from_id,
            connects_to_id=connects_to_id,
            door_type=dtype,
            width_m=width_m,
            is_bidirectional=is_bidirectional,
            is_emergency_exit=is_emergency_exit,
            verification_status=status,
            associated_landmark_id=associated_landmark_id,
            metadata=metadata or {},
        )
        floor.doors[door_id] = door

        # Link door reference in room if applicable
        if connects_from_id in floor.rooms:
            if door_id not in floor.rooms[connects_from_id].entrance_door_ids:
                floor.rooms[connects_from_id].entrance_door_ids.append(door_id)
        if connects_to_id in floor.rooms:
            if door_id not in floor.rooms[connects_to_id].entrance_door_ids:
                floor.rooms[connects_to_id].entrance_door_ids.append(door_id)

        # Link door reference in corridor if applicable
        if connects_from_id in floor.corridors:
            if door_id not in floor.corridors[connects_from_id].connected_door_ids:
                floor.corridors[connects_from_id].connected_door_ids.append(door_id)
        if connects_to_id in floor.corridors:
            if door_id not in floor.corridors[connects_to_id].connected_door_ids:
                floor.corridors[connects_to_id].connected_door_ids.append(door_id)

        return self

    def add_vertical_transition(
        self,
        transition_id: str,
        transition_type: Union[ConnectionType, str],
        name: str,
        floor_positions: Dict[int, Union[Point2D, Tuple[float, float], Dict[str, float]]],
        is_accessible: bool = False,
        verification_status: Union[MapVerificationStatus, str] = MapVerificationStatus.UNVERIFIED,
        associated_landmark_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "MapBuilder":
        """Add a multi-floor vertical transition (stairwell or elevator shaft)."""
        pos_dict: Dict[int, Point2D] = {}
        for f_id, p in floor_positions.items():
            fid_int = int(f_id)
            self._ensure_floor_exists(fid_int)
            if isinstance(p, Point2D):
                pos_dict[fid_int] = p
            elif isinstance(p, (tuple, list)):
                pos_dict[fid_int] = Point2D(float(p[0]), float(p[1]))
            elif isinstance(p, dict):
                pos_dict[fid_int] = Point2D(float(p["x"]), float(p["y"]))

        ttype = (
            transition_type
            if isinstance(transition_type, ConnectionType)
            else ConnectionType.from_string(transition_type)
        )
        status = (
            verification_status
            if isinstance(verification_status, MapVerificationStatus)
            else MapVerificationStatus.from_string(verification_status)
        )

        vt = VerticalTransition(
            transition_id=transition_id,
            building_id=self.building_id,
            transition_type=ttype,
            name=name,
            floor_positions=pos_dict,
            is_accessible=is_accessible,
            verification_status=status,
            associated_landmark_ids=list(associated_landmark_ids or []),
            metadata=metadata or {},
        )
        self.vertical_transitions[transition_id] = vt
        return self

    def add_connection(
        self,
        source_id: str,
        destination_id: str,
        source_floor_id: int,
        destination_floor_id: int,
        connection_type: Union[ConnectionType, str],
        distance_m: Optional[float] = None,
        connection_id: Optional[str] = None,
        is_bidirectional: bool = True,
        accessibility: Union[AccessibilityType, str] = AccessibilityType.WHEELCHAIR_ACCESSIBLE,
        verification_status: Union[MapVerificationStatus, str] = MapVerificationStatus.UNVERIFIED,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "MapBuilder":
        """Explicitly add a topological navigation connection edge."""
        cid = connection_id or f"conn_{source_id}_{destination_id}"
        ctype = (
            connection_type
            if isinstance(connection_type, ConnectionType)
            else ConnectionType.from_string(connection_type)
        )
        acc = (
            accessibility
            if isinstance(accessibility, AccessibilityType)
            else AccessibilityType(accessibility)
        )
        status = (
            verification_status
            if isinstance(verification_status, MapVerificationStatus)
            else MapVerificationStatus.from_string(verification_status)
        )

        # Estimate distance if not explicitly provided
        dist = float(distance_m) if distance_m is not None else 1.0

        conn = SpatialConnection(
            connection_id=cid,
            building_id=self.building_id,
            source_id=source_id,
            destination_id=destination_id,
            source_floor_id=source_floor_id,
            destination_floor_id=destination_floor_id,
            connection_type=ctype,
            distance_m=dist,
            is_bidirectional=is_bidirectional,
            accessibility=acc,
            verification_status=status,
            metadata=metadata or {},
        )
        self.connections[cid] = conn
        return self

    def import_wifi_reference_locations(
        self,
        floor_id: int,
        wifi_references: List[Dict[str, Any]],
        auto_associate_rooms: bool = True,
    ) -> "MapBuilder":
        """Import Wi-Fi radio-map survey reference locations into the floor map.

        Associates points with enclosing room polygons if within boundaries.
        """
        floor = self._ensure_floor_exists(floor_id)
        for ref in wifi_references:
            ref_id = str(ref.get("reference_id") or ref.get("id") or uuid.uuid4().hex[:8])
            pos_data = ref.get("position", {})
            px = float(pos_data.get("x", ref.get("x", 0.0)))
            py = float(pos_data.get("y", ref.get("y", 0.0)))
            pt = Point2D(px, py)

            associated_room: Optional[str] = ref.get("room_id")
            if auto_associate_rooms and not associated_room:
                for r_id, room in floor.rooms.items():
                    if room.polygon.contains_point(pt):
                        associated_room = r_id
                        if ref_id not in room.wifi_reference_ids:
                            room.wifi_reference_ids.append(ref_id)
                        break

            floor.wifi_reference_locations[ref_id] = {
                "reference_id": ref_id,
                "label": ref.get("label", ref_id),
                "position": {"x": px, "y": py},
                "floor_id": floor_id,
                "room_id": associated_room,
                "fingerprint_count": int(ref.get("fingerprint_count", len(ref.get("fingerprints", [])))),
                "source": ref.get("source", "TASK_23_RADIO_MAP"),
            }
        return self

    def import_verified_landmarks(
        self,
        floor_id: int,
        landmarks: List[Dict[str, Any]],
        auto_associate_doors: bool = True,
    ) -> "MapBuilder":
        """Import verified computer vision landmark detections (Task 25) into the floor map."""
        floor = self._ensure_floor_exists(floor_id)
        for lm in landmarks:
            lm_id = str(lm.get("landmark_id") or lm.get("id") or uuid.uuid4().hex[:8])
            pos_data = lm.get("position", {})
            px = float(pos_data.get("x", lm.get("x", 0.0)))
            py = float(pos_data.get("y", lm.get("y", 0.0)))
            pt = Point2D(px, py)

            category = lm.get("category", "OTHER")
            status = lm.get("verification_status", "VERIFIED")

            # Link with door if landmark is near a door
            if auto_associate_doors and ("DOOR" in category or "ENTRANCE" in category):
                closest_door_id: Optional[str] = None
                closest_dist = 2.0  # within 2m threshold
                for d_id, door in floor.doors.items():
                    dist = door.position.distance_to(pt)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_door_id = d_id

                if closest_door_id:
                    floor.doors[closest_door_id].associated_landmark_id = lm_id

            floor.landmarks[lm_id] = {
                "landmark_id": lm_id,
                "name": lm.get("name", lm.get("label", f"Landmark {lm_id}")),
                "category": category,
                "floor_id": floor_id,
                "position": {"x": px, "y": py},
                "verification_status": status,
                "source": lm.get("source", "TASK_25_COMPUTER_VISION"),
            }
        return self

    def auto_link_doors_and_spaces(self) -> "MapBuilder":
        """Generate topological navigation connections from doors to their connected spaces."""
        for floor in self.floors.values():
            for door_id, door in floor.doors.items():
                door_node_id = f"door_{door_id}"

                # Link connects_from
                src_pos = self._get_spatial_entity_centroid(floor, door.connects_from_id)
                if src_pos:
                    dist_from = round(door.position.distance_to(src_pos), 2)
                    cid_from = f"conn_{door.connects_from_id}_{door_node_id}"
                    self.add_connection(
                        source_id=door.connects_from_id,
                        destination_id=door_node_id,
                        source_floor_id=floor.floor_id,
                        destination_floor_id=floor.floor_id,
                        connection_type=ConnectionType.DOOR,
                        distance_m=dist_from,
                        connection_id=cid_from,
                        is_bidirectional=door.is_bidirectional,
                        accessibility=AccessibilityType.WHEELCHAIR_ACCESSIBLE,
                        verification_status=door.verification_status,
                    )

                # Link connects_to
                dst_pos = self._get_spatial_entity_centroid(floor, door.connects_to_id)
                if dst_pos:
                    dist_to = round(door.position.distance_to(dst_pos), 2)
                    cid_to = f"conn_{door_node_id}_{door.connects_to_id}"
                    self.add_connection(
                        source_id=door_node_id,
                        destination_id=door.connects_to_id,
                        source_floor_id=floor.floor_id,
                        destination_floor_id=floor.floor_id,
                        connection_type=ConnectionType.DOOR,
                        distance_m=dist_to,
                        connection_id=cid_to,
                        is_bidirectional=door.is_bidirectional,
                        accessibility=AccessibilityType.WHEELCHAIR_ACCESSIBLE,
                        verification_status=door.verification_status,
                    )
        return self

    def auto_link_vertical_transitions(self, vertical_distance_per_floor_m: float = 3.5) -> "MapBuilder":
        """Generate vertical topological navigation connections across consecutive floor landings."""
        for vt_id, vt in self.vertical_transitions.items():
            sorted_floors = vt.connected_floors
            acc = (
                AccessibilityType.WHEELCHAIR_ACCESSIBLE
                if vt.is_accessible
                else AccessibilityType.STAIRS_ONLY
            )
            for i in range(len(sorted_floors) - 1):
                f_low = sorted_floors[i]
                f_high = sorted_floors[i + 1]
                pos_low = vt.floor_positions[f_low]
                pos_high = vt.floor_positions[f_high]

                dx = pos_high.x - pos_low.x
                dy = pos_high.y - pos_low.y
                dz = abs(f_high - f_low) * vertical_distance_per_floor_m
                dist_3d = round(math.sqrt(dx * dx + dy * dy + dz * dz), 2)

                node_low = f"vt_{vt.transition_id}_f{f_low}"
                node_high = f"vt_{vt.transition_id}_f{f_high}"
                cid = f"conn_vt_{vt.transition_id}_{f_low}_{f_high}"

                self.add_connection(
                    source_id=node_low,
                    destination_id=node_high,
                    source_floor_id=f_low,
                    destination_floor_id=f_high,
                    connection_type=vt.transition_type,
                    distance_m=dist_3d,
                    connection_id=cid,
                    is_bidirectional=True,
                    accessibility=acc,
                    verification_status=vt.verification_status,
                )
        return self

    def _get_spatial_entity_centroid(self, floor: FloorMap, entity_id: str) -> Optional[Point2D]:
        """Resolve the spatial point or centroid of a room, corridor, or vertical landing."""
        if entity_id in floor.rooms:
            return floor.rooms[entity_id].centroid
        if entity_id in floor.corridors:
            corr = floor.corridors[entity_id]
            if corr.centerline:
                # Midpoint of corridor centerline
                mid_idx = len(corr.centerline) // 2
                return corr.centerline[mid_idx]
        for vt in self.vertical_transitions.values():
            if vt.transition_id == entity_id or f"vt_{vt.transition_id}" in entity_id:
                if floor.floor_id in vt.floor_positions:
                    return vt.floor_positions[floor.floor_id]
        return None

    def build(self, validate: bool = True) -> Tuple[BuildingMap, MapValidationReport]:
        """Construct the BuildingMap instance and run structural map validation."""
        building_map = BuildingMap(
            building_id=self.building_id,
            name=self.name,
            coordinate_frame=self.coordinate_frame,
            floors=self.floors,
            vertical_transitions=self.vertical_transitions,
            connections=self.connections,
            survey_provenance=self.survey_provenance,
            metadata=self.metadata,
        )

        validator = MapValidator()
        report = validator.validate(building_map)

        if validate and not report.is_valid:
            # We still return the map alongside the report so caller can inspect errors
            pass

        return building_map, report
