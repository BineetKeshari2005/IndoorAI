"""Map evaluation framework and synthetic indoor benchmark fixtures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.mapping.geometry import Point2D, Polygon2D
from src.mapping.map_builder import MapBuilder
from src.mapping.models import (
    AccessibilityType,
    BuildingMap,
    ConnectionType,
    DoorType,
    MapVerificationStatus,
    RoomCategory,
)
from src.mapping.topology import IndoorTopologyGraph


@dataclass
class MapEvaluationMetrics:
    """Quantitative quality, topological connectivity, and coverage metrics of an indoor map."""
    building_id: str
    total_floors: int
    total_rooms: int
    total_corridors: int
    total_doors: int
    total_vertical_transitions: int
    total_connections: int
    total_surveyed_area_sq_m: float
    graph_node_count: int
    graph_edge_count: int
    graph_connected_components: int
    isolated_nodes_count: int
    is_fully_connected: bool
    verification_ratio: float  # [0.0, 1.0]
    accessibility_ratio: float  # [0.0, 1.0]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "building_id": self.building_id,
            "total_floors": self.total_floors,
            "total_rooms": self.total_rooms,
            "total_corridors": self.total_corridors,
            "total_doors": self.total_doors,
            "total_vertical_transitions": self.total_vertical_transitions,
            "total_connections": self.total_connections,
            "total_surveyed_area_sq_m": round(self.total_surveyed_area_sq_m, 2),
            "graph_node_count": self.graph_node_count,
            "graph_edge_count": self.graph_edge_count,
            "graph_connected_components": self.graph_connected_components,
            "isolated_nodes_count": self.isolated_nodes_count,
            "is_fully_connected": self.is_fully_connected,
            "verification_ratio": round(self.verification_ratio, 3),
            "accessibility_ratio": round(self.accessibility_ratio, 3),
            "metadata": self.metadata,
        }


class MapEvaluator:
    """Calculates spatial coverage, topological connectivity, and survey verification metrics."""

    @staticmethod
    def evaluate(building_map: BuildingMap) -> MapEvaluationMetrics:
        graph = IndoorTopologyGraph.from_building_map(building_map)
        components = graph.get_connected_components()
        isolated = graph.get_isolated_nodes()

        total_rooms = building_map.total_rooms_count
        total_doors = building_map.total_doors_count
        total_corridors = sum(len(f.corridors) for f in building_map.floors.values())
        total_vts = len(building_map.vertical_transitions)
        total_conns = len(building_map.connections)

        total_area = sum(f.surveyed_room_area_sq_m for f in building_map.floors.values())

        # Verification ratio across rooms, corridors, doors, and transitions
        verified_count = 0
        total_entities = 0
        for floor in building_map.floors.values():
            for r in floor.rooms.values():
                total_entities += 1
                if r.verification_status == MapVerificationStatus.VERIFIED:
                    verified_count += 1
            for c in floor.corridors.values():
                total_entities += 1
                if c.verification_status == MapVerificationStatus.VERIFIED:
                    verified_count += 1
            for d in floor.doors.values():
                total_entities += 1
                if d.verification_status == MapVerificationStatus.VERIFIED:
                    verified_count += 1
        for vt in building_map.vertical_transitions.values():
            total_entities += 1
            if vt.verification_status == MapVerificationStatus.VERIFIED:
                verified_count += 1

        v_ratio = (verified_count / total_entities) if total_entities > 0 else 0.0

        # Accessibility ratio across graph edges
        accessible_edges = sum(
            1 for e in graph.edges.values()
            if e.accessibility == AccessibilityType.WHEELCHAIR_ACCESSIBLE
        )
        acc_ratio = (accessible_edges / len(graph.edges)) if graph.edges else 1.0

        return MapEvaluationMetrics(
            building_id=building_map.building_id,
            total_floors=building_map.floor_count,
            total_rooms=total_rooms,
            total_corridors=total_corridors,
            total_doors=total_doors,
            total_vertical_transitions=total_vts,
            total_connections=total_conns,
            total_surveyed_area_sq_m=total_area,
            graph_node_count=len(graph.nodes),
            graph_edge_count=len(graph.edges),
            graph_connected_components=len(components),
            isolated_nodes_count=len(isolated),
            is_fully_connected=len(components) <= 1,
            verification_ratio=v_ratio,
            accessibility_ratio=acc_ratio,
            metadata={
                "map_version": building_map.map_version,
                "building_name": building_map.name,
            },
        )


def generate_controlled_synthetic_building_map(
    building_id: str = "SYNTH_BLDG_01",
    name: str = "IndoorAI Synthetic Research Complex",
) -> BuildingMap:
    """Generate a fully validated 2-floor synthetic building map for testing and evaluation.

    Layout Specifications:
    Floor 0:
      - Boundary: 30m x 20m rectangle
      - Corridor 0: 24m centerline from (3, 10) to (27, 10), width 2m
      - Room 101: Lobby (category LOBBY, 8m x 6m at [3, 2] to [11, 8])
      - Room 102: Robotics Lab (category LAB, 12m x 6m at [13, 2] to [25, 8])
      - Room 103: Restroom (category RESTROOM, 6m x 6m at [3, 12] to [9, 18])
      - Doors:
        - Door D101: connects Room 101 to Corridor 0 at (7, 10)
        - Door D102: connects Room 102 to Corridor 0 at (19, 10)
        - Door D103: connects Room 103 to Corridor 0 at (6, 10)
      - Stairwell A Landing at (27, 10)
      - Elevator East Landing at (27, 14)
      - Wi-Fi Reference Locations: 2 surveyed points
      - Visual Landmarks: 2 verified landmarks

    Floor 1:
      - Boundary: 30m x 20m rectangle
      - Corridor 1: 24m centerline from (3, 10) to (27, 10), width 2m
      - Room 201: Conference Hall (category CONFERENCE, 8m x 6m at [3, 2] to [11, 8])
      - Room 202: Faculty Office (category OFFICE, 12m x 6m at [13, 2] to [25, 8])
      - Room 203: Computing Lab (category LAB, 6m x 6m at [3, 12] to [9, 18])
      - Doors:
        - Door D201: connects Room 201 to Corridor 1 at (7, 10)
        - Door D202: connects Room 202 to Corridor 1 at (19, 10)
        - Door D203: connects Room 203 to Corridor 1 at (6, 10)
      - Stairwell A Landing at (27, 10)
      - Elevator East Landing at (27, 14)
      - Wi-Fi Reference Locations: 2 surveyed points
      - Visual Landmarks: 2 verified landmarks
    """
    builder = MapBuilder(building_id=building_id, name=name)

    # ------------------ Floor 0 ------------------
    floor0_bound = Polygon2D.create_rectangle(min_x=0.0, min_y=0.0, max_x=30.0, max_y=20.0)
    builder.add_floor(
        floor_id=0,
        label="Ground Floor",
        floor_number=0,
        elevation_m=0.0,
        boundary=floor0_bound,
        verification_status=MapVerificationStatus.VERIFIED,
    )

    # Corridor 0
    builder.add_corridor(
        corridor_id="corr_f0_main",
        floor_id=0,
        name="Main Ground Corridor",
        centerline=[Point2D(3.0, 10.0), Point2D(15.0, 10.0), Point2D(27.0, 10.0)],
        width_m=2.0,
        verification_status=MapVerificationStatus.VERIFIED,
    )

    # Rooms Floor 0
    builder.add_room(
        room_id="room_101",
        floor_id=0,
        name="Reception & Lobby",
        polygon=Polygon2D.create_rectangle(min_x=3.0, min_y=2.0, max_x=11.0, max_y=8.0),
        category=RoomCategory.LOBBY,
        room_number="101",
        verification_status=MapVerificationStatus.VERIFIED,
    )
    builder.add_room(
        room_id="room_102",
        floor_id=0,
        name="Robotics Research Lab",
        polygon=Polygon2D.create_rectangle(min_x=13.0, min_y=2.0, max_x=25.0, max_y=8.0),
        category=RoomCategory.LAB,
        room_number="102",
        verification_status=MapVerificationStatus.VERIFIED,
    )
    builder.add_room(
        room_id="room_103",
        floor_id=0,
        name="Ground Restroom",
        polygon=Polygon2D.create_rectangle(min_x=3.0, min_y=12.0, max_x=9.0, max_y=18.0),
        category=RoomCategory.RESTROOM,
        room_number="103",
        verification_status=MapVerificationStatus.VERIFIED,
    )

    # Doors Floor 0
    builder.add_door(
        door_id="door_d101",
        floor_id=0,
        name="Lobby Entrance Door",
        position=Point2D(7.0, 9.0),
        connects_from_id="room_101",
        connects_to_id="corr_f0_main",
        door_type=DoorType.DOUBLE_SWING,
        verification_status=MapVerificationStatus.VERIFIED,
    )
    builder.add_door(
        door_id="door_d102",
        floor_id=0,
        name="Robotics Lab Door",
        position=Point2D(19.0, 9.0),
        connects_from_id="room_102",
        connects_to_id="corr_f0_main",
        door_type=DoorType.SINGLE_SWING,
        verification_status=MapVerificationStatus.VERIFIED,
    )
    builder.add_door(
        door_id="door_d103",
        floor_id=0,
        name="Restroom Door",
        position=Point2D(6.0, 11.0),
        connects_from_id="room_103",
        connects_to_id="corr_f0_main",
        door_type=DoorType.SINGLE_SWING,
        verification_status=MapVerificationStatus.VERIFIED,
    )

    # ------------------ Floor 1 ------------------
    floor1_bound = Polygon2D.create_rectangle(min_x=0.0, min_y=0.0, max_x=30.0, max_y=20.0)
    builder.add_floor(
        floor_id=1,
        label="First Floor",
        floor_number=1,
        elevation_m=3.8,
        boundary=floor1_bound,
        verification_status=MapVerificationStatus.VERIFIED,
    )

    # Corridor 1
    builder.add_corridor(
        corridor_id="corr_f1_main",
        floor_id=1,
        name="Upper First Floor Corridor",
        centerline=[Point2D(3.0, 10.0), Point2D(15.0, 10.0), Point2D(27.0, 10.0)],
        width_m=2.0,
        verification_status=MapVerificationStatus.VERIFIED,
    )

    # Rooms Floor 1
    builder.add_room(
        room_id="room_201",
        floor_id=1,
        name="Conference Auditorium",
        polygon=Polygon2D.create_rectangle(min_x=3.0, min_y=2.0, max_x=11.0, max_y=8.0),
        category=RoomCategory.CONFERENCE,
        room_number="201",
        verification_status=MapVerificationStatus.VERIFIED,
    )
    builder.add_room(
        room_id="room_202",
        floor_id=1,
        name="Faculty Suite",
        polygon=Polygon2D.create_rectangle(min_x=13.0, min_y=2.0, max_x=25.0, max_y=8.0),
        category=RoomCategory.OFFICE,
        room_number="202",
        verification_status=MapVerificationStatus.VERIFIED,
    )
    builder.add_room(
        room_id="room_203",
        floor_id=1,
        name="Advanced Computing Lab",
        polygon=Polygon2D.create_rectangle(min_x=3.0, min_y=12.0, max_x=9.0, max_y=18.0),
        category=RoomCategory.LAB,
        room_number="203",
        verification_status=MapVerificationStatus.VERIFIED,
    )

    # Doors Floor 1
    builder.add_door(
        door_id="door_d201",
        floor_id=1,
        name="Auditorium Double Door",
        position=Point2D(7.0, 9.0),
        connects_from_id="room_201",
        connects_to_id="corr_f1_main",
        door_type=DoorType.DOUBLE_SWING,
        verification_status=MapVerificationStatus.VERIFIED,
    )
    builder.add_door(
        door_id="door_d202",
        floor_id=1,
        name="Faculty Suite Door",
        position=Point2D(19.0, 9.0),
        connects_from_id="room_202",
        connects_to_id="corr_f1_main",
        door_type=DoorType.SINGLE_SWING,
        verification_status=MapVerificationStatus.VERIFIED,
    )
    builder.add_door(
        door_id="door_d203",
        floor_id=1,
        name="Computing Lab Door",
        position=Point2D(6.0, 11.0),
        connects_from_id="room_203",
        connects_to_id="corr_f1_main",
        door_type=DoorType.SINGLE_SWING,
        verification_status=MapVerificationStatus.VERIFIED,
    )

    # ------------------ Vertical Transitions ------------------
    # Stairwell A connecting Floor 0 and Floor 1
    builder.add_vertical_transition(
        transition_id="stairwell_a",
        transition_type=ConnectionType.STAIRCASE,
        name="East Stairwell A",
        floor_positions={
            0: Point2D(27.0, 10.0),
            1: Point2D(27.0, 10.0),
        },
        is_accessible=False,
        verification_status=MapVerificationStatus.VERIFIED,
    )

    # Elevator East connecting Floor 0 and Floor 1
    builder.add_vertical_transition(
        transition_id="elevator_east",
        transition_type=ConnectionType.ELEVATOR,
        name="Main Passenger Elevator",
        floor_positions={
            0: Point2D(27.0, 14.0),
            1: Point2D(27.0, 14.0),
        },
        is_accessible=True,
        verification_status=MapVerificationStatus.VERIFIED,
    )

    # Connect corridors to vertical transition landings
    builder.add_connection(
        source_id="corr_f0_main",
        destination_id="vt_stairwell_a_f0",
        source_floor_id=0,
        destination_floor_id=0,
        connection_type=ConnectionType.CORRIDOR,
        distance_m=0.5,
        verification_status=MapVerificationStatus.VERIFIED,
    )
    builder.add_connection(
        source_id="corr_f0_main",
        destination_id="vt_elevator_east_f0",
        source_floor_id=0,
        destination_floor_id=0,
        connection_type=ConnectionType.CORRIDOR,
        distance_m=4.0,
        verification_status=MapVerificationStatus.VERIFIED,
    )
    builder.add_connection(
        source_id="corr_f1_main",
        destination_id="vt_stairwell_a_f1",
        source_floor_id=1,
        destination_floor_id=1,
        connection_type=ConnectionType.CORRIDOR,
        distance_m=0.5,
        verification_status=MapVerificationStatus.VERIFIED,
    )
    builder.add_connection(
        source_id="corr_f1_main",
        destination_id="vt_elevator_east_f1",
        source_floor_id=1,
        destination_floor_id=1,
        connection_type=ConnectionType.CORRIDOR,
        distance_m=4.0,
        verification_status=MapVerificationStatus.VERIFIED,
    )

    # ------------------ Wi-Fi Reference Locations ------------------
    builder.import_wifi_reference_locations(
        floor_id=0,
        wifi_references=[
            {"reference_id": "wifi_ref_0_1", "label": "Lobby AP Point", "x": 7.0, "y": 5.0, "fingerprint_count": 12},
            {"reference_id": "wifi_ref_0_2", "label": "Robotics Center", "x": 19.0, "y": 5.0, "fingerprint_count": 15},
        ],
    )
    builder.import_wifi_reference_locations(
        floor_id=1,
        wifi_references=[
            {"reference_id": "wifi_ref_1_1", "label": "Auditorium AP", "x": 7.0, "y": 5.0, "fingerprint_count": 10},
            {"reference_id": "wifi_ref_1_2", "label": "Faculty Hall AP", "x": 19.0, "y": 5.0, "fingerprint_count": 14},
        ],
    )

    # ------------------ Visual Landmarks ------------------
    builder.import_verified_landmarks(
        floor_id=0,
        landmarks=[
            {"landmark_id": "lm_f0_door101", "name": "Lobby Signboard", "category": "SIGNBOARD", "x": 7.0, "y": 9.0},
            {"landmark_id": "lm_f0_door102", "name": "Robotics Lab Sign", "category": "ROOM_SIGN", "x": 19.0, "y": 9.0},
        ],
    )
    builder.import_verified_landmarks(
        floor_id=1,
        landmarks=[
            {"landmark_id": "lm_f1_door201", "name": "Auditorium Plaque", "category": "SIGNBOARD", "x": 7.0, "y": 9.0},
            {"landmark_id": "lm_f1_door202", "name": "Faculty Office Sign", "category": "ROOM_SIGN", "x": 19.0, "y": 9.0},
        ],
    )

    # Auto-link doors and vertical transitions
    builder.auto_link_doors_and_spaces()
    builder.auto_link_vertical_transitions()

    building_map, _ = builder.build(validate=True)
    return building_map
