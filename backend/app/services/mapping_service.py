"""Service layer for indoor mapping, spatial representation, and building topology."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.app.core.dataset_config import DatasetConfig, default_dataset_config
from backend.app.schemas.mapping import (
    CreateBuildingMapRequest,
    CreateConnectionRequest,
    CreateCorridorRequest,
    CreateDoorRequest,
    CreateFloorRequest,
    CreateRoomRequest,
    CreateVerticalTransitionRequest,
    ImportLandmarksRequest,
    ImportWifiReferencesRequest,
)
from src.core.logging import setup_logger
from src.mapping.evaluation import (
    MapEvaluationMetrics,
    MapEvaluator,
    generate_controlled_synthetic_building_map,
)
from src.mapping.geometry import Point2D, Polygon2D
from src.mapping.map_builder import MapBuilder
from src.mapping.map_export import MapExporter
from src.mapping.map_storage import MapStorage
from src.mapping.map_validator import MapValidationReport, MapValidator
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
from src.mapping.pipeline import MappingPipeline
from src.mapping.topology import IndoorTopologyGraph

logger = setup_logger("MappingService")


class MappingService:
    """Manages building maps, floorplans, spatial entities, validation, and versioning."""

    def __init__(self, config: Optional[DatasetConfig] = None, storage: Optional[MapStorage] = None):
        self.config = config or default_dataset_config
        self.maps_dir = self.config.base_dir / "indoor_maps"
        self.maps_dir.mkdir(parents=True, exist_ok=True)
        self.storage = storage or MapStorage(base_dir=self.maps_dir)
        self.pipeline = MappingPipeline(storage=self.storage)

    def list_buildings(self) -> List[Dict[str, Any]]:
        """Return all persisted building maps with high-level summary metadata."""
        summaries = self.storage.list_buildings()
        # If storage is empty, initialize synthetic fixture so dashboard and tests always have sample data
        if not summaries:
            synth_map = generate_controlled_synthetic_building_map()
            self.storage.save_building_map(synth_map)
            self.storage.publish_version(synth_map.building_id, version_tag="v1.0.0")
            summaries = self.storage.list_buildings()
        return summaries

    def get_building_map(self, building_id: str) -> Optional[BuildingMap]:
        """Retrieve building map by ID."""
        bmap = self.storage.load_building_map(building_id)
        if not bmap and building_id in ("SYNTH_BLDG_01", "SYNTH_BENCHMARK"):
            bmap = generate_controlled_synthetic_building_map(building_id=building_id)
            self.storage.save_building_map(bmap)
        return bmap

    def create_or_update_building(self, req: CreateBuildingMapRequest) -> BuildingMap:
        """Create a new building map or update an existing one's root attributes."""
        existing = self.storage.load_building_map(req.building_id)
        if existing:
            existing.name = req.name
            if req.metadata:
                existing.metadata.update(req.metadata)
            return self.storage.save_building_map(existing)
        else:
            bmap = BuildingMap(
                building_id=req.building_id,
                name=req.name,
                metadata=req.metadata or {},
            )
            return self.storage.save_building_map(bmap)

    def add_floor(self, building_id: str, req: CreateFloorRequest) -> BuildingMap:
        """Add or update a floor in the specified building."""
        bmap = self.get_building_map(building_id)
        if not bmap:
            bmap = BuildingMap(building_id=building_id, name=f"Building {building_id}")

        bound_poly = None
        if req.boundary:
            bound_poly = Polygon2D([Point2D(p.x, p.y) for p in req.boundary])

        fnum = req.floor_number if req.floor_number is not None else req.floor_id
        fmap = FloorMap(
            floor_id=req.floor_id,
            building_id=building_id,
            floor_number=fnum,
            label=req.label,
            elevation_m=req.elevation_m,
            boundary=bound_poly,
            verification_status=MapVerificationStatus.from_string(req.verification_status),
            metadata=req.metadata or {},
        )
        bmap.floors[req.floor_id] = fmap
        return self.storage.save_building_map(bmap)

    def add_room(self, building_id: str, req: CreateRoomRequest) -> BuildingMap:
        """Add or update a room polygon in the specified building floor."""
        bmap = self.get_building_map(building_id)
        if not bmap:
            raise ValueError(f"Building {building_id} not found")
        if req.floor_id not in bmap.floors:
            # Create floor automatically if absent
            self.add_floor(
                building_id,
                CreateFloorRequest(floor_id=req.floor_id, label=f"Floor {req.floor_id}"),
            )
            bmap = self.get_building_map(building_id)

        poly = Polygon2D([Point2D(p.x, p.y) for p in req.polygon])
        room = Room(
            room_id=req.room_id,
            building_id=building_id,
            floor_id=req.floor_id,
            name=req.name,
            polygon=poly,
            category=RoomCategory.from_string(req.category),
            room_number=req.room_number,
            entrance_door_ids=list(req.entrance_door_ids),
            verification_status=MapVerificationStatus.from_string(req.verification_status),
            metadata=req.metadata or {},
        )
        bmap.floors[req.floor_id].rooms[req.room_id] = room
        return self.storage.save_building_map(bmap)

    def add_corridor(self, building_id: str, req: CreateCorridorRequest) -> BuildingMap:
        """Add or update a corridor centerline in the specified building floor."""
        bmap = self.get_building_map(building_id)
        if not bmap:
            raise ValueError(f"Building {building_id} not found")
        if req.floor_id not in bmap.floors:
            self.add_floor(
                building_id,
                CreateFloorRequest(floor_id=req.floor_id, label=f"Floor {req.floor_id}"),
            )
            bmap = self.get_building_map(building_id)

        pts = [Point2D(p.x, p.y) for p in req.centerline]
        poly = Polygon2D([Point2D(p.x, p.y) for p in req.polygon]) if req.polygon else None

        corridor = Corridor(
            corridor_id=req.corridor_id,
            building_id=building_id,
            floor_id=req.floor_id,
            name=req.name,
            centerline=pts,
            polygon=poly,
            width_m=req.width_m,
            connected_room_ids=list(req.connected_room_ids),
            connected_door_ids=list(req.connected_door_ids),
            verification_status=MapVerificationStatus.from_string(req.verification_status),
            metadata=req.metadata or {},
        )
        bmap.floors[req.floor_id].corridors[req.corridor_id] = corridor
        return self.storage.save_building_map(bmap)

    def add_door(self, building_id: str, req: CreateDoorRequest) -> BuildingMap:
        """Add or update a door connecting spaces in the building floor."""
        bmap = self.get_building_map(building_id)
        if not bmap:
            raise ValueError(f"Building {building_id} not found")
        if req.floor_id not in bmap.floors:
            self.add_floor(
                building_id,
                CreateFloorRequest(floor_id=req.floor_id, label=f"Floor {req.floor_id}"),
            )
            bmap = self.get_building_map(building_id)

        pos = Point2D(req.position.x, req.position.y)
        door = Door(
            door_id=req.door_id,
            building_id=building_id,
            floor_id=req.floor_id,
            name=req.name,
            position=pos,
            connects_from_id=req.connects_from_id,
            connects_to_id=req.connects_to_id,
            door_type=DoorType(req.door_type),
            width_m=req.width_m,
            is_bidirectional=req.is_bidirectional,
            is_emergency_exit=req.is_emergency_exit,
            verification_status=MapVerificationStatus.from_string(req.verification_status),
            associated_landmark_id=req.associated_landmark_id,
            metadata=req.metadata or {},
        )
        bmap.floors[req.floor_id].doors[req.door_id] = door

        # Also auto-link entrance IDs in room
        floor = bmap.floors[req.floor_id]
        if req.connects_from_id in floor.rooms:
            if req.door_id not in floor.rooms[req.connects_from_id].entrance_door_ids:
                floor.rooms[req.connects_from_id].entrance_door_ids.append(req.door_id)
        if req.connects_to_id in floor.rooms:
            if req.door_id not in floor.rooms[req.connects_to_id].entrance_door_ids:
                floor.rooms[req.connects_to_id].entrance_door_ids.append(req.door_id)

        return self.storage.save_building_map(bmap)

    def add_vertical_transition(self, building_id: str, req: CreateVerticalTransitionRequest) -> BuildingMap:
        """Add or update a vertical transition (staircase / elevator) across floors."""
        bmap = self.get_building_map(building_id)
        if not bmap:
            raise ValueError(f"Building {building_id} not found")

        pos_dict = {int(k): Point2D(v.x, v.y) for k, v in req.floor_positions.items()}
        vt = VerticalTransition(
            transition_id=req.transition_id,
            building_id=building_id,
            transition_type=ConnectionType.from_string(req.transition_type),
            name=req.name,
            floor_positions=pos_dict,
            is_accessible=req.is_accessible,
            verification_status=MapVerificationStatus.from_string(req.verification_status),
            associated_landmark_ids=list(req.associated_landmark_ids),
            metadata=req.metadata or {},
        )
        bmap.vertical_transitions[req.transition_id] = vt
        return self.storage.save_building_map(bmap)

    def add_connection(self, building_id: str, req: CreateConnectionRequest) -> BuildingMap:
        """Add a navigation topological connection edge."""
        bmap = self.get_building_map(building_id)
        if not bmap:
            raise ValueError(f"Building {building_id} not found")

        cid = f"conn_{req.source_id}_{req.destination_id}"
        dist = float(req.distance_m) if req.distance_m is not None else 1.0
        conn = SpatialConnection(
            connection_id=cid,
            building_id=building_id,
            source_id=req.source_id,
            destination_id=req.destination_id,
            source_floor_id=req.source_floor_id,
            destination_floor_id=req.destination_floor_id,
            connection_type=ConnectionType.from_string(req.connection_type),
            distance_m=dist,
            is_bidirectional=req.is_bidirectional,
            accessibility=AccessibilityType(req.accessibility),
            verification_status=MapVerificationStatus.from_string(req.verification_status),
            metadata=req.metadata or {},
        )
        bmap.connections[cid] = conn
        return self.storage.save_building_map(bmap)

    def import_wifi_references(self, building_id: str, req: ImportWifiReferencesRequest) -> BuildingMap:
        """Import Wi-Fi survey reference points into the floor map."""
        bmap = self.get_building_map(building_id)
        if not bmap:
            raise ValueError(f"Building {building_id} not found")
        builder = MapBuilder(building_id=bmap.building_id, name=bmap.name)
        builder.floors = bmap.floors
        builder.vertical_transitions = bmap.vertical_transitions
        builder.connections = bmap.connections
        builder.import_wifi_reference_locations(
            floor_id=req.floor_id,
            wifi_references=req.wifi_references,
            auto_associate_rooms=req.auto_associate_rooms,
        )
        bmap.floors = builder.floors
        return self.storage.save_building_map(bmap)

    def import_landmarks(self, building_id: str, req: ImportLandmarksRequest) -> BuildingMap:
        """Import verified visual landmarks into the floor map."""
        bmap = self.get_building_map(building_id)
        if not bmap:
            raise ValueError(f"Building {building_id} not found")
        builder = MapBuilder(building_id=bmap.building_id, name=bmap.name)
        builder.floors = bmap.floors
        builder.vertical_transitions = bmap.vertical_transitions
        builder.connections = bmap.connections
        builder.import_verified_landmarks(
            floor_id=req.floor_id,
            landmarks=req.landmarks,
            auto_associate_doors=req.auto_associate_doors,
        )
        bmap.floors = builder.floors
        return self.storage.save_building_map(bmap)

    def validate_building_map(self, building_id: str) -> MapValidationReport:
        """Validate structural geometry, topology, and integrity of a building map."""
        bmap = self.get_building_map(building_id)
        if not bmap:
            raise ValueError(f"Building {building_id} not found")
        validator = MapValidator()
        return validator.validate(bmap)

    def evaluate_building_map(self, building_id: str) -> MapEvaluationMetrics:
        """Evaluate coverage, connectivity, verification, and accessibility metrics."""
        bmap = self.get_building_map(building_id)
        if not bmap:
            raise ValueError(f"Building {building_id} not found")
        return MapEvaluator.evaluate(bmap)

    def publish_version(self, building_id: str, version_tag: Optional[str] = None) -> BuildingMap:
        """Publish an immutable numbered release snapshot for a building map."""
        return self.storage.publish_version(building_id=building_id, version_tag=version_tag)

    def list_versions(self, building_id: str) -> List[Dict[str, Any]]:
        """List all published version snapshots for a building map."""
        return self.storage.list_versions(building_id=building_id)

    def export_map(
        self,
        building_id: str,
        export_format: str = "json",
        floor_id: Optional[int] = None,
    ) -> Any:
        """Export building map in JSON or GeoJSON FeatureCollection format."""
        bmap = self.get_building_map(building_id)
        if not bmap:
            raise ValueError(f"Building {building_id} not found")
        exporter = MapExporter(bmap)
        fmt = (export_format or "json").lower().strip()
        if fmt == "geojson":
            return exporter.export_geojson(floor_id=floor_id)
        elif fmt == "json":
            return exporter.export_json(floor_id=floor_id)
        else:
            raise ValueError(f"Unsupported format: {export_format}. Use 'json' or 'geojson'.")
