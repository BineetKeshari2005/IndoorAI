"""End-to-end mapping pipeline orchestrating spatial ingestion, validation, and storage."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from src.mapping.evaluation import MapEvaluator, MapEvaluationMetrics
from src.mapping.map_builder import MapBuilder
from src.mapping.map_export import MapExporter
from src.mapping.map_storage import MapStorage
from src.mapping.map_validator import MapValidator, MapValidationReport
from src.mapping.models import BuildingMap, MapVerificationStatus

logger = logging.getLogger(__name__)


class MappingPipeline:
    """Orchestrates ingestion of survey annotations, Wi-Fi radio-maps, and visual landmarks into indoor maps."""

    def __init__(self, storage: Optional[MapStorage] = None):
        self.storage = storage or MapStorage()

    def construct_map(
        self,
        building_id: str,
        name: str,
        floors_spec: List[Dict[str, Any]],
        rooms_spec: List[Dict[str, Any]],
        corridors_spec: List[Dict[str, Any]],
        doors_spec: List[Dict[str, Any]],
        vertical_transitions_spec: Optional[List[Dict[str, Any]]] = None,
        wifi_references_by_floor: Optional[Dict[int, List[Dict[str, Any]]]] = None,
        landmarks_by_floor: Optional[Dict[int, List[Dict[str, Any]]]] = None,
        auto_link: bool = True,
        validate: bool = True,
    ) -> Tuple[BuildingMap, MapValidationReport, MapEvaluationMetrics]:
        """Construct a validated multi-floor BuildingMap from structured survey specifications."""
        builder = MapBuilder(building_id=building_id, name=name)

        # 1. Floors
        for f_spec in floors_spec:
            builder.add_floor(
                floor_id=int(f_spec["floor_id"]),
                label=f_spec.get("label", f"Floor {f_spec['floor_id']}"),
                floor_number=f_spec.get("floor_number"),
                elevation_m=float(f_spec.get("elevation_m", 0.0)),
                boundary=f_spec.get("boundary"),
                verification_status=f_spec.get("verification_status", MapVerificationStatus.PARTIALLY_SURVEYED),
                metadata=f_spec.get("metadata", {}),
            )

        # 2. Corridors
        for c_spec in corridors_spec:
            builder.add_corridor(
                corridor_id=c_spec["corridor_id"],
                floor_id=int(c_spec["floor_id"]),
                name=c_spec.get("name", c_spec["corridor_id"]),
                centerline=c_spec["centerline"],
                width_m=float(c_spec.get("width_m", 2.0)),
                polygon=c_spec.get("polygon"),
                connected_room_ids=c_spec.get("connected_room_ids"),
                connected_door_ids=c_spec.get("connected_door_ids"),
                verification_status=c_spec.get("verification_status", MapVerificationStatus.UNVERIFIED),
                metadata=c_spec.get("metadata", {}),
            )

        # 3. Rooms
        for r_spec in rooms_spec:
            builder.add_room(
                room_id=r_spec["room_id"],
                floor_id=int(r_spec["floor_id"]),
                name=r_spec.get("name", r_spec["room_id"]),
                polygon=r_spec["polygon"],
                category=r_spec.get("category", "OTHER"),
                room_number=r_spec.get("room_number"),
                entrance_door_ids=r_spec.get("entrance_door_ids"),
                verification_status=r_spec.get("verification_status", MapVerificationStatus.UNVERIFIED),
                metadata=r_spec.get("metadata", {}),
            )

        # 4. Doors
        for d_spec in doors_spec:
            builder.add_door(
                door_id=d_spec["door_id"],
                floor_id=int(d_spec["floor_id"]),
                name=d_spec.get("name", d_spec["door_id"]),
                position=d_spec["position"],
                connects_from_id=d_spec["connects_from_id"],
                connects_to_id=d_spec["connects_to_id"],
                door_type=d_spec.get("door_type", "SINGLE_SWING"),
                width_m=float(d_spec.get("width_m", 0.90)),
                is_bidirectional=bool(d_spec.get("is_bidirectional", True)),
                is_emergency_exit=bool(d_spec.get("is_emergency_exit", False)),
                verification_status=d_spec.get("verification_status", MapVerificationStatus.UNVERIFIED),
                associated_landmark_id=d_spec.get("associated_landmark_id"),
                metadata=d_spec.get("metadata", {}),
            )

        # 5. Vertical Transitions
        if vertical_transitions_spec:
            for vt_spec in vertical_transitions_spec:
                builder.add_vertical_transition(
                    transition_id=vt_spec["transition_id"],
                    transition_type=vt_spec.get("transition_type", "STAIRCASE"),
                    name=vt_spec.get("name", vt_spec["transition_id"]),
                    floor_positions=vt_spec["floor_positions"],
                    is_accessible=bool(vt_spec.get("is_accessible", False)),
                    verification_status=vt_spec.get("verification_status", MapVerificationStatus.UNVERIFIED),
                    associated_landmark_ids=vt_spec.get("associated_landmark_ids"),
                    metadata=vt_spec.get("metadata", {}),
                )

        # 6. Wi-Fi Reference Locations
        if wifi_references_by_floor:
            for floor_id, refs in wifi_references_by_floor.items():
                builder.import_wifi_reference_locations(floor_id=floor_id, wifi_references=refs)

        # 7. Landmarks
        if landmarks_by_floor:
            for floor_id, lms in landmarks_by_floor.items():
                builder.import_verified_landmarks(floor_id=floor_id, landmarks=lms)

        # 8. Auto-link topological connections
        if auto_link:
            builder.auto_link_doors_and_spaces()
            builder.auto_link_vertical_transitions()

        # 9. Build and Validate
        building_map, report = builder.build(validate=validate)
        metrics = MapEvaluator.evaluate(building_map)

        return building_map, report, metrics

    def save_map(self, building_map: BuildingMap, publish_version: Optional[str] = None) -> BuildingMap:
        """Persist the building map to atomic storage and optionally publish a version snapshot."""
        saved_map = self.storage.save_building_map(building_map)
        if publish_version:
            saved_map = self.storage.publish_version(building_map.building_id, version_tag=publish_version)
        return saved_map

    def export_map(
        self,
        building_id: str,
        export_format: str = "json",
        floor_id: Optional[int] = None,
    ) -> Any:
        """Export stored map to JSON string or GeoJSON dict."""
        building_map = self.storage.load_building_map(building_id)
        if not building_map:
            raise FileNotFoundError(f"Building map {building_id} not found in storage")

        exporter = MapExporter(building_map)
        fmt = (export_format or "json").lower().strip()

        if fmt == "geojson":
            return exporter.export_geojson(floor_id=floor_id)
        elif fmt == "json":
            return exporter.export_json(floor_id=floor_id)
        else:
            raise ValueError(f"Unsupported export format: {export_format}. Use 'json' or 'geojson'.")
