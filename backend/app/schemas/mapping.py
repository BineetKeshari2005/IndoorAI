"""Pydantic request and response schemas for indoor map generation and spatial representations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Point2DSchema(BaseModel):
    x: float
    y: float


class CreateBuildingMapRequest(BaseModel):
    building_id: str = Field(..., min_length=1, description="Unique building identifier")
    name: str = Field(..., min_length=1, description="Human-readable building name")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class CreateFloorRequest(BaseModel):
    floor_id: int = Field(..., description="Integer floor index (e.g. 0, 1, 2)")
    label: str = Field(..., min_length=1, description="Floor label (e.g. Ground Floor)")
    floor_number: Optional[int] = None
    elevation_m: float = 0.0
    boundary: Optional[List[Point2DSchema]] = None
    verification_status: str = "PARTIALLY_SURVEYED"
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class CreateRoomRequest(BaseModel):
    room_id: str = Field(..., min_length=1)
    floor_id: int
    name: str = Field(..., min_length=1)
    polygon: List[Point2DSchema] = Field(..., min_length=3)
    category: str = "OTHER"
    room_number: Optional[str] = None
    entrance_door_ids: List[str] = Field(default_factory=list)
    verification_status: str = "UNVERIFIED"
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class CreateCorridorRequest(BaseModel):
    corridor_id: str = Field(..., min_length=1)
    floor_id: int
    name: str = Field(..., min_length=1)
    centerline: List[Point2DSchema] = Field(..., min_length=2)
    width_m: float = 2.0
    polygon: Optional[List[Point2DSchema]] = None
    connected_room_ids: List[str] = Field(default_factory=list)
    connected_door_ids: List[str] = Field(default_factory=list)
    verification_status: str = "UNVERIFIED"
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class CreateDoorRequest(BaseModel):
    door_id: str = Field(..., min_length=1)
    floor_id: int
    name: str = Field(..., min_length=1)
    position: Point2DSchema
    connects_from_id: str
    connects_to_id: str
    door_type: str = "SINGLE_SWING"
    width_m: float = 0.90
    is_bidirectional: bool = True
    is_emergency_exit: bool = False
    verification_status: str = "UNVERIFIED"
    associated_landmark_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class CreateVerticalTransitionRequest(BaseModel):
    transition_id: str = Field(..., min_length=1)
    transition_type: str = "STAIRCASE"
    name: str = Field(..., min_length=1)
    floor_positions: Dict[int, Point2DSchema]
    is_accessible: bool = False
    verification_status: str = "UNVERIFIED"
    associated_landmark_ids: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class CreateConnectionRequest(BaseModel):
    source_id: str
    destination_id: str
    source_floor_id: int
    destination_floor_id: int
    connection_type: str = "CORRIDOR"
    distance_m: Optional[float] = None
    is_bidirectional: bool = True
    accessibility: str = "WHEELCHAIR_ACCESSIBLE"
    verification_status: str = "UNVERIFIED"
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ImportWifiReferencesRequest(BaseModel):
    floor_id: int
    wifi_references: List[Dict[str, Any]]
    auto_associate_rooms: bool = True


class ImportLandmarksRequest(BaseModel):
    floor_id: int
    landmarks: List[Dict[str, Any]]
    auto_associate_doors: bool = True


class PublishVersionRequest(BaseModel):
    version_tag: Optional[str] = None


class MapValidationResponse(BaseModel):
    is_valid: bool
    error_count: int
    warning_count: int
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    summary: str


class MapMetricsResponse(BaseModel):
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
    verification_ratio: float
    accessibility_ratio: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BuildingMapSummaryResponse(BaseModel):
    building_id: str
    name: str
    map_version: str
    floor_count: int
    total_rooms_count: int
    total_doors_count: int
    updated_at: str
