"""FastAPI Pydantic schemas for Indoor Navigation & Route Planning (Task 28)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DestinationResponse(BaseModel):
    """Single verified indoor navigation destination target."""
    destination_id: str
    building_id: str
    floor_id: int
    display_name: str
    category: str
    entity_id: str
    verified_coordinates_m: List[float]
    verification_status: str
    room_number: Optional[str] = None
    is_accessible: bool = True
    map_version: str = "1.0.0"
    aliases: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DestinationListResponse(BaseModel):
    """Collection of navigable destinations."""
    total_destinations: int
    building_id: str
    destinations: List[DestinationResponse]


class RoutePlanRequest(BaseModel):
    """Payload for requesting an optimal indoor route between two spaces."""
    building_id: str = Field("SYNTH_BLDG_01", description="Building identifier")
    origin_node_id: Optional[str] = Field(None, description="Starting topological node ID")
    origin_x_m: Optional[float] = Field(None, description="Starting metric X coordinate")
    origin_y_m: Optional[float] = Field(None, description="Starting metric Y coordinate")
    origin_floor_id: Optional[int] = Field(0, description="Starting floor integer ID")
    destination_id: Optional[str] = Field(None, description="Target destination ID or Room ID")
    destination_x_m: Optional[float] = Field(None, description="Target metric X coordinate")
    destination_y_m: Optional[float] = Field(None, description="Target metric Y coordinate")
    destination_floor_id: Optional[int] = Field(None, description="Target floor integer ID")
    algorithm: str = Field("ASTAR", description="Routing algorithm ('ASTAR' or 'DIJKSTRA')")
    preference: str = Field("SHORTEST", description="Route preference ('SHORTEST', 'FASTEST', 'WHEELCHAIR_ACCESSIBLE', 'AVOID_STAIRS', 'PREFER_ELEVATORS')")


class WaypointResponse(BaseModel):
    """A geometric point along an indoor walking route."""
    waypoint_id: str
    node_id: str
    floor_id: int
    x_m: float
    y_m: float
    label: str
    is_portal: bool = False
    is_floor_transition: bool = False
    transition_type: Optional[str] = None


class InstructionResponse(BaseModel):
    """Human-readable guidance directive."""
    instruction_id: str
    sequence_number: int
    instruction_type: str
    text: str
    waypoint: WaypointResponse
    floor_id: int
    target_heading_deg: Optional[float] = None
    distance_to_next_m: float = 0.0
    cumulative_distance_m: float = 0.0


class PlannedRouteResponse(BaseModel):
    """Complete planned and validated indoor route."""
    route_id: str
    building_id: str
    origin_name: str
    destination_name: str
    origin_floor_id: int
    destination_floor_id: int
    destination_id: str
    total_distance_m: float
    estimated_time_s: float
    waypoints: List[WaypointResponse]
    instructions: List[InstructionResponse]
    algorithm_used: str
    preference: str
    is_accessible: bool
    floors_traversed: List[int]
    map_version: str = "1.0.0"
    creation_timestamp_ms: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CreateSessionRequest(BaseModel):
    """Payload to instantiate an active navigation session."""
    route_id: Optional[str] = Field(None, description="ID of pre-computed route")
    # Alternatively provide inline routing params:
    building_id: Optional[str] = Field(None, description="Building ID")
    origin_node_id: Optional[str] = Field(None, description="Origin node")
    destination_id: Optional[str] = Field(None, description="Destination ID")
    preference: str = Field("SHORTEST", description="Routing preference")


class NavigationProgressRequest(BaseModel):
    """Payload delivering live pedestrian localization fixes to active session."""
    x_m: float = Field(..., description="Estimated X coordinate in meters")
    y_m: float = Field(..., description="Estimated Y coordinate in meters")
    floor_id: int = Field(0, description="Estimated floor index")
    localization_status: str = Field("TRACKING", description="Status from localization engine")
    timestamp_epoch_ms: Optional[int] = Field(None, description="Epoch timestamp in ms")


class NavigationProgressResponse(BaseModel):
    """Real-time progress response with waypoint index and possible dynamic reroute."""
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
    localization_status: str
    confidence_level: str
    status_message: str = "OK"
    updated_route: Optional[PlannedRouteResponse] = None


class NavigationSessionResponse(BaseModel):
    """Active or historic navigation trip session record."""
    session_id: str
    dataset_id: str
    building_id: str
    origin_name: str
    destination_id: str
    destination_name: str
    status: str
    active_route: PlannedRouteResponse
    current_progress: Optional[Dict[str, Any]] = None
    start_time_ms: int
    end_time_ms: Optional[int] = None
    reroute_count: int = 0
    reroute_history: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NavigationEvaluateRequest(BaseModel):
    """Payload to trigger comparative navigation routing benchmark."""
    building_id: str = Field("SYNTH_BLDG_01", description="Building to evaluate")
    use_synthetic_benchmark: bool = Field(True, description="Run on controlled multi-floor synthetic fixture")


class NavigationEvaluateResponse(BaseModel):
    """Comparative routing benchmark report."""
    evaluation_id: str
    building_id: str
    report: Dict[str, Any]
