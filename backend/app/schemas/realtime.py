"""FastAPI Pydantic schemas for real-time multimodal streaming API (Task 29)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RealtimeEventSchema(BaseModel):
    """Schema for an individual incoming typed real-time event."""
    event_id: Optional[str] = Field(default=None, description="Unique client UUID for event idempotency")
    session_id: Optional[str] = Field(default=None, description="Target capture or navigation session ID")
    event_type: str = Field(..., description="Observation or lifecycle event type (e.g. WIFI_SCAN, IMU_BATCH)")
    device_id: Optional[str] = Field(default="unknown_device", description="Pseudonymous client device identifier")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Typed payload dictionary")
    client_timestamp_monotonic: Optional[int] = Field(default=None, description="Client monotonic clock value")
    client_timestamp_utc: Optional[int] = Field(default=None, description="Client UTC epoch milliseconds")
    payload_schema_version: Optional[str] = Field(default="1.0.0", description="Schema version of event payload")


class BatchEventsRequest(BaseModel):
    """Schema for batch ingestion of events via REST fallback."""
    events: List[RealtimeEventSchema] = Field(..., description="Ordered list of events to ingest")


class ServerAckResponse(BaseModel):
    """Per-event acknowledgement returned to client."""
    type: str = "ACK"
    event_id: str
    session_id: str
    status: str
    server_receive_timestamp: int
    rejection_reason: str
    message: str
    sequence_number: Optional[int] = None


class BatchAckResponse(BaseModel):
    """Summary and itemized acknowledgements for batch event submission."""
    session_id: str
    total_events: int
    accepted: int
    rejected: int
    dropped: int
    duplicates: int
    acknowledgements: List[ServerAckResponse]


class StreamStatusResponse(BaseModel):
    """Real-time streaming metrics and connection status."""
    session_id: str
    is_active: bool
    connected_clients: int
    events_received: int
    events_processed: int
    events_rejected: int
    events_dropped: int
    last_event_timestamp_ms: Optional[int] = None
    active_navigation_session_id: Optional[str] = None
    building_id: str = "default_building"
    floor_id: int = 0
    buffer_size: int = 0
    buffer_capacity: int = 5000
    buffer_utilization_pct: float = 0.0


class SessionControlRequest(BaseModel):
    """Lifecycle control action for a streaming session."""
    action: str = Field(..., description="Lifecycle action: START, PAUSE, RESUME, STOP, CANCEL, RESET")
    note: Optional[str] = None
    building_id: Optional[str] = None
    floor_id: Optional[int] = None


class SimulateStepRequest(BaseModel):
    """Development and test simulation step injection."""
    x_m: float
    y_m: float
    floor_id: int
    heading_rad: Optional[float] = 0.0
    uncertainty_radius_m: Optional[float] = 2.0


class AttachNavigationRequest(BaseModel):
    """Request linking an active Task 28 navigation session to a streaming session."""
    navigation_session_id: str
