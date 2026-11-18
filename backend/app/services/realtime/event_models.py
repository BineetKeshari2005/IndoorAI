"""Typed event models and schemas for real-time multimodal ingestion (Task 29).

Defines typed observation events, server acknowledgements, outgoing localization,
and navigation updates for live WebSocket and REST streaming.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Dict, List, Optional
import uuid


class RealtimeEventType(str, Enum):
    """Supported real-time observation and lifecycle event types."""
    WIFI_SCAN = "WIFI_SCAN"
    IMU_BATCH = "IMU_BATCH"
    ORIENTATION_UPDATE = "ORIENTATION_UPDATE"
    PHOTO_CAPTURE = "PHOTO_CAPTURE"
    LANDMARK_OBSERVATION = "LANDMARK_OBSERVATION"
    ANNOTATION = "ANNOTATION"
    HEARTBEAT = "HEARTBEAT"
    SESSION_STATUS = "SESSION_STATUS"


class StreamConnectionStatus(str, Enum):
    """Lifecycle status of a streaming client connection."""
    CONNECTED = "CONNECTED"
    DISCONNECTED = "DISCONNECTED"
    RECONNECTING = "RECONNECTING"
    CLOSED = "CLOSED"


class EventProcessingStatus(str, Enum):
    """Ingestion status for an individual event."""
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    DROPPED = "DROPPED"
    DUPLICATE = "DUPLICATE"


class RejectionReason(str, Enum):
    """Explicit taxonomy of event rejection and drop reasons."""
    NONE = "NONE"
    DUPLICATE_EVENT_ID = "DUPLICATE_EVENT_ID"
    SESSION_INACTIVE = "SESSION_INACTIVE"
    MALFORMED_PAYLOAD = "MALFORMED_PAYLOAD"
    TIMESTAMP_ANOMALY = "TIMESTAMP_ANOMALY"
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"
    NAN_OR_INF_DETECTED = "NAN_OR_INF_DETECTED"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    BACKPRESSURE_DROP = "BACKPRESSURE_DROP"
    RATE_LIMITED = "RATE_LIMITED"
    UNSUPPORTED_SCHEMA_VERSION = "UNSUPPORTED_SCHEMA_VERSION"
    UNAUTHORIZED = "UNAUTHORIZED"
    UNKNOWN_EVENT_TYPE = "UNKNOWN_EVENT_TYPE"


@dataclass
class RealtimeEvent:
    """Canonical server representation of an incoming typed real-time event."""
    event_id: str
    session_id: str
    event_type: RealtimeEventType
    device_id: str
    payload: Dict[str, Any]
    client_timestamp_monotonic: Optional[int] = None
    client_timestamp_utc: Optional[int] = None
    server_receive_timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    payload_schema_version: str = "1.0.0"
    sequence_number: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to serialized JSON-compatible dictionary."""
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "event_type": self.event_type.value if isinstance(self.event_type, RealtimeEventType) else str(self.event_type),
            "device_id": self.device_id,
            "client_timestamp_monotonic": self.client_timestamp_monotonic,
            "client_timestamp_utc": self.client_timestamp_utc,
            "server_receive_timestamp": self.server_receive_timestamp,
            "payload_schema_version": self.payload_schema_version,
            "sequence_number": self.sequence_number,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RealtimeEvent:
        """Instantiate RealtimeEvent from dictionary."""
        raw_type = data.get("event_type", "HEARTBEAT")
        try:
            event_type = RealtimeEventType(raw_type)
        except ValueError:
            event_type = RealtimeEventType.HEARTBEAT

        return cls(
            event_id=str(data.get("event_id") or str(uuid.uuid4())),
            session_id=str(data.get("session_id", "default")),
            event_type=event_type,
            device_id=str(data.get("device_id", "unknown_device")),
            client_timestamp_monotonic=data.get("client_timestamp_monotonic"),
            client_timestamp_utc=data.get("client_timestamp_utc"),
            server_receive_timestamp=int(data.get("server_receive_timestamp") or int(time.time() * 1000)),
            payload_schema_version=str(data.get("payload_schema_version", "1.0.0")),
            sequence_number=data.get("sequence_number"),
            payload=dict(data.get("payload", {})),
        )


@dataclass
class ServerAckMessage:
    """Immediate acknowledgement returned to client upon event receipt."""
    event_id: str
    session_id: str
    status: EventProcessingStatus
    server_receive_timestamp: int
    rejection_reason: RejectionReason = RejectionReason.NONE
    message: str = ""
    sequence_number: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "ACK",
            "event_id": self.event_id,
            "session_id": self.session_id,
            "status": self.status.value,
            "server_receive_timestamp": self.server_receive_timestamp,
            "rejection_reason": self.rejection_reason.value,
            "message": self.message,
            "sequence_number": self.sequence_number,
        }


@dataclass
class LiveLocalizationUpdateMessage:
    """Real-time 2D position and uncertainty published to connected streaming clients."""
    session_id: str
    timestamp_ms: int
    x_m: float
    y_m: float
    floor_id: int
    heading_rad: float
    uncertainty_radius_m: float
    confidence_level: str
    source: str
    source_event_id: Optional[str] = None
    building_id: str = "default_building"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "LOCALIZATION_UPDATE",
            "session_id": self.session_id,
            "building_id": self.building_id,
            "timestamp_ms": self.timestamp_ms,
            "x_m": round(self.x_m, 3),
            "y_m": round(self.y_m, 3),
            "floor_id": self.floor_id,
            "heading_rad": round(self.heading_rad, 4),
            "uncertainty_radius_m": round(self.uncertainty_radius_m, 2),
            "confidence_level": self.confidence_level,
            "source": self.source,
            "source_event_id": self.source_event_id,
        }


@dataclass
class LiveNavigationUpdateMessage:
    """Real-time turn-by-turn guidance and progress published to streaming clients."""
    session_id: str
    navigation_session_id: str
    timestamp_ms: int
    current_instruction: str
    distance_to_next_waypoint_m: float
    remaining_distance_m: float
    cross_track_error_m: float
    arrival_status: bool
    reroute_occurred: bool
    route_status: str
    target_destination: str = ""
    target_floor_id: int = 0
    instruction_type: str = "CONTINUE"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "NAVIGATION_UPDATE",
            "session_id": self.session_id,
            "navigation_session_id": self.navigation_session_id,
            "timestamp_ms": self.timestamp_ms,
            "current_instruction": self.current_instruction,
            "instruction_type": self.instruction_type,
            "distance_to_next_waypoint_m": round(self.distance_to_next_waypoint_m, 2),
            "remaining_distance_m": round(self.remaining_distance_m, 2),
            "cross_track_error_m": round(self.cross_track_error_m, 2),
            "arrival_status": self.arrival_status,
            "reroute_occurred": self.reroute_occurred,
            "route_status": self.route_status,
            "target_destination": self.target_destination,
            "target_floor_id": self.target_floor_id,
        }


@dataclass
class StreamStatusMessage:
    """Status summary describing connection and streaming metrics."""
    session_id: str
    is_active: bool
    connected_clients: int
    events_received: int
    events_processed: int
    events_rejected: int
    events_dropped: int
    last_event_timestamp_ms: Optional[int]
    active_navigation_session_id: Optional[str] = None
    building_id: str = "default_building"
    floor_id: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "STREAM_STATUS",
            "session_id": self.session_id,
            "is_active": self.is_active,
            "connected_clients": self.connected_clients,
            "events_received": self.events_received,
            "events_processed": self.events_processed,
            "events_rejected": self.events_rejected,
            "events_dropped": self.events_dropped,
            "last_event_timestamp_ms": self.last_event_timestamp_ms,
            "active_navigation_session_id": self.active_navigation_session_id,
            "building_id": self.building_id,
            "floor_id": self.floor_id,
        }
