"""Real-time multimodal ingestion and streaming package (Task 29)."""

from backend.app.services.realtime.connection_manager import WebSocketConnectionManager
from backend.app.services.realtime.event_buffer import BackpressurePolicy, BoundedEventBuffer
from backend.app.services.realtime.event_models import (
    EventProcessingStatus,
    LiveLocalizationUpdateMessage,
    LiveNavigationUpdateMessage,
    RealtimeEvent,
    RealtimeEventType,
    RejectionReason,
    ServerAckMessage,
    StreamConnectionStatus,
    StreamStatusMessage,
)
from backend.app.services.realtime.event_processor import EventProcessor
from backend.app.services.realtime.event_validator import EventValidator
from backend.app.services.realtime.realtime_pipeline import RealtimePipeline, get_realtime_pipeline
from backend.app.services.realtime.replay_manager import ReplayManager
from backend.app.services.realtime.session_stream import SessionStream

__all__ = [
    "RealtimeEventType",
    "StreamConnectionStatus",
    "EventProcessingStatus",
    "RejectionReason",
    "RealtimeEvent",
    "ServerAckMessage",
    "LiveLocalizationUpdateMessage",
    "LiveNavigationUpdateMessage",
    "StreamStatusMessage",
    "EventValidator",
    "BackpressurePolicy",
    "BoundedEventBuffer",
    "ReplayManager",
    "WebSocketConnectionManager",
    "SessionStream",
    "EventProcessor",
    "RealtimePipeline",
    "get_realtime_pipeline",
]
