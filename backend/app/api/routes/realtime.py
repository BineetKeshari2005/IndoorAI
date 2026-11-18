"""Real-Time Ingestion and Multimodal Streaming API Routes (Task 29).

Implements WebSocket bidirectional streaming and REST fallback endpoints for live
sensor, Wi-Fi, landmark observations, and navigation progress.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, status

from src.core.logging import setup_logger
from backend.app.schemas.realtime import (
    AttachNavigationRequest,
    BatchAckResponse,
    BatchEventsRequest,
    RealtimeEventSchema,
    ServerAckResponse,
    SessionControlRequest,
    SimulateStepRequest,
    StreamStatusResponse,
)
from backend.app.services.navigation_service import get_navigation_service
from backend.app.services.realtime.event_models import (
    EventProcessingStatus,
    RealtimeEvent,
    RealtimeEventType,
    RejectionReason,
)
from backend.app.services.realtime.realtime_pipeline import (
    RealtimePipeline,
    get_realtime_pipeline,
)

logger = setup_logger("RealtimeRoutes")

router = APIRouter(prefix="/realtime", tags=["realtime"])


@router.websocket("/sessions/{session_id}/stream")
async def websocket_session_stream(
    websocket: WebSocket,
    session_id: str,
    last_ack_event_id: Optional[str] = Query(None, description="Last event ID successfully acknowledged by client"),
    pipeline: RealtimePipeline = Depends(get_realtime_pipeline),
):
    """Bidirectional WebSocket streaming interface for real-time sensor, Wi-Fi, and navigation data.

    Supports live event ingestion, server acks, reconnection recovery via last_ack_event_id,
    and asynchronous broadcasts of localization and navigation updates.
    """
    await pipeline.handle_websocket_connection(
        websocket=websocket,
        session_id=session_id,
        last_ack_event_id=last_ack_event_id,
    )


@router.post(
    "/sessions/{session_id}/events/batch",
    response_model=BatchAckResponse,
    summary="Batch Ingestion for HTTP Clients",
)
def ingest_events_batch(
    session_id: str,
    request: BatchEventsRequest,
    pipeline: RealtimePipeline = Depends(get_realtime_pipeline),
) -> BatchAckResponse:
    """Ingests a batch of observation events via REST POST for clients unable to maintain WebSockets."""
    domain_events: List[RealtimeEvent] = []
    for s in request.events:
        domain_events.append(
            RealtimeEvent(
                event_id=s.event_id or str(uuid.uuid4()),
                session_id=session_id,
                event_type=RealtimeEventType(s.event_type) if s.event_type in RealtimeEventType.__members__ else RealtimeEventType.HEARTBEAT,
                device_id=s.device_id or "unknown_device",
                client_timestamp_monotonic=s.client_timestamp_monotonic,
                client_timestamp_utc=s.client_timestamp_utc,
                server_receive_timestamp=int(time.time() * 1000),
                payload_schema_version=s.payload_schema_version or "1.0.0",
                payload=s.payload,
            )
        )

    acks = pipeline.ingest_batch(session_id=session_id, events=domain_events)

    accepted = sum(1 for a in acks if a.status == EventProcessingStatus.ACCEPTED)
    rejected = sum(1 for a in acks if a.status == EventProcessingStatus.REJECTED)
    dropped = sum(1 for a in acks if a.status == EventProcessingStatus.DROPPED)
    duplicates = sum(1 for a in acks if a.status == EventProcessingStatus.DUPLICATE)

    ack_responses = [
        ServerAckResponse(
            type="ACK",
            event_id=a.event_id,
            session_id=a.session_id,
            status=a.status.value,
            server_receive_timestamp=a.server_receive_timestamp,
            rejection_reason=a.rejection_reason.value,
            message=a.message,
            sequence_number=a.sequence_number,
        )
        for a in acks
    ]

    return BatchAckResponse(
        session_id=session_id,
        total_events=len(acks),
        accepted=accepted,
        rejected=rejected,
        dropped=dropped,
        duplicates=duplicates,
        acknowledgements=ack_responses,
    )


@router.get(
    "/sessions/{session_id}/status",
    response_model=StreamStatusResponse,
    summary="Get Stream Metrics & Status",
)
def get_stream_status(
    session_id: str,
    pipeline: RealtimePipeline = Depends(get_realtime_pipeline),
) -> StreamStatusResponse:
    """Returns streaming throughput metrics, buffer utilization, and connection state for a session."""
    stream = pipeline.get_or_create_stream(session_id)
    connected = pipeline.connections.get_client_count(session_id)
    msg = stream.get_status_message(connected_clients=connected)
    buf_stats = stream.buffer.get_stats()

    return StreamStatusResponse(
        session_id=msg.session_id,
        is_active=msg.is_active,
        connected_clients=msg.connected_clients,
        events_received=msg.events_received,
        events_processed=msg.events_processed,
        events_rejected=msg.events_rejected,
        events_dropped=msg.events_dropped,
        last_event_timestamp_ms=msg.last_event_timestamp_ms,
        active_navigation_session_id=msg.active_navigation_session_id,
        building_id=msg.building_id,
        floor_id=msg.floor_id,
        buffer_size=buf_stats["current_size"],
        buffer_capacity=buf_stats["max_capacity"],
        buffer_utilization_pct=buf_stats["utilization_pct"],
    )


@router.post(
    "/sessions/{session_id}/control",
    summary="Lifecycle Control Actions",
)
def control_session_stream(
    session_id: str,
    request: SessionControlRequest,
    pipeline: RealtimePipeline = Depends(get_realtime_pipeline),
) -> Dict[str, Any]:
    """Applies lifecycle actions (START, PAUSE, RESUME, STOP, CANCEL, RESET) to a streaming session."""
    stream = pipeline.get_or_create_stream(session_id)
    act = request.action.upper()

    if act == "PAUSE":
        stream.pause()
    elif act == "RESUME":
        stream.resume()
    elif act in ("STOP", "CANCEL"):
        stream.stop()
    elif act == "START":
        stream.is_active = True
        if request.building_id:
            stream.building_id = request.building_id
        if request.floor_id is not None:
            stream.floor_id = request.floor_id
    elif act == "RESET":
        stream.buffer.clear()
        stream.is_active = True
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported lifecycle action '{request.action}'.",
        )

    return {
        "session_id": session_id,
        "action": act,
        "is_active": stream.is_active,
        "note": request.note,
        "timestamp_ms": int(time.time() * 1000),
    }


@router.get(
    "/sessions/{session_id}/history",
    summary="Get Historical Events / Replay Slice",
)
def get_stream_history(
    session_id: str,
    last_ack_event_id: Optional[str] = Query(None, description="Fetch items strictly after this event ID"),
    last_ack_seq: Optional[int] = Query(None, description="Fetch items strictly after this sequence number"),
    max_items: int = Query(100, ge=1, le=500),
    pipeline: RealtimePipeline = Depends(get_realtime_pipeline),
) -> Dict[str, Any]:
    """Retrieves buffered replay history for synchronization or reconnection inspection."""
    stream = pipeline.get_or_create_stream(session_id)
    items, is_gap = stream.replay.get_replay_since(
        last_ack_event_id=last_ack_event_id,
        last_ack_seq=last_ack_seq,
        max_items=max_items,
    )
    return {
        "session_id": session_id,
        "item_count": len(items),
        "is_gap": is_gap,
        "items": items,
    }


@router.post(
    "/sessions/{session_id}/simulate",
    summary="Inject Synthetic Simulation Step",
)
def simulate_step(
    session_id: str,
    request: SimulateStepRequest,
    pipeline: RealtimePipeline = Depends(get_realtime_pipeline),
) -> Dict[str, Any]:
    """Injects a synthetic simulation step into the session pipeline for development and testing."""
    ack, updates = pipeline.inject_synthetic_step(
        session_id=session_id,
        x_m=request.x_m,
        y_m=request.y_m,
        floor_id=request.floor_id,
        heading_rad=request.heading_rad or 0.0,
        uncertainty_radius_m=request.uncertainty_radius_m or 2.0,
    )
    return {
        "ack": ack.to_dict(),
        "updates": updates,
    }


@router.post(
    "/sessions/{session_id}/attach-navigation",
    summary="Attach Task 28 Navigation Session",
)
def attach_navigation_session(
    session_id: str,
    request: AttachNavigationRequest,
    pipeline: RealtimePipeline = Depends(get_realtime_pipeline),
) -> Dict[str, Any]:
    """Links an active Task 28 navigation session to stream so location fixes drive turn-by-turn HUD updates."""
    nav_service = get_navigation_service()
    session_obj = nav_service.get_session(dataset_id="default", session_id=request.navigation_session_id)
    session_mgr = None
    if session_obj:
        pip = nav_service.get_pipeline(building_id=session_obj.building_id, dataset_id="default")
        session_mgr = pip.sessions

    success = pipeline.attach_navigation_session(
        session_id=session_id,
        nav_session_manager=session_mgr,
        nav_session_id=request.navigation_session_id,
    )
    return {
        "session_id": session_id,
        "attached_navigation_session_id": request.navigation_session_id,
        "success": success,
    }


@router.get(
    "/sessions",
    summary="List Active Streaming Sessions",
)
def list_active_streaming_sessions(
    pipeline: RealtimePipeline = Depends(get_realtime_pipeline),
) -> Dict[str, Any]:
    """Returns all active streaming session IDs."""
    sessions = pipeline.list_active_sessions()
    return {
        "count": len(sessions),
        "active_session_ids": sessions,
    }
