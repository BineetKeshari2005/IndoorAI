"""Real-time pipeline orchestrator coordinating streaming, validation, and engines (Task 29).

Central integration hub connecting WebSocket and REST streaming clients to
in-memory buffers, validators, event processors, and downstream engines.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from src.core.logging import setup_logger
from backend.app.services.realtime.connection_manager import WebSocketConnectionManager
from backend.app.services.realtime.event_models import (
    EventProcessingStatus,
    RealtimeEvent,
    RealtimeEventType,
    RejectionReason,
    ServerAckMessage,
    StreamStatusMessage,
)
from backend.app.services.realtime.event_processor import EventProcessor
from backend.app.services.realtime.event_validator import EventValidator
from backend.app.services.realtime.session_stream import SessionStream

logger = setup_logger("RealtimePipeline")


class RealtimePipeline:
    """Singleton service orchestrating real-time streaming sessions."""

    _instance: Optional[RealtimePipeline] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs) -> RealtimePipeline:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self.validator = EventValidator()
        self.processor = EventProcessor()
        self.connections = WebSocketConnectionManager()
        self._streams: Dict[str, SessionStream] = {}
        self._streams_lock = threading.Lock()
        self._initialized = True
        logger.info("RealtimePipeline initialized successfully.")

    def get_or_create_stream(
        self,
        session_id: str,
        building_id: str = "default_building",
        floor_id: int = 0,
        max_buffer_capacity: int = 5000,
    ) -> SessionStream:
        """Retrieves or creates an in-memory SessionStream for a session."""
        with self._streams_lock:
            if session_id not in self._streams:
                self._streams[session_id] = SessionStream(
                    session_id=session_id,
                    building_id=building_id,
                    floor_id=floor_id,
                    max_buffer_capacity=max_buffer_capacity,
                )
                logger.info("Created new SessionStream for session: %s", session_id)
            return self._streams[session_id]

    def get_stream(self, session_id: str) -> Optional[SessionStream]:
        with self._streams_lock:
            return self._streams.get(session_id)

    def list_active_sessions(self) -> List[str]:
        with self._streams_lock:
            return [sid for sid, s in self._streams.items() if s.is_active]

    def ingest_event(
        self,
        event: RealtimeEvent,
    ) -> Tuple[ServerAckMessage, List[Dict[str, Any]]]:
        """Ingests, validates, buffers, and executes an individual event.

        Returns:
            Tuple of (ServerAckMessage, list_of_outgoing_updates: List[Dict]).
        """
        stream = self.get_or_create_stream(event.session_id)
        now_ms = int(time.time() * 1000)
        event.server_receive_timestamp = now_ms

        # 1. Deduplication check
        if stream.is_event_duplicate(event.event_id):
            ack = ServerAckMessage(
                event_id=event.event_id,
                session_id=event.session_id,
                status=EventProcessingStatus.DUPLICATE,
                server_receive_timestamp=now_ms,
                rejection_reason=RejectionReason.DUPLICATE_EVENT_ID,
                message=f"Event {event.event_id} already received (idempotent ignore).",
            )
            return ack, []

        # 2. Validation
        is_valid, reason, msg = self.validator.validate_event(
            event=event,
            active_session_ids=None if stream.is_active else {s for s, st in self._streams.items() if st.is_active},
        )
        if not is_valid:
            stream.record_rejection(reason)
            ack = ServerAckMessage(
                event_id=event.event_id,
                session_id=event.session_id,
                status=EventProcessingStatus.REJECTED,
                server_receive_timestamp=now_ms,
                rejection_reason=reason,
                message=msg,
            )
            return ack, []

        # 3. Buffer queuing & backpressure check
        queued, drop_reason, queue_msg = stream.buffer.push(event)
        if not queued:
            stream.record_rejection(drop_reason)
            ack = ServerAckMessage(
                event_id=event.event_id,
                session_id=event.session_id,
                status=EventProcessingStatus.DROPPED,
                server_receive_timestamp=now_ms,
                rejection_reason=drop_reason,
                message=queue_msg,
            )
            return ack, []

        # Record receipt for deduplication
        stream.record_received_event(event)

        # 4. Immediate synchronous processing of buffered event
        outgoing_updates = self.processor.process_event(event, stream)

        # Build positive ACK
        ack = ServerAckMessage(
            event_id=event.event_id,
            session_id=event.session_id,
            status=EventProcessingStatus.ACCEPTED,
            server_receive_timestamp=now_ms,
            rejection_reason=RejectionReason.NONE,
            message="Event accepted and processed.",
            sequence_number=event.sequence_number,
        )
        return ack, outgoing_updates

    def ingest_batch(
        self,
        session_id: str,
        events: List[RealtimeEvent],
    ) -> List[ServerAckMessage]:
        """Ingests a batch of events via REST endpoint.

        Returns:
            List of ServerAckMessage objects corresponding to each event.
        """
        acks: List[ServerAckMessage] = []
        for ev in events:
            ev.session_id = session_id
            ack, _ = self.ingest_event(ev)
            acks.append(ack)
        return acks

    async def handle_websocket_connection(
        self,
        websocket: WebSocket,
        session_id: str,
        last_ack_event_id: Optional[str] = None,
    ) -> None:
        """Handles the complete bidirectional WebSocket event loop for a streaming client."""
        stream = self.get_or_create_stream(session_id)
        await self.connections.connect(session_id, websocket)

        try:
            # Reconnection replay
            if last_ack_event_id:
                replay_items, is_gap = stream.replay.get_replay_since(last_ack_event_id=last_ack_event_id)
                if replay_items:
                    await websocket.send_json({
                        "type": "REPLAY_START",
                        "session_id": session_id,
                        "item_count": len(replay_items),
                        "is_gap": is_gap,
                    })
                    for item in replay_items:
                        for update in item.get("updates", []):
                            await websocket.send_json(update)
                    await websocket.send_json({"type": "REPLAY_COMPLETE", "session_id": session_id})

            # Send initial stream status
            status_msg = stream.get_status_message(connected_clients=self.connections.get_client_count(session_id))
            await websocket.send_json(status_msg.to_dict())

            # Event ingestion loop
            while True:
                data = await websocket.receive_json()
                if not isinstance(data, dict):
                    await websocket.send_json({
                        "type": "ERROR",
                        "message": "Invalid message: must be JSON object.",
                    })
                    continue

                msg_type = data.get("type")

                # Handle ping/heartbeat probe
                if msg_type == "PING":
                    await websocket.send_json({"type": "PONG", "timestamp_ms": int(time.time() * 1000)})
                    continue

                # Parse into RealtimeEvent
                event = RealtimeEvent.from_dict(data)
                event.session_id = session_id

                ack, updates = self.ingest_event(event)
                await websocket.send_json(ack.to_dict())

                # Broadcast generated updates to all connected subscribers in the session
                for update in updates:
                    await self.connections.broadcast_to_session(session_id, update)

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected gracefully from session %s", session_id)
        except Exception as e:
            logger.error("Error in WebSocket session loop %s: %s", session_id, e)
        finally:
            await self.connections.disconnect(session_id, websocket)

    def attach_navigation_session(
        self,
        session_id: str,
        nav_session_manager: Any,
        nav_session_id: str,
    ) -> bool:
        """Links an active Task 28 navigation session to this streaming session."""
        stream = self.get_or_create_stream(session_id)
        stream.attached_navigation_session = nav_session_manager
        stream.active_navigation_session_id = nav_session_id
        logger.info("Attached navigation session %s to streaming session %s", nav_session_id, session_id)
        return True

    def attach_particle_filter(
        self,
        session_id: str,
        particle_filter: Any,
    ) -> bool:
        """Links a Task 27 Map-Constrained Particle Filter to this streaming session."""
        stream = self.get_or_create_stream(session_id)
        stream.attached_particle_filter = particle_filter
        return True

    def attach_radio_map(
        self,
        session_id: str,
        radio_map: Any,
    ) -> bool:
        """Links a Task 23 RadioMap to this streaming session."""
        stream = self.get_or_create_stream(session_id)
        stream.attached_radio_map = radio_map
        return True

    def attach_building_map(
        self,
        session_id: str,
        building_map: Any,
    ) -> bool:
        """Links a Task 26 BuildingMap to this streaming session."""
        stream = self.get_or_create_stream(session_id)
        stream.attached_building_map = building_map
        stream.building_id = building_map.building_id
        return True

    def inject_synthetic_step(
        self,
        session_id: str,
        x_m: float,
        y_m: float,
        floor_id: int,
        heading_rad: float = 0.0,
        uncertainty_radius_m: float = 2.0,
    ) -> Tuple[ServerAckMessage, List[Dict[str, Any]]]:
        """Injects a synthetic simulation step for development and automated testing."""
        import uuid
        ev = RealtimeEvent(
            event_id=f"SIM-{uuid.uuid4().hex[:8]}",
            session_id=session_id,
            event_type=RealtimeEventType.IMU_BATCH,
            device_id="SIMULATOR",
            payload={
                "samples": [{"sensor_type": "ACCELEROMETER", "x": 0.0, "y": 0.0, "z": 9.8}],
                "step": {
                    "step_length_m": 0.75,
                    "heading_rad": heading_rad,
                },
                "x_m": x_m,
                "y_m": y_m,
                "floor_id": floor_id,
                "uncertainty_radius_m": uncertainty_radius_m,
                "is_simulation": True,
            },
        )
        return self.ingest_event(ev)


# Global singleton pipeline accessor
def get_realtime_pipeline() -> RealtimePipeline:
    return RealtimePipeline()
