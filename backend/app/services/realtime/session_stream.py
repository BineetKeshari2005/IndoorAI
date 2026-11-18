"""Per-session streaming state, buffer coordination, and lifecycle tracking (Task 29).

Maintains in-memory session contexts, deduplication caches, and attached localization /
navigation engines.
"""

from __future__ import annotations

from collections import deque
import threading
import time
from typing import Any, Dict, List, Optional, Set

from src.core.logging import setup_logger
from backend.app.services.realtime.event_buffer import (
    BackpressurePolicy,
    BoundedEventBuffer,
)
from backend.app.services.realtime.event_models import (
    EventProcessingStatus,
    LiveLocalizationUpdateMessage,
    LiveNavigationUpdateMessage,
    RealtimeEvent,
    RealtimeEventType,
    RejectionReason,
    StreamStatusMessage,
)
from backend.app.services.realtime.replay_manager import ReplayManager

logger = setup_logger("RealtimeSessionStream")

MAX_DEDUP_CACHE_SIZE = 10000


class SessionStream:
    """Encapsulates the real-time streaming pipeline state for a single session."""

    def __init__(
        self,
        session_id: str,
        building_id: str = "default_building",
        floor_id: int = 0,
        max_buffer_capacity: int = 5000,
        buffer_policy: BackpressurePolicy = BackpressurePolicy.DROP_OLDEST,
    ) -> None:
        self.session_id = session_id
        self.building_id = building_id
        self.floor_id = floor_id
        self.is_active = True
        self.created_at_epoch_ms = int(time.time() * 1000)
        self.last_event_timestamp_ms: Optional[int] = None

        self.buffer = BoundedEventBuffer(
            session_id=session_id,
            max_capacity=max_buffer_capacity,
            policy=buffer_policy,
        )
        self.replay = ReplayManager(session_id=session_id, max_history_size=1000)

        # Bounded deduplication tracking
        self._dedup_set: Set[str] = set()
        self._dedup_queue: deque[str] = deque()
        self._lock = threading.Lock()

        # Telemetry metrics
        self.events_received = 0
        self.events_processed = 0
        self.events_rejected = 0
        self.event_counts_by_type: Dict[str, int] = {t.value: 0 for t in RealtimeEventType}
        self.rejection_counts: Dict[str, int] = {}

        # Latest state estimates
        self.latest_localization: Optional[LiveLocalizationUpdateMessage] = None
        self.latest_navigation: Optional[LiveNavigationUpdateMessage] = None
        self.active_navigation_session_id: Optional[str] = None

        # Attached real-time algorithm state containers
        self.attached_building_map: Optional[Any] = None
        self.attached_radio_map: Optional[Any] = None
        self.attached_particle_filter: Optional[Any] = None
        self.attached_navigation_session: Optional[Any] = None
        self.last_pdr_timestamp_ms: Optional[int] = None
        self.cumulative_pdr_distance_m: float = 0.0

    def is_event_duplicate(self, event_id: str) -> bool:
        """Checks if event_id has been received recently."""
        with self._lock:
            return event_id in self._dedup_set

    def record_received_event(self, event: RealtimeEvent) -> None:
        """Records an incoming event into deduplication cache and metrics."""
        with self._lock:
            self.events_received += 1
            type_str = event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type)
            self.event_counts_by_type[type_str] = self.event_counts_by_type.get(type_str, 0) + 1
            self.last_event_timestamp_ms = event.server_receive_timestamp

            # Deduplication cache
            if event.event_id not in self._dedup_set:
                self._dedup_set.add(event.event_id)
                self._dedup_queue.append(event.event_id)
                if len(self._dedup_queue) > MAX_DEDUP_CACHE_SIZE:
                    evicted = self._dedup_queue.popleft()
                    self._dedup_set.discard(evicted)

    def record_rejection(self, reason: RejectionReason) -> None:
        with self._lock:
            self.events_rejected += 1
            self.rejection_counts[reason.value] = self.rejection_counts.get(reason.value, 0) + 1

    def record_processed(self) -> None:
        with self._lock:
            self.events_processed += 1

    def pause(self) -> None:
        with self._lock:
            self.is_active = False

    def resume(self) -> None:
        with self._lock:
            self.is_active = True

    def stop(self) -> None:
        with self._lock:
            self.is_active = False

    def get_status_message(self, connected_clients: int = 0) -> StreamStatusMessage:
        with self._lock:
            return StreamStatusMessage(
                session_id=self.session_id,
                is_active=self.is_active,
                connected_clients=connected_clients,
                events_received=self.events_received,
                events_processed=self.events_processed,
                events_rejected=self.events_rejected,
                events_dropped=self.buffer.total_dropped,
                last_event_timestamp_ms=self.last_event_timestamp_ms,
                active_navigation_session_id=self.active_navigation_session_id,
                building_id=self.building_id,
                floor_id=self.floor_id,
            )

    def get_detailed_stats(self, connected_clients: int = 0) -> Dict[str, Any]:
        with self._lock:
            return {
                "session_id": self.session_id,
                "building_id": self.building_id,
                "floor_id": self.floor_id,
                "is_active": self.is_active,
                "connected_clients": connected_clients,
                "events_received": self.events_received,
                "events_processed": self.events_processed,
                "events_rejected": self.events_rejected,
                "events_dropped": self.buffer.total_dropped,
                "last_event_timestamp_ms": self.last_event_timestamp_ms,
                "event_counts_by_type": dict(self.event_counts_by_type),
                "rejection_counts": dict(self.rejection_counts),
                "buffer": self.buffer.get_stats(),
                "replay": self.replay.get_history_summary(),
                "active_navigation_session_id": self.active_navigation_session_id,
                "latest_localization": self.latest_localization.to_dict() if self.latest_localization else None,
                "latest_navigation": self.latest_navigation.to_dict() if self.latest_navigation else None,
            }
