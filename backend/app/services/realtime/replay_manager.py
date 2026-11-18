"""Replay and reconnection recovery manager (Task 29).

Maintains a bounded historical sequence of processed events and outgoing state updates
to allow disconnected clients to resume seamlessly upon reconnection.
"""

from __future__ import annotations

from collections import deque
import threading
from typing import Any, Dict, List, Optional, Tuple

from src.core.logging import setup_logger
from backend.app.services.realtime.event_models import RealtimeEvent

logger = setup_logger("RealtimeReplayManager")


class ReplayManager:
    """Manages replay log for reconnection recovery per session."""

    def __init__(self, session_id: str, max_history_size: int = 1000) -> None:
        self.session_id = session_id
        self.max_history_size = max(50, max_history_size)
        self._history: deque[Dict[str, Any]] = deque()
        self._event_id_to_seq: Dict[str, int] = {}
        self._next_seq: int = 1
        self._lock = threading.Lock()

    @property
    def next_sequence_number(self) -> int:
        with self._lock:
            return self._next_seq

    def record_processed_item(
        self,
        event: RealtimeEvent,
        outgoing_updates: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Records an ingested event and its associated downstream state updates.

        Returns:
            The assigned monotonically increasing sequence number.
        """
        with self._lock:
            seq = self._next_seq
            self._next_seq += 1
            event.sequence_number = seq

            record = {
                "sequence_number": seq,
                "event_id": event.event_id,
                "event_type": event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type),
                "timestamp_ms": event.server_receive_timestamp,
                "event": event.to_dict(),
                "updates": outgoing_updates or [],
            }

            if len(self._history) >= self.max_history_size:
                evicted = self._history.popleft()
                evicted_id = evicted.get("event_id")
                if evicted_id in self._event_id_to_seq:
                    del self._event_id_to_seq[evicted_id]

            self._history.append(record)
            self._event_id_to_seq[event.event_id] = seq
            return seq

    def get_replay_since(
        self,
        last_ack_event_id: Optional[str] = None,
        last_ack_seq: Optional[int] = None,
        max_items: int = 500,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Retrieves history of updates since the given event ID or sequence number.

        Args:
            last_ack_event_id: Event ID client last acknowledged.
            last_ack_seq: Sequence number client last acknowledged.
            max_items: Maximum items to return in replay batch.

        Returns:
            Tuple of (replay_items: List[Dict], is_complete_gap: bool).
            is_complete_gap is True if client's last ACK was evicted and history has a gap.
        """
        with self._lock:
            target_seq = 0
            if last_ack_seq is not None and last_ack_seq > 0:
                target_seq = last_ack_seq
            elif last_ack_event_id and last_ack_event_id in self._event_id_to_seq:
                target_seq = self._event_id_to_seq[last_ack_event_id]
            elif last_ack_event_id:
                # Event ID was evicted or never seen:
                target_seq = -1

            if not self._history:
                return [], False

            earliest_seq = self._history[0]["sequence_number"]
            is_gap = False
            if target_seq > 0 and target_seq < earliest_seq:
                is_gap = True

            replay: List[Dict[str, Any]] = []
            for record in self._history:
                if record["sequence_number"] > target_seq:
                    replay.append(record)
                    if len(replay) >= max_items:
                        break

            return replay, is_gap

    def get_history_summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "session_id": self.session_id,
                "recorded_items": len(self._history),
                "earliest_seq": self._history[0]["sequence_number"] if self._history else None,
                "latest_seq": self._history[-1]["sequence_number"] if self._history else None,
                "max_history_size": self.max_history_size,
            }
