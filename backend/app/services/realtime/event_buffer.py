"""Bounded in-memory event buffer with backpressure management (Task 29).

Provides thread-safe FIFO queuing, drop policy enforcement, and memory monitoring.
"""

from __future__ import annotations

from collections import deque
from enum import Enum
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from src.core.logging import setup_logger
from backend.app.services.realtime.event_models import (
    RealtimeEvent,
    RejectionReason,
)

logger = setup_logger("RealtimeEventBuffer")


class BackpressurePolicy(str, Enum):
    """Action to take when buffer reaches maximum capacity."""
    DROP_OLDEST = "DROP_OLDEST"
    REJECT_NEW = "REJECT_NEW"


class BoundedEventBuffer:
    """Thread-safe bounded FIFO buffer with backpressure policies."""

    def __init__(
        self,
        session_id: str,
        max_capacity: int = 5000,
        policy: BackpressurePolicy = BackpressurePolicy.DROP_OLDEST,
    ) -> None:
        self.session_id = session_id
        self.max_capacity = max(1, max_capacity)
        self.policy = policy
        self._queue: deque[RealtimeEvent] = deque()
        self._lock = threading.Lock()
        self._total_pushed: int = 0
        self._total_dropped: int = 0
        self._drop_reasons: Dict[str, int] = {}

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._queue)

    @property
    def is_full(self) -> bool:
        with self._lock:
            return len(self._queue) >= self.max_capacity

    @property
    def total_pushed(self) -> int:
        with self._lock:
            return self._total_pushed

    @property
    def total_dropped(self) -> int:
        with self._lock:
            return self._total_dropped

    def push(self, event: RealtimeEvent) -> Tuple[bool, RejectionReason, str]:
        """Pushes an event into the bounded buffer applying backpressure policy.

        Returns:
            Tuple of (accepted: bool, rejection_reason: RejectionReason, message: str).
        """
        with self._lock:
            if len(self._queue) >= self.max_capacity:
                if self.policy == BackpressurePolicy.REJECT_NEW:
                    self._total_dropped += 1
                    self._drop_reasons[RejectionReason.BACKPRESSURE_DROP.value] = (
                        self._drop_reasons.get(RejectionReason.BACKPRESSURE_DROP.value, 0) + 1
                    )
                    return (
                        False,
                        RejectionReason.BACKPRESSURE_DROP,
                        f"Buffer for session '{self.session_id}' reached capacity {self.max_capacity}. New event rejected.",
                    )
                else:  # DROP_OLDEST
                    evicted = self._queue.popleft()
                    self._total_dropped += 1
                    self._drop_reasons[RejectionReason.BACKPRESSURE_DROP.value] = (
                        self._drop_reasons.get(RejectionReason.BACKPRESSURE_DROP.value, 0) + 1
                    )
                    logger.debug(
                        "Session %s buffer full: dropped oldest event %s",
                        self.session_id,
                        evicted.event_id,
                    )

            self._queue.append(event)
            self._total_pushed += 1
            return True, RejectionReason.NONE, "Queued successfully."

    def pop(self) -> Optional[RealtimeEvent]:
        """Pops and returns the oldest event from the buffer, or None if empty."""
        with self._lock:
            if not self._queue:
                return None
            return self._queue.popleft()

    def pop_batch(self, max_items: int = 50) -> List[RealtimeEvent]:
        """Pops up to `max_items` events atomically."""
        with self._lock:
            items: List[RealtimeEvent] = []
            while self._queue and len(items) < max_items:
                items.append(self._queue.popleft())
            return items

    def peek(self) -> Optional[RealtimeEvent]:
        """Returns the oldest event without removing it."""
        with self._lock:
            if not self._queue:
                return None
            return self._queue[0]

    def clear(self) -> int:
        """Empties the buffer and returns number of cleared items."""
        with self._lock:
            count = len(self._queue)
            self._queue.clear()
            return count

    def get_stats(self) -> Dict[str, Any]:
        """Returns buffer utilization metrics."""
        with self._lock:
            return {
                "session_id": self.session_id,
                "current_size": len(self._queue),
                "max_capacity": self.max_capacity,
                "utilization_pct": round((len(self._queue) / self.max_capacity) * 100, 2),
                "total_pushed": self._total_pushed,
                "total_dropped": self._total_dropped,
                "drop_reasons": dict(self._drop_reasons),
                "policy": self.policy.value,
            }
