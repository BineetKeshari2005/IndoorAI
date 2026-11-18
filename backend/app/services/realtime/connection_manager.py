"""WebSocket connection manager for real-time streaming (Task 29).

Tracks active WebSocket sessions, multiplexes clients (mobile device + dashboard observers),
delivers asynchronous messages, and handles disconnections safely.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional, Set

from fastapi import WebSocket
from starlette.websockets import WebSocketState

from src.core.logging import setup_logger

logger = setup_logger("RealtimeConnectionManager")


class WebSocketConnectionManager:
    """Manages active WebSocket connections partitioned by session ID."""

    def __init__(self) -> None:
        # Maps session_id -> Set of active WebSockets
        self._active_connections: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, session_id: str, websocket: WebSocket) -> None:
        """Registers an established WebSocket under a session ID."""
        await websocket.accept()
        async with self._lock:
            if session_id not in self._active_connections:
                self._active_connections[session_id] = set()
            self._active_connections[session_id].add(websocket)
        logger.info(
            "WebSocket client connected to session %s (total clients in session: %d)",
            session_id,
            len(self._active_connections[session_id]),
        )

    async def disconnect(self, session_id: str, websocket: WebSocket) -> None:
        """Removes a disconnected WebSocket from active registry."""
        async with self._lock:
            if session_id in self._active_connections:
                self._active_connections[session_id].discard(websocket)
                if not self._active_connections[session_id]:
                    del self._active_connections[session_id]
        logger.info("WebSocket client disconnected from session %s", session_id)

    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket) -> bool:
        """Sends a JSON message directly to an individual WebSocket."""
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json(message)
                return True
            return False
        except Exception as e:
            logger.debug("Failed to send direct WebSocket message: %s", e)
            return False

    async def broadcast_to_session(self, session_id: str, message: Dict[str, Any]) -> int:
        """Broadcasts a JSON message to all clients connected to the given session.

        Returns:
            Number of clients successfully delivered to.
        """
        async with self._lock:
            sockets = list(self._active_connections.get(session_id, set()))

        if not sockets:
            return 0

        delivered = 0
        dead_sockets: List[WebSocket] = []

        for ws in sockets:
            try:
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.send_json(message)
                    delivered += 1
                else:
                    dead_sockets.append(ws)
            except Exception as e:
                logger.debug("Error delivering message to WebSocket in session %s: %s", session_id, e)
                dead_sockets.append(ws)

        if dead_sockets:
            async with self._lock:
                if session_id in self._active_connections:
                    for dead in dead_sockets:
                        self._active_connections[session_id].discard(dead)
                    if not self._active_connections[session_id]:
                        del self._active_connections[session_id]

        return delivered

    def get_client_count(self, session_id: str) -> int:
        """Returns number of active connected clients for a session."""
        return len(self._active_connections.get(session_id, set()))

    def get_total_connected_sessions(self) -> int:
        """Returns number of distinct sessions with connected clients."""
        return len(self._active_connections)
