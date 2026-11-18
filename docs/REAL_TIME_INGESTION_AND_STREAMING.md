# Real-Time Multimodal Ingestion & Streaming Integration Architecture (Task 29)

This document provides the complete technical specification for **Task 29: Real-Time Multimodal Ingestion and Streaming Integration** for IndoorAI.

---

## 1. Executive Summary

Task 29 connects the authorized Android mobile client (`ai.indoor.app`) directly to the IndoorAI FastAPI backend via high-performance bidirectional WebSockets and an HTTP/REST fallback batch ingestion endpoint.

The real-time pipeline ingests live Wi-Fi RSSI scans, IMU sensor batches, orientation updates, photo capture metadata, and landmark observations. It validates incoming payloads, manages bounded in-memory FIFO buffers with backpressure protection, coordinates deduplication, and streams low-latency localization fixes (Task 23 / 24 / 27) and live turn-by-turn guidance directives (Task 28) back to the device and connected dashboard observers.

```
+-----------------------------------------------------------------------------------+
|                           Android Client (ai.indoor.app)                          |
|  [WifiScanner]    [SensorManager]    [CameraX Privacy]    [PendingEventQueue (1k)]|
+-----------------------------------------------------------------------------------+
                                   │                 ▲
          WebSocket Frame (JSON)   │                 │ Outgoing State Updates
       /api/realtime/.../stream    ▼                 │ (ACK, LOCALIZATION, NAV)
+───────────────────────────────────────────────────────────────────────────────────+
|                               FastAPI Backend                                     |
|  ┌──────────────────┐    ┌──────────────────┐    ┌─────────────────────────────┐  |
|  |  EventValidator  |───►| BoundedBuffer(5k)|───►|       EventProcessor        |  |
|  | (Schema/NaN/OOB) |    |  (Backpressure)  |    |  - Site RadioMap (Task 23)  |  |
|  └──────────────────┘    └──────────────────┘    |  - PDR Kinematics (Task 21) |  |
|                                                  |  - Particle Filter (Task 27)|  |
|  ┌──────────────────┐    ┌──────────────────┐    |  - Navigation HUD (Task 28) |  |
|  |  ReplayManager   |◄───| WebSocketManager |◄───┴─────────────────────────────┘  |
|  | (Seq # Reconnect)|    | (Multiplex/Bcast)|                                     |
|  └──────────────────┘    └──────────────────┘                                     |
+───────────────────────────────────────────────────────────────────────────────────+
                                   ▲
                                   │ WebSocket / REST Stream
+──────────────────────────────────┴────────────────────────────────────────────────+
|                     Streamlit Dashboard (Section 19)                              |
|   Live Stream Telemetry | Wi-Fi / IMU Feed | 2D Localization | Turn-by-Turn HUD   |
+───────────────────────────────────────────────────────────────────────────────────+
```

---

## 2. Ingestion Architecture & Data Flow

The real-time ingestion layer is modularized under `backend/app/services/realtime/`:

| Module | Responsibility |
| :--- | :--- |
| `event_models.py` | Typed dataclasses, enums, payload structures, server acknowledgements, and state broadcasts. |
| `event_validator.py` | Structural validation, timestamp bounds, Wi-Fi RF validation, sensor NaN/Inf rejection, and privacy checks. |
| `event_buffer.py` | Thread-safe FIFO ring buffer with configurable drop policies (`DROP_OLDEST`, `REJECT_NEW`) and memory bounds. |
| `replay_manager.py` | Ordered sequence logger supporting reconnection resumption from `last_ack_event_id` or sequence number. |
| `connection_manager.py` | Asynchronous WebSocket connection multiplexer handling client tracking, personal messaging, and session broadcasts. |
| `session_stream.py` | In-memory session context tracking telemetry, deduplication caches, attached maps, and active navigation trips. |
| `event_processor.py` | Chronological event dispatcher routing observations to Wi-Fi matching, PDR step detection, particle filtering, and navigation. |
| `realtime_pipeline.py` | Singleton coordinator orchestrating ingestion, WebSocket loops, REST fallback batches, and simulation injection. |

---

## 3. Supported Typed Events & Payloads

Every event conforms to `RealtimeEvent` and includes provenance headers:

```json
{
  "event_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "session_id": "LIVE_SURVEY_01",
  "event_type": "WIFI_SCAN",
  "device_id": "PIXEL_8_PRO",
  "client_timestamp_monotonic": 12847192847,
  "client_timestamp_utc": 1700000000000,
  "server_receive_timestamp": 1700000000045,
  "payload_schema_version": "1.0.0",
  "sequence_number": 42,
  "payload": { ... }
}
```

### Event Type Schema Definitions

1. **`WIFI_SCAN`**:
   - `bssid_scans`: List of observed APs. Each contains `bssid` (MAC address), `rssi_dbm` (float, bounded $[-120, 0]$ dBm), `frequency_mhz` (validated against 2.4 GHz, 5 GHz, or 6 GHz channels).
2. **`IMU_BATCH`**:
   - `samples`: List of sensor readings. Each contains `sensor_type` (`ACCELEROMETER`, `GYROSCOPE`, `MAGNETOMETER`, `ROTATION_VECTOR`), `x`, `y`, `z` (validated for finite numbers; NaN and Inf strictly rejected), `accuracy`, `timestamp_epoch_ms`.
3. **`ORIENTATION_UPDATE`**:
   - `heading_rad` / `azimuth_deg`: Planar orientation. Checked for finite floating-point values.
4. **`PHOTO_CAPTURE`**:
   - `photo_id`, `filename`, `exif_sanitized` (mandatory `True` flag; unsanitized captures rejected), `detected_faces_count`.
   - Continuous camera frame streaming is **forbidden** by architectural design. Only user-initiated, privacy-sanitized captures are accepted.
5. **`LANDMARK_OBSERVATION`**:
   - `landmark_type` (e.g. `DOOR`, `STAIRCASE`, `ELEVATOR`, `SIGNBOARD`), `landmark_id`, `is_verified` (boolean flag).
   - Unverified landmarks are recorded for surveyor inspection but **never** update localization anchors (anti-fabrication rule).
6. **`ANNOTATION`**:
   - `label`, `category`, `x_m`, `y_m`, `floor_id`.
7. **`HEARTBEAT`**:
   - Client uptime, battery level, sequence tracking.
8. **`SESSION_STATUS`**:
   - Lifecycle triggers: `START`, `PAUSE`, `RESUME`, `STOP`, `CANCEL`.

---

## 4. Streaming API Specification

### 4.1 Bidirectional WebSocket Interface
`GET /api/realtime/sessions/{session_id}/stream`
- **Query Parameter**: `last_ack_event_id` (optional): If supplied, the server immediately replays state updates and events emitted since this ID.
- **Client $\rightarrow$ Server**: Ingestion frames (`RealtimeEvent`), `PING` heartbeat probes.
- **Server $\rightarrow$ Client**:
  - `STREAM_STATUS`: Connection state, buffer utilization, active navigation session.
  - `ACK`: Acknowledgement with `event_id`, status (`ACCEPTED`, `REJECTED`, `DROPPED`, `DUPLICATE`), sequence number, and server receive timestamp.
  - `LOCALIZATION_UPDATE`: 2D ENU position, floor ID, heading, 95% circular uncertainty radius, source engine (`PARTICLE_FILTER`, `WIFI_RADIO_MAP`, `PDR_DEAD_RECKONING`, `VERIFIED_LANDMARK`).
  - `NAVIGATION_UPDATE`: Current instruction narrative, directional arrow type, distance to next waypoint, remaining distance, cross-track error, arrival status, and reroute alert flag.
  - `PONG`: Heartbeat acknowledgement.

### 4.2 REST Fallback Batch Ingestion Endpoint
`POST /api/realtime/sessions/{session_id}/events/batch`
- **Request Body**: `{"events": [...]}`
- **Response**: `BatchAckResponse` containing total count, accepted count, rejected count, duplicate count, and itemized `ServerAckResponse` records with assigned sequence numbers.

### 4.3 Management & Telemetry Endpoints
- `GET /api/realtime/sessions/{session_id}/status`: Real-time throughput metrics, buffer utilization percentage, connected client count.
- `POST /api/realtime/sessions/{session_id}/control`: Lifecycle control (`START`, `PAUSE`, `RESUME`, `STOP`, `CANCEL`, `RESET`).
- `GET /api/realtime/sessions/{session_id}/history`: Replay slice inspection with query parameters `last_ack_event_id` and `max_items`.
- `POST /api/realtime/sessions/{session_id}/simulate`: Development step injection.
- `POST /api/realtime/sessions/{session_id}/attach-navigation`: Links an active Task 28 navigation session to the stream.
- `GET /api/realtime/sessions`: Lists all active streaming sessions.

---

## 5. Buffering, Backpressure & Replay Synchronization

### 5.1 Bounded In-Memory Queue
To guarantee server stability without relying on external message brokers (such as Redis or Kafka), each active session manages a `BoundedEventBuffer` capped at 5,000 events (configurable).
- Under `DROP_OLDEST` policy, incoming bursts gracefully evict the oldest pending event while recording an explicit `BACKPRESSURE_DROP` telemetry event.
- Under `REJECT_NEW` policy, excess events are rejected with an explicit rejection acknowledgement.
- A sliding deduplication window (10,000 items) guarantees idempotent retransmissions across retries.

### 5.2 Monotonic Sequence Ordering & Replay
The `ReplayManager` assigns a strictly increasing `sequence_number` ($1, 2, 3, \dots$) to every accepted event.
When a mobile device reconnects after a tunnel or network drop:
1. Client connects with `last_ack_event_id`.
2. Server queries `ReplayManager.get_replay_since(last_ack_event_id)`.
3. If the event ID exists in the circular history buffer (1,000 items), the server transmits `REPLAY_START`, streams the missing outgoing updates in chronological order, and concludes with `REPLAY_COMPLETE`.
4. If the gap exceeds the buffer limit, `is_gap=True` is communicated, instructing the client to perform a full baseline refresh.

---

## 6. Live Engine Integration & Anti-Fabrication Principles

### 6.1 Wi-Fi & PDR Fusion
1. **Wi-Fi**: Observations are processed against the site-specific `RadioMap` (Task 23). Localization fixes are only produced when $\ge 2$ APs match the radio map. Incompatible coordinate frames are rejected.
2. **PDR Kinematics**: Acceleration vectors are inspected for step peaks. Stepping intervals propagate dead-reckoning displacement without assuming device orientation directly provides absolute Cartesian coordinates.
3. **Map-Constrained Particle Filter**: When attached, Wi-Fi fixes trigger Gaussian measurement updates, PDR steps trigger motion propagation, and solid walls reject physically implausible particles (Task 27).

### 6.2 Anti-Flapping Live Navigation
Task 28's route planner and progress tracker are integrated via `EventProcessor._update_navigation`:
- **Confidence Gating**: Waypoint advancement is **suppressed** if the localization fix reports `confidence_level == "LOW"`. Noisy or weak fixes cannot cause spurious instruction changes.
- **Dynamic Rerouting**: If the cross-track error exceeds $4.0\text{m}$ with `HIGH` or `MEDIUM` confidence, the `ReroutingEngine` recalculates a valid route respecting the 5.0-second cooldown debounce.

---

## 7. Android Client Architecture (`android/`)

The Android integration is implemented under `android/app/src/main/java/ai/indoor/app/realtime/`:
- `RealtimeEventModels.kt`: Kotlin data classes mirroring backend schemas.
- `PendingEventQueue.kt`: Bounded in-memory FIFO queue (1,000 capacity) storing unacknowledged events during network drops.
- `RealtimeWebSocketClient.kt`: OkHttp WebSocket client featuring exponential backoff reconnect, 15-second heartbeat pinging, and coroutine StateFlow / SharedFlow streams.
- `RealtimeBatchFallbackClient.kt`: HTTP/REST fallback batch uploader when WebSockets are unavailable.
- `RealtimeStreamingManager.kt`: High-level manager coordinating capture usecases, queue drainage, and UI state flows.

> [!NOTE]
> **Physical-Device Verification Status**: Marked as **PENDING**. In this development and automated test environment, physical Android hardware was not connected, and no local Java Runtime SDK was available on the host PATH. All Kotlin serialization, pending queue logic, and lifecycle state management were validated through comprehensive synthetic unit tests (`RealtimeEventSerializationTest`, `PendingEventQueueTest`). Physical-device testing with live Wi-Fi scans and hardware IMUs must be conducted during field trials.

---

## 8. Streamlit Dashboard Section 19

The Streamlit explorer has been extended with **Section 19: Real-Time Multimodal Ingestion & Live Navigation**:
1. **Stream Telemetry & Buffer Tab**: Live connection pill, throughput KPI metrics (Received, Processed, Rejected, Dropped), buffer utilization bar, and session pause/resume/reset controls.
2. **Live Sensor & Wi-Fi Feed Tab**: Chronological event feed, recent Wi-Fi AP RSSI horizontal bar chart, and vertical acceleration kinematic waveform.
3. **Live Map Localization Canvas Tab**: 2D floorplan plot with real-time user location $(x, y)$, heading arrow, and 95% circular confidence ellipse.
4. **Turn-by-Turn Navigation HUD Tab**: Guidance banner with directional arrow icons, current instruction, distance to next waypoint, remaining distance, cross-track deviation, and reroute alert banner.
5. **Simulation & Replay Harness Tab**: Safe deterministic simulation controls with a prominent yellow/red warning banner:
   `⚠️ SIMULATION MODE ACTIVE: Deterministic synthetic data generated for testing. Do not use for physical verification.`

---

## 9. Verification Summary

- **Unit & Integration Suite (`tests/test_realtime.py`):**
  - 15 test cases covering event validation, duplicate idempotency, bounded buffer backpressure, REST batch ingestion, WebSocket streaming, reconnection replay, session isolation, unverified landmark rejection, live navigation tracking with anti-flapping, session lifecycle control, and development simulator injection.
  - Result: `Ran 15 tests in 0.043s. OK`
- **Full Repository Regression Suite:**
  - 315 test cases executed across all 29 tasks.
  - Result: `Ran 315 tests in 14.128s. OK` (100% pass, 0 failures, 0 errors).
