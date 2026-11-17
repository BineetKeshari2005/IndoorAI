# IndoorAI — Unified Multimodal Session Architecture & Dataset Builder

This document details the architectural design, synchronization theory, persistent storage model, and dataset export validation rules implemented for **Task 17: Unified Multimodal Capture Session & Dataset Builder** in the IndoorAI platform.

---

## 1. High-Level Architecture Overview

IndoorAI unifies multiple asynchronous telemetry streams into a synchronized survey session:

```
+---------------------------------------------------------------------------------------------------+
|                                      IndoorAI Android Client                                      |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  [ Wi-Fi Radio Collector ]    [ Inertial & Magnetic Sensors ]     [ Manual Camera Landmark Capture ]|
|  - WifiScanner API            - SensorEventListener               - CameraX User-Initiated         |
|  - Real 802.11 Beacons        - Accel, Gyro, Mag, Heading         - On-Device Face Blur & Stripping|
|            |                                |                                    |                |
|            +--------------------------------+------------------------------------+                |
|                                             |                                                     |
|                                             v                                                     |
|                           +-----------------------------------+                                   |
|                           |   Timestamp Normalization Layer   |                                   |
|                           |   - ClockAnchor Alignment         |                                   |
|                           |   - Monotonic to Wall-Clock (UTC) |                                   |
|                           |   - Uncertainty Modeling (±ms)    |                                   |
|                           +-----------------+-----------------+                                   |
|                                             |                                                     |
|                                             v                                                     |
|                           +-----------------------------------+                                   |
|                           |       SurveySession Engine        |                                   |
|                           |   - Status: CREATED..COMPLETED    |                                   |
|                           |   - Manual Ground-Truth Labels    |                                   |
|                           |   - Real vs Mock Provenance       |                                   |
|                           +-----------------+-----------------+                                   |
|                                             |                                                     |
|                      +----------------------+----------------------+                              |
|                      |                                             |                              |
|                      v                                             v                              |
|         +-------------------------+                   +-------------------------+                 |
|         | SessionPersistenceMgr   |                   |    DatasetExporter      |                 |
|         | - filesDir/survey_sess/ |                   | - Pre-export Validation |                 |
|         | - JSONL observation log |                   | - SHA-256 Checksums     |                 |
|         | - Interruption Recovery |                   | - manifest.json         |                 |
|         | - Image Reference Sync  |                   | - session_<id>.zip      |                 |
|         +-------------------------+                   +-------------------------+                 |
|                                                                                                   |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Session Lifecycle State Machine

Every survey session transitions through a finite state machine:

```
       [ Create ]
           |
           v
     +-----------+         [ Start ]
     |  CREATED  | ------------------------+
     +-----------+                         |
                                           v
     +-----------+         [ Resume ]  +--------+
     |  PAUSED   | <=================> | ACTIVE |
     +-----------+         [ Pause ]   +--------+
           |                               |
           | [ Complete / Cancel ]         | [ Complete / Cancel ]
           v                               v
     +-----------+                   +-----------+
     | CANCELLED |                   | COMPLETED |
     +-----------+                   +-----+-----+
                                           |
                                           v  [ Export ]
                                     +-----------+
                                     | ZIP Bundle|
                                     +-----------+
```

### Lifecycle States
1. **`CREATED`**: Session is configured with building, floor, room, and survey name. Clock anchor is calibrated, but observations are not yet ingested.
2. **`ACTIVE`**: Survey is running. Wi-Fi sweeps, high-frequency IMU events, manual landmark captures, and surveyor annotations are associated with the session.
3. **`PAUSED`**: Survey is temporarily suspended (e.g. surveyor is moving between wings). Sensor listening is unhooked to save battery.
4. **`COMPLETED`**: Survey has completed successfully. Observation journals are sealed and ready for dataset packaging and validation.
5. **`CANCELLED`**: Survey was aborted. Data can be reviewed or permanently deleted with associated media files.

---

## 3. Timestamp Normalization & Synchronization

Different Android subsystems emit timestamps from independent clock sources:
- **Motion Sensors (`SensorEvent.timestamp`)**: Emits nanoseconds elapsed since device boot (`CLOCK_BOOTTIME`).
- **Wi-Fi Scans (`ScanResult.timestamp`)**: Emits microseconds elapsed since device boot (`CLOCK_BOOTTIME`).
- **Camera Captures & Annotations**: Emits standard UNIX epoch milliseconds (`CLOCK_REALTIME`).

Directly comparing nanoseconds to epoch milliseconds or assuming clock alignment causes temporal skew of decades.

### Clock Anchor Strategy
At session initialization, `ClockAnchor` samples both clocks atomically:
$$t_{\text{epoch\_anchor}} = \text{System.currentTimeMillis()}$$
$$t_{\text{boot\_anchor}} = \text{SystemClock.elapsedRealtimeNanos()}$$

Given any subsequent hardware sensor reading at $t_{\text{sensor\_ns}}$:
$$\Delta t_{\text{ns}} = t_{\text{sensor\_ns}} - t_{\text{boot\_anchor}}$$
$$t_{\text{norm\_ms}} = t_{\text{epoch\_anchor}} + \left\lfloor \frac{\Delta t_{\text{ns}}}{1\,000\,000} \right\rfloor$$

### Uncertainty Bounds & Timing Reality
IndoorAI does **not** claim microsecond-perfect synchronization across separate hardware busses. Every normalized timestamp record explicitly tracks its error bounds:

| Stream | Original Source | Sync Method | Estimated Uncertainty |
| :--- | :--- | :--- | :--- |
| **Inertial (IMU)** | `CLOCK_BOOTTIME_NANOS` | `BOOTTIME_ANCHOR_OFFSET` | $\pm 3.5\text{ ms}$ (kernel dispatch & thread scheduling) |
| **Magnetometer** | `CLOCK_BOOTTIME_NANOS` | `BOOTTIME_ANCHOR_OFFSET` | $\pm 3.5\text{ ms}$ |
| **Fresh Wi-Fi Scan** | `CLOCK_BOOTTIME_MICROS` | `SCAN_AGE_CONVERTED` | $\pm 30.0\text{ ms}$ (radio channel dwell time) |
| **Cached Wi-Fi Scan** | `CLOCK_BOOTTIME_MICROS` | `SCAN_AGE_CONVERTED` | $\pm 250.0\text{ ms}$ (OS throttling delay) |
| **Camera Photo** | `CLOCK_REALTIME_MILLIS` | `WALLCLOCK_DIRECT` | $\pm 1.5\text{ ms}$ |
| **Manual Annotation** | `CLOCK_REALTIME_MILLIS` | `WALLCLOCK_DIRECT` | $\pm 1.5\text{ ms}$ |
| **Invalid / $\le 0$** | `UNKNOWN` | `UNSYNCHRONIZED` | Undefined (`null`) |

---

## 4. Persistent Local Storage & Interruption Recovery

### On-Disk Session Layout
Sessions are stored in application-private storage (`context.filesDir/survey_sessions/<sessionId>/`):
```text
filesDir/survey_sessions/SES-12345678/
├── session_metadata.json          # Complete SurveySession state, counts, anchor
├── wifi_journal.jsonl             # Appended Wi-Fi scan observations
├── imu_journal.jsonl              # Appended IMU sensor samples
├── photos_journal.jsonl           # Appended photo metadata records
└── annotations_journal.jsonl      # Appended manual annotations
```

### Crash & Interruption Recovery
If the Android OS terminates the application process (e.g., low-memory kill, device reboot) while a session is `ACTIVE`:
1. Upon restart, `SessionPersistenceManager` scans all session folders.
2. Any session with `status == ACTIVE` is converted to `status = PAUSED`.
3. The session is tagged with `isInterrupted = true` and `interruptionNote = "Session recovered after application termination; review or resume."`.
4. The user is alerted with a dedicated banner allowing them to **Resume**, **Complete**, or **Cancel** the session.
5. Under no circumstances is an interrupted session silently completed or discarded.

### Referential Integrity on Deletion
When `deleteSession(sessionId)` is invoked:
1. `photos_journal.jsonl` is inspected.
2. All referenced image files on disk (`context.filesDir/survey_photos/PHOTO_..._sanitized.jpg`) are permanently deleted.
3. The session directory is recursively removed.
4. This guarantees zero orphaned images or broken file references.

---

## 5. Dataset ZIP Export & Manifest Schema

A completed session is packaged as a standard self-contained ZIP:
```text
indoorai_session_<sessionId>.zip
├── session.json              # Full session metadata, anchor, and location labels
├── wifi_observations.csv     # Access point sweeps (BSSID, RSSI, channel, timestamps)
├── imu_observations.csv      # 3-axis motion/magnetic samples with normalized time
├── photo_metadata.csv        # Landmark photo records and sanitization provenance
├── annotations.csv           # Manual ground-truth architectural markers
├── images/                   # Confirmed sanitized JPEG files
│   ├── PHOTO_20261117_153000_sanitized.jpg
│   └── ...
└── manifest.json             # Cryptographic manifest with SHA-256 hashes
```

### Manifest Schema (`manifest.json`)
```json
{
  "exportVersion": "1.0.0",
  "sessionId": "SES-FULL01",
  "exportedAtMs": 1716000000000,
  "exportedAtIso": "2024-05-18T02:40:00.000Z",
  "recordCounts": {
    "wifiCount": 12,
    "imuCount": 845,
    "photoCount": 3,
    "annotationCount": 4,
    "totalObservations": 864
  },
  "sourceCounts": {
    "realObservationCount": 864,
    "mockObservationCount": 0,
    "containsRealData": true,
    "containsMockData": false
  },
  "files": [
    {
      "filename": "session.json",
      "sizeBytes": 1240,
      "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    },
    {
      "filename": "images/PHOTO_20261117_153000_sanitized.jpg",
      "sizeBytes": 184320,
      "sha256": "4a5b6c..."
    }
  ],
  "validationStatus": "VALIDATED"
}
```

### Export Validation Checklist
Before generating the ZIP archive, `ExportValidator` performs mandatory checks:
- [x] Every observation references the valid `sessionId`.
- [x] Every observation has a non-empty, unique ID.
- [x] All photo files exist on disk with positive file size.
- [x] **Mandatory Privacy Gate**: Every image must have `isExifSanitized == true`. If an image is unsanitized or unknown, export aborts immediately.
- [x] Timestamps have valid values or are explicitly tagged `UNSYNCHRONIZED`.
- [x] Real and mock observations are clearly distinguished.
- [x] SHA-256 digests are computed for all files.

---

## 6. Real vs. Mock Data Separation

IndoorAI enforces clear provenance distinction:
- Every observation record carries an explicit `isMock: Boolean` attribute.
- The session computes `DataSourceType`:
  - `REAL`: Only real hardware observations.
  - `MOCK`: Only synthetic test data.
  - `HYBRID`: Mixed session.
- Synthetic mock observations are visibly tagged with yellow `MOCK` badges in the UI and timeline.
- Real hardware data is never silently combined with synthetic benchmarks.

---

## 7. Current System Limitations

1. **Multi-Floor Automatic Detection**: Vertical transitions (elevators, stairs) rely on manual surveyor annotations or pressure sensor fusion (planned for subsequent PDR / filter tasks).
2. **Wi-Fi Throttle Latency**: Real-world Wi-Fi radio scans are throttled by Android (4 scans / 2 minutes in foreground); timestamp uncertainty reflects cache aging.
3. **Clock Drift**: Device boot clocks experience low drift ($\approx 10\text{--}50\text{ ppm}$) across multi-hour sessions; periodic clock anchor recalibration is recommended for surveys exceeding 60 minutes.
