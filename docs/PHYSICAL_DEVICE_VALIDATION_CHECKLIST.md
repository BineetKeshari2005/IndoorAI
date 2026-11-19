# IndoorAI — Physical Device Validation Checklist

**Document Version**: 1.0.0  
**Current Status**: **PENDING PHYSICAL DEVICE EXECUTION**  
**Host Toolchain Audit**: Native host environment currently lacks Java Development Kit (`javac`) and Android SDK binaries; physical on-device executions could not be automated in this headless container. All items below must be verified manually on a physical Android device prior to general production rollout.

---

## 1. Prerequisites & Test Equipment

- **Target Device**: Physical Android smartphone running **Android 10 (API 29)** or higher (Recommended: Google Pixel 7/8 or Samsung Galaxy S22+).
- **Required Sensors**:
  - 3-axis Accelerometer (`TYPE_ACCELEROMETER`)
  - 3-axis Gyroscope (`TYPE_GYROSCOPE`)
  - 3-axis Magnetometer (`TYPE_MAGNETIC_FIELD`)
  - Wi-Fi 802.11 b/g/n/ac/ax receiver
  - Back-facing Camera with autofocus
- **Connectivity**: Local Wi-Fi network routing to the running IndoorAI backend server (e.g. `http://192.168.1.100:8000`).

---

## 2. On-Device Verification Checklist

### Phase 1: Build & APK Installation
- [ ] **1.1 Android SDK Build**: Run `./gradlew assembleDebug` in `android/` directory and verify build outputs `app-debug.apk` with zero errors.  
  *Status*: `[PENDING]`
- [ ] **1.2 APK Installation**: Install APK via ADB: `adb install -r android/app/build/outputs/apk/debug/app-debug.apk`.  
  *Status*: `[PENDING]`
- [ ] **1.3 App Launch & Splash**: Verify the app launches smoothly without runtime crash or uncaught exceptions.  
  *Status*: `[PENDING]`

### Phase 2: Runtime Permissions & Explicit Consent
- [ ] **2.1 Permission Requests**: Verify app displays standard Android dialogs requesting:
  - `ACCESS_FINE_LOCATION` (Required for Wi-Fi scanning on Android 10+)
  - `ACCESS_WIFI_STATE` & `CHANGE_WIFI_STATE`
  - `BODY_SENSORS` & `HIGH_SAMPLING_RATE_SENSORS`
  - `CAMERA` (Required for landmark collection)  
  *Status*: `[PENDING]`
- [ ] **2.2 Explicit Consent Banner**: Verify that the survey/capture screen displays explicit user consent dialogs before collecting any data.  
  *Status*: `[PENDING]`
- [ ] **2.3 Graceful Permission Denial**: Verify that denying camera permission allows sensor/Wi-Fi survey to continue without app crashing.  
  *Status*: `[PENDING]`

### Phase 3: Background Wi-Fi Survey Collection
- [ ] **3.1 Wi-Fi Scan Throttling Handling**: Ensure Wi-Fi scans trigger every 2–4 seconds without crashing due to Android Wi-Fi scan throttling limits (4 scans per 2-minute window when in background).  
  *Status*: `[PENDING]`
- [ ] **3.2 BSSID & RSSI Capture**: Verify that scan results contain valid BSSIDs (`aa:bb:cc:dd:ee:ff`), valid RSSI levels between $-95\text{ dBm}$ and $-30\text{ dBm}$, and standard operating frequencies ($2412\text{ MHz}$, $5180\text{ MHz}$).  
  *Status*: `[PENDING]`
- [ ] **3.3 Manual Reference Location Tagging**: Verify user can pin their current $(x, y, \text{floor})$ location on screen during survey sweeps.  
  *Status*: `[PENDING]`

### Phase 4: Camera Capture & Privacy Sanitization
- [ ] **4.1 Manual Photo Triggering**: Verify camera preview displays cleanly and captures photos only upon explicit manual shutter tap.  
  *Status*: `[PENDING]`
- [ ] **4.2 On-Device Face Blurring**: Test camera capture with a person in the frame. Verify that face detection runs and applies a Gaussian blur box over detected human faces before image file storage.  
  *Status*: `[PENDING]`
- [ ] **4.3 EXIF Metadata Scrubbing**: Inspect saved image files using `exiftool` or metadata readers. Verify all GPS location tags, device serial numbers, and camera owner names are completely stripped.  
  *Status*: `[PENDING]`

### Phase 5: Live Streaming & WebSocket Handshake
- [ ] **5.1 Server Connection Setup**: Configure server endpoint URL (`ws://<host-ip>:8000/api/realtime/sessions/{id}/stream`).  
  *Status*: `[PENDING]`
- [ ] **5.2 JWT Authentication Handshake**: Verify client passes JWT token in connection query. Verify server accepts authenticated handshake and returns initial `STREAM_STATUS`.  
  *Status*: `[PENDING]`
- [ ] **5.3 Real-Time Ping/Pong**: Verify periodic heartbeat messages maintain active connection without timeout drops.  
  *Status*: `[PENDING]`

### Phase 6: PDR Tracking & Motion Estimation
- [ ] **6.1 Step Detection Accuracy**: Walk 50 steps along a measured corridor with device held in hand. Verify step detector records $50 \pm 2$ steps.  
  *Status*: `[PENDING]`
- [ ] **6.2 Stride Length Adaptation**: Walk with varying paces (slow walk vs fast stride). Verify estimated distance corresponds to actual corridor length ($< 5\%$ error).  
  *Status*: `[PENDING]`
- [ ] **6.3 Heading Stability & Magnetic Interference**: Walk through a corridor with metal door frames. Verify complementary filter rejects transient magnetic spikes without abrupt orientation flips.  
  *Status*: `[PENDING]`

### Phase 7: Real-Time Navigation & Turn-by-Turn HUD
- [ ] **7.1 Destination Selection**: Select a room on Floor 2. Verify server computes route and client receives structured waypoint coordinates.  
  *Status*: `[PENDING]`
- [ ] **7.2 Turn Instruction Display**: Walk along the planned route. Verify client HUD updates turn instructions (`CONTINUE_STRAIGHT`, `TURN_LEFT`) within 2 meters of turning points.  
  *Status*: `[PENDING]`
- [ ] **7.3 Off-Route Recalculation**: Intentionally take a wrong hallway. Verify that upon exceeding 4.0 meters of lateral deviation, the HUD alerts and updates with a rerouted trajectory.  
  *Status*: `[PENDING]`
- [ ] **7.4 Destination Arrival**: Reach destination door. Verify HUD displays `ARRIVE_DESTINATION` and prompts session completion.  
  *Status*: `[PENDING]`

### Phase 8: Disconnection Recovery & Replay
- [ ] **8.1 Airplane Mode Simulation**: Enable Airplane Mode for 15 seconds during active tracking, then disable it.  
  *Status*: `[PENDING]`
- [ ] **8.2 Reconnection Handshake**: Verify client re-establishes WebSocket connection with `last_ack_event_id`.  
  *Status*: `[PENDING]`
- [ ] **8.3 Replay Stream Ingestion**: Verify server sends missed localization updates without duplicating processed steps or dropping position tracking.  
  *Status*: `[PENDING]`

---

## 3. Physical Field Test Sign-Off Protocol

When running physical validations on site:
1. Record device model, Android OS version, and test building ID.
2. Store the captured survey dataset in `data/raw/surveys/<session-id>/`.
3. Validate that resulting positional error against ground-truth surveyed points remains within **$< 3.0\text{ meters}$ median error**.
4. Update this checklist by marking verified checkboxes and changing overall status to `VERIFIED ON HARDWARE`.
