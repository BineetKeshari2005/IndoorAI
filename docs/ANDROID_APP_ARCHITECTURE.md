# IndoorAI — Android Application Architecture & Scaffolding Specification

## 1. Executive Summary & Architectural Overview

Task 13 establishes the Android mobile foundation for **IndoorAI**, providing a modular, reactive, and privacy-first edge platform for indoor positioning and floorplan survey capture. The mobile client is structured using **Modern Android Architecture (MVVM)**, **Jetpack Compose (Material 3)**, **Kotlin Coroutines / StateFlow**, and **Retrofit 2**.

```
                           +-----------------------------------------------+
                           |          Jetpack Compose Presentation         |
                           |   (Screens A-F: Welcome, Permissions, Wi-Fi,  |
                           |       Sensors, Manual Camera, Session)        |
                           +-----------------------------------------------+
                                                  |
                                                  v
                           +-----------------------------------------------+
                           |            StateFlow / ViewModels             |
                           |   (WifiScanVM, SensorVM, SessionVM, MainVM)   |
                           +-----------------------------------------------+
                                                  |
                                                  v
                           +-----------------------------------------------+
                           |                 Domain Layer                  |
                           |  (UseCases: ScanWifi, ObserveSensors, Session)|
                           +-----------------------------------------------+
                                    /                             \
                                   /                               \
                                  v                                 v
    +------------------------------------------+    +------------------------------------------+
    |         Hardware Abstraction Layer       |    |         Data & Networking Layer          |
    |  - WifiScanner (Android vs Mock)         |    |  - Retrofit IndoorApiService             |
    |  - SensorProvider (Android vs Mock)      |    |  - LocalizationRepository                |
    |  - CameraX / ActivityResultContracts     |    |  - DTOs (Health, Predict, Telemetry)     |
    +------------------------------------------+    +------------------------------------------+
```

### Key Architectural Invariants
1. **Decoupled Hardware Boundary:** All sensor and Wi-Fi operations are mediated through explicit abstract interfaces (`WifiScanner`, `SensorProvider`). On hardware lacking physical radios or gyroscopes (such as emulators), the app gracefully switches to high-fidelity synthetic mock providers (`MockWifiScanner`, `MockSensorProvider`).
2. **Explicit User Consent & Anti-Surveillance:** No data collection occurs in the background. Wi-Fi scanning and sensor streaming run strictly in the foreground during explicit user interaction or recorded capture sessions.
3. **No Fabricated Scan Claims:** All synthetic demonstration data is explicitly flagged with `isMock = true` and clearly badged in the UI.
4. **Resilience to OS Throttling:** Strict adherence to Android 9+ rate limits (4 scans / 2 minutes in foreground) with cached result fallback.

---

## 2. Project Directory Structure

The Android client is organized within the `android/` directory according to clean architecture guidelines:

```
android/
├── app/
│   ├── build.gradle.kts                       # Module-level Gradle configuration & dependencies
│   ├── proguard-rules.pro                     # R8/ProGuard obfuscation rules
│   └── src/
│       ├── main/
│       │   ├── AndroidManifest.xml            # Permissions, activities, hardware features
│       │   ├── res/                           # Drawables, themes, colors, string resources
│       │   │   ├── drawable/ic_location_pin.xml
│       │   │   └── values/{colors.xml, strings.xml, themes.xml}
│       │   └── java/ai/indoor/app/
│       │       ├── MainActivity.kt            # Single Activity container & DI wiring
│       │       ├── data/
│       │       │   ├── api/
│       │       │   │   ├── ApiClient.kt       # Retrofit & OkHttp client factory
│       │       │   │   └── IndoorApiService.kt# REST contracts (Predict, Health, Telemetry)
│       │       │   ├── models/                # Network Data Transfer Objects (DTOs)
│       │       │   │   ├── HealthResponseDto.kt
│       │       │   │   ├── ImuObservationDto.kt
│       │       │   │   ├── LocalizationRequestDto.kt
│       │       │   │   ├── LocalizationResponseDto.kt
│       │       │   │   ├── SessionTelemetryDto.kt
│       │       │   │   └── WifiScanResultDto.kt
│       │       │   └── repository/
│       │       │       ├── LocalizationRepository.kt
│       │       │       └── LocalizationRepositoryImpl.kt
│       │       ├── domain/
│       │       │   ├── models/                # Pure Kotlin domain entities
│       │       │   │   ├── AccessPoint.kt
│       │       │   │   ├── CaptureSession.kt
│       │       │   │   ├── ImuSample.kt
│       │       │   │   ├── LocalizationResult.kt
│       │       │   │   └── WifiFingerprint.kt
│       │       │   └── usecases/              # Clean architecture business logic
│       │       │       ├── ManageCaptureSessionUseCase.kt
│       │       │       ├── ObserveSensorsUseCase.kt
│       │       │       └── ScanWifiUseCase.kt
│       │       ├── sensors/                   # Motion & Magnetic sensor abstraction
│       │       │   ├── SensorProvider.kt      # Core interface & availability matrix
│       │       │   ├── AndroidSensorProvider.kt# Lifecycle-safe SensorManager implementation
│       │       │   └── MockSensorProvider.kt  # Synthetic 10 Hz gait simulation
│       │       ├── wifi/                      # 802.11 beacon observation abstraction
│       │       │   ├── WifiScanner.kt         # Core interface & WifiScanState sealed class
│       │       │   ├── AndroidWifiScanner.kt  # Throttling-aware WifiManager implementation
│       │       │   └── MockWifiScanner.kt     # Synthetic campus AP provider
│       │       ├── ui/
│       │       │   ├── components/            # Reusable Compose widgets
│       │       │   │   ├── AccessPointItem.kt # AP card with signal meter
│       │       │   │   ├── PermissionCard.kt  # Permission rationale card
│       │       │   │   ├── SensorReadingCard.kt# 3-axis telemetric gauge
│       │       │   │   └── StatusBadge.kt     # Mock & online status indicators
│       │       │   ├── navigation/
│       │       │   │   ├── AppNavHost.kt      # Scaffold with BottomNavigationBar
│       │       │   │   └── Screen.kt          # Sealed destination definitions
│       │       │   ├── screens/               # Compose screen composables
│       │       │   │   ├── WelcomeScreen.kt       # Screen A: Branding & overview
│       │       │   │   ├── PermissionSetupScreen.kt # Screen B: Permission rationales
│       │       │   │   ├── WifiScanScreen.kt      # Screen C: Wi-Fi beacon capture
│       │       │   │   ├── SensorMonitorScreen.kt # Screen D: IMU & compass monitor
│       │       │   │   ├── ImageCaptureScreen.kt  # Screen E: Manual camera capture
│       │       │   │   └── CaptureSessionScreen.kt# Screen F: Session telemetry & sync
│       │       │   └── theme/
│       │       │       ├── Color.kt           # Material 3 color palette
│       │       │       ├── Theme.kt           # Dynamic / dark / light color scheme
│       │       │       └── Type.kt            # Typography definitions
│       │       └── viewmodel/                 # AndroidX ViewModels exposing StateFlow
│       │           ├── ImageCaptureViewModel.kt
│       │           ├── MainViewModel.kt
│       │           ├── SensorViewModel.kt
│       │           ├── SessionViewModel.kt
│       │           └── WifiScanViewModel.kt
│       └── test/java/ai/indoor/app/           # Local JVM Unit Tests
│           ├── data/ApiSerializationTest.kt
│           ├── sensors/MockSensorProviderTest.kt
│           ├── viewmodel/SensorViewModelTest.kt
│           ├── viewmodel/SessionViewModelTest.kt
│           ├── viewmodel/WifiScanViewModelTest.kt
│           └── wifi/MockWifiScannerTest.kt
├── build.gradle.kts                           # Root Gradle build script
├── settings.gradle.kts                        # Plugin management & project inclusion
├── gradle.properties                          # JVM memory & AndroidX flags
├── gradlew                                    # POSIX wrapper script
└── gradle/wrapper/gradle-wrapper.properties   # Gradle 8.6 distribution specification
```

---

## 3. Toolchain & SDK Requirements

| Component | Target Version | Rationale |
| :--- | :--- | :--- |
| **Android compileSdk** | **34 (Android 14)** | Required for latest Android 14 APIs, sensor permission contracts, and security features. |
| **Android minSdk** | **26 (Android 8.0 Oreo)** | Covers >95% of active global Android devices while guaranteeing hardware sensor vector support. |
| **Android targetSdk** | **34 (Android 14)** | Full compliance with Google Play Store 2024+ deployment requirements. |
| **Java Toolchain** | **Java 17 (LTS)** | Mandatory requirement for Android Gradle Plugin 8.x and modern Gradle daemons. |
| **Android Gradle Plugin** | **8.4.1** | Stable release supporting Kotlin 1.9+ and Compose Compiler 1.5+. |
| **Gradle Distribution** | **8.6** | Optimized for AGP 8.4 and configuration caching. |
| **Kotlin Language** | **1.9.23** | Binary and compiler compatible with Compose Compiler 1.5.11. |
| **Jetpack Compose BOM** | **2024.05.00** | Stable Material 3 UI component library. |
| **Retrofit / OkHttp** | **2.11.0 / 4.12.0** | Modern asynchronous HTTP client with Kotlin Coroutines adapter. |

---

## 4. Permission Architecture & Anti-Surveillance Guarantees

Android enforces strict permission gating around ambient radio signals because beacon addresses can be reverse-engineered into physical locations.

### Permission Breakdown

```
+-----------------------------------+--------------------+-----------------------------------------------------+
| Android Permission                | Protection Level   | Architectural Purpose                               |
+-----------------------------------+--------------------+-----------------------------------------------------+
| android.permission.INTERNET       | Normal (Automatic) | Communicating with the IndoorAI backend inference   |
|                                   |                    | and telemetry upload services.                      |
+-----------------------------------+--------------------+-----------------------------------------------------+
| android.permission.ACCESS_NETWORK_STATE | Normal (Automatic) | Checking network connectivity state before uploads. |
+-----------------------------------+--------------------+-----------------------------------------------------+
| android.permission.ACCESS_WIFI_STATE | Normal (Automatic) | Inspecting Wi-Fi radio status and reading connection|
|                                   |                    | parameters.                                         |
+-----------------------------------+--------------------+-----------------------------------------------------+
| android.permission.CHANGE_WIFI_STATE | Normal (Automatic) | Invoking `WifiManager.startScan()` to trigger an     |
|                                   |                    | active radio beacon sweep.                          |
+-----------------------------------+--------------------+-----------------------------------------------------+
| android.permission.ACCESS_FINE_LOCATION | Dangerous (Runtime)| Mandatory since Android 9 (API 28) to read access   |
|                                   |                    | point BSSIDs and RSSI levels via `getScanResults()`.|
+-----------------------------------+--------------------+-----------------------------------------------------+
| android.permission.CAMERA         | Dangerous (Runtime)| STRICTLY opt-in. Requested ONLY when the user taps  |
|                                   |                    | "Take Photo" on Screen E. Never run in background.  |
+-----------------------------------+--------------------+-----------------------------------------------------+
| HIGH_SAMPLING_RATE_SENSORS        | Normal (Manifest)  | Allows IMU sampling > 200 Hz on Android 12+ (API 31)|
|                                   |                    | for high-resolution Dead Reckoning step analysis.   |
+-----------------------------------+--------------------+-----------------------------------------------------+
```

### Privacy & Anti-Surveillance Invariants
1. **Just-In-Time (JIT) Elevation:** Permissions are never requested at application launch. Location is only prompted when the user triggers a Wi-Fi scan; camera is only prompted when the user taps the camera shutter.
2. **Zero Covert Background Execution:** No `ForegroundService` or background workers poll the camera or microphone.
3. **Client-Side Privacy Sanitization (Screen E):**
   - **EXIF Stripping:** GPS latitude/longitude, handset serial, and lens tags are stripped from image bytes before transmission.
   - **Facial Blurring:** Privacy toggle enables on-device blurring of human faces and license plates.
   - **Ephemeral Raw Media:** Raw RGB photos are deleted immediately after visual landmark feature extraction.

---

## 5. Wi-Fi Scanning Architecture & OS Limitations

### Android OS Throttling Constraints
Starting in Android 9 (Pie, API 28) and hardened in Android 10+:
- **Foreground Rate Limiting:** A foreground application may trigger `WifiManager.startScan()` a maximum of **4 times in a 2-minute window**. Any subsequent scan within that window fails (`startScan()` returns `false`).
- **Background Rate Limiting:** Background applications are restricted to **1 scan every 30 minutes**.
- **Location Setting Dependency:** Even with `ACCESS_FINE_LOCATION` granted, system-wide **Location Services (GPS)** must be toggled **ON** in Android Settings for `getScanResults()` to return access points.
- **BSSID MAC Redaction:** On Android 10+ devices without fine location permission, BSSIDs are masked as `02:00:00:00:00:00`.

### Implementation: `AndroidWifiScanner` vs `MockWifiScanner`
The application abstracts Wi-Fi operations via `WifiScanner.kt`:

```kotlin
sealed class WifiScanState {
    object Idle : WifiScanState()
    object Scanning : WifiScanState()
    data class Success(val fingerprint: WifiFingerprint) : WifiScanState()
    data class Throttled(val message: String, val cachedFingerprint: WifiFingerprint?) : WifiScanState()
    data class Error(val message: String) : WifiScanState()
}
```

- When `startScan()` succeeds, `AndroidWifiScanner` listens for `SCAN_RESULTS_AVAILABLE_ACTION` via an asynchronous `BroadcastReceiver`, parses BSSID, SSID, RSSI (dBm), frequency (MHz), and elapsed nanosecond timestamps into pure domain `AccessPoint` entities.
- When `startScan()` is throttled by the OS, `AndroidWifiScanner` transitions to `WifiScanState.Throttled` and gracefully serves the **cached** previous scan results with an explicit warning banner, preventing UI crashes.
- For testing on the Android Studio Emulator (which lacks physical 802.11 radios), `MockWifiScanner` generates realistic 11-AP fingerprints with simulated ±2 dBm multipath Gaussian jitter and 1.2-second radio sweep latency.

---

## 6. Inertial & Magnetic Sensor Architecture

The application monitors three key on-board hardware sensors:
1. **Tri-Axial Accelerometer (`Sensor.TYPE_ACCELEROMETER`):** Measures proper acceleration in $\text{m/s}^2$ across X, Y, and Z axes. Used for step detection and gait analysis.
2. **Tri-Axial Gyroscope (`Sensor.TYPE_GYROSCOPE`):** Measures angular velocity in $\text{rad/s}$ across X, Y, and Z axes. Used for turn detection and angular heading integration.
3. **Tri-Axial Magnetometer (`Sensor.TYPE_MAGNETIC_FIELD`):** Measures ambient geomagnetic field in $\mu\text{T}$ across X, Y, and Z axes. Used for absolute compass orientation.

### Lifecycle-Safe Management
- **Safe Registration:** Sensors are registered using `SensorManager.SENSOR_DELAY_UI` (~15–20 ms interval) only when the user explicitly enters the monitoring screen or starts an active session.
- **Automatic Teardown:** In `MainActivity.onStop()`, all sensor listeners are immediately unregistered via `sensorProvider.stopListening()`. This prevents parasitic background battery drain.
- **Synthetic Fallback:** `MockSensorProvider` simulates walking gait dynamics using trigonometric equations ($9.81\,\text{m/s}^2$ vertical oscillation + $0.4\,\text{m/s}^2$ forward step stride) at 10 Hz when physical sensors are absent.

---

## 7. Backend Communication Foundation

### Network Architecture: Physical Device vs. Android Emulator

```
               [ Workstation Running FastAPI Backend at 127.0.0.1:8000 ]
                                      ^
                                     / \
                                    /   \
  [ Android Studio Emulator ]      /     \      [ Physical Android Device ]
  Connects to Host via:          /       \     Connects to Workstation via:
  http://10.0.2.2:8000/          /         \    http://192.168.1.150:8000/
                                /           \   (Workstation LAN IP)
```

> [!WARNING]
> On Android, `localhost` and `127.0.0.1` refer to the **phone/emulator itself**, not the developer's computer.
> - **In the Android Emulator:** Use `http://10.0.2.2:8000/` (special alias for host loopback).
> - **On a Physical Handset:** Use the computer's LAN IP address (e.g. `http://192.168.1.X:8000/`). Both handset and workstation must be on the same Wi-Fi subnet, and workstation firewalls must allow port 8000.

### REST API Service Contract
Configured in `IndoorApiService.kt` and managed via `ApiClient.kt`:

1. **`GET /api/v1/health`**
   - **Purpose:** Verifies backend connectivity and model availability.
   - **Response:** `HealthResponseDto(status="healthy", service="indoor-ai-backend", version="1.0.0", active_models=[...])`
2. **`POST /api/v1/predict`**
   - **Purpose:** Submits an ambient Wi-Fi observation for real-time model inference.
   - **Request:** `LocalizationRequestDto(device_id, timestamp, model_preference, wifi_fingerprint=[...], imu_sample=...)`
   - **Response:** `LocalizationResponseDto(building_id, floor, longitude, latitude, confidence, model_used, inference_latency_ms)`
3. **`POST /api/v1/telemetry/session`**
   - **Purpose:** Batched upload of multi-modal session recordings.
   - **Request:** `SessionTelemetryDto(session_id, device_id, start_time_ms, end_time_ms, wifi_scans=[...], imu_stream=[...], photo_count)`
   - **Response:** `SessionUploadResponseDto(session_id, status="ingested", records_ingested=..., message=...)`

---

## 8. Screen Implementation Summary

| Screen | File | Responsibilities & UI Controls |
| :--- | :--- | :--- |
| **Screen A: Welcome / Home** | `WelcomeScreen.kt` | • Location-pin vector branding.<br>• Indoor positioning vs GPS rationale.<br>• "Start Wi-Fi Scan & Localization" primary button.<br>• Live backend health status pill (`ONLINE` / `OFFLINE`).<br>• Navigation tiles for Wi-Fi, Sensors, Camera, and Session.<br>• Privacy guarantee banner. |
| **Screen B: Permission Setup** | `PermissionSetupScreen.kt` | • Detailed rationales for Location, Wi-Fi, Camera, and Sensors.<br>• Live granted/denied indicators.<br>• Interactive permission prompt triggers using Jetpack Compose `rememberLauncherForActivityResult`.<br>• Explicit explanation of Android 9+ Wi-Fi location mandate.<br>• Anti-surveillance manifesto. |
| **Screen C: Wi-Fi Fingerprint Capture** | `WifiScanScreen.kt` | • Manual "Scan" button with indeterminate circular progress indicator.<br>• Detected AP counter badge.<br>• OS Throttling warning banner with cached fallback.<br>• Scrollable LazyColumn displaying BSSID, SSID, RSSI (dBm), signal bar graph, and frequency.<br>• "Predict Location with Multitask MLP" action button.<br>• Formatted prediction card showing Building, Floor, and native Coordinates (X, Y). |
| **Screen D: Sensor Monitoring** | `SensorMonitorScreen.kt` | • Hardware availability matrix (Accelerometer, Gyroscope, Magnetometer).<br>• Live Start/Stop sampling control toggle.<br>• 3-axis telemetric gauges with dynamic magnitude calculation ($L_2$ norm).<br>• Clear note confirming foreground-only polling to protect battery. |
| **Screen E: Manual Image Capture** | `ImageCaptureScreen.kt` | • User-initiated shutter button ("Take Photo").<br>• Camera access triggered strictly on user press with runtime permission prompt.<br>• Preview placeholder with timestamp and EXIF status.<br>• Clear photo control.<br>• Privacy checklist: EXIF stripping, face blurring, and raw photo ephemerality. |
| **Screen F: Capture Session** | `CaptureSessionScreen.kt` | • Session Start/Pause/Stop lifecycle controls.<br>• Large elapsed duration timer (`HH:MM:SS`).<br>• Real-time observation counters: Wi-Fi scans, IMU samples, Photos.<br>• "Simulate Ambient Telemetry Packet" button.<br>• Configurable Backend Base URL text field (supports emulator `10.0.2.2` and physical LAN IP).<br>• "Sync Session" button with upload status readout. |

---

## 9. Physical-Device Testing Instructions

To test on a physical Android smartphone:
1. **Enable Developer Options:** Navigate to Android Settings $\to$ About Phone $\to$ Tap "Build Number" 7 times.
2. **Enable USB Debugging:** In Developer Options, enable "USB Debugging".
3. **(Optional) Disable Wi-Fi Throttling for Testing:** In Developer Options $\to$ Networking $\to$ Disable "Wi-Fi scan throttling". (Allows rapid testing without waiting for 2-minute rate-limit windows).
4. **Configure Local Network:**
   - Connect the Android device to the same local Wi-Fi router as your development computer.
   - Determine your computer's local LAN IP:
     - macOS: `ipconfig getifaddr en0` (or `en1`)
     - Linux: `hostname -I | awk '{print $1}'`
     - Windows: `ipconfig` (IPv4 Address)
   - Ensure the workstation firewall allows inbound TCP traffic on port 8000.
5. **Open Project in Android Studio:**
   - Open the `/android` folder in Android Studio Hedgehog (2023.1.1) or newer.
   - Ensure JDK 17 is selected in **Settings $\to$ Build, Execution, Deployment $\to$ Build Tools $\to$ Gradle $\to$ Gradle JDK**.
   - Sync Gradle project files.
6. **Launch & Test:**
   - Select your physical phone in the device target dropdown.
   - Tap **Run 'app'** (`Shift + F10`).
   - Navigate to **Screen F (Capture Session)**, enter `http://<YOUR_LAN_IP>:8000/`, and tap "Apply URL".

---

## 10. Features Implemented vs. Future Placeholders

| Architectural Feature | Status in Task 13 | Future Scope / Next Phase |
| :--- | :--- | :--- |
| **Gradle Multi-Project Build** | **Complete** (AGP 8.4.1, Kotlin 1.9.23, Compose BOM) | Task 14–16 dependency maintenance |
| **Material 3 Navigation & Theme** | **Complete** (6 screens, AppNavHost, custom theme) | Visual refinement |
| **Permission Management** | **Complete** (JIT contracts for Fine Location & Camera) | Background notification permissions (if needed) |
| **Wi-Fi Scanner Boundary** | **Complete** (Interfaces, Throttling-aware Android scanner, Mock scanner) | Task 14: continuous surveyor batch collector |
| **IMU Sensor Streamer** | **Complete** (Lifecycle-safe 3-axis provider, Mock provider) | Task 15: PDR step-counting filter |
| **Manual Photo Shutter** | **Complete** (ActivityResultContract, EXIF/blur UI) | Task 16: CameraX preview & on-device face detector |
| **Retrofit Network Layer** | **Complete** (IndoorApiService, ApiClient, 6 DTOs) | Phase 8: live connection to FastAPI endpoints |
| **Local Unit Test Suite** | **Complete** (6 test classes covering ViewModels, Mocks, and Serialization) | Instrumented Espresso & Compose UI tests |
