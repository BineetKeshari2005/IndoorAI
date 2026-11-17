# IndoorAI — Real Android Wi-Fi Fingerprint Collector Specification

## 1. Executive Summary

Task 14 implements **real Wi-Fi fingerprint collection** within the IndoorAI Android application. While Task 13 established the project scaffolding, navigation, and hardware boundary interfaces, Task 14 implements:

1. **Production Wi-Fi Scanning:** Interacting directly with Android's `WifiManager` via user-initiated radio sweeps and asynchronous `BroadcastReceiver` callbacks.
2. **Strict Permission & OS Preconditions:** Comprehensive handling of `ACCESS_FINE_LOCATION`, system Location Services (GPS), Wi-Fi radio status, and Android 9+ rate throttling.
3. **Freshness & Throttling Detection:** Differentiating between fresh radio sweep observations (`EXTRA_RESULTS_UPDATED == true`) and cached/throttled results.
4. **Deduplication & Channel Derivation:** Filtering duplicate BSSIDs by selecting the strongest RSSI observation and computing standard 802.11 channel numbers across 2.4 GHz, 5.0 GHz, and 6.0 GHz bands.
5. **Local Ground-Truth Dataset Capture:** Coupling ambient 802.11 observations with manual surveyor ground-truth annotations (Building, Floor, Room, Notes) and exporting to JSON and CSV.
6. **Critical UJIIndoorLoc Calibration Invariant:** Strict prevention of arbitrary or fabricated mapping between real-world BSSIDs and the UJI campus dataset's abstract `WAP001`–`WAP520` columns.

```
                           +-----------------------------------------------+
                           |           User Taps "Start Scan"              |
                           +-----------------------------------------------+
                                                  |
                                                  v
                           +-----------------------------------------------+
                           |           Precondition Validation             |
                           |   1. ACCESS_FINE_LOCATION granted?            |
                           |   2. Wi-Fi radio enabled?                     |
                           |   3. System Location Services (GPS) active?   |
                           +-----------------------------------------------+
                                                  | Passed
                                                  v
                           +-----------------------------------------------+
                           |          WifiManager.startScan()              |
                           +-----------------------------------------------+
                                     /                           \
                           Accepted /                             \ Throttled / Rejected
                                   v                               v
            +-------------------------------+    +----------------------------------+
            | BroadcastReceiver on          |    | Return Cached Results            |
            | SCAN_RESULTS_AVAILABLE_ACTION |    | State: WifiScanState.Throttled   |
            +-------------------------------+    +----------------------------------+
                           |
                           v
            +-------------------------------+
            | Inspect EXTRA_RESULTS_UPDATED |
            | true  -> Fresh Sweep          |
            | false -> Cached / Throttled   |
            +-------------------------------+
                           |
                           v
            +---------------------------------------------------------------+
            | Map & Deduplicate Raw ScanResults                             |
            | - Retain strongest RSSI per BSSID                             |
            | - Derive 802.11 Channel from Frequency                        |
            | - Convert boot elapsed time to epoch wall-clock ms            |
            +---------------------------------------------------------------+
                                   |
                                   v
            +---------------------------------------------------------------+
            | UJIIndoorLoc Compatibility Boundary                           |
            | (WifiFingerprintModelAdapter)                                 |
            | - Real BSSIDs != WAP001..WAP520                               |
            | - Refuse arbitrary mapping; prompt for calibration survey     |
            +---------------------------------------------------------------+
                                   |
                                   v
            +---------------------------------------------------------------+
            | Local Capture & Ground-Truth Annotation                       |
            | (Manual Building, Floor, Room -> Export JSON / CSV)           |
            +---------------------------------------------------------------+
```

---

## 2. Android Permissions & System API Requirements

### Permission Specification

| Android Permission | Protection Level | Purpose & Android Version Constraints |
| :--- | :--- | :--- |
| `android.permission.ACCESS_FINE_LOCATION` | Dangerous (Runtime) | **Mandatory.** Android considers nearby Wi-Fi beacon signals to be spatial identifiers. Since Android 9 (API 28), calling `WifiManager.getScanResults()` requires `ACCESS_FINE_LOCATION`. |
| `android.permission.ACCESS_WIFI_STATE` | Normal (Install-time) | Required to query Wi-Fi status and inspect connection parameters. |
| `android.permission.CHANGE_WIFI_STATE` | Normal (Install-time) | Required to initiate active radio sweeps via `WifiManager.startScan()`. |
| `android.permission.INTERNET` | Normal (Install-time) | For future backend telemetry and model calibration synchronization. |

### Why `NEARBY_WIFI_DEVICES` Is NOT Used for Wi-Fi Scanning
In Android 13 (API 33), Google introduced `NEARBY_WIFI_DEVICES` with an optional `neverForLocation` flag. However:
- `neverForLocation` is strictly for P2P protocols (Wi-Fi Direct, Local-Only Hotspot, Nan/Aware) where the application promises **never** to derive physical location.
- An Indoor Positioning System (IPS) explicitly processes beacon signal strengths to derive spatial coordinates.
- Using `neverForLocation` legally and technically prohibits the OS from returning location-bearing scan results.
- Therefore, `ACCESS_FINE_LOCATION` remains the legally and technically correct runtime permission for indoor Wi-Fi scanning.

### System Preconditions Handled in `AndroidWifiScanner.kt`
1. **Wi-Fi Hardware Availability:** Checks if `Context.getSystemService(Context.WIFI_SERVICE)` returns non-null. If null, emits `WifiScanState.Error`.
2. **Wi-Fi Radio Enabled:** Verifies `wifiManager.isWifiEnabled`. If disabled, emits `WifiScanState.WifiDisabled` with guidance to enable Wi-Fi in Quick Settings.
3. **Location Services Enabled:** On Android 9+, even with runtime permission granted, system-wide Location Services (GPS) must be turned on. Verifies via `LocationManager.isLocationEnabled` (API 28+) or provider checks. If disabled, emits `WifiScanState.LocationServicesRequired`.
4. **Runtime Permission Granted:** Verifies `ContextCompat.checkSelfPermission`. If missing, emits `WifiScanState.PermissionRequired(listOf(ACCESS_FINE_LOCATION))` and launches an interactive permission prompt.

---

## 3. Scan Throttling & Freshness Handling

### Android OS Throttling Constraints
Starting in Android 9 (API 28) and continuing through Android 14+:
- **Foreground Limits:** A foreground application may trigger `startScan()` at most **4 times within a 2-minute rolling window**.
- **Background Limits:** Background applications are restricted to **1 scan every 30 minutes**.
- **OS Rejection Behavior:** When throttled, `wifiManager.startScan()` immediately returns `false` without triggering a radio sweep or dispatching a broadcast.

### Freshness Identification (`EXTRA_RESULTS_UPDATED`)
When `startScan()` succeeds, Android executes an asynchronous radio sweep and broadcasts `WifiManager.SCAN_RESULTS_AVAILABLE_ACTION`.
The intent contains the boolean extra:
```kotlin
val isFresh = intent.getBooleanExtra(WifiManager.EXTRA_RESULTS_UPDATED, false)
```
- `isFresh == true`: Indicates a fresh hardware radio sweep completed across all supported frequencies.
- `isFresh == false`: Indicates the OS did not perform a new radio sweep and instead served cached scan results from a previous cycle.

`AndroidWifiScanner` maps this extra directly into each `AccessPoint.isFresh` and `WifiFingerprint.isFresh` domain attribute. The UI renders green `FRESH` and amber `CACHED` badge indicators accordingly.

---

## 4. Deduplication & Channel Derivation

### BSSID Deduplication
Multi-band routers (e.g. 2.4 GHz + 5 GHz mesh systems) or repeaters often emit beacons on multiple channels with the same BSSID or produce multiple scan result records in rapid succession. `AndroidWifiScanner.mapAndDeduplicateScanResults` groups incoming records by lowercase BSSID:
```kotlin
rawResults
    .filter { !it.BSSID.isNullOrBlank() }
    .groupBy { it.BSSID.lowercase() }
    .map { (_, group) ->
        val best = group.maxByOrNull { it.level } ?: group.first()
        // Map to AccessPoint...
    }
    .sortedByDescending { it.rssi }
```
This guarantees that each physical BSSID appears at most once in a fingerprint, retaining its strongest received signal strength ($RSSI_{\max}$).

### 802.11 Channel Number Derivation
Given an operating frequency $f$ in MHz, standard channel numbers are calculated deterministically:

$$\text{Channel}(f) = \begin{cases} 
14 & \text{if } f = 2484 \text{ MHz} \\
\frac{f - 2407}{5} & \text{if } f \in [2412, 2472] \text{ MHz (2.4 GHz)} \\
\frac{f - 5000}{5} & \text{if } f \in [5000, 5895] \text{ MHz (5.0 GHz)} \\
\frac{f - 5950}{5} & \text{if } f \in [5925, 7125] \text{ MHz (6.0 GHz / Wi-Fi 6E)} \\
0 & \text{otherwise}
\end{cases}$$

Example verified mappings:
- $2412\text{ MHz} \to \text{Channel 1}$
- $2437\text{ MHz} \to \text{Channel 6}$
- $5180\text{ MHz} \to \text{Channel 36}$
- $5240\text{ MHz} \to \text{Channel 48}$
- $5745\text{ MHz} \to \text{Channel 149}$
- $5955\text{ MHz} \to \text{Channel 1 (6 GHz)}$

---

## 5. Critical Invariant: Real BSSIDs vs UJIIndoorLoc WAP Identifiers

> [!WARNING]
> **Strict Scientific Invariant:** Real-world Wi-Fi scans return physical MAC addresses (BSSIDs such as `00:14:bf:b1:97:8a`), whereas the UJIIndoorLoc dataset was curated with 520 abstract feature columns: `WAP001` through `WAP520` corresponding to specific access points deployed across the Jaume I University campus in Spain in 2013.

### Why Direct Model Inference Is Strictly Blocked
1. **No Intrinsic Correspondence:** A real-world access point in an office or home has zero semantic relationship to `WAP042` on the UJI campus.
2. **Prevention of Fabricated Localization:** Feeding uncalibrated real-world signals into the Multitask MLP, Random Forest, or k-NN models would result in arbitrary coordinates on the UJI campus, producing meaningless outputs.
3. **Architectural Boundary:** The application introduces `WifiFingerprintModelAdapter` ([WifiFingerprintModelAdapter.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/adapter/WifiFingerprintModelAdapter.kt)). When an uncalibrated fingerprint is evaluated, it returns `ModelAdaptationResult.Uncalibrated`, and the UI displays a prominent warning notice.
4. **Calibration Phase Requirement:** To localize in a new building, an empirical site survey must first be collected (using the export features in Task 14) and mapped to a new spatial index or feature coordinate space.

---

## 6. Ground-Truth Capture & Export Schema

To facilitate building empirical calibration datasets, Task 14 introduces the `CapturedFingerprint` model ([CapturedFingerprint.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/models/CapturedFingerprint.kt)).

### Data Model Attributes
- `captureId`: Unique UUID for the observation.
- `sessionId`: Unique UUID for the survey session.
- `timestamp`: Wall-clock epoch timestamp in milliseconds.
- `fingerprint`: `WifiFingerprint` containing the list of observed `AccessPoint` records.
- `manualLabel`: `ManualLocationLabel` entered by the surveyor:
  - `building`: String (e.g. `"Engineering-North"`)
  - `floor`: String (e.g. `"2"`)
  - `room`: String (e.g. `"Room 204"`)
  - `notes`: String (e.g. `"Facing west corridor"`)
- `deviceMetadata`: Device manufacturer, model, and Android SDK level.
- `isMock`: Boolean strictly indicating synthetic demo data vs real hardware.

### CSV Export Specification
Exported CSV files use standard RFC 4180 formatting with the following header:
```csv
capture_id,session_id,timestamp_epoch_ms,is_fresh,is_mock,bssid,ssid,rssi_dbm,frequency_mhz,channel,band,manual_building,manual_floor,manual_room,manual_notes
```

#### Example CSV Output
```csv
capture_id,session_id,timestamp_epoch_ms,is_fresh,is_mock,bssid,ssid,rssi_dbm,frequency_mhz,channel,band,manual_building,manual_floor,manual_room,manual_notes
CAP-001,SES-001,1700000000000,true,false,00:14:bf:b1:97:8a,eduroam,-48,5180,36,5.0 GHz,Engineering-North,2,Lab-204,Survey point near entrance
CAP-001,SES-001,1700000000000,true,false,00:14:bf:b1:97:8b,Campus-Staff,-52,5240,48,5.0 GHz,Engineering-North,2,Lab-204,Survey point near entrance
CAP-001,SES-001,1700000000000,true,false,70:69:79:8c:12:4a,TI-Lab-AP04,-64,5200,40,5.0 GHz,Engineering-North,2,Lab-204,Survey point near entrance
```

### JSON Export Specification
```json
[
  {
    "captureId": "CAP-001",
    "sessionId": "SES-001",
    "timestamp": 1700000000000,
    "fingerprint": {
      "accessPoints": [
        {
          "bssid": "00:14:bf:b1:97:8a",
          "ssid": "eduroam",
          "rssi": -48,
          "frequency": 5180,
          "channel": 36,
          "timestamp": 1700000000000,
          "isFresh": true,
          "isMock": false
        }
      ],
      "timestamp": 1700000000000,
      "scanDurationMs": 1200,
      "isFresh": true,
      "isMock": false,
      "scanId": "SCAN-001"
    },
    "manualLabel": {
      "building": "Engineering-North",
      "floor": "2",
      "room": "Lab-204",
      "notes": "Survey point near entrance"
    },
    "deviceMetadata": {
      "manufacturer": "Google",
      "model": "Pixel 8",
      "sdkInt": 34,
      "appVersion": "1.0.0"
    },
    "isMock": false
  }
]
```

---

## 7. Backend Ingestion Contract (Future Phase 8)

Inspection of the existing backend confirmed that FastAPI endpoints are not yet live in Phase 4. When the backend ingestion service is activated in Phase 8, the client will synchronize captured fingerprints using the following contract:

- **Endpoint:** `POST /api/v1/wifi/capture`
- **Headers:** `Content-Type: application/json`
- **Request Payload:** Identical to the `CapturedFingerprint` JSON schema above.
- **Expected Response:**
  ```json
  {
    "status": "ingested",
    "capture_id": "CAP-001",
    "records_recorded": 14,
    "timestamp": 1700000001000
  }
  ```

---

## 8. Physical Android Device Testing Guide

To test real Wi-Fi scanning on a physical Android handset:
1. **Enable Developer Options:** Settings $\to$ About Phone $\to$ Tap "Build Number" 7 times.
2. **Enable USB Debugging:** Developer Options $\to$ Enable "USB Debugging".
3. **(Optional) Disable Wi-Fi Throttling:** Developer Options $\to$ Networking $\to$ Turn OFF "Wi-Fi scan throttling". This allows triggering scans more frequently than the 4 scans/2-minute OS limit during testing.
4. **Enable Hardware Radios:**
   - Turn **ON** Wi-Fi in Quick Settings.
   - Turn **ON** Location (GPS) in Quick Settings.
5. **Install & Run Application:**
   - Open `/android` in Android Studio Hedgehog or newer.
   - Run `app` onto the connected handset.
6. **Execute Verification Flow:**
   - Navigate to **Screen C (Wi-Fi)**.
   - Tap **"Start Scan"**.
   - If prompted, tap **"Grant Location Permission"**.
   - Confirm that detected APs display real local SSIDs, BSSIDs, signal meters (dBm), channel frequencies, and green `FRESH` badges.
   - Tap **"Edit Labels"**, enter building/floor/room labels, and tap **"Save"**.
   - Tap **"JSON"** or **"CSV"** to inspect and copy the serialized dataset export.

---

## 9. Test Suite & Verification Results

### Unit Test Summary
- [MockWifiScannerTest.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/test/java/ai/indoor/app/wifi/MockWifiScannerTest.kt): Verifies state flow transitions, synthetic AP counts, signal jitter, and frequency bands.
- [AndroidWifiScannerMappingTest.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/test/java/ai/indoor/app/wifi/AndroidWifiScannerMappingTest.kt): Verifies 802.11 channel derivation across 2.4 GHz, 5.0 GHz, and 6.0 GHz bands, signal level categorization, and wall-clock timestamp calculations.
- [WifiFingerprintModelAdapterTest.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/test/java/ai/indoor/app/domain/adapter/WifiFingerprintModelAdapterTest.kt): Verifies that real-world uncalibrated scans strictly refuse arbitrary mapping to UJI WAP columns, and that calibrated mappings correctly populate the 520-float feature vector.
- [CapturedFingerprintExportTest.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/test/java/ai/indoor/app/domain/models/CapturedFingerprintExportTest.kt): Verifies JSON serialization, CSV generation with header integrity, comma/quote escaping, and strict preservation of `isMock` flags.
- [WifiScanViewModelTest.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/test/java/ai/indoor/app/viewmodel/WifiScanViewModelTest.kt): Verifies ViewModel state transitions, scan triggers, ground-truth label updates, export previews, and error/permission/throttling state handling.

### Regression Verification
- **Python ML Test Suite:** **105 / 105 tests passed** across all Phase 1–3 models and pipelines. Zero regressions.
