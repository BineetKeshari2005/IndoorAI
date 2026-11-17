# IndoorAI — Android User-Initiated Photo Capture & Privacy Sanitizer

This document details the architectural design, implementation, and privacy safeguards of **Task 16: User-Initiated Photo Capture & Privacy Sanitizer** in the IndoorAI Android client.

---

## 1. Architectural Overview & Component Design

The photo capture subsystem is designed around strict privacy, explicit user initiation, offline execution, and integration with the active survey session telemetry.

```
                           +--------------------------------+
                           |       ImageCaptureScreen       |
                           |   (Jetpack Compose Material 3) |
                           +---------------+----------------+
                                           |
                                  ViewModel Actions
                                           v
                           +--------------------------------+
                           |     ImageCaptureViewModel      |
                           +---------------+----------------+
                                           |
        +----------------------------------+----------------------------------+
        |                                  |                                  |
        v                                  v                                  v
+---------------+                +--------------------+             +--------------------+
| CameraManager |                |   Privacy Engine   |             | PhotoStorageManager|
+-------+-------+                +---------+----------+             +---------+----------+
        |                                  |                                  |
   +----+----+                    +--------+--------+                         |
   |         |                    |                 |                         v
   v         v                    v                 v                 Local App Storage:
CameraX     Mock          ExifInterface     OnDeviceFaceDetector     context.filesDir/
Hardware  Fallback       (GPS Stripping)   (android.media.FaceDetector)survey_photos/
```

### Component Responsibilities

1. **`CameraManager` (`CameraManager.kt`, `AndroidCameraXManager.kt`, `MockCameraManager.kt`)**:
   - Hardware abstraction isolating CameraX APIs (`Preview`, `ImageCapture`, `ProcessCameraProvider`).
   - Guarantees preview and capture execute **only** upon explicit user request.
   - Provides `MockCameraManager` for emulator environments lacking physical camera hardware.

2. **`ImagePrivacySanitizer` (`ImagePrivacySanitizer.kt`)**:
   - Strips all EXIF GPS coordinates and geolocation tags.
   - Removes identifying hardware and software tags (device model, manufacturer, serial numbers, camera software).
   - Normalizes image rotation physically into pixel orientation.
   - Re-encodes cleanly to JPEG via atomic temporary file operations.

3. **`OnDeviceFaceDetector` & `FaceBlurApplicator` (`AndroidMediaFaceDetector.kt`, `FaceBlurApplicator.kt`)**:
   - Evaluates images for human faces using Android framework's native `android.media.FaceDetector`.
   - Runs 100% locally on-device with zero network requests or cloud AI dependencies.
   - Applies pixelation over detected face regions before saving.

4. **`PhotoStorageManager` (`PhotoStorageManager.kt`)**:
   - Persists sanitized images and metadata inside application-private storage (`context.filesDir/survey_photos/`).
   - Maintains synchronized JSON (`survey_photos_index.json`) and CSV (`survey_photos_index.csv`) indices.
   - Supports single-photo deletion and batch export without cloud storage requirements.

5. **`ManageCaptureSessionUseCase` (`ManageCaptureSessionUseCase.kt`)**:
   - Links captured photos to the active survey session (`sessionId`).
   - Increments `photoCount` telemetry in `CaptureSession`.

---

## 2. Privacy & Anti-Surveillance Guarantees

IndoorAI enforces a strict privacy contract for camera operations:

| Privacy Principle | Implementation Guarantee |
| :--- | :--- |
| **No Background Capture** | The camera preview is **never** started on app launch or screen open. It initializes strictly when the user taps "Start Camera Preview". When navigating away, the camera lifecycle unbinds immediately. |
| **No Covert Photography** | The shutter fires only when the user taps the capture button. There is no automated interval timer or background trigger. |
| **Zero EXIF Geolocation** | All GPS metadata tags (`TAG_GPS_LATITUDE`, `TAG_GPS_LONGITUDE`, `TAG_GPS_ALTITUDE`, `TAG_GPS_TIMESTAMP`, etc.) are permanently removed. |
| **Hardware Privacy** | Device identifiers (`TAG_MAKE`, `TAG_MODEL`, `TAG_SOFTWARE`, `TAG_SERIAL_NUMBER`) are purged from image headers. |
| **Local Face Obfuscation** | Face detection operates purely in memory on the handset. Bystander faces are blurred using on-device pixelation. |
| **No Facial Recognition** | No biometric templates, embeddings, or identity tracking are generated or retained. |
| **Review & Consent** | Every capture enters a review stage where the surveyor inspects the image, verifies detected face count, toggles blur, and enters landmark labels before saving. |
| **Safe Failure Handling** | If sanitization fails, the destination file is not written and the raw frame is not persisted. |

> [!IMPORTANT]
> **Face Detection Limitations Disclaimer**: On-device face detection uses algorithmic landmark geometry (`android.media.FaceDetector`). While effective for upright and profile faces in typical indoor lighting, it is not guaranteed to catch 100% of bystanders (e.g. extreme angles, occlusions, or low light). Surveyors are reminded to review photographs before saving.

---

## 3. Indoor Landmark Metadata Schema

Manual landmark annotations are explicitly segregated from automated sensor observations:

```kotlin
data class CapturedPhotoMetadata(
    val photoId: String,               // e.g. "IMG-8C31D8F1"
    val sessionId: String,             // e.g. "SES-49B2A1C3"
    val timestampMs: Long,             // Epoch timestamp
    val filename: String,              // e.g. "PHOTO_20261117_153000_sanitized.jpg"
    val filePath: String,              // Local absolute file path
    val building: String,              // e.g. "TI-Engineering"
    val floor: String,                 // e.g. "2"
    val room: String,                  // e.g. "TI-204"
    val landmarkType: LandmarkType,    // Enum: CORRIDOR, DOOR, STAIRCASE, etc.
    val description: String,           // e.g. "East wing door sign with room number"
    val notes: String,                 // Surveyor notes (lighting, walking direction)
    val isExifSanitized: Boolean,      // true
    val facesDetectedCount: Int,       // Count of detected faces
    val faceBlurApplied: Boolean,      // true if pixelation filter was applied
    val faceDetectionStatus: FaceDetectionStatus, // SUCCEEDED, NO_FACES_DETECTED, UNAVAILABLE
    val fileSizeBytes: Long,           // Byte size on disk
    val imageWidth: Int,               // Pixel width
    val imageHeight: Int,              // Pixel height
    val isMock: Boolean                // true if generated via MockCameraManager
)
```

### Supported Landmark Types (`LandmarkType.kt`)
- `CORRIDOR`: Hallways, junctions, corridors, and pathways.
- `DOOR`: Office, classroom, and lab doors.
- `STAIRCASE`: Stairwell entrances and flights.
- `ELEVATOR`: Elevator banks and call buttons.
- `ROOM_SIGN`: Door plaques, overhead directional signs, and directory maps.
- `ENTRANCE`: Building main/side exterior entrances.
- `EXIT`: Emergency and standard exits.
- `LOBBY`: Atriums, reception desks, and waiting areas.
- `OTHER`: Unclassified indoor landmarks.

---

## 4. Export Formats & Data Exchange

Survey photos and metadata can be exported directly from the handset without internet or server connectivity:

### JSON Export Format (`exportBatchJson`)
```json
{
  "batch_size": 1,
  "exported_at_ms": 1716000000000,
  "photos": [
    {
      "photo_id": "IMG-8C31D8F1",
      "session_id": "SES-49B2A1C3",
      "timestamp_ms": 1716000000000,
      "timestamp_iso": "2024-05-18T02:40:00.000Z",
      "filename": "PHOTO_20261117_153000_sanitized.jpg",
      "file_path": "/data/user/0/ai.indoor.app/files/survey_photos/PHOTO_20261117_153000_sanitized.jpg",
      "building": "TI-Engineering",
      "floor": "2",
      "room": "Room 204",
      "landmark_type": "CORRIDOR",
      "landmark_display": "Corridor",
      "description": "Corridor sign TI-204",
      "notes": "Surveyor walking east",
      "is_exif_sanitized": true,
      "faces_detected_count": 0,
      "face_blur_applied": false,
      "face_detection_status": "NO_FACES_DETECTED",
      "file_size_bytes": 184320,
      "image_width": 1920,
      "image_height": 1080,
      "is_mock": false
    }
  ]
}
```

### CSV Export Format (`exportBatchCsv`)
Adheres strictly to RFC 4180 standard escaping:
```csv
photo_id,session_id,timestamp_ms,timestamp_iso,filename,building,floor,room,landmark_type,description,notes,exif_sanitized,faces_detected,face_blur_applied,face_status,file_size_bytes,width,height,is_mock
IMG-8C31D8F1,SES-49B2A1C3,1716000000000,2024-05-18T02:40:00.000Z,PHOTO_20261117_153000_sanitized.jpg,TI-Engineering,2,Room 204,CORRIDOR,Corridor sign TI-204,Surveyor walking east,true,0,false,NO_FACES_DETECTED,184320,1920,1080,false
```

---

## 5. Physical Device Testing & Verification Guide

### Physical Android Device Run Instructions
1. Enable **Developer Options** and **USB Debugging** on the target handset.
2. Connect the phone via USB and verify with `adb devices`.
3. Open `android/` in Android Studio (Giraffe or Jellyfish) or build via Gradle:
   ```bash
   cd android
   ./gradlew installDebug
   ```
4. Verify CameraX Preview:
   - Launch IndoorAI. Observe that the camera is completely dormant (preview is `INACTIVE`).
   - Tap **Camera** bottom navigation tab. Note the "Zero-Background Privacy Guarantee" card.
   - Tap **Start Camera Preview**. Grant the runtime camera permission prompt.
   - Confirm that the live viewfinder streams with the alignment crosshair overlay.
   - Point at a room sign or door and press **Capture Landmark Frame**.
   - Review image in dialog: confirm landmark fields, face detection status badge, and face blur switch.
   - Tap **Confirm & Save**. Confirm that the photo card appears in the catalog, EXIF is stripped, and session counter increments.
   - Tap **Stop Camera** or navigate away: verify that the camera indicator LED turns off immediately.

### Headless / Emulator Testing (Mock Camera Provider)
On virtual devices without camera hardware:
- The system automatically detects missing camera hardware and displays the "No Camera Hardware Detected" state.
- Tapping **Test Mock Capture** invokes `MockCameraManager`, generating synthetic hallway benchmark frames to test the full review, EXIF sanitization, landmark entry, and CSV/JSON export workflow.

---

## 6. Unit Testing & Verification Matrix

The implementation is verified across domain models, sanitization algorithms, and ViewModel state transitions:

| Test File | Test Case | Target Behavior |
| :--- | :--- | :--- |
| `CapturedPhotoExportTest.kt` | `testJsonSerializationContainsAllFields` | Verifies all 19 metadata fields are serialized in JSON. |
| `CapturedPhotoExportTest.kt` | `testJsonSpecialCharactersEscaping` | Validates escaping of quotes, newlines, tabs, and backslashes in JSON strings. |
| `CapturedPhotoExportTest.kt` | `testCsvHeaderAndRowFormatting` | Asserts standard CSV header and column alignment. |
| `CapturedPhotoExportTest.kt` | `testCsvEscapingWithCommasAndQuotes` | Ensures RFC 4180 quotes are applied when commas or quotes occur in text. |
| `CapturedPhotoExportTest.kt` | `testBatchExportJsonAndCsv` | Validates batch JSON and CSV structure with multiple records. |
| `CapturedPhotoExportTest.kt` | `testDisplayLocationFormatting` | Tests formatted human-readable location string fallbacks. |
| `CapturedPhotoExportTest.kt` | `testLandmarkTypeFromStringFallback` | Tests case-insensitive mapping and fallback to `OTHER`. |
| `ImagePrivacySanitizerTest.kt` | `testFaceBoundsCalculations` | Verifies face width, height, and confidence calculation. |
| `ImagePrivacySanitizerTest.kt` | `testFaceBoundsZeroOrInvertedDimensions` | Tests safe zero clamping for invalid face coordinates. |
| `ImagePrivacySanitizerTest.kt` | `testSanitizerFailsGracefullyWhenFileDoesNotExist` | Ensures sanitization fails safely if raw file is missing. |
| `ImagePrivacySanitizerTest.kt` | `testSanitizerFailsGracefullyWhenFileIsEmpty` | Tests safe failure on zero-byte files without writing destination file. |
| `ImagePrivacySanitizerTest.kt` | `testFaceDetectionStatusesDescriptions` | Asserts human-readable status descriptions. |
| `ImageCaptureViewModelTest.kt` | `testInitialStateCameraIsStrictlyInactive` | Asserts camera is strictly `INACTIVE` at launch. |
| `ImageCaptureViewModelTest.kt` | `testStartPreviewRequiresPermission` | Verifies transition to `PERMISSION_REQUIRED` if permission is denied. |
| `ImageCaptureViewModelTest.kt` | `testCameraUnavailableTransitionsCorrectly` | Verifies fallback when camera hardware is absent. |
| `ImageCaptureViewModelTest.kt` | `testStopPreviewReleasesCamera` | Ensures `stopPreview` unbinds resources and resets state. |
| `ImageCaptureViewModelTest.kt` | `testUpdateDraftAnnotations` | Tests live updating of landmark annotation draft fields. |
| `ImageCaptureViewModelTest.kt` | `testToggleFaceBlur` | Verifies toggling privacy blur setting. |
| `ImageCaptureViewModelTest.kt` | `testRetakePhotoDeletesTempFileAndResetsToIdle` | Confirms retake removes temporary file and resets flow to `Idle`. |
| `ImageCaptureViewModelTest.kt` | `testResetCaptureState` | Tests reset to `Idle`. |
