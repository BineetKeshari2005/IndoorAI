# Multimodal Dataset Explorer & Visualization Dashboard

## 1. Overview & Architectural Role

The **Multimodal Dataset Explorer & Visualization Dashboard** represents **Task 19** of the IndoorAI system. It extends the unified Streamlit frontend application (`frontend/app.py`) with deep exploration, diagnostics, visualization, and management capabilities for multimodal survey datasets ingested via the Task 18 FastAPI backend.

```
       +---------------------------------------------------------------+
       |             IndoorAI Unified Streamlit Dashboard              |
       |                      (frontend/app.py)                        |
       +-------------------------------+-------------------------------+
                                       |
                   Workspace Selector in Sidebar
                  /                             \
                 v                               v
    [Multimodal Survey Explorer]     [UJIIndoorLoc Benchmark & Inference]
          (Tasks 17-19)                         (Tasks 07-12)
                 |                                    |
      +----------+----------+                 +-------+-------+
      |  DatasetApiClient   |                 | Models & Eval |
      | (frontend/api_client|                 |  Pipelines    |
      +----------+----------+                 +---------------+
                 | HTTP REST
                 v
    +-----------------------------------------------------------------+
    |                IndoorAI FastAPI Backend Service                 |
    |                   (backend/app/main.py)                         |
    +-----------------------------------------------------------------+
    |  /health                                                        |
    |  /api/datasets                                                  |
    |  /api/datasets/upload (multipart ZIP)                           |
    |  /api/datasets/{id}                                             |
    |  /api/datasets/{id}/quality                                     |
    |  /api/datasets/{id}/validate                                    |
    |  /api/datasets/{id}/observations/{wifi|sensors|photos|annot...} |
    |  /api/datasets/{id}/images/{filename}                           |
    |  DELETE /api/datasets/{id}                                      |
    +-----------------------------------------------------------------+
```

### Key Design Goals
1. **Preserve Legacy UJIIndoorLoc Benchmark UI**: 100% of Tasks 07–12 (model loading, inference, coordinate regression, floorplan visualizations, and evaluation comparisons) remain intact, accessible via the top-level sidebar workspace selector.
2. **Modular View Component Architecture**: Each visualization and diagnostic screen is isolated in `frontend/components/`, ensuring single-responsibility design, clean maintainability, and easy testability.
3. **Robust Backend API Client**: All backend HTTP operations are abstracted via `frontend/api_client.py` (`DatasetApiClient`) with graceful degradation, connection health monitoring, request timeout protection, and testable dependency injection.
4. **Strict Privacy Invariant Enforcement**: Survey photos are only rendered if client-side EXIF sanitization (`is_exif_sanitized == true`) is cryptographically verified by the backend. Raw photos without verified sanitization return HTTP 403 Forbidden.
5. **Clear Audit vs. Validation Separation**: Structural validation status (`PASSED`, `WARNINGS`, `FAILED`) is kept clearly distinct from human/model visual privacy audits (`PENDING_AUDIT`, `APPROVED`, `REJECTED`).
6. **No Data Fabrication**: Sensor timestamps, sampling gaps, and missing observations are displayed truthfully without silent interpolation, synthetic signal generation, or geolocation guessing from BSSIDs.

---

## 2. Component Architecture

The frontend is structured into modular components:

```
frontend/
├── app.py                       # Main Streamlit application & navigation router
├── api_client.py                # DatasetApiClient with httpx & error mapping
└── components/
    ├── __init__.py              # Component package exports
    ├── overview.py              # Catalog KPI cards, registry table, selector
    ├── upload.py                # Ingestion uploader, dry-run mode, report badge
    ├── quality_view.py          # 6-tab quality & integrity diagnostic dashboard
    ├── wifi_view.py             # Sweep selector, AP table, RSSI histogram, Top-N
    ├── sensor_view.py           # Tri-axial IMU/magnetometer Plotly time-series
    ├── timeline_view.py         # Multi-modal concurrency Gantt, clock drift, feed
    ├── photo_view.py            # Sanitized photo viewer, metadata, annotations
    ├── comparison_view.py       # Cross-dataset comparative KPI & density tables
    └── management_view.py       # Revalidation trigger, log view, confirmed delete
```

### Module Responsibilities

| Component | Responsibility | Primary APIs Used |
| :--- | :--- | :--- |
| `overview.py` | Renders catalog summary metrics, session cards, active dataset dropdown, and quality disclaimer | `GET /api/datasets` |
| `upload.py` | Handles ZIP drag-and-drop, client-side size pre-flight check, dry-run validation toggle, and live upload progress | `POST /api/datasets/upload` |
| `quality_view.py` | 6-domain QA inspection: Integrity, Wi-Fi, IMU, Photos, Annotations, and Diagnostic Logs | `GET /api/datasets/{id}/quality` |
| `wifi_view.py` | Explores Wi-Fi scans per sweep, AP frequency, BSSID distribution, and RSSI spread | `GET /api/datasets/{id}/observations/wifi` |
| `sensor_view.py` | Visualizes Accelerometer, Gyroscope, Magnetometer, and Rotation Vector with axis filtering and zoom/pan | `GET /api/datasets/{id}/observations/sensors` |
| `timeline_view.py` | Multi-modality Gantt timeline, concurrent overlap analysis, clock uncertainty metrics, chronological stream | `GET /api/datasets/{id}/observations/{modality}` |
| `photo_view.py` | Renders sanitized landmark photos, verification badges, capture timestamps, and ground-truth labels | `GET /api/datasets/{id}/images/{fn}`, `GET .../photos` |
| `comparison_view.py` | Side-by-side comparison of 2–5 datasets across AP density, IMU sampling rate, and total observations | `GET /api/datasets` & `GET .../quality` |
| `management_view.py` | Revalidates extracted sessions, displays raw error logs, and handles confirmed deletion | `POST .../validate`, `DELETE .../datasets/{id}` |

---

## 3. Backend Integration & Observation APIs

To support interactive exploration without overloading browser memory on multi-megabyte datasets, Task 19 extended the FastAPI backend (`backend/app/api/routes/datasets.py`) with two dedicated endpoints:

### 3.1 Paged Observation Retrieval
- **Route**: `GET /api/datasets/{dataset_id}/observations/{modality}`
- **Supported Modalities**: `wifi`, `sensors`, `photos`, `annotations`
- **Query Parameters**:
  - `limit` (int, default: 500, max: 2000): Pagination window size.
  - `offset` (int, default: 0): Starting row offset.
- **Behavior**: Streams rows directly from extracted CSV files (`wifi_scans.csv`, `sensor_readings.csv`, `photo_metadata.csv`, `annotations.csv`), mapping string representations into typed dictionaries and returning total available count.

### 3.2 Privacy-Sanitized Image Streaming
- **Route**: `GET /api/datasets/{dataset_id}/images/{filename}`
- **Security Protections**:
  - **Directory Traversal Defense**: Resolves filename strictly against `<extracted>/photos/`, rejecting path separators (`../`, `..\\`).
  - **Sanitization Invariant Verification**: Checks extracted `photo_metadata.csv` to ensure `is_exif_sanitized.lower() == "true"` for the requested file.
  - **Status Code Mapping**:
    - `400 Bad Request`: Malformed or invalid filename.
    - `403 Forbidden`: Image exists but `is_exif_sanitized` is false or unverified.
    - `404 Not Found`: Image does not exist in dataset storage.
    - `200 OK`: Serves JPEG image with `Content-Type: image/jpeg`.

---

## 4. Privacy & Data Integrity Rules

1. **Client Sanitization vs. Server Visual Audit**:
   - The dashboard explicitly distinguishes between **structural/manifest verification** (`PASSED`) and **visual privacy auditing** (`PENDING_AUDIT`).
   - A dataset marked `PASSED` indicates that all files are structurally intact, timestamps are monotonically valid, and client metadata asserts EXIF sanitization.
   - It does *not* imply that photos have undergone computer-vision blurring of faces, computer monitors, or badges. This is clearly signposted with an amber warning banner.
2. **Forbidden Photo Fallback**:
   - If an image fails the server-side sanitization check, the dashboard renders an error placeholder explaining that unverified photos cannot be displayed.
3. **No Synthetic Extrapolations**:
   - Gaps in sensor data (> 500 ms) are rendered as discontinuities in Plotly charts rather than interpolated straight lines.
   - BSSID MAC addresses are never geolocated or assumed to map to UJIIndoorLoc WAP identifiers.

---

## 5. Performance & Large Dataset Handling

Survey sessions collected on Android devices can yield over 50,000 IMU records and several thousand Wi-Fi scan records. The dashboard employs several optimizations:

1. **Subsampling for Interactive Rendering**:
   - For sensor time-series with $N > 1,000$ points per sensor type, the UI applies deterministic equidistant decimation before building Plotly traces to prevent Streamlit browser lag.
2. **Lazy Loading of Images**:
   - Photo images are retrieved on demand only when the user expands or navigates the Photo Gallery tab.
3. **Stateless Pagination**:
   - Data tables display the first page of observations with clear total row counts, avoiding high memory overhead on the client.

---

## 6. Limitations & Future Work (Tasks 20+)

- **Current Scope**: Visualization, exploratory analytics, QA inspection, and dataset management only.
- **Out of Scope (Task 20+)**:
  - Live model retraining or fine-tuning from survey sessions.
  - Live cross-building localization using uploaded survey fingerprints.
  - Automated visual privacy redaction (e.g., face detection, license plate blurring).
