# Multimodal Data Preprocessing & Temporal Alignment Architecture

## 1. Overview & Architectural Role

The **Multimodal Data Preprocessing & Temporal Alignment Pipeline** represents **Task 20** of the IndoorAI system. It takes raw multimodal mobile survey datasets ingested through Task 18 (`data/survey_datasets/extracted/<dataset_id>/`), validates their structure, and transforms them into clean, standardized, analysis-ready Parquet feature tables and JSON manifests stored under `data/processed/multimodal/<dataset_id>/`.

```
           +-------------------------------------------------------------+
           |       Raw Ingested Dataset (Tasks 17-18 Workspace)          |
           |      (data/survey_datasets/extracted/<dataset_id>/)         |
           +------------------------------+------------------------------+
                                          |
                                          v
                    +-------------------------------------+
                    |       MultimodalPreValidator        |
                    |     (pre-flight sanity checks)      |
                    +---------------------+---------------+
                                          |
               +--------------------------+--------------------------+
               |                          |                          |
               v                          v                          v
      +-----------------+        +-----------------+        +-----------------+
      | WifiPreprocessor|        |SensorPreprocessr|        |PhotoMetadataProc|
      |  - RSSI filter  |        |  - NaN/inf drop |        |  - EXIF check   |
      |  - Deduplicate  |        |  - Gap detect   |        |  - Disk verify  |
      |  - Sweep group  |        |  - Magnitude    |        |  - Audit status |
      +--------+--------+        +--------+--------+        +--------+--------+
               |                          |                          |
               +--------------------------+--------------------------+
                                          |
                                          v
                    +-------------------------------------+
                    |          TimestampAligner           |
                    |   (Configurable temporal windows:   |
                    |     100ms, 250ms, 500ms, 1000ms)    |
                    |    Boundary uncertainty tracking    |
                    +---------------------+---------------+
                                          |
                                          v
                    +-------------------------------------+
                    |          SessionSegmenter           |
                    |   (Fixed duration, gap-based,       |
                    |     or annotation boundary split)   |
                    +---------------------+---------------+
                                          |
                                          v
           +-------------------------------------------------------------+
           |             Processed Multimodal Artifacts                  |
           |          (data/processed/multimodal/<dataset_id>/)          |
           +-------------------------------------------------------------+
           |  wifi/wifi_cleaned.parquet, wifi_features.parquet           |
           |  sensors/sensor_cleaned.parquet, sensor_features.parquet    |
           |  photos/photo_metadata.parquet                              |
           |  annotations/annotations_cleaned.parquet                    |
           |  aligned/temporal_windows.parquet                           |
           |  segments/segments.json                                     |
           |  preprocessing_report.json, preprocessing_manifest.json     |
           +-------------------------------------------------------------+
```

### Key Engineering Principles
1. **Preserve Raw Provenance**: Raw survey archives and extracted CSVs/JSONs are strictly immutable and never overwritten or mutated.
2. **Separation from UJIIndoorLoc**: The academic UJIIndoorLoc baseline pipeline (`src/preprocessing/`, `data/processed/ujiindoorloc/`) remains 100% untouched. Real-world BSSIDs are never mapped to synthetic `WAP001–WAP520` IDs.
3. **Zero Data Fabrication**: Missing sensor observations or sampling gaps are never silently interpolated. Gaps are explicitly flagged and kinematics are reset across discontinuities.
4. **No BSSID Geolocation Guessing**: Physical coordinates are never inferred from BSSID MAC addresses alone.
5. **Privacy Enforcement**: Photos without verified client EXIF sanitization (`is_exif_sanitized == true`) are gated and excluded from approved image sets. Client-side EXIF removal is explicitly distinguished from independent visual privacy auditing (`visual_audit_status: PENDING_AUDIT`).
6. **Deterministic Reproducibility**: Given identical input files and configuration parameters, the pipeline produces bit-for-bit identical Parquet feature tables and cryptographic SHA-256 hashes.

---

## 2. Component Structure

The package is located at `src/multimodal_preprocessing/`:

```
src/multimodal_preprocessing/
├── __init__.py           # Package exports
├── config.py             # MultimodalPreprocessingConfig dataclass
├── validation.py         # MultimodalPreValidator pre-flight validation
├── wifi.py               # WifiPreprocessor: cleaning, dedup, sweeps, AP features
├── sensors.py            # SensorPreprocessor: tri-axial IMU, gaps, kinematics
├── timestamps.py         # TimestampAligner: cross-modal window synchronization
├── photos.py             # PhotoMetadataProcessor: EXIF gate & file verification
├── annotations.py        # AnnotationProcessor: ground-truth standardization
├── segmentation.py       # SessionSegmenter: fixed-duration, gap, annotation splits
├── reporting.py          # PreprocessingReport & PreprocessingManifest
└── pipeline.py           # MultimodalPreprocessingPipeline orchestrator
```

---

## 3. Wi-Fi Preprocessing & Fingerprint Extraction

Located in `src/multimodal_preprocessing/wifi.py`.

### 3.1 Cleaning & Normalization
- **BSSID Normalization**: Standardizes MAC addresses to lowercase colon-separated hex (`aa:bb:cc:dd:ee:ff`). Non-compliant or missing BSSIDs are rejected.
- **RSSI Gating**: Rejects RSSI values outside configurable bounds (default: $-105$ dBm to $-20$ dBm). Positive values and unparseable strings are filtered out.
- **Frequency Validation**: Rejects frequencies outside Wi-Fi bands (default: $2,400$ MHz to $7,125$ MHz, covering 2.4 GHz, 5 GHz, and 6 GHz Wi-Fi 6E).
- **Sweep Clustering**: Groups individual AP readings into discrete hardware sweeps by clustering records separated by $\le 200$ ms. Each sweep receives a unique identifier (`SWEEP-00001`).
- **In-Sweep Deduplication**:
  - `keep_strongest` (default): Retains the AP observation with the highest RSSI if the same BSSID appears multiple times within a single sweep.
  - `keep_first`: Keeps the first chronological record.
  - `reject`: Drops all duplicate occurrences.

### 3.2 Extracted Wi-Fi Features
Per-sweep summary features in `wifi/wifi_features.parquet`:
- `num_visible_aps`: Total distinct visible access points in the sweep.
- `rssi_mean`, `rssi_max`, `rssi_min`, `rssi_std`: First- and second-order signal strength moments.
- `strongest_bssid`, `strongest_rssi`: Primary serving cell / dominant beacon identifier and signal level.
- `building`, `floor`, `room`: Spatial context.

---

## 4. IMU & Magnetometer Preprocessing

Located in `src/multimodal_preprocessing/sensors.py`.

### 4.1 Sensor Physical Units & Types
| Sensor Type | Physical Units | Transformation |
| :--- | :--- | :--- |
| `ACCELEROMETER` | $m/s^2$ | Magnitude $\sqrt{x^2+y^2+z^2}$, jerk $\Delta x/\Delta t$, rolling stats |
| `GYROSCOPE` | $rad/s$ | Magnitude, angular velocity variation, rolling stats |
| `MAGNETIC_FIELD` | $\mu T$ | Field magnitude, directional variance, rolling stats |
| `ROTATION_VECTOR` | Unitless | Norm verification, quaternion component extraction |

### 4.2 Discontinuity & Gap Handling
- **No Silent Interpolation**: Missing sensor readings are never filled by synthetic linear or spline interpolations.
- **Sampling Gaps**: If consecutive sensor readings exceed `sensor_gap_threshold_ms` (default 500 ms), the record is flagged with `is_gap = True`.
- **Kinematic Rolling Reset**: Rolling statistical windows (mean, standard deviation) and finite-difference jerk values are segmented by gap occurrences so that statistics never smooth over discontinuous jumps.

---

## 5. Temporal Normalization & Cross-Modal Alignment

Located in `src/multimodal_preprocessing/timestamps.py`.

### 5.1 Alignment Strategy
The session is discretized into uniform temporal windows $[t_k, t_k + W_{size})$:
- Supported window sizes: $100$ ms, $250$ ms, $500$ ms (default), $1,000$ ms.
- Stride: Configurable (default: $500$ ms, non-overlapping).

### 5.2 Modality Collection & Boundary Ambiguity
- For each window, the pipeline collects observation IDs belonging to Wi-Fi, sensors, photos, and annotations that fall strictly within the window interval.
- **Clock Uncertainty Monitoring**: Smartphone time sources report clock synchronization uncertainties $\pm \epsilon$. If an observation's uncertainty window $[t - \epsilon, t + \epsilon]$ crosses either the start or end of the temporal window, it is flagged as `is_timing_ambiguous = True`.
- Window output schema in `aligned/temporal_windows.parquet`:
  - `window_id`, `window_start_ms`, `window_end_ms`, `window_duration_ms`
  - `has_wifi`, `has_sensors`, `has_photos`, `has_annotations`
  - `is_multimodal_concurrent` ($\ge 2$ modalities present)
  - `wifi_obs_count`, `sensor_obs_count`, `photo_obs_count`, `annotation_obs_count`
  - `wifi_obs_ids`, `sensor_obs_ids`, `photo_obs_ids`, `annotation_obs_ids`
  - `is_timing_ambiguous`, `ambiguous_obs_count`

---

## 6. Photo Metadata Preprocessing & Privacy Gating

Located in `src/multimodal_preprocessing/photos.py`.

- **Zero Image Modification**: Raw JPEG/PNG image files are never altered or compressed during preprocessing.
- **Disk File Verification**: Verifies that every referenced image file physically exists in `<extracted>/images/<filename>`.
- **EXIF Sanitization Filter**: When `filter_unsanitized_photos = True`, photos whose `is_exif_sanitized` flag is false, missing, or unverified are excluded from the approved output table `photos/photo_metadata.parquet`.
- **Visual Privacy Separation**: Metadata records explicitly assign `visual_audit_status: PENDING_AUDIT`, reinforcing that EXIF removal does not prove visual privacy of faces, badges, or monitors.

---

## 7. Annotation Processing & Ground-Truth Standardization

Located in `src/multimodal_preprocessing/annotations.py`.

- **Canonical Types**: Normalizes user labels into canonical architectural categories: `CORRIDOR`, `DOOR`, `ROOM_ENTRANCE`, `STAIRCASE`, `ELEVATOR`, `LANDMARK`, `REFERENCE_POINT`, `MANUAL_LOCATION_CORRECTION`, and `CUSTOM`.
- **Surveyor Provenance**: Explicitly sets `label_provenance: SURVEYOR_MANUAL`. The pipeline never creates synthetic labels not provided by the surveyor.

---

## 8. Session Segmentation

Located in `src/multimodal_preprocessing/segmentation.py`.

Supports 4 configurable partitioning strategies:
1. `fixed_duration`: Splits session into segments of fixed duration (e.g. 60 seconds).
2. `observation_count`: Splits session when total observations reach a threshold (e.g. 100 observations).
3. `annotation_boundaries`: Creates a new segment boundary at each surveyor annotation event.
4. `temporal_gaps`: Splits session whenever sensor inactivity or capture gap exceeds a threshold (e.g. 5.0 seconds).

Output is written to `segments/segments.json`.

---

## 9. Processed Storage Layout & Manifest Schema

```text
data/processed/multimodal/<dataset_id>/
├── wifi/
│   ├── wifi_cleaned.parquet       # Deduplicated, filtered AP records
│   └── wifi_features.parquet      # Sweep-level aggregated signal statistics
├── sensors/
│   ├── sensor_cleaned.parquet     # Calibrated readings with magnitude & gaps
│   └── sensor_features.parquet    # Per-stream sampling rates & moments
├── photos/
│   └── photo_metadata.parquet     # Sanitization-verified landmark metadata
├── annotations/
│   └── annotations_cleaned.parquet# Standardized surveyor ground-truth tags
├── aligned/
│   └── temporal_windows.parquet   # Synchronized cross-modal window catalog
├── segments/
│   └── segments.json              # Analytical temporal/topological segments
├── preprocessing_report.json       # Diagnostic QA report
└── preprocessing_manifest.json     # Cryptographic audit manifest (SHA-256)
```

---

## 10. Backend API & Dashboard Integration

### 10.1 REST Endpoints
- `POST /api/datasets/{dataset_id}/preprocess`: Accepts JSON `PreprocessRequest` and executes the pipeline asynchronously.
- `GET /api/datasets/{dataset_id}/preprocessing`: Returns `PreprocessingStatusResponse` with latest report and manifest.
- `GET /api/datasets/{dataset_id}/preprocessing/files`: Returns list of generated Parquet and JSON files with sizes and SHA-256 checksums.

### 10.2 Streamlit Dashboard
Section **10. Preprocessing & Alignment** in `frontend/app.py` allows users to:
- Adjust thresholds (RSSI bounds, gap thresholds, window duration, segmentation strategy).
- Trigger execution via an explicit button.
- Inspect diagnostic KPI metric cards and tabbed breakdowns (cleaning, alignment, segments, generated artifacts, report JSON).

---

## 11. Known Limitations & Future Work (Task 21+)

1. **Orientation Alignment**: Sensor kinematics currently operate in the device body frame. World-frame coordinate transformations require fused orientation quaternions or magnetometers, to be integrated into the PDR engine (Task 21).
2. **Visual Feature Extraction**: This pipeline processes photo metadata only; deep visual embeddings (ResNet/EfficientNet/YOLO) will be extracted in Task 21+.
3. **Producer Uncertainty**: Temporal window ambiguity relies on producer-reported timestamp uncertainties ($\pm\epsilon$), which represent hardware driver estimates rather than laboratory-calibrated sync.
