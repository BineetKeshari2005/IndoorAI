# Computer Vision-Based Indoor Landmark Detection & Recognition

## 1. Executive Summary & Architecture Overview

The IndoorAI **Computer Vision Module** (`src/computer_vision/`) provides visual perception and indoor landmark identification for mobile survey photographs collected through the Android client (Task 16). The module recognizes key indoor architectural features and signage to support semantic mapping and future visual-inertial localization.

### Key Architectural Objectives:
- **Privacy-by-Design:** Strictly verifies `is_exif_sanitized` metadata and rejects un-sanitized or corrupt photos prior to decoding.
- **Architectural Taxonomy:** Detects 11 indoor categories including doors, room entrances, staircases, elevators, corridors, intersections, and room/direction/exit signs.
- **Signboard OCR:** Locally extracts text from room plates and directional boards, maintaining candidate text as unverified until manual confirmation.
- **Persistent Landmark Registry:** Maintains physical landmark records under `data/landmarks/{building_id}/landmarks.json`, preventing automatic merging of visually similar fixtures.
- **Visual Descriptor Matching:** Compares query photos against registered physical landmarks using ORB keypoint descriptors and global HSV appearance embeddings.
- **Strict Provenance:** Inherits building and floor IDs solely from verified survey session annotations—never inventing spatial labels from visual appearance alone.

```
                                  ┌─────────────────────────────────────────┐
                                  │   Multimodal Survey Session Storage     │
                                  │ (photos/<photo_id>.jpg, photo_metadata) │
                                  └────────────────────┬────────────────────┘
                                                       │
                                                       ▼
                                  ┌─────────────────────────────────────────┐
                                  │        Image Privacy Validator          │
                                  │ (Check is_exif_sanitized, dimensions)   │
                                  └────────────────────┬────────────────────┘
                                                       │  (If Valid & Sanitized)
                                                       ▼
                                  ┌─────────────────────────────────────────┐
                                  │        Safe Image Loader (PIL/CV2)      │
                                  │   (EXIF Transpose & RGB/BGR Conversion) │
                                  └────────────────────┬────────────────────┘
                                                       │
                           ┌───────────────────────────┴───────────────────────────┐
                           ▼                                                       ▼
            ┌─────────────────────────────┐                         ┌─────────────────────────────┐
            │   Landmark Detection Engine │                         │  Visual Feature Extractor   │
            │   (YOLO / Architectural)    │                         │  (ORB + 512-dim HSV Embed)  │
            └──────────────┬──────────────┘                         └──────────────┬──────────────┘
                           │                                                       │
                           ▼                                                       │
            ┌─────────────────────────────┐                                        │
            │  Signboard OCR Engine       │                                        │
            │  (EasyOCR / Morphological)  │                                        │
            └──────────────┬──────────────┘                                        │
                           │                                                       │
                           ▼                                                       ▼
            ┌─────────────────────────────────────────────────────────────────────────────┐
            │                 Persistent Landmark Registry (data/landmarks/)              │
            │        (Lifecycle: UNVERIFIED -> MANUALLY_VERIFIED / REJECTED)              │
            └──────────────────────────────────────┬──────────────────────────────────────┘
                                                   │
                                                   ▼
            ┌─────────────────────────────────────────────────────────────────────────────┐
            │                       Visual Landmark Matcher                               │
            │           (BFMatcher + Cosine Similarity -> STRONG/WEAK/NO_MATCH)           │
            └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Privacy Requirements & Image Ingestion Security

Indoor survey photography inherently captures physical spaces that may contain bystanders, personal items, or sensitive signage. The vision pipeline enforces strict security boundaries:

1. **Mandatory EXIF Sanitization:**
   Before decoding any image, `ImagePrivacyValidator.validate_photo_file` checks `photo_metadata.csv` to ensure `is_exif_sanitized == True`. If sanitization is missing or false, processing is aborted immediately with `HTTP 403 Forbidden` / `ImageLoadingError`.
2. **Local Processing Only:**
   All detection, OCR, descriptor extraction, and matching execute 100% locally on the device / host server. No images or feature embeddings are transmitted to external third-party cloud APIs.
3. **Format & Path Traversal Guards:**
   Only verified extensions (`.jpg`, `.jpeg`, `.png`, `.webp`) matching declared file magic numbers are loaded. Path climbs (`..`, absolute paths outside the dataset) are rejected.
4. **Dimension & File Size Limits:**
   Images must be between $512\text{ bytes}$ and $25\text{ MB}$, with spatial dimensions between $32\text{ px}$ and $8192\text{ px}$.

---

## 3. Indoor Landmark Taxonomy

The system defines 11 standardized indoor landmark categories:

| Category Code | Description | Visual Characteristics | Navigation Role |
|:---|:---|:---|:---|
| `DOOR` | Closed or standard doorway | Vertical rectangular frame ($H/W \approx 1.8-3.0$), door jambs, handle | Waypoint boundary, room entry |
| `ROOM_ENTRANCE` | Open room portal / archway | Wide opening without door leaf, leading into distinct functional room | Access point, area transition |
| `STAIRCASE` | Set of indoor stairs | Periodic parallel horizontal edges with consistent perspective spacing | Vertical transition (floor change) |
| `ELEVATOR` | Elevator entrance | Metallic double sliding doors, call button panel, floor indicator | Vertical transition (floor change) |
| `CORRIDOR` | Long hallway or passageway | Converging perspective lines, receding parallel wall boundaries | Linear topological edge |
| `INTERSECTION` | Hallway junction / T-junction | Branching corridor paths, multiple open vistas | Decision point in routing graph |
| `ROOM_NUMBER_SIGN` | Room identification plate | High-contrast rectangular plaque near door ($y \approx 1.4-1.8\text{m}$) | Ground-truth semantic label |
| `DIRECTION_SIGN` | Wayfinding signboard | Arrows, room ranges (e.g., "101–120 →"), departmental signs | Topological directional guidance |
| `EXIT_SIGN` | Emergency exit indicator | Green/white luminescent sign mounted high ($y > 2.0\text{m}$) | Safety reference landmark |
| `DISTINCTIVE_STRUCTURE` | Unique architectural column or sculpture | Distinctive geometry, high feature density, unique color pattern | High-confidence visual anchor |
| `UNKNOWN` | Unclassified candidate | Visual anomaly not matching standard categories | Flagged for manual inspection |

---

## 4. Detection Engine & Architectural Feature Heuristics

The detector (`LandmarkDetector`) employs a dual-stage architecture:

### 4.1 Fine-Tuned Neural Model Support (Ultralytics YOLO)
When fine-tuned model weights (e.g. `indoor_yolo.pt`) are available, the detector loads the YOLO model to detect indoor classes directly with GPU/Metal acceleration.

### 4.2 Structural Architectural Heuristics (Zero-Shot Fallback)
For environments where custom weights are not present, the system applies computer vision heuristics:
1. **Vertical Door Frames:** Applies morphological vertical dilation (`cv2.MORPH_RECT(1, 15)`) to connect door jamb edges from Canny edge maps, filtering bounding boxes with aspect ratio $1.4 \le H/W \le 3.8$ and height $> 0.25 H_{img}$.
2. **Corridor Perspective Lines:** Uses Probabilistic Hough Line Transform (`cv2.HoughLinesP`), identifying perspective convergence of left-slanted and right-slanted receding wall lines.
3. **Signboard Plaques:** Applies Gaussian adaptive thresholding and morphological rectangular dilation to locate high-contrast plaques in the upper and middle thirds of the image. Color classification separates Exit signs (green HSV mask) and Direction signs (blue HSV mask) from standard room number plaques.
4. **Non-Maximum Suppression (NMS):** Eliminates overlapping duplicate candidate boxes using greedy IoU suppression at a configurable threshold ($\text{IoU} \ge 0.45$).

### 4.3 Strict Provenance Guarantee
**The detector never infers building or floor labels from visual imagery.** Spatial metadata (`building_id`, `floor_id`, `reference_location_id`) is strictly inherited from verified survey annotations associated with the session.

---

## 5. Room Number and Signboard OCR Engine

The OCR engine (`SignboardOcrEngine`) extracts textual labels from detected signage:
- **Primary Engine:** EasyOCR reader operating on contrast-enhanced (CLAHE) grayscale signboard crops.
- **Fallback Engine:** Morphological stroke gradient density detector for offline or resource-constrained test environments.
- **Strict Verification Rule:** OCR output is stored in `OcrTextRecord` with `is_manually_verified = False`. Extracted text is treated as a candidate suggestion and **never** automatically promoted to ground-truth room labels without explicit human confirmation.

---

## 6. Persistent Landmark Registry

The registry (`LandmarkRegistry`) maintains confirmed physical landmarks in `data/landmarks/{building_id}/landmarks.json`:

```json
{
  "landmark_id": "LM-BLDG-1-F02-DOO-0001",
  "building_id": "BLDG-1",
  "floor_id": 2,
  "category": "DOOR",
  "name": "Room 201 Entrance Door",
  "description": "Lab entrance along north corridor",
  "reference_location_id": "ROOM-201",
  "verified_coordinates_m": [15.5, 30.2],
  "source_image_ids": ["photo_001.jpg", "photo_014.jpg"],
  "associated_detection_ids": ["det-uuid-1", "det-uuid-2"],
  "ocr_text": "ROOM 201",
  "verification_status": "MANUALLY_VERIFIED",
  "created_at": "2026-11-18T10:00:00Z",
  "updated_at": "2026-11-18T10:05:00Z",
  "template_count": 2
}
```

### Verification Lifecycle:
1. **`UNVERIFIED` (Default):** Candidate detected during automated survey analysis. Displayed with an orange badge in the dashboard.
2. **`MANUALLY_VERIFIED`:** Confirmed by an operator, assigned physical metric coordinates $(x, y)$, and validated room name. Displayed with a green badge.
3. **`REJECTED`:** Flagged as false positive or unusable. Excluded from visual matching.
4. **Anti-Duplication Policy:** Two visually identical doors or exit signs are never automatically merged into a single physical landmark. Distinct physical locations must be confirmed via session provenance or human review.

---

## 7. Visual Landmark Matching

The visual matcher (`VisualLandmarkMatcher`) enables image-to-landmark localization queries:
- **ORB Local Descriptors:** Detects up to 500 Oriented FAST and Rotated BRIEF keypoints with 32-byte binary descriptors.
- **Global HSV Embedding:** Computes a normalized 512-dimensional HSV color histogram capturing ambient illumination and material tone.
- **Descriptor Matching:** Uses Brute-Force Matcher with Hamming distance and Lowe's ratio test ($0.75$) to evaluate feature inliers.
- **Match Status Taxonomy:**
  - `STRONG_MATCH`: Similarity $\ge 0.70$ and Inliers $\ge 12$.
  - `WEAK_MATCH`: Similarity $\ge 0.45$ and Inliers $\ge 5$.
  - `NO_MATCH`: Insufficient visual correspondence.
- **Caveat:** Similarity scores are uncalibrated visual feature metrics, **not** positioning probabilities.

---

## 8. Evaluation Framework & Benchmark Results

The evaluation engine (`VisionEvaluator`) computes detection precision, recall, F1, and confusion matrices:
- **Session-Level Splitting:** Evaluations use independent session holdouts to ensure near-duplicate images do not leak between gallery and test sets.
- **Metrics Computed:**
  - Detection Precision, Recall, and F1 at $\text{IoU} \ge 0.50$.
  - Per-category breakdown.
  - Multi-class confusion matrix with `BACKGROUND` class.
  - OCR Character Error Rate (CER) via normalized Levenshtein distance.
  - Visual matching precision and recall.

### Controlled Synthetic Benchmark Results (Strictly Partitioned 3-Image Fixture):
| Metric | Value | Notes |
|:---|:---:|:---|
| **Overall Detection Precision** | **$80.0\%$** | 4 True Positives, 1 False Positive |
| **Overall Detection Recall** | **$100.0\%$** | 4 / 4 Ground Truth Boxes Detected |
| **Overall Detection F1-Score** | **$88.9\%$** | Harmonic mean |
| **Mean IoU** | **$0.83$** | High spatial bounding box overlap |
| **OCR Exact Match Rate** | **$100.0\%$** | Perfect string matching on test signs |
| **OCR Character Error Rate (CER)** | **$0.0\%$** | 0 edit operations |
| **Visual Matching Precision** | **$100.0\%$** | Correct gallery landmark retrieved |

---

## 9. Backend REST API Reference

The service exposes 6 endpoints under `/api/datasets/{dataset_id}/vision`:

| Method | Endpoint | Description |
|:---|:---|:---|
| `POST` | `/api/datasets/{dataset_id}/vision/analyze` | Executes landmark detection and sign OCR on survey photos. |
| `GET` | `/api/datasets/{dataset_id}/vision/detections` | Retrieves computed landmark candidates with bounding boxes and OCR text. |
| `GET` | `/api/datasets/{dataset_id}/vision/landmarks` | Queries registered landmarks by building, floor, category, or status. |
| `POST` | `/api/datasets/{dataset_id}/vision/landmarks/verify` | Manually updates or confirms a landmark record with coordinates. |
| `POST` | `/api/datasets/{dataset_id}/vision/match` | Compares a survey photo against registered landmarks. |
| `POST` | `/api/datasets/{dataset_id}/vision/evaluate` | Runs quantitative evaluation against ground-truth benchmarks. |

---

## 10. Dashboard User Guide (Streamlit Section 15)

Section 15 of the Streamlit dashboard (`frontend/components/computer_vision_view.py`) provides:
1. **📸 Image Gallery & Detections:** Interactive photo browser rendering overlay bounding boxes, confidence scores, and candidate OCR text with EXIF sanitization status badges.
2. **⚙️ Pipeline Execution:** Controls for confidence threshold, OCR toggle, and auto-registration triggers.
3. **🏛️ Landmark Registry:** Searchable table of registered landmarks, highlighting `MANUALLY_VERIFIED` (green) vs `UNVERIFIED` (orange), with an interactive manual verification form.
4. **🔍 Visual Matcher:** Testbed to query any survey photo against the registry and inspect matched candidates, inlier counts, and match status pills.
5. **📊 Evaluation & Confusion Matrix:** Benchmarking dashboard showing overall precision/recall/F1, OCR CER, per-category bar charts, and interactive Plotly confusion matrices.
