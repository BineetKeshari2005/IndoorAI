# IndoorAI — 30-Task Development Roadmap

This roadmap organizes the development of the **IndoorAI** platform into 30 manageable, sequential, and testable engineering tasks. Each task specifies explicit objectives, technical scope, dependencies, and concrete acceptance criteria.

---

## Phase 1: Foundation, Data Ingestion & Preprocessing (Tasks 1–6)

### Task 01: Project Foundation, Architecture, and Repository Scaffold
- **Objective:** Establish the repository structure, architecture documentation, environment configurations, and core Python package scaffold.
- **Dependencies:** None.
- **Deliverables:** Directory tree, `docs/` architecture documents, `pyproject.toml`, `.env.example`, `.gitignore`, `IMPLEMENTATION_STATUS.md`.
- **Acceptance Criteria:** Directory integrity tests pass; configuration and logger modules initialize cleanly.

### Task 02: UJIIndoorLoc Benchmark Dataset Acquisition & Ingestion Pipeline
- **Objective:** Build an automated, checksum-verified ingestion pipeline for the UJIIndoorLoc dataset.
- **Dependencies:** Task 01.
- **Deliverables:** `src/data/download_uji.py`, `src/data/loader.py`, raw CSV landing in `data/raw/`.
- **Acceptance Criteria:** Download is reproducible, SHA256 verified, loads training (19,937 rows) and validation (1,111 rows) without data corruption.

### Task 03: Wi-Fi Fingerprint Exploratory Data Analysis (EDA)
- **Objective:** Perform exploratory data analysis on the Wi-Fi benchmark to uncover signal distributions and spatial properties.
- **Dependencies:** Task 02.
- **Deliverables:** Research notebook `notebooks/01_uji_wifi_eda.ipynb` and statistical report.
- **Acceptance Criteria:** Distributions of RSSI, missing values (+100 sentinel), device model heterogeneity, and floor balance are analyzed and documented.

### Task 04: Data Validation, Quality Checks, and Cleaning Pipeline
- **Objective:** Implement rigorous validation assertions and missing value converters.
- **Dependencies:** Task 03.
- **Deliverables:** `src/preprocessing/validator.py`, `src/preprocessing/cleaner.py`.
- **Acceptance Criteria:** Replaces sentinel `+100` with standard physical floor ($-104\text{ dBm}$), drops uninformative WAPs (zero variance), and validates coordinate bounds.

### Task 05: Wi-Fi Signal Representation & Normalization Engine
- **Objective:** Implement and benchmark multiple mathematical representations of Wi-Fi RSSI vectors.
- **Dependencies:** Task 04.
- **Deliverables:** `src/preprocessing/scaler.py` (MinMax, Standard, Exponential/Power scaling).
- **Acceptance Criteria:** Unit tests prove signal representations map strictly within expected ranges and preserve relative signal strength distances.

### Task 06: Spatial Feature Engineering (WAP Statistics & Differencing)
- **Objective:** Engineer robust spatial features from raw RSSI vectors.
- **Dependencies:** Task 05.
- **Deliverables:** `src/features/wifi_features.py` (mean RSSI of detected APs, AP count, difference of strongest APs, spatial clustering).
- **Acceptance Criteria:** Feature extraction pipeline produces clean tabular feature matrices ready for ML training.

---

## Phase 2: Traditional ML Baselines & Evaluation (Tasks 7–9)

### Task 07: Traditional ML Baselines — Building & Floor Classification
- **Objective:** Build Random Forest and XGBoost classifiers for hierarchical building and floor classification.
- **Dependencies:** Task 06.
- **Deliverables:** `src/models/traditional_ml/classifier.py`.
- **Acceptance Criteria:** Achieves $\ge 99\%$ building accuracy and $\ge 88\%$ floor accuracy on UJIIndoorLoc validation set.

### Task 08: Traditional ML Baselines — k-Nearest Neighbors Localization Baseline (COMPLETED)
- **Objective:** Build a non-parametric spatial retrieval baseline using k-NN across classification (Building, Floor, Building-Floor) and 2D coordinate regression (Longitude, Latitude), evaluate distance metrics (cosine, euclidean, manhattan), weighting schemes, and feature variants with zero leakage, and conduct a head-to-head comparison against Random Forest.
- **Dependencies:** Task 07.
- **Deliverables:** `configs/knn_config.yaml`, `src/models/traditional_ml/knn.py`, `src/evaluation/knn_evaluation.py`, `src/training/train_knn.py`, `tests/test_knn.py`, `experiments/knn/reports/experiment_report.md`.
- **Acceptance Criteria:** Validated on official validation split without data leakage; achieves 99.91% building accuracy, 84.43% floor accuracy, and 10.05 units mean 2D Euclidean coordinate error (outperforming Random Forest's 11.53 units).

### Task 09: Model Benchmarking, Comparative Evaluation & Selection (COMPLETED)
- **Objective:** Independently audit and benchmark Random Forest (Task 07) and k-NN (Task 08) models, resolve documentation and hyperparameter selection discrepancies, audit dataset phone/user counts, perform sub-domain error diagnostics across buildings and floors, and formulate evidence-based model recommendations for future phases.
- **Dependencies:** Task 07, Task 08.
- **Deliverables:** `src/evaluation/model_benchmark.py`, `experiments/model_benchmark/metrics/benchmark_results.json`, `experiments/model_benchmark/metrics/benchmark_comparison.csv`, `experiments/model_benchmark/plots/` (3 diagnostic charts), `experiments/model_benchmark/reports/model_benchmark_report.md`, `tests/test_model_benchmark.py`.
- **Acceptance Criteria:** Sample alignment verified on all 1,111 validation records; all metrics recalculated from raw predictions; phone ID discrepancy resolved; evidence-based task recommendations delivered (RF for Floor, k-NN for Coordinates, Hybrid proposed); full test suite passing.

### Task 10: Localization Error Analysis and Hybrid Model Investigation (COMPLETED)
- **Objective:** Analyze building, floor, and coordinate failure modes for Random Forest and k-NN, investigate difficult fingerprint archetypes, calibrate classification confidence and spatial dispersion uncertainty, and evaluate a Two-Stage Hierarchical Hybrid Localizer.
- **Dependencies:** Task 07, Task 08, Task 09.
- **Deliverables:** `src/models/traditional_ml/hybrid_localizer.py`, `src/analysis/localization_error_analysis.py`, `experiments/error_analysis/metrics/` (JSON, CSVs), `experiments/error_analysis/plots/` (7 diagnostic plots), `experiments/error_analysis/reports/experiment_report.md`, `tests/test_hybrid_localizer.py`, `tests/test_error_analysis.py`.
- **Acceptance Criteria:** Quantifies adjacent floor error dominance (>89%); isolates difficult fingerprints via Jaccard overlap (0.5648 vs 0.6915); implements confidence-gated hybrid architecture achieving 9.98 units mean Euclidean error; 86 tests passing.

---

## Phase 3: Deep Learning Architectures (Tasks 11–12)

### Task 11: First Interactive Web Dashboard (COMPLETED)
- **Objective:** Build the first interactive, user-facing web application demonstrating IndoorAI localization, fingerprint exploration, signal strength analysis, 2D coordinate mapping, error evaluation, and verified model benchmarking.
- **Dependencies:** Task 07, Task 08, Task 09, Task 10.
- **Deliverables:** `frontend/app.py`, `backend/app/services/prediction_service.py`, `docs/DASHBOARD_GUIDE.md`, `experiments/dashboard/` (visuals & generator), `tests/test_dashboard.py`.
- **Acceptance Criteria:** Interactive Streamlit dashboard operational on port 8501; multi-model selection (RF, k-NN, Hybrid); custom CSV upload with 520 WAP validation; 2D coordinate scatter map in native dataset coordinate space (no fabricated floorplans); 95 unit tests passing.

### Task 12: Model Explainability, SHAP Analysis & Error Diagnostics
- **Objective:** Integrate TreeSHAP and GradientSHAP to diagnose feature importance and failure modes.
- **Dependencies:** Task 11.
- **Deliverables:** `src/explainability/shap_explainer.py`, `notebooks/02_explainability_analysis.ipynb`.
- **Acceptance Criteria:** Top influential WAPs identified per building; systematic error heatmaps generated per floor.

---

## Phase 4: Android Mobile Client Development (Tasks 13–16)

### Task 13: Android Project Scaffolding & Permissions Architecture
- **Objective:** Initialize Kotlin/Android project with granular runtime permissions (Fine Location, Background Wi-Fi, Sensor Manager).
- **Dependencies:** Task 01.
- **Deliverables:** `android/` application module, manifest with explicit user consent dialogs.
- **Acceptance Criteria:** App compiles, displays explicit consent screen, and requests only necessary Android permissions.

### Task 14: Android Wi-Fi Fingerprint Collector
- **Objective:** Implement passive Wi-Fi scanner capturing BSSID, RSSI, frequency, and timestamp while respecting Android scan throttling.
- **Dependencies:** Task 13.
- **Deliverables:** `android/.../wifi/WifiScanner.kt`.
- **Acceptance Criteria:** Collects valid scans matching backend ingestion schema without crashing or memory leaks.

### Task 15: Android IMU & Magnetometer Streamer
- **Objective:** Implement high-frequency (50 Hz) sensor listener for accelerometer, gyroscope, and magnetometer.
- **Dependencies:** Task 13.
- **Deliverables:** `android/.../sensors/ImuStreamer.kt`.
- **Acceptance Criteria:** Captures synchronized 3-axis motion packets, buffering safely when offline.

### Task 16: Android User-Initiated Photo Capture & Privacy Sanitizer
- **Objective:** Build manual, on-demand camera module that immediately strips EXIF and executes face blurring.
- **Dependencies:** Task 13.
- **Deliverables:** `android/.../camera/PhotoCaptureManager.kt`.
- **Acceptance Criteria:** Camera runs ONLY on explicit user button tap (never in background); EXIF metadata is completely removed.

---

## Phase 5: Computer Vision Modules (Tasks 17–20)

### Task 17: Pretrained Visual Feature Extraction Backbone
- **Objective:** Implement ResNet/EfficientNet visual embedding pipeline for indoor scene representations.
- **Dependencies:** Task 01.
- **Deliverables:** `src/models/vision/embeddings.py`.
- **Acceptance Criteria:** Transforms RGB indoor images into dense $L_2$-normalized embedding vectors.

### Task 18: Indoor Object and Landmark Detection (YOLO)
- **Objective:** Configure and fine-tune YOLO detector for indoor signage, doors, stairs, and room numbers.
- **Dependencies:** Task 17.
- **Deliverables:** `src/models/vision/yolo_detector.py`.
- **Acceptance Criteria:** Accurately detects and classifies indoor navigational elements with confidence scores.

### Task 19: Semantic Segmentation for Walkable Areas & Boundaries
- **Objective:** Implement lightweight segmentation to segment walkable floor areas from obstacles and structural walls.
- **Dependencies:** Task 17.
- **Deliverables:** `src/models/vision/segmentation.py`.
- **Acceptance Criteria:** Generates binary walkability masks from perspective indoor images.

### Task 20: Visual-Spatial Retrieval & k-NN Pose Estimation
- **Objective:** Index landmark embeddings in a vector index (`pgvector` / Faiss) for spatial image retrieval.
- **Dependencies:** Task 17, Task 18.
- **Deliverables:** `src/models/vision/visual_indexer.py`.
- **Acceptance Criteria:** Given a test query photo, retrieves top-k nearest spatial anchors with metric distance scores.

---

## Phase 6: Kinematic Sensor Fusion & Tracking (Tasks 21–23)

### Task 21: Pedestrian Dead Reckoning (PDR) Engine
- **Objective:** Implement step detection, Weinberg step length estimation, and heading calculation from IMU feeds.
- **Dependencies:** Task 15.
- **Deliverables:** `src/sensor_fusion/pdr.py`.
- **Acceptance Criteria:** Accurately detects human steps and produces relative $(dx, dy)$ displacement vectors.

### Task 22: Extended Kalman Filter (EKF) for Wi-Fi + PDR
- **Objective:** Build an Extended Kalman Filter fusing high-frequency PDR motion models with intermittent Wi-Fi position fixes.
- **Dependencies:** Task 11, Task 21.
- **Deliverables:** `src/sensor_fusion/ekf.py`.
- **Acceptance Criteria:** Smooths jerky Wi-Fi predictions, maintains trajectory during Wi-Fi signal dropouts.

### Task 23: Map-Constrained Particle Filter
- **Objective:** Develop a particle filter that incorporates floorplan obstacle constraints (preventing wall-crossing).
- **Dependencies:** Task 22.
- **Deliverables:** `src/sensor_fusion/particle_filter.py`.
- **Acceptance Criteria:** Particles crossing non-walkable wall segments are killed during resampling, bounding cumulative drift.

---

## Phase 7: Multimodal Learning & Spatial Mapping (Tasks 24–26)

### Task 24: Multimodal Cross-Attention Fusion Network
- **Objective:** Design a deep neural network uniting Wi-Fi, IMU temporal representations, and visual embeddings.
- **Dependencies:** Task 11, Task 17, Task 21.
- **Deliverables:** `src/models/multimodal/fusion_network.py`.
- **Acceptance Criteria:** Joint model handles missing modalities via dynamic gating without failure.

### Task 25: Multimodal Joint Training & Uncertainty Calibration
- **Objective:** Train the multimodal network and calibrate predictive uncertainty (variance / confidence bounds).
- **Dependencies:** Task 24.
- **Deliverables:** `src/models/multimodal/uncertainty.py`, `src/training/train_multimodal.py`.
- **Acceptance Criteria:** Outputs calibrated 2D covariance ellipses correlating with true empirical error.

### Task 26: Topological-Metric Spatial Floorplan Mapping Engine
- **Objective:** Develop the spatial graph manager representing corridors, rooms, and topological transitions.
- **Dependencies:** Task 19, Task 23.
- **Deliverables:** `src/mapping/floorplan_graph.py`, `src/mapping/spatial_index.py`.
- **Acceptance Criteria:** Generates navigable graph structures with shortest-path routing and coordinate projections.

---

## Phase 8: Full-Stack Systems, UI & Integration (Tasks 27–30)

### Task 27: FastAPI Core Backend, Database Models & Device Auth
- **Objective:** Build production FastAPI backend with PostgreSQL/SQLAlchemy schemas and device registration.
- **Dependencies:** Task 01.
- **Deliverables:** `backend/app/main.py`, `backend/app/core/security.py`, `backend/app/services/device_service.py`.
- **Acceptance Criteria:** Secure JWT device authentication, database migrations with Alembic, health check endpoints.

### Task 28: Real-Time Ingestion & Inference Endpoints
- **Objective:** Create REST and WebSocket endpoints for Wi-Fi scans, IMU streaming, and photo uploads with real-time inference dispatch.
- **Dependencies:** Task 25, Task 26, Task 27.
- **Deliverables:** `backend/app/api/v1/localization.py`, `backend/app/api/v1/telemetry.py`.
- **Acceptance Criteria:** Endpoints handle concurrent streams, responding with position estimates in $\le 50\text{ ms}$.

### Task 29: Next.js Frontend Interactive Dashboard
- **Objective:** Build Next.js React web application visualizing building floorplans, device markers, and uncertainty ellipses.
- **Dependencies:** Task 28.
- **Deliverables:** `frontend/` components (SVG/Canvas map renderer, floor selector, live device monitor).
- **Acceptance Criteria:** Real-time WebSocket connection renders moving device coordinates with smooth interpolation.

### Task 30: End-to-End System Integration, Security Audit & Verification
- **Objective:** Conduct end-to-end integration tests, privacy audits (EXIF, face blur, device pseudonymization), and benchmark reporting.
- **Dependencies:** All preceding tasks (Tasks 01–29).
- **Deliverables:** `tests/integration/`, end-to-end audit report, final documentation update.
- **Acceptance Criteria:** Full simulated walk from Android payload through backend to frontend canvas executes flawlessly under load.

