# IndoorAI — Multimodal AI Indoor Localization & Spatial Mapping

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

**IndoorAI** is an advanced, research-oriented, production-quality indoor positioning and progressive spatial mapping platform. Operating where GPS fails due to heavy attenuation and multipath interference, IndoorAI combines three distinct sensing streams from authorized participating mobile devices:

1. **Ambient Wi-Fi Signals:** Passive observation of nearby 802.11 access points (BSSID, RSSI, channel frequency, timestamps).
2. **Smartphone Sensors:** High-frequency tri-axial accelerometer, gyroscope, and magnetometer measurements.
3. **User-Initiated Photographs:** On-demand photographs of surrounding indoor landmarks (corridors, doors, signs, room numbers, structural boundaries).

---

## 1. Core Architectural Pillars

- **Hierarchical Multitask Learning:** Joint prediction of Building ID, Floor Level, and $(x, y)$ continuous metric coordinates.
- **Kinematic State Estimation:** Extended Kalman Filtering (EKF) and Map-Constrained Particle Filtering fusing high-frequency pedestrian dead reckoning (PDR) with intermittent global Wi-Fi and visual fixes.
- **Visual-Spatial Semantic Grounding:** Deep visual metric embeddings (ResNet/EfficientNet) and object detection (YOLO) against a pre-indexed spatial landmark database.
- **Topological-Metric Progressive Mapping:** Incremental construction of navigable indoor floorplans using walkability segmentation and pose-graph optimization.
- **Strict Privacy & Anti-Surveillance Invariants:** No covert tracking, no background camera execution, client-side EXIF stripping and face blurring, and ephemeral device pseudonymization.

---

## 2. Project Directory Layout

```
IndoorAI/
├── data/                       # Dataset storage (kept out of version control)
│   ├── raw/                    # Immutable raw benchmark downloads & field logs
│   ├── processed/              # Cleaned, standardized Parquet feature tables
│   └── external/               # Architectural reference blueprints and calibration
│
├── notebooks/                  # Prototyping and exploratory research notebooks
│
├── src/                        # Production Python source package
│   ├── core/                   # Central configuration & structured logging
│   ├── data/                   # Dataset downloaders and schema loaders
│   ├── preprocessing/          # RSSI normalization, IMU cleaning, image sanitizers
│   ├── features/               # WAP visibility matrices, PDR features, visual tokens
│   ├── models/                 # Model architectures
│   │   ├── traditional_ml/     # Random Forest, XGBoost baselines
│   │   ├── deep_learning/      # Multitask MLP and coordinate regressors
│   │   ├── vision/             # ResNet/EfficientNet embeddings, YOLO detectors
│   │   └── multimodal/         # Cross-attention fusion networks & uncertainty heads
│   ├── training/               # Training pipelines, multi-task loss balancing
│   ├── evaluation/             # Positioning error metrics (Euclidean error, CEP50/95)
│   ├── explainability/         # SHAP value diagnostics and error heatmaps
│   ├── sensor_fusion/          # PDR step counter, EKF, Map-Constrained Particle Filter
│   └── mapping/                # Floorplan graph models and metric coordinate systems
│
├── backend/                    # FastAPI production backend
│   └── app/
│       ├── api/                # REST & WebSocket route handlers
│       ├── services/           # Ingestion, inference, and device session services
│       ├── schemas/            # Pydantic validation schemas
│       └── core/               # Security, token generation, database sessions
│
├── android/                    # Android client application (Kotlin / Android SDK)
├── frontend/                   # Web dashboard (Next.js, React, TypeScript, Tailwind)
│
├── experiments/                # Model checkpoints, MLflow runs, experiment artifacts
├── tests/                      # Unit, integration, and structural validation tests
├── configs/                    # Central YAML configuration files
├── docs/                       # Comprehensive architecture and strategy specifications
│
├── README.md                   # Project overview and quick start guide
├── IMPLEMENTATION_STATUS.md    # 30-task progress tracker
├── .gitignore                  # Git exclusions for ML, Python, Android, Node
├── .env.example                # Environment variable configuration template
└── pyproject.toml              # Modern Python build and dependency specifications
```

---

## 3. Technology Stack

| Domain | Technologies |
| :--- | :--- |
| **AI / Machine Learning** | Python 3.11+, NumPy, Pandas, Scikit-learn, XGBoost, PyTorch, OpenCV, Plotly, MLflow |
| **Backend Service** | FastAPI, Pydantic, PostgreSQL, SQLAlchemy, Asyncpg, Alembic |
| **Frontend Dashboard** | Next.js 14, React 18, TypeScript, Tailwind CSS, Lucide Icons |
| **Mobile Client** | Kotlin, Android SDK (Wi-Fi Scan API, SensorManager, CameraX) |

---

## 4. Documentation Index

Detailed architectural and engineering documentation is maintained in the `docs/` directory:

- [System Architecture](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/SYSTEM_ARCHITECTURE.md): Complete data flow, sequence diagrams, and microservice communication.
- [ML Architecture](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/ML_ARCHITECTURE.md): Mathematical formulations, deep multitask MLP, vision backbones, and sensor fusion.
- [Data Strategy](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/DATA_STRATEGY.md): Benchmark specifications (UJIIndoorLoc), multimodal data protocols, and validation.
- [UJIIndoorLoc Dataset Report](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/UJI_INDOORLOC_DATASET_REPORT.md): Empirical inspection, SHA-256 hashes, feature schema, quality audit, and initial EDA plots.
- [Advanced EDA Report](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/advanced_eda/advanced_eda_report.md): In-depth Wi-Fi signal analysis, AP co-occurrence, spatial footprints, duplicate root cause, and domain shifts.
- [Preprocessing Pipeline](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/PREPROCESSING_PIPELINE.md): Data validation, signal floor replacement (-105 dBm), visibility masks, duplicate policies, leakage-free scaling, and target encoding.
- [Development Roadmap](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/DEVELOPMENT_ROADMAP.md): Detailed 30-task granular implementation sequence.
- [Privacy & Security Framework](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/PRIVACY_AND_SECURITY.md): Non-negotiable anti-surveillance constraints, cryptographic access control, and physical RF limitations.
- [Implementation Status](file:///Users/bineetkeshari/Desktop/IndoorAI/IMPLEMENTATION_STATUS.md): Live progress tracker across all 30 implementation tasks.
- [Dashboard User Guide](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/DASHBOARD_GUIDE.md): Interactive Streamlit dashboard architecture, CSV schema, and user guide.
- [Multitask MLP Architecture](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/MLP_ARCHITECTURE.md): Deep multitask neural network specification, layer dimensions, target standardization, and loss balancing.
- [Android App Architecture](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/ANDROID_APP_ARCHITECTURE.md): Jetpack Compose mobile client architecture, permissions, hardware abstraction, Wi-Fi throttling mitigation, and API integration.
- [Multimodal Session Architecture](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/MULTIMODAL_SESSION_ARCHITECTURE.md): Unified multimodal capture sessions, timestamp normalization, crash recovery, and dataset packaging.
- [Dataset Ingestion Architecture](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/DATASET_INGESTION_ARCHITECTURE.md): Backend pipeline for ZIP verification, manifest integrity, multimodal QA, and dataset registry.
- [Multimodal Dataset Explorer](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/MULTIMODAL_DATASET_EXPLORER.md): Multimodal survey explorer and visualization dashboard, API client, privacy-sanitized photo viewer, and QA inspection.

---

## 5. Quick Start & Setup Instructions

### 5.1 Environment Configuration
```bash
# Clone the repository and enter the project directory
cd IndoorAI

# Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Copy the environment variable configuration template
cp .env.example .env
```

### 5.2 Dependency Installation
IndoorAI modularizes dependencies using `pyproject.toml` extra groups:
```bash
# Install core dependencies (data, config, logging)
pip install -e .

# Install dependencies for specific development tracks:
pip install -e ".[ml]"       # Traditional ML & MLflow
pip install -e ".[dl]"       # PyTorch & deep learning
pip install -e ".[vision]"   # OpenCV, YOLO, Albumentations
pip install -e ".[backend]"  # FastAPI, SQLAlchemy, asyncpg
pip install -e ".[dev]"      # Pytest, black, ruff, mypy

# Or install the full research & engineering bundle:
pip install -e ".[all]"
```

### 5.3 Run Interactive Web Dashboard
Launch the IndoorAI interactive demonstration dashboard (Task 11):
```bash
streamlit run frontend/app.py
```
Open `http://localhost:8501` to explore Wi-Fi fingerprints, run multi-model localization inference, and inspect 2D coordinate maps.

### 5.4 Run Validation Tests
To verify all models, features, services, and structural invariants:
```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

### 5.5 Open Android Application
The Android mobile application is located in `android/`:
1. Launch **Android Studio** (Jellyfish / Koala / Ladybug).
2. Select **Open** and select the `/Users/bineetkeshari/Desktop/IndoorAI/android` directory.
3. Allow Gradle to synchronize dependencies (Gradle 8.7, Kotlin 1.9.22, Compose BOM 2024.02.00).
4. Run on a virtual device (API 34) or connected hardware.

---

## 6. Implementation Status
Track our progress in [IMPLEMENTATION_STATUS.md](file:///Users/bineetkeshari/Desktop/IndoorAI/IMPLEMENTATION_STATUS.md).
- **Task 01 (Project Foundation, Architecture & Repository Scaffold):** COMPLETED
- **Task 02 (UJIIndoorLoc Benchmark Dataset Acquisition & Inspection):** COMPLETED
- **Task 03 (Advanced Wi-Fi Fingerprint Exploratory Data Analysis):** COMPLETED
- **Task 04 (Data Validation, Quality Checks, and Cleaning Pipeline):** COMPLETED
- **Task 05 (Advanced Wi-Fi Feature Engineering & Signal Representation):** COMPLETED
- **Task 06 (Independent Audit & Numerical Verification of Wi-Fi Features):** COMPLETED
- **Task 07 (Traditional ML Baselines — Random Forest Baseline Models):** COMPLETED
- **Task 08 (Traditional ML Baselines — k-Nearest Neighbors Localization Baseline):** COMPLETED
- **Task 09 (Model Benchmarking, Comparative Evaluation & Selection):** COMPLETED
- **Task 10 (Localization Error Analysis and Hybrid Model Investigation):** COMPLETED
- **Task 11 (First Interactive Web Dashboard — Streamlit Demonstration Engine):** COMPLETED
- **Task 12 (Multitask Deep Learning [MLP] Architecture & Training Loop):** COMPLETED
- **Task 13 (Android Project Scaffolding & Permissions Architecture):** COMPLETED
- **Task 14 (Android Wi-Fi Fingerprint Collector):** COMPLETED
- **Task 15 (Android IMU & Magnetometer Streamer):** COMPLETED
- **Task 16 (Android User-Initiated Photo Capture & Privacy Sanitizer):** COMPLETED
- **Task 17 (Unified Multimodal Capture Session & Dataset Builder):** COMPLETED
- **Task 18 (Multimodal Dataset Ingestion & Quality Assurance Backend):** COMPLETED
- **Tasks 19 through 30:** Pending subsequent implementation prompts.
