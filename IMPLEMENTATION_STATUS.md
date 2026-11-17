# IndoorAI — Implementation Status & Task Tracker

This document tracks progress across all 30 implementation tasks defined in the [Development Roadmap](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/DEVELOPMENT_ROADMAP.md).

**Summary Progress:**
- **Completed:** 18 / 30 tasks (60.0%)
- **In Progress:** 0 / 30 tasks
- **Not Started:** 12 / 30 tasks

---

## Task Progress Matrix

| Task ID | Task Description | Phase | Status | Target / Completion Date |
| :--- | :--- | :--- | :--- | :--- |
| **Task 01** | **Project Foundation, Architecture & Repository Scaffold** | **Phase 1** | **COMPLETED** | **2026-11-16** |
| **Task 02** | **UJIIndoorLoc Benchmark Dataset Acquisition & Ingestion Pipeline** | **Phase 1** | **COMPLETED** | **2026-11-16** |
| **Task 03** | **Wi-Fi Fingerprint Exploratory Data Analysis (EDA)** | **Phase 1** | **COMPLETED** | **2026-11-16** |
| **Task 04** | **Data Validation, Quality Checks, and Cleaning Pipeline** | **Phase 1** | **COMPLETED** | **2026-11-16** |
| **Task 05** | **Advanced Wi-Fi Feature Engineering & Signal Representation** | **Phase 1** | **COMPLETED** | **2026-11-16** |
| **Task 06** | **Independent Audit & Numerical Verification of Wi-Fi Features** | **Phase 1** | **COMPLETED** | **2026-11-16** |
| **Task 07** | **Traditional ML Baselines — Random Forest Baseline Models** | **Phase 2** | **COMPLETED** | **2026-11-16** |
| **Task 08** | **Traditional ML Baselines — k-Nearest Neighbors Localization Baseline** | **Phase 2** | **COMPLETED** | **2026-11-16** |
| **Task 09** | **Model Benchmarking, Comparative Evaluation & Selection** | **Phase 2** | **COMPLETED** | **2026-11-16** |
| **Task 10** | **Localization Error Analysis & Hybrid Model Investigation** | **Phase 2** | **COMPLETED** | **2026-11-16** |
| **Task 11** | **First Interactive Web Dashboard (Streamlit Demonstration Engine)** | **Phase 3** | **COMPLETED** | **2026-11-16** |
| **Task 12** | **Multitask Deep MLP Architecture & Training Loop** | **Phase 3** | **COMPLETED** | **2026-11-16** |
| **Task 13** | **Android Project Scaffolding & Permissions Architecture** | **Phase 4** | **COMPLETED** | **2026-11-17** |
| **Task 14** | **Android Wi-Fi Fingerprint Collector** | **Phase 4** | **COMPLETED** | **2026-11-17** |
| **Task 15** | **Android IMU & Magnetometer Streamer** | **Phase 4** | **COMPLETED** | **2026-11-17** |
| **Task 16** | **Android User-Initiated Photo Capture & Privacy Sanitizer** | **Phase 4** | **COMPLETED** | **2026-11-17** |
| **Task 17** | **Unified Multimodal Capture Session & Dataset Builder** | **Phase 4** | **COMPLETED** | **2026-11-17** |
| **Task 18** | **Multimodal Dataset Ingestion & Quality Assurance Backend** | **Phase 4** | **COMPLETED** | **2026-11-17** |
| Task 19 | Indoor Object and Landmark Detection (YOLO) | Phase 5 | NOT STARTED | Planned |
| Task 20 | Visual-Spatial Retrieval & k-NN Pose Estimation | Phase 5 | NOT STARTED | Planned |
| Task 21 | Pedestrian Dead Reckoning (PDR) Engine | Phase 6 | NOT STARTED | Planned |
| Task 22 | Extended Kalman Filter (EKF) for Wi-Fi + PDR | Phase 6 | NOT STARTED | Planned |
| Task 23 | Map-Constrained Particle Filter | Phase 6 | NOT STARTED | Planned |
| Task 24 | Multimodal Cross-Attention Fusion Network | Phase 7 | NOT STARTED | Planned |
| Task 25 | Multimodal Joint Training & Uncertainty Calibration | Phase 7 | NOT STARTED | Planned |
| Task 26 | Topological-Metric Spatial Floorplan Mapping Engine | Phase 7 | NOT STARTED | Planned |
| Task 27 | FastAPI Core Backend, Database Models & Device Auth | Phase 8 | NOT STARTED | Planned |
| Task 28 | Real-Time Ingestion & Inference Endpoints | Phase 8 | NOT STARTED | Planned |
| Task 29 | Next.js Frontend Interactive Dashboard | Phase 8 | NOT STARTED | Planned |
| Task 30 | End-to-End System Integration, Security Audit & Verification | Phase 8 | NOT STARTED | Planned |

---

## Detailed Task Log

### Task 01: Project Foundation, Architecture, and Repository Scaffold
- **Status:** COMPLETED
- **Completed On:** 2026-11-16
- **Deliverables:** Directory structure, architecture specs, package config, logging, and structural tests.

### Task 02: UJIIndoorLoc Benchmark Dataset Acquisition & Ingestion Pipeline
- **Status:** COMPLETED
- **Completed On:** 2026-11-16
- **Deliverables:** Official UCI download module, raw immutable storage with SHA-256 hashes, inspection module, EDA tables, plots, and dataset report.

### Task 03: Wi-Fi Fingerprint Exploratory Data Analysis (EDA)
- **Status:** COMPLETED
- **Completed On:** 2026-11-16
- **Deliverables:** Advanced EDA module, 18 high-resolution visualization plots, AP co-occurrence analysis, duplicate investigation, CRS verification report, and unit test suite.

### Task 04: Data Validation, Quality Checks, and Cleaning Pipeline
- **Status:** COMPLETED
- **Completed On:** 2026-11-16
- **Deliverables Created:**
  - Validation module: [src/preprocessing/validation.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/preprocessing/validation.py) (schema, missing values, signal boundaries, hierarchical building-floor checks).
  - Duplicate handler: [src/preprocessing/duplicate_handler.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/preprocessing/duplicate_handler.py) (`keep_all`, `remove_exact`, `sample_weight`).
  - Wi-Fi signal engineering: [src/preprocessing/wifi_preprocessing.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/preprocessing/wifi_preprocessing.py) ($-105.0\text{ dBm}$ noise floor, binary visibility masks).
  - Feature quality cleaner: [src/preprocessing/cleaning.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/preprocessing/cleaning.py) (zero-variance audit, schema preservation).
  - Scaler module: [src/preprocessing/scaling.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/preprocessing/scaling.py) (MinMaxScaler, StandardScaler, RobustScaler fitted strictly on training data).
  - Target encoding: [src/preprocessing/target_encoding.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/preprocessing/target_encoding.py) (building, floor, composite building-floor targets; metadata separation; programmatic leakage assertion).
  - End-to-end pipeline: [src/preprocessing/pipeline.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/preprocessing/pipeline.py) (automated execution with single command `python -m src.preprocessing.pipeline`).
  - Configuration: [configs/preprocessing_config.yaml](file:///Users/bineetkeshari/Desktop/IndoorAI/configs/preprocessing_config.yaml).
  - Processed datasets: `data/processed/ujiindoorloc/` (`X_train.parquet`, `X_validation.parquet`, `y_train.parquet`, `y_validation.parquet`, `metadata_train.parquet`, `metadata_validation.parquet`).
  - Preprocessing artifacts: `experiments/preprocessing/` (`validation_report.json`, `duplicate_handling_report.json`, `ap_feature_quality.csv`, `preprocessing_summary.json`, `artifacts/scaler.joblib`, `artifacts/target_encoders.json`, `artifacts/preprocessing_metadata.json`).
  - Automated tests: [tests/test_preprocessing.py](file:///Users/bineetkeshari/Desktop/IndoorAI/tests/test_preprocessing.py) (12 unit tests passing; 29 total project tests passing).
  - Pipeline documentation: [docs/PREPROCESSING_PIPELINE.md](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/PREPROCESSING_PIPELINE.md).

### Task 05: Advanced Wi-Fi Feature Engineering & Signal Representation
- **Status:** COMPLETED
- **Completed On:** 2026-11-16
- **Deliverables Created:**
  - Signal statistics module: [src/features/signal_statistics.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/features/signal_statistics.py) (18 per-sample summary features: visibility counts, moments, top-N signals, and propagation quality bands).
  - Access point moments & relative features: [src/features/ap_features.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/features/ap_features.py) (training-only AP detection rates, RSSI moments, and relative differential features).
  - Feature selection analyzer: [src/features/feature_selection.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/features/feature_selection.py) (training-only variance and detection threshold evaluation, policy reports).
  - Unified transformer: [src/features/wifi_features.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/features/wifi_features.py) (stateful `WiFiFeatureTransformer` combining baseline and engineered features, zero-leakage training-only scaling).
  - End-to-end pipeline: [src/features/feature_pipeline.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/features/feature_pipeline.py) (automated execution via `python -m src.features.feature_pipeline`).
  - Configuration: [configs/feature_config.yaml](file:///Users/bineetkeshari/Desktop/IndoorAI/configs/feature_config.yaml).
  - Engineered datasets: `data/processed/ujiindoorloc/feature_engineered/` (`X_train_engineered.parquet` [19,937 x 1,058], `X_validation_engineered.parquet` [1,111 x 1,058], `feature_names.json`, `feature_selection_summary.json`).
  - Experiment artifacts: `experiments/feature_engineering/` (`ap_statistics.csv`, `feature_summary.json`, `feature_correlation_summary.csv`, `feature_selection_report.md`, `feature_selection_summary.json`).
  - Diagnostic visualizations: `experiments/feature_engineering/plots/` (6 publication-quality charts: visible AP count distribution, average RSSI distribution, strongest RSSI distribution, feature correlation heatmap, AP detection frequency chart, AP signal variability chart).
  - Automated tests: [tests/test_feature_engineering.py](file:///Users/bineetkeshari/Desktop/IndoorAI/tests/test_feature_engineering.py) (12 unit tests passing; 41 total project tests passing).
  - Documentation: [docs/WIFI_FEATURE_ENGINEERING.md](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/WIFI_FEATURE_ENGINEERING.md).

### Task 06: Independent Audit & Numerical Verification of Wi-Fi Features
- **Status:** COMPLETED
- **Completed On:** 2026-11-16
- **Deliverables Created:**
  - Audit execution engine: [src/features/feature_audit.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/features/feature_audit.py) (recalculates all 18 features from raw data, evaluates all 6 invariants, compares AP statistics with Task 03).
  - Audit report JSON: [experiments/feature_engineering/audit/feature_audit_report.json](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/feature_engineering/audit/feature_audit_report.json).
  - Audit report Markdown: [experiments/feature_engineering/audit/feature_audit_report.md](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/feature_engineering/audit/feature_audit_report.md).
  - Corrected summary JSON: [experiments/feature_engineering/audit/corrected_feature_summary.json](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/feature_engineering/audit/corrected_feature_summary.json).
  - AP statistics verification table: [experiments/feature_engineering/audit/ap_statistics_verification.csv](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/feature_engineering/audit/ap_statistics_verification.csv) (100% verified across all 520 WAPs).
  - Invariant checks JSON: [experiments/feature_engineering/audit/feature_invariant_checks.json](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/feature_engineering/audit/feature_invariant_checks.json) (0 violations across 19,937 train and 1,111 val samples).
  - Automated audit tests: [tests/test_feature_audit.py](file:///Users/bineetkeshari/Desktop/IndoorAI/tests/test_feature_audit.py) (8 unit tests passing; 49 total project tests passing).
  - Updated documentation: [docs/WIFI_FEATURE_ENGINEERING.md](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/WIFI_FEATURE_ENGINEERING.md).

### Task 07: Traditional ML Baselines — Random Forest Baseline Models
- **Status:** COMPLETED
- **Completed On:** 2026-11-16
- **Deliverables Created:**
  - Configuration: [configs/random_forest_config.yaml](file:///Users/bineetkeshari/Desktop/IndoorAI/configs/random_forest_config.yaml).
  - Model architecture & wrapper: [src/models/traditional_ml/random_forest.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/models/traditional_ml/random_forest.py) (`IndoorAIRandomForest` orchestrating 5 model heads: Building, Floor, Building-Floor, Longitude, Latitude, with programmatic leakage assertion).
  - Evaluation module: [src/evaluation/random_forest_evaluation.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/evaluation/random_forest_evaluation.py) (macro/weighted F1, confusion matrices, MAE, RMSE, 2D Euclidean coordinate error, hierarchical breakdowns, and publication plotting).
  - Training pipeline: [src/training/train_random_forest.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/training/train_random_forest.py) (5-fold `StratifiedGroupKFold` on `USERID`, Variant A vs Variant B comparison, final model training, single-pass validation evaluation).
  - Serialized model suite: [experiments/random_forest/models/](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/random_forest/models/) (`model_building.joblib`, `model_floor.joblib`, `model_building_floor.joblib`, `model_longitude.joblib`, `model_latitude.joblib`, `model_metadata.joblib`).
  - Prediction outputs: [experiments/random_forest/predictions/](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/random_forest/predictions/) (`val_predictions.parquet`, `val_predictions.csv`).
  - Metrics artifacts: [experiments/random_forest/metrics/](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/random_forest/metrics/) (`random_forest_results.json`, `random_forest_results.csv`, `classification_report.json`, `regression_metrics.json`, `feature_importance.csv`).
  - Publication-quality diagnostic plots: [experiments/random_forest/plots/](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/random_forest/plots/) (6 PNGs: confusion matrices for building, floor, building-floor, top-20 feature importances, actual vs predicted coordinates, coordinate error distribution).
  - Experiment report: [experiments/random_forest/reports/experiment_report.md](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/random_forest/reports/experiment_report.md).
  - Automated tests: [tests/test_random_forest.py](file:///Users/bineetkeshari/Desktop/IndoorAI/tests/test_random_forest.py) (8 unit tests passing; 57 total project tests passing).

### Task 08: Traditional ML Baselines — k-Nearest Neighbors Localization Baseline
- **Status:** COMPLETED
- **Completed On:** 2026-11-16
- **Deliverables Created:**
  - Configuration: [configs/knn_config.yaml](file:///Users/bineetkeshari/Desktop/IndoorAI/configs/knn_config.yaml).
  - Model architecture & wrapper: [src/models/traditional_ml/knn.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/models/traditional_ml/knn.py) (`IndoorAIKNN` orchestrating 5 nearest-neighbor heads: Building, Floor, Building-Floor, Longitude, Latitude; `SummaryFeatureScaler` for training-only scaling of engineered summary statistics; strict programmatic leakage assertion).
  - Evaluation module: [src/evaluation/knn_evaluation.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/evaluation/knn_evaluation.py) (re-exports standard indoor positioning metrics; diagnostic plotting for $k$-parameter sensitivity curves and RF vs k-NN head-to-head grouped bar comparisons).
  - Training pipeline: [src/training/train_knn.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/training/train_knn.py) (5-fold `StratifiedGroupKFold` on `USERID`, full CV grid search across distance metrics, weighting schemes, neighbor counts, and feature variants; final model training; single-pass validation evaluation).
  - Serialized model suite: [experiments/knn/models/](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/knn/models/) (`model_building.joblib`, `model_floor.joblib`, `model_building_floor.joblib`, `model_longitude.joblib`, `model_latitude.joblib`, `summary_scaler.joblib`, `knn_metadata.joblib`, `knn_metadata.json` — 17.68 MB total footprint, 96.8% smaller than Random Forest).
  - Prediction outputs: [experiments/knn/predictions/](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/knn/predictions/) (`val_predictions.parquet`, `val_predictions.csv`).
  - Metrics artifacts: [experiments/knn/metrics/](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/knn/metrics/) (`knn_results.json`, `knn_results.csv`, `classification_report.json`, `regression_metrics.json`, `model_comparison.csv`, `cv_grid_results.csv`).
  - Diagnostic visualizations: [experiments/knn/plots/](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/knn/plots/) (7 PNGs: confusion matrices for building, floor, building-floor, actual vs predicted coordinates, coordinate error distribution, $k$ sensitivity comparison curve, RF vs k-NN head-to-head comparison).
  - Experiment report: [experiments/knn/reports/experiment_report.md](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/knn/reports/experiment_report.md).
  - Automated tests: [tests/test_knn.py](file:///Users/bineetkeshari/Desktop/IndoorAI/tests/test_knn.py) (10 unit tests passing; 67 total project tests passing).

### Task 09: Model Benchmarking, Comparative Evaluation & Selection
- **Status:** COMPLETED
- **Completed On:** 2026-11-16
- **Deliverables Created:**
  - Benchmark & evaluation engine: [src/evaluation/model_benchmark.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/evaluation/model_benchmark.py) (independent metric recalculation from parquet predictions, validation sample alignment audit, raw device ID verification, k-NN selection investigation, sub-domain hierarchical breakdowns, and publication plotting).
  - Benchmark master JSON: [experiments/model_benchmark/metrics/benchmark_results.json](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/model_benchmark/metrics/benchmark_results.json).
  - Unified comparison table: [experiments/model_benchmark/metrics/benchmark_comparison.csv](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/model_benchmark/metrics/benchmark_comparison.csv).
  - Publication-quality diagnostic plots: [experiments/model_benchmark/plots/](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/model_benchmark/plots/)
    - `model_accuracy_comparison.png` (Task-by-task classification accuracy comparison)
    - `coordinate_error_comparison.png` (Empirical CDF and boxplot with mean markers for 2D Euclidean coordinate error)
    - `building_floor_comparison.png` (Campus building floor accuracy and mean coordinate error breakdown)
  - Comprehensive report: [experiments/model_benchmark/reports/model_benchmark_report.md](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/model_benchmark/reports/model_benchmark_report.md).
  - Automated unit test suite: [tests/test_model_benchmark.py](file:///Users/bineetkeshari/Desktop/IndoorAI/tests/test_model_benchmark.py) (9 unit tests passing; 76 total project tests passing).
- **Key Resolutions & Findings:**
  - **Sample Alignment:** 100% verified identical alignment across all 1,111 validation records (0 target discrepancies).
  - **Metric Recalculation:** Recomputed from predictions; exact match to previous JSONs down to 4 decimal places.
  - **k-NN Selection Audit:** Resolved $k=7$ vs $k=5$ discrepancy. Selection used multi-objective lexicographic sorting: primary objective was minimizing mean 2D Euclidean error. At full float precision, $k=7$ achieved global minimum ($10.039661\text{ units}$) across all 60 combinations; accuracy difference ($0.16\%$) was within CV standard deviation ($\pm 6.14\%$).
  - **Device Counts Audit:** Training set has 18 users and 16 distinct phones (`max(PHONEID)=24` caused prior documentation confusion). Validation set has 1 user and 11 distinct phones (9 unseen in training). Dataset union has exactly 25 phones (0-24).
  - **Model Recommendations:** Recommended Random Forest for Floor classification (88.66% vs 84.43%, +4.23%); recommended k-NN for 2D Coordinate regression (Mean Euclidean: 10.05 vs 11.53 units, Median: 7.33 vs 8.65 units, 96.8% smaller model footprint). Proposed two-stage hierarchical hybrid localization architecture for future integration.

### Task 10: Localization Error Analysis and Hybrid Model Investigation
- **Status:** COMPLETED
- **Completed On:** 2026-11-16
- **Deliverables Created:**
  - Analysis engine: [src/analysis/localization_error_analysis.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/analysis/localization_error_analysis.py) (structural confusion analysis, adjacent floor quantification, top-50 high error isolation, Jaccard overlap with training neighbors, confidence calibration, spatial uncertainty dispersion, and diagnostic plotting).
  - Hybrid model architecture: [src/models/traditional_ml/hybrid_localizer.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/models/traditional_ml/hybrid_localizer.py) (`HybridLocalizer` orchestrating Stage 1 RF classification, Stage 2 local k-NN coordinate regression across 13 building-floor subsets, confidence-gating threshold tau, and global k-NN fallback).
  - Master error analysis JSON: [experiments/error_analysis/metrics/error_analysis.json](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/error_analysis/metrics/error_analysis.json).
  - High-error samples CSV: [experiments/error_analysis/metrics/high_error_samples.csv](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/error_analysis/metrics/high_error_samples.csv) (Top 50 high-error cases for RF and k-NN with full target, coordinate, signal, and device metadata).
  - Floor confusion CSV: [experiments/error_analysis/metrics/floor_confusion_analysis.csv](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/error_analysis/metrics/floor_confusion_analysis.csv).
  - Building error CSV: [experiments/error_analysis/metrics/building_error_analysis.csv](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/error_analysis/metrics/building_error_analysis.csv).
  - Hybrid model results JSON: [experiments/error_analysis/metrics/hybrid_model_results.json](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/error_analysis/metrics/hybrid_model_results.json).
  - Hybrid validation predictions: [experiments/error_analysis/predictions/val_predictions.parquet](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/error_analysis/predictions/val_predictions.parquet) and [val_predictions.csv](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/error_analysis/predictions/val_predictions.csv).
  - Diagnostic visualizations (7 PNGs): [experiments/error_analysis/plots/](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/error_analysis/plots/)
    - `floor_error_heatmap.png` (Confusion matrix heatmaps for RF and k-NN)
    - `building_coordinate_error_comparison.png` (Mean coordinate error across buildings)
    - `coordinate_error_distribution.png` (Error density KDE and empirical CDF)
    - `actual_vs_predicted_positions.png` (2D scatter plot with red error vectors for top 20 outliers)
    - `visible_aps_vs_coordinate_error.png` (Scatter plot and polynomial trend line)
    - `rf_vs_knn_error_comparison.png` (Sample-by-sample error parity scatter plot)
    - `hybrid_vs_standalone_comparison.png` (Grouped bar chart comparing RF, k-NN, and Hybrid)
  - Comprehensive markdown report: [experiments/error_analysis/reports/experiment_report.md](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/error_analysis/reports/experiment_report.md).
  - Automated unit tests: [tests/test_hybrid_localizer.py](file:///Users/bineetkeshari/Desktop/IndoorAI/tests/test_hybrid_localizer.py) (5 tests passing) and [tests/test_error_analysis.py](file:///Users/bineetkeshari/Desktop/IndoorAI/tests/test_error_analysis.py) (5 tests passing; 86 total project tests passing).
- **Key Empirical Findings:**
  - **Adjacent Floor Dominance:** 89.68% of Random Forest floor errors and 91.91% of k-NN floor errors are adjacent floor confusions ($|\text{true} - \text{pred}| = 1$), verifying vertical concrete slab penetration as the primary cause of floor error. Non-adjacent errors represent $< 1.3\%$ of samples.
  - **Building 1 & Floor 4 Bottlenecks:** Over 55% of mutual floor misclassifications concentrate in Building 1. On Floor 4 (Building 2, $N=39$), downward vertical bias pulls $>83\%$ of misclassified fingerprints into Floor 3.
  - **Difficult Fingerprint Archetypes:** Top 50 high-error fingerprints share significantly fewer access points with training data (Jaccard similarity 0.5648 vs 0.6915 for low-error samples) and suffer from low signal contrast (`rssi_std` correlation -0.244).
  - **Hybrid Model Performance:** Direct unthresholded hybrid suffered from Stage 1 error propagation (when RF misclassifies floor, local k-NN coordinates explode to 18.51 units). Implementing confidence gating ($\tau = 0.70$) resolved this by falling back to global k-NN on uncertain predictions, yielding an overall Mean Euclidean Error of **9.98 units** (outperforming Global k-NN's 10.05 units and Random Forest's 11.53 units) and capping maximum error at 67.98 units.
  - **Spatial Uncertainty Metric:** Retrieved neighbor spatial dispersion ($\sigma_{\text{spatial}}$) exhibits a strong positive correlation (+0.499) with actual Euclidean coordinate error (Low dispersion: 7.34 units vs High dispersion: 15.96 units).

### Task 11: First Interactive Web Dashboard & Task 11.1 Audit
- **Status:** COMPLETED & AUDITED
- **Completed On:** 2026-11-16
- **Deliverables Created:**
  - Interactive Web Application: [frontend/app.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/app.py) (Streamlit dashboard with modern responsive UI, multi-model selection, fingerprint explorer, 2D coordinate map, prediction evaluation, and benchmark comparison).
  - Prediction & Preprocessing Service: [backend/app/services/prediction_service.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/services/prediction_service.py) (thread-safe, cached model inference for RF, k-NN, and Hybrid; raw 520-WAP CSV validation and feature transformer; evaluation metrics).
  - Operational & User Guide: [docs/DASHBOARD_GUIDE.md](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/DASHBOARD_GUIDE.md) (comprehensive architectural and user documentation).
  - Task 11.1 Audit & Consistency Report: [docs/TASK_11_1_AUDIT_REPORT.md](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/TASK_11_1_AUDIT_REPORT.md) (root cause analysis, corrected k-NN k=7 configuration, audited benchmark tables, live sample verification, and uncalibrated confidence notes).
  - Visual Evaluation Artifacts: [experiments/dashboard/](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/dashboard/)
    - `dashboard_coordinate_map_sample0.png` (2D scatter map with ground truth, RF prediction, and Hybrid prediction)
    - `dashboard_signal_profile_sample150.png` (Top Wi-Fi access point RSSI distribution)
    - `dashboard_model_benchmarks.png` (Comparative mean and median 2D coordinate error)
  - Visual Generator Script: [experiments/dashboard/generate_dashboard_visuals.py](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/dashboard/generate_dashboard_visuals.py).
  - Automated Unit Test Suite: [tests/test_dashboard.py](file:///Users/bineetkeshari/Desktop/IndoorAI/tests/test_dashboard.py) (11 unit tests passing; 97 total project tests passing in 13.0s).
- **Key Capabilities & Invariants Enforced:**
  - **Six Dedicated Sections:** Section A (System & Dataset Overview with 6 KPI metrics and mandatory benchmark disclosure), Section B (Fingerprint Explorer with validation browser and raw CSV uploader), Section C (Location Prediction across RF, k-NN [k=7], and Hybrid), Section D (2D Coordinate Scatter Map in native coordinate space), Section E (Prediction Evaluation against ground truth), Section F (Model Benchmarking with Task 09 verified metrics and separated experimental hybrid table).
  - **Strict Coordinate Invariant:** Axes explicitly labeled as `Dataset Longitude (Coordinate X)` and `Dataset Latitude (Coordinate Y)`; coordinate errors reported in `dataset coordinate units`, never metric metres; no architectural walls or rooms fabricated.
  - **Zero Data Leakage:** Target and metadata columns are strictly excluded from input feature matrices during CSV ingestion and inference.
  - **Offline Evaluation Disclosure:** Prominent callout explicitly stating the system demonstrates offline benchmark localization and does not perform live phone tracking or GPS trilateration.

### Task 12: Multitask Deep Learning (MLP) Architecture & Training Loop
- **Status:** COMPLETED
- **Completed On:** 2026-11-16
- **Deliverables Created:**
  - Neural Network Architecture & Model Orchestrator: [src/models/deep_learning/multitask_mlp.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/models/deep_learning/multitask_mlp.py) (`MultitaskMLP`, `IndoorAIMultitaskMLP`, `CoordinateStandardizer`, `SummaryFeatureScaler`, `MultitaskLoss`, `MLPHyperparameters`).
  - PyTorch Deep Learning Package Module: [src/models/deep_learning/__init__.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/models/deep_learning/__init__.py).
  - Training & Grouped Cross-Validation Pipeline: [src/training/train_multitask.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/training/train_multitask.py).
  - Diagnostic & Benchmark Plotting Engine: [src/evaluation/mlp_evaluation.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/evaluation/mlp_evaluation.py).
  - Hyperparameter Configuration: [configs/multitask_mlp_config.yaml](file:///Users/bineetkeshari/Desktop/IndoorAI/configs/multitask_mlp_config.yaml).
  - Architectural Specification: [docs/MLP_ARCHITECTURE.md](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/MLP_ARCHITECTURE.md).
  - Experiment Report: [experiments/mlp/reports/mlp_experiment_report.md](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/mlp/reports/mlp_experiment_report.md).
  - Trained Model Checkpoint: [experiments/mlp/models/multitask_mlp.pt](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/mlp/models/multitask_mlp.pt) (2.7 MB).
  - Validation Predictions: [experiments/mlp/predictions/val_predictions.parquet](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/mlp/predictions/val_predictions.parquet) and [val_predictions.csv](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/mlp/predictions/val_predictions.csv).
  - Evaluation Metrics JSON: [experiments/mlp/metrics/mlp_results.json](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/mlp/metrics/mlp_results.json).
  - Diagnostic Visualizations (6 PNGs): [experiments/mlp/visualizations/](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/mlp/visualizations/)
    - `loss_curves.png` (Total loss and component loss breakdown)
    - `accuracy_curves.png` (Building and Floor validation accuracy across epochs)
    - `confusion_matrices.png` (Building 3x3 and Floor 5x5 confusion matrix heatmaps)
    - `coordinate_predictions.png` (Actual vs predicted Longitude and Latitude scatter plots)
    - `error_distribution.png` (2D Euclidean spatial error distribution and empirical CDF)
    - `model_comparison_benchmark.png` (Grouped bar chart comparing RF, k-NN, and Multitask MLP)
  - Unit Test Suite: [tests/test_multitask_mlp.py](file:///Users/bineetkeshari/Desktop/IndoorAI/tests/test_multitask_mlp.py) (8 unit tests passing; 105 total project tests passing in 10.28s).
- **Key Empirical Results & Benchmarking:**
  - **New State-of-the-Art Floor Classification:** Achieved **90.37% Floor Accuracy** (Macro F1: 0.8988) on the official validation dataset, outperforming both Random Forest (88.66%) and k-NN (84.43%).
  - **Building-Floor Joint Accuracy:** Achieved **90.37%** (identical to floor accuracy as Building Accuracy was 99.82%), outperforming Random Forest (88.48%) and k-NN (84.43%).
  - **Coordinate Regression:** Longitude MAE: 8.71 ($R^2 = 0.9760$), Latitude MAE: 6.83 ($R^2 = 0.9634$), Mean 2D Euclidean Error: 12.18 dataset units, Median 2D Euclidean Error: 10.06 dataset units, 95th Percentile Error: 28.77 dataset units.
  - **Extreme Model Efficiency:** Model checkpoint size is only **2.7 MB** (207x smaller than Random Forest's 561 MB and 6.5x smaller than k-NN's 17.7 MB).
  - **Sub-Millisecond Latency:** Per-sample CPU inference latency of **0.0163 ms** (~61,000 predictions/sec), enabling real-time edge execution.
  - **Zero Data Leakage:** All summary feature and coordinate target standardizers were fitted strictly on training data; 5-Fold Grouped CV grouped strictly on `USERID` verified generalization across unseen surveyors (Building: 98.90±1.17%, Floor: 70.86±13.97%, Mean 2D Error: 19.97±3.23 units).

### Task 13: Android Application Scaffolding & Permissions Architecture
- **Status:** COMPLETED
- **Completed On:** 2026-11-17
- **Deliverables Created:**
  - Modern Gradle Multi-Project Configuration: [android/build.gradle.kts](file:///Users/bineetkeshari/Desktop/IndoorAI/android/build.gradle.kts), [android/settings.gradle.kts](file:///Users/bineetkeshari/Desktop/IndoorAI/android/settings.gradle.kts), [android/gradle.properties](file:///Users/bineetkeshari/Desktop/IndoorAI/android/gradle.properties), [android/gradle/wrapper/gradle-wrapper.properties](file:///Users/bineetkeshari/Desktop/IndoorAI/android/gradle/wrapper/gradle-wrapper.properties), [android/app/build.gradle.kts](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/build.gradle.kts), [android/app/proguard-rules.pro](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/proguard-rules.pro).
  - App Manifest & Resources: [android/app/src/main/AndroidManifest.xml](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/AndroidManifest.xml), [android/app/src/main/res/drawable/ic_location_pin.xml](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/res/drawable/ic_location_pin.xml), [android/app/src/main/res/values/colors.xml](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/res/values/colors.xml), [android/app/src/main/res/values/strings.xml](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/res/values/strings.xml), [android/app/src/main/res/values/themes.xml](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/res/values/themes.xml).
  - Main Activity & Navigation: [android/app/src/main/java/ai/indoor/app/MainActivity.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/MainActivity.kt), [android/app/src/main/java/ai/indoor/app/ui/navigation/Screen.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/ui/navigation/Screen.kt), [android/app/src/main/java/ai/indoor/app/ui/navigation/AppNavHost.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/ui/navigation/AppNavHost.kt).
  - Six Dedicated Jetpack Compose Screens:
    - [android/app/src/main/java/ai/indoor/app/ui/screens/WelcomeScreen.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/ui/screens/WelcomeScreen.kt) (Branding, feature overview, quick actions, privacy statement).
    - [android/app/src/main/java/ai/indoor/app/ui/screens/PermissionSetupScreen.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/ui/screens/PermissionSetupScreen.kt) (Granular rationale for Location, Wi-Fi, IMU Sensors, Camera, and Network).
    - [android/app/src/main/java/ai/indoor/app/ui/screens/WifiScanScreen.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/ui/screens/WifiScanScreen.kt) (Scan trigger, AP counter, BSSID/RSSI list, live/mock hardware toggle).
    - [android/app/src/main/java/ai/indoor/app/ui/screens/SensorMonitorScreen.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/ui/screens/SensorMonitorScreen.kt) (Tri-axial Accelerometer, Gyroscope, Magnetometer live telemetry, hardware availability).
    - [android/app/src/main/java/ai/indoor/app/ui/screens/ImageCaptureScreen.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/ui/screens/ImageCaptureScreen.kt) (Manual user capture, EXIF stripping, face blurring toggle, no background photography).
    - [android/app/src/main/java/ai/indoor/app/ui/screens/CaptureSessionScreen.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/ui/screens/CaptureSessionScreen.kt) (Session lifecycle, elapsed duration timer, observation counters, configurable backend URL [10.0.2.2 vs LAN IP]).
  - Custom UI Components & Material 3 Theme:
    - [android/app/src/main/java/ai/indoor/app/ui/components/AccessPointItem.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/ui/components/AccessPointItem.kt), [PermissionCard.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/ui/components/PermissionCard.kt), [SensorReadingCard.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/ui/components/SensorReadingCard.kt), [StatusBadge.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/ui/components/StatusBadge.kt).
    - [android/app/src/main/java/ai/indoor/app/ui/theme/Color.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/ui/theme/Color.kt), [Theme.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/ui/theme/Theme.kt), [Type.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/ui/theme/Type.kt).
  - MVVM ViewModels:
    - [android/app/src/main/java/ai/indoor/app/viewmodel/MainViewModel.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/viewmodel/MainViewModel.kt), [WifiScanViewModel.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/viewmodel/WifiScanViewModel.kt), [SensorViewModel.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/viewmodel/SensorViewModel.kt), [SessionViewModel.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/viewmodel/SessionViewModel.kt), [ImageCaptureViewModel.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/viewmodel/ImageCaptureViewModel.kt).
  - Domain & Clean Architecture Layer:
    - Domain Models: [AccessPoint.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/models/AccessPoint.kt), [CaptureSession.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/models/CaptureSession.kt), [ImuSample.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/models/ImuSample.kt), [LocalizationResult.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/models/LocalizationResult.kt), [WifiFingerprint.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/models/WifiFingerprint.kt).
    - Domain Use Cases: [ScanWifiUseCase.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/usecases/ScanWifiUseCase.kt), [ObserveSensorsUseCase.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/usecases/ObserveSensorsUseCase.kt), [ManageCaptureSessionUseCase.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/usecases/ManageCaptureSessionUseCase.kt).
  - Hardware & Mock Scanner Layer:
    - Wi-Fi Interfaces & Providers: [WifiScanner.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/wifi/WifiScanner.kt), [AndroidWifiScanner.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/wifi/AndroidWifiScanner.kt) (throttling-aware), [MockWifiScanner.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/wifi/MockWifiScanner.kt) (campus benchmark data).
    - Sensor Interfaces & Providers: [SensorProvider.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/sensors/SensorProvider.kt), [AndroidSensorProvider.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/sensors/AndroidSensorProvider.kt) (lifecycle safe), [MockSensorProvider.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/sensors/MockSensorProvider.kt).
  - Data & Networking Layer:
    - Retrofit API Service: [IndoorApiService.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/data/api/IndoorApiService.kt), [ApiClient.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/data/api/ApiClient.kt).
    - DTOs: `HealthResponseDto`, `ImuObservationDto`, `LocalizationRequestDto`, `LocalizationResponseDto`, `SessionTelemetryDto`, `WifiScanResultDto`.
    - Repository Pattern: [LocalizationRepository.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/data/repository/LocalizationRepository.kt), [LocalizationRepositoryImpl.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/data/repository/LocalizationRepositoryImpl.kt).
  - Android Unit Test Suite:
    - [MockWifiScannerTest.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/test/java/ai/indoor/app/wifi/MockWifiScannerTest.kt)
    - [MockSensorProviderTest.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/test/java/ai/indoor/app/sensors/MockSensorProviderTest.kt)
    - [WifiScanViewModelTest.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/test/java/ai/indoor/app/viewmodel/WifiScanViewModelTest.kt)
    - [SensorViewModelTest.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/test/java/ai/indoor/app/viewmodel/SensorViewModelTest.kt)
    - [SessionViewModelTest.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/test/java/ai/indoor/app/viewmodel/SessionViewModelTest.kt)
    - [ApiSerializationTest.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/test/java/ai/indoor/app/data/ApiSerializationTest.kt)
  - Architectural Documentation: [docs/ANDROID_APP_ARCHITECTURE.md](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/ANDROID_APP_ARCHITECTURE.md).
- **Key Capabilities & Architectural Invariants:**
  - **Clean Architecture & Decoupled Hardware:** All presentation components consume abstract use cases and interfaces, enabling immediate execution in emulator or hardware-free testing via mock providers without modifying application logic.
  - **Strict Android Security & Privacy:** Granular runtime permissions requested strictly just-in-time; camera operation strictly user-initiated; zero background tracking or automatic photography.
  - **Android Wi-Fi Throttling Mitigation:** Full handling of Android 9+ rate limits (4 scans/2 mins foreground; 1 scan/30 mins background) with graceful fallback to cached results and explicit scan timestamps.
  - **Lifecycle-Safe Sensor Streaming:** Listeners automatically unregistered when screens leave the foreground, eliminating battery drain.
  - **Flexible Backend Networking:** Configurable base URL supporting both Android Emulator host loopback (`http://10.0.2.2:8000/`) and physical LAN IP (`http://192.168.1.X:8000/`).
  - **Regression Safety:** Python test suite maintained at 105/105 passing tests across all preceding tasks.

### Task 14: Real Android Wi-Fi Fingerprint Collector
- **Status:** COMPLETED
- **Completed On:** 2026-11-17
- **Deliverables Created & Enhanced:**
  - Production Hardware Wi-Fi Scanner: [android/app/src/main/java/ai/indoor/app/wifi/AndroidWifiScanner.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/wifi/AndroidWifiScanner.kt) (WifiManager integration, SCAN_RESULTS_AVAILABLE_ACTION broadcast receiver, EXTRA_RESULTS_UPDATED freshness verification, BSSID deduplication, wall-clock epoch ms timestamp derivation, and foreground rate-throttling fallback).
  - Domain Data Models:
    - [android/app/src/main/java/ai/indoor/app/domain/models/AccessPoint.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/models/AccessPoint.kt) (BSSID, SSID, RSSI dBm, frequency MHz, 802.11 channel derivation, band identification, signal quality level, fresh vs cached status, mock flag).
    - [android/app/src/main/java/ai/indoor/app/domain/models/WifiFingerprint.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/models/WifiFingerprint.kt) (Epoch timestamp, duration, fresh/cached counts, strongest AP, mean RSSI, frequency bands).
    - [android/app/src/main/java/ai/indoor/app/domain/models/ManualLocationLabel.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/models/ManualLocationLabel.kt) (Surveyor ground-truth annotations: building, floor, room/area, notes).
    - [android/app/src/main/java/ai/indoor/app/domain/models/DeviceMetadata.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/models/DeviceMetadata.kt) (Non-identifying device telemetry: manufacturer, model, SDK level).
    - [android/app/src/main/java/ai/indoor/app/domain/models/CapturedFingerprint.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/models/CapturedFingerprint.kt) (Single and batch JSON and RFC-4180 CSV serialization).
  - Model Adaptation & Calibration Invariant:
    - [android/app/src/main/java/ai/indoor/app/domain/adapter/WifiFingerprintModelAdapter.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/adapter/WifiFingerprintModelAdapter.kt) (Architectural boundary explicitly preventing arbitrary mapping of real-world BSSIDs to UJIIndoorLoc WAP001–WAP520 feature columns; mandates empirical site survey calibration).
  - Enhanced UI & State Management:
    - [android/app/src/main/java/ai/indoor/app/ui/screens/WifiScanScreen.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/ui/screens/WifiScanScreen.kt) (Start Scan CTA, progress spinner, fresh vs cached breakdown, rate-throttling banners, interactive location permission launcher, ground-truth label editor, JSON/CSV export modals, and calibration warning banners).
    - [android/app/src/main/java/ai/indoor/app/ui/components/AccessPointItem.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/ui/components/AccessPointItem.kt) (SSID, BSSID monospace, RSSI dBm signal meter, frequency, channel number, fresh/cached pill tags).
    - [android/app/src/main/java/ai/indoor/app/viewmodel/WifiScanViewModel.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/viewmodel/WifiScanViewModel.kt) (Scan trigger, ground-truth capture management, JSON/CSV exports, calibration notice evaluation).
  - Comprehensive Unit Test Suites:
    - [android/app/src/test/java/ai/indoor/app/wifi/AndroidWifiScannerMappingTest.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/test/java/ai/indoor/app/wifi/AndroidWifiScannerMappingTest.kt) (Channel calculations, band identification, signal categorization, timestamp conversions).
    - [android/app/src/test/java/ai/indoor/app/domain/adapter/WifiFingerprintModelAdapterTest.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/test/java/ai/indoor/app/domain/adapter/WifiFingerprintModelAdapterTest.kt) (Uncalibrated real-world rejection, 520-WAP feature vector mapping).
    - [android/app/src/test/java/ai/indoor/app/domain/models/CapturedFingerprintExportTest.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/test/java/ai/indoor/app/domain/models/CapturedFingerprintExportTest.kt) (JSON/CSV serialization, header preservation, quote/comma escaping, mock/real flag separation).
    - [android/app/src/test/java/ai/indoor/app/viewmodel/WifiScanViewModelTest.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/test/java/ai/indoor/app/viewmodel/WifiScanViewModelTest.kt) (State transitions, ground-truth labeling, export flows, permission and throttling handling).
  - Architectural Documentation:
    - [docs/ANDROID_WIFI_COLLECTION.md](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/ANDROID_WIFI_COLLECTION.md) (Full workflow, permission analysis, throttling mechanics, schema definitions, export formats, calibration requirements, physical device testing guide).
- **Key Capabilities & Architectural Invariants:**
  - **No Fabricated Real Data:** All synthetic test data is strictly segregated under `isMock = true` and badged `DEMO DATA`; real scans require actual physical 802.11 beacon reception.
  - **No Arbitrary WAP Mapping:** Real-world BSSIDs are never mapped to UJIIndoorLoc WAP001–WAP520 feature columns arbitrarily; empirical survey calibration is explicitly required before ML inference can run.
  - **OS Throttling Resilience:** Fully handles Android 9+ rate limits (4 scans / 2 mins foreground) with cached result fallback and explicit UI indicators.
  - **Surveyor Ground-Truth Labeling & Local Export:** Allows surveyors to capture and export labeled training datasets to JSON and RFC-4180 CSV without cloud dependencies.

### Task 15: Android IMU & Magnetometer Streamer
- **Status:** COMPLETED
- **Completed On:** 2026-11-17
- **Deliverables Created & Enhanced:**
  - Production Hardware Sensor Streamer: [android/app/src/main/java/ai/indoor/app/sensors/AndroidSensorProvider.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/sensors/AndroidSensorProvider.kt) (Android SensorManager integration, TYPE_ACCELEROMETER, TYPE_GYROSCOPE, TYPE_MAGNETIC_FIELD, TYPE_ROTATION_VECTOR, nanosecond timestamp preservation, dynamic sampling rate profile configuration, low-pass gravity separation, and triad orientation fusion).
  - Synthetic Sensor Streamer: [android/app/src/main/java/ai/indoor/app/sensors/MockSensorProvider.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/sensors/MockSensorProvider.kt) (Realistic walking gait dynamics, ~1.2 Hz step cadence, ~9.81 m/s² gravity vector, compass orientation, and strict `isMock = true` tagging).
  - Physics & Sensor Fusion Foundation:
    - [android/app/src/main/java/ai/indoor/app/sensors/fusion/SensorFusionManager.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/sensors/fusion/SensorFusionManager.kt) (Cross-product triad orientation vector fusion computing Azimuth/Yaw, Pitch, Roll in degrees, and low-pass exponential moving average gravity filter `LowPassGravityFilter`).
  - Domain Data Models:
    - [android/app/src/main/java/ai/indoor/app/domain/models/SensorType.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/models/SensorType.kt) (ACCELEROMETER, GYROSCOPE, MAGNETIC_FIELD, ROTATION_VECTOR).
    - [android/app/src/main/java/ai/indoor/app/domain/models/SensorAccuracy.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/models/SensorAccuracy.kt) (HIGH, MEDIUM, LOW, UNRELIABLE).
    - [android/app/src/main/java/ai/indoor/app/domain/models/SensorSamplingRate.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/models/SensorSamplingRate.kt) (UI ~16 Hz, Game ~50 Hz, Fastest max rate).
    - [android/app/src/main/java/ai/indoor/app/domain/models/SingleSensorReading.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/models/SingleSensorReading.kt) (Nanosecond timestamp, accuracy, axis metrics, magnitude).
    - [android/app/src/main/java/ai/indoor/app/domain/models/CapturedSensorBatch.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/models/CapturedSensorBatch.kt) (JSON and RFC-4180 CSV serialization).
    - [android/app/src/main/java/ai/indoor/app/domain/models/ImuSample.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/models/ImuSample.kt) (Composite multimodal sample with raw axes, low-pass gravity, linear dynamic acceleration, and orientation angles).
  - UI & Telemetry Management:
    - [android/app/src/main/java/ai/indoor/app/ui/screens/SensorMonitorScreen.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/ui/screens/SensorMonitorScreen.kt) (Start/Stop controls, sampling profile exposed dropdown, 4 distinct visual cards for Accelerometer, Gyroscope, Magnetometer, and Orientation, nanosecond timestamp readouts, CSV/JSON export dialog).
    - [android/app/src/main/java/ai/indoor/app/ui/components/SensorReadingCard.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/ui/components/SensorReadingCard.kt) (Magnitude meters, accuracy badges, OrientationReadingCard with compass heading and cardinal direction).
    - [android/app/src/main/java/ai/indoor/app/viewmodel/SensorViewModel.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/viewmodel/SensorViewModel.kt) (State management, sampling profile switching, local exports).
    - [android/app/src/main/java/ai/indoor/app/ui/screens/CaptureSessionScreen.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/ui/screens/CaptureSessionScreen.kt) & [SessionViewModel.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/viewmodel/SessionViewModel.kt) (Capture session integration with explicit IMU streaming toggle and telemetry counter synchronization).
  - Unit Test Suites:
    - [android/app/src/test/java/ai/indoor/app/sensors/fusion/LowPassGravityFilterTest.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/test/java/ai/indoor/app/sensors/fusion/LowPassGravityFilterTest.kt) (Static gravity convergence, linear dynamic motion separation, filter reset).
    - [android/app/src/test/java/ai/indoor/app/sensors/fusion/SensorFusionManagerTest.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/test/java/ai/indoor/app/sensors/fusion/SensorFusionManagerTest.kt) (Triad orientation math, pitch/roll/yaw calculation, free-fall degenerate handling).
    - [android/app/src/test/java/ai/indoor/app/domain/models/CapturedSensorBatchExportTest.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/test/java/ai/indoor/app/domain/models/CapturedSensorBatchExportTest.kt) (Nanosecond timestamp preservation, RFC-4180 CSV header and rows, JSON batching, mock separation).
    - [android/app/src/test/java/ai/indoor/app/sensors/MockSensorProviderTest.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/test/java/ai/indoor/app/sensors/MockSensorProviderTest.kt) (Availability, sampling rates, physical plausibility, buffer clearing, exports).
    - [android/app/src/test/java/ai/indoor/app/viewmodel/SensorViewModelTest.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/test/java/ai/indoor/app/viewmodel/SensorViewModelTest.kt) (State transitions, sampling rate updates, CSV/JSON exports).
    - [android/app/src/test/java/ai/indoor/app/viewmodel/SessionViewModelTest.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/test/java/ai/indoor/app/viewmodel/SessionViewModelTest.kt) (Session lifecycle integration, sensor streaming toggle, counter updates).
  - Architectural Documentation:
    - [docs/ANDROID_SENSOR_COLLECTION.md](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/ANDROID_SENSOR_COLLECTION.md) (Sensor physics, units, anti-drift guarantees, sampling rates, low-pass filter equations, triad orientation math, export schemas, testing steps).
- **Key Capabilities & Architectural Invariants:**
  - **No IMU-Only Positioning Fallacy:** Clearly prevents and documents why double integration of raw acceleration causes quadratic position drift ($O(t^2)$); establishes groundwork for step-based PDR (Task 16+).
  - **Nanosecond Timestamp Preservation:** Preserves raw hardware `event.timestamp` in nanoseconds directly from the hardware FIFO, preventing distorted differential calculus ($\Delta t$).
  - **Lifecycle-Safe & User-Controlled:** Sensors stream strictly in the foreground during explicit monitoring or active capture sessions; immediately unregistered on `onStop()`.
  - **Graceful Hardware Degradation:** Safely monitors devices lacking gyroscopes or magnetometers without crashing.

### Task 16: Android User-Initiated Photo Capture & Privacy Sanitizer
- **Status:** COMPLETED
- **Completed On:** 2026-11-17
- **Deliverables Created & Enhanced:**
  - Production CameraX Integration & Hardware Isolation:
    - [android/app/src/main/java/ai/indoor/app/camera/CameraManager.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/camera/CameraManager.kt) (Camera hardware abstraction decoupling UI from CameraX).
    - [android/app/src/main/java/ai/indoor/app/camera/AndroidCameraXManager.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/camera/AndroidCameraXManager.kt) (CameraX `Preview`, `ImageCapture`, `ProcessCameraProvider`, lifecycle-safe binding, low-latency shutter, and automatic resource release).
    - [android/app/src/main/java/ai/indoor/app/camera/MockCameraManager.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/camera/MockCameraManager.kt) (Synthetic benchmark frame generation depicting hallway/doorway landmarks for camera-less devices and virtual testing).
  - Client-Side Privacy Sanitizer & On-Device Face Blur Engine:
    - [android/app/src/main/java/ai/indoor/app/camera/privacy/ImagePrivacySanitizer.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/camera/privacy/ImagePrivacySanitizer.kt) (EXIF GPS tag purge, hardware model/serial stripping, pixel orientation normalization, atomic temp-file write, safe failure handling).
    - [android/app/src/main/java/ai/indoor/app/camera/privacy/OnDeviceFaceDetector.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/camera/privacy/OnDeviceFaceDetector.kt) & [FaceBounds.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/camera/privacy/FaceBounds.kt) (Purely offline client-side face detection contracts).
    - [android/app/src/main/java/ai/indoor/app/camera/privacy/AndroidMediaFaceDetector.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/camera/privacy/AndroidMediaFaceDetector.kt) (100% offline Android framework `android.media.FaceDetector` implementation, zero network requests, zero cloud AI dependencies).
    - [android/app/src/main/java/ai/indoor/app/camera/privacy/FaceBlurApplicator.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/camera/privacy/FaceBlurApplicator.kt) (On-device pixelation and solid tint filters over detected face regions).
  - Application-Private Storage & Session Accounting:
    - [android/app/src/main/java/ai/indoor/app/camera/storage/PhotoStorageManager.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/camera/storage/PhotoStorageManager.kt) (Scoped storage in `context.filesDir/survey_photos/`, synchronized JSON/CSV indices, single photo deletion, batch exports).
    - [android/app/src/main/java/ai/indoor/app/domain/models/LandmarkType.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/models/LandmarkType.kt) (Enum: `CORRIDOR`, `DOOR`, `STAIRCASE`, `ELEVATOR`, `ROOM_SIGN`, `ENTRANCE`, `EXIT`, `LOBBY`, `OTHER`).
    - [android/app/src/main/java/ai/indoor/app/domain/models/FaceDetectionStatus.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/models/FaceDetectionStatus.kt) (Status: `SUCCEEDED`, `NO_FACES_DETECTED`, `UNAVAILABLE`, `FAILED`, `NOT_ATTEMPTED`).
    - [android/app/src/main/java/ai/indoor/app/domain/models/CapturedPhotoMetadata.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/models/CapturedPhotoMetadata.kt) (19 metadata attributes, RFC 4180 CSV serialization, indented JSON batching).
  - Jetpack Compose Material 3 UI & ViewModel:
    - [android/app/src/main/java/ai/indoor/app/viewmodel/ImageCaptureViewModel.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/viewmodel/ImageCaptureViewModel.kt) (JIT camera preview management, capture flow state machine: `Idle` -> `Capturing` -> `Reviewing` -> `Sanitizing` -> `Saved`, draft updates, face blur toggle, retake handling, active session telemetry).
    - [android/app/src/main/java/ai/indoor/app/ui/screens/ImageCaptureScreen.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/ui/screens/ImageCaptureScreen.kt) (JIT inactive viewfinder, alignment crosshair overlay, review modal dialog, landmark annotation inputs, privacy checklist badges, captured photo catalog, JSON/CSV export modals).
    - [android/app/src/main/java/ai/indoor/app/MainActivity.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/MainActivity.kt) & [AppNavHost.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/ui/navigation/AppNavHost.kt) (Integration with `SessionViewModel`, DI wiring).
  - Unit Test Suites:
    - [android/app/src/test/java/ai/indoor/app/camera/CapturedPhotoExportTest.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/test/java/ai/indoor/app/camera/CapturedPhotoExportTest.kt) (JSON escaping, RFC 4180 CSV escaping, batch exports, location string formatting, LandmarkType mapping).
    - [android/app/src/test/java/ai/indoor/app/camera/privacy/ImagePrivacySanitizerTest.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/test/java/ai/indoor/app/camera/privacy/ImagePrivacySanitizerTest.kt) (Face bounds math, safe failure on missing/empty files, status reporting).
    - [android/app/src/test/java/ai/indoor/app/viewmodel/ImageCaptureViewModelTest.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/test/java/ai/indoor/app/viewmodel/ImageCaptureViewModelTest.kt) (Strictly inactive initial state, permission transitions, draft updates, retake cleanup, face blur toggling).
  - Documentation:
    - [docs/ANDROID_IMAGE_CAPTURE.md](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/ANDROID_IMAGE_CAPTURE.md) (Architecture, anti-surveillance guarantees, landmark schema, JSON/CSV exports, physical device testing, headless emulator steps).
- **Key Capabilities & Architectural Invariants:**
  - **No Background Photography:** The camera is strictly inactive at startup and opens exclusively when the surveyor taps "Start Camera Preview".
  - **Zero EXIF Geolocation Leakage:** All GPS tags and device identifying attributes are permanently stripped.
  - **100% On-Device Face Obfuscation:** Uses Android's native framework face detection without cloud inference or biometric identification.
  - **Review Before Persist:** Surveyor reviews the photo, face blur, and landmark labels before saving.
  - **Session Integration:** Increments `photoCount` on the active survey session and stores all data in app-private scoped storage.

### Task 17: Unified Multimodal Capture Session & Dataset Builder
- **Status:** COMPLETED
- **Completed On:** 2026-11-17
- **Deliverables Created & Enhanced:**
  - Unified Session State Machine & Models:
    - [android/app/src/main/java/ai/indoor/app/domain/models/SurveySession.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/models/SurveySession.kt) (Central session model with UUID, lifecycle state, building/floor/room labels, observation telemetry counts, duration, and clock anchor).
    - [android/app/src/main/java/ai/indoor/app/domain/models/SurveySessionStatus.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/models/SurveySessionStatus.kt) (Enum: `CREATED`, `ACTIVE`, `PAUSED`, `COMPLETED`, `CANCELLED`).
    - [android/app/src/main/java/ai/indoor/app/domain/models/DataSourceType.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/models/DataSourceType.kt) (Enum: `REAL`, `MOCK`, `HYBRID`).
    - [android/app/src/main/java/ai/indoor/app/domain/models/SurveyAnnotation.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/models/SurveyAnnotation.kt) (Ground-truth annotations: Corridor, Door, Room Entrance, Staircase, Elevator, Landmark, Reference Point, Location Correction, Note).
    - [android/app/src/main/java/ai/indoor/app/domain/models/SurveyWifiObservation.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/models/SurveyWifiObservation.kt) (Wi-Fi sweep model with normalized timestamps, AP list, and CSV serialization).
    - [android/app/src/main/java/ai/indoor/app/domain/models/SurveySensorObservation.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/models/SurveySensorObservation.kt) (IMU/Magnetometer 3-axis readings with original nanoseconds and normalized timestamps).
    - [android/app/src/main/java/ai/indoor/app/domain/models/SurveyPhotoObservation.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/models/SurveyPhotoObservation.kt) (Sanitized photo observation metadata referencing local files).
  - Timestamp Normalization & Synchronization Layer:
    - [android/app/src/main/java/ai/indoor/app/domain/timestamp/ClockAnchor.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/timestamp/ClockAnchor.kt) (Synchronized atomic snapshot of wall-clock epoch ms and boottime nanoseconds).
    - [android/app/src/main/java/ai/indoor/app/domain/timestamp/TimeSource.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/timestamp/TimeSource.kt) & `SyncMethod.kt` (Original clock domain tracking and conversion method enumeration).
    - [android/app/src/main/java/ai/indoor/app/domain/timestamp/TimestampRecord.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/timestamp/TimestampRecord.kt) (Preserves raw timestamps, records normalized UTC epoch ms, ISO-8601 string, and uncertainty bounds).
    - [android/app/src/main/java/ai/indoor/app/domain/timestamp/TimestampNormalizer.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/timestamp/TimestampNormalizer.kt) (Normalization logic across boottime nanos, boottime micros, and wall-clock millis).
  - Unified Chronological Timeline:
    - [android/app/src/main/java/ai/indoor/app/domain/timeline/SessionTimelineItem.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/timeline/SessionTimelineItem.kt) (Unified timeline item model with inspection metadata).
    - [android/app/src/main/java/ai/indoor/app/domain/timeline/ObservationTimelineBuilder.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/timeline/ObservationTimelineBuilder.kt) (Chronological multi-stream aggregator and sorter).
  - Persistence & Interrupted Session Recovery:
    - [android/app/src/main/java/ai/indoor/app/domain/storage/SessionPersistenceManager.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/storage/SessionPersistenceManager.kt) (App-private storage in `filesDir/survey_sessions/<id>/`, JSONL streaming observation journals, crash/interruption recovery converting `ACTIVE` $\rightarrow$ `PAUSED` with explicit flag, and zero-orphan image cleanup).
  - Dataset Packaging & Verification:
    - [android/app/src/main/java/ai/indoor/app/domain/export/ExportManifest.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/export/ExportManifest.kt) (Manifest schema tracking files, record counts, real/mock source counts, and SHA-256 hashes).
    - [android/app/src/main/java/ai/indoor/app/domain/export/ExportValidator.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/export/ExportValidator.kt) (Pre-export validation enforcing parent session references, unique IDs, photo file existence, and mandatory unsanitized image rejection).
    - [android/app/src/main/java/ai/indoor/app/domain/export/DatasetExporter.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/export/DatasetExporter.kt) (Builds `indoorai_session_<id>.zip` with `session.json`, `wifi_observations.csv`, `imu_observations.csv`, `photo_metadata.csv`, `annotations.csv`, `images/`, and `manifest.json`).
  - ViewModels & UI:
    - [android/app/src/main/java/ai/indoor/app/domain/usecases/ManageCaptureSessionUseCase.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/domain/usecases/ManageCaptureSessionUseCase.kt) (Multimodal usecase coordinating lifecycle, ingestion, timeline, and export).
    - [android/app/src/main/java/ai/indoor/app/viewmodel/SessionViewModel.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/viewmodel/SessionViewModel.kt) (Lifecycle actions, timeline StateFlow, export triggering, IMU sensor stream buffering).
    - [android/app/src/main/java/ai/indoor/app/ui/screens/CaptureSessionScreen.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/ui/screens/CaptureSessionScreen.kt) (Material 3 UI: active session timer, interrupted recovery alert, observation counters with real/mock breakdown, add annotation dialog, interactive timeline inspection dialog, ZIP export dialog, session history).
  - Unit Test Suites:
    - [android/app/src/test/java/ai/indoor/app/domain/timestamp/TimestampNormalizerTest.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/test/java/ai/indoor/app/domain/timestamp/TimestampNormalizerTest.kt) (Anchor math, boottime to wall-clock, invalid timestamp safety, Wi-Fi cache uncertainty).
    - [android/app/src/test/java/ai/indoor/app/domain/models/SurveySessionLifecycleTest.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/test/java/ai/indoor/app/domain/models/SurveySessionLifecycleTest.kt) (Lifecycle transitions, duration, source type classification, JSON roundtrip).
    - [android/app/src/test/java/ai/indoor/app/domain/storage/SessionPersistenceTest.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/test/java/ai/indoor/app/domain/storage/SessionPersistenceTest.kt) (Disk persistence, crash recovery from `ACTIVE` to `PAUSED`, image cleanup on delete).
    - [android/app/src/test/java/ai/indoor/app/domain/export/DatasetExportAndValidationTest.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/test/java/ai/indoor/app/domain/export/DatasetExportAndValidationTest.kt) (Full ZIP generation, SHA-256 verification, manifest validation, rejection of unsanitized images and missing files).
    - [android/app/src/test/java/ai/indoor/app/domain/timeline/ObservationTimelineTest.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/test/java/ai/indoor/app/domain/timeline/ObservationTimelineTest.kt) (Timeline sorting, real vs mock differentiation, uncertain timing handling).
  - Documentation:
    - [docs/MULTIMODAL_SESSION_ARCHITECTURE.md](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/MULTIMODAL_SESSION_ARCHITECTURE.md) (Unified session architecture, clock anchor synchronization, persistence, recovery, ZIP export, and validation).

### Task 18: Multimodal Dataset Ingestion & Quality Assurance Backend
- **Status:** COMPLETED
- **Completed On:** 2026-11-17
- **Deliverables Created & Enhanced:**
  - Secure Extraction & Storage Layer:
    - [backend/app/core/dataset_config.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/core/dataset_config.py) (Configurable size thresholds, allowed extensions, required files, and directory layout).
    - [backend/app/services/dataset_extraction.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/services/dataset_extraction.py) (`DatasetExtractor` enforcing PK signature check, path traversal prevention, symlink rejection, ZIP bomb 100:1 ratio limit, 64KB chunk streaming, extension whitelisting, and atomic cleanup on error).
  - Validation & Quality Assurance Engine:
    - [backend/app/services/dataset_validation.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/services/dataset_validation.py) (`DatasetValidator` verifying SHA-256 digests vs `manifest.json`, CSV headers, session ID referential integrity, privacy policy sanitization, statistical Wi-Fi/IMU/photo/annotation quality metrics, and cross-modal temporal overlap).
  - Registry & Ingestion Orchestration:
    - [backend/app/services/dataset_registry.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/services/dataset_registry.py) (`DatasetRegistry` providing thread-safe, atomic disk persistence in `data/survey_datasets/registry.json`).
    - [backend/app/services/dataset_ingestion.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/services/dataset_ingestion.py) (`DatasetIngestionService` coordinating ZIP preservation, extraction, QA report generation, and cataloging).
  - Schemas & REST API Endpoints:
    - [backend/app/schemas/dataset.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/schemas/dataset.py) (Pydantic models for QA reports, modality metrics, integrity checks, and summaries).
    - [backend/app/api/routes/datasets.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/api/routes/datasets.py) (FastAPI endpoints: `POST /api/datasets/upload`, `GET /api/datasets`, `GET /api/datasets/{id}`, `GET /api/datasets/{id}/quality`, `POST /api/datasets/{id}/validate`, `DELETE /api/datasets/{id}`).
    - [backend/app/main.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/main.py) (FastAPI application entrypoint with CORS, health probes, and OpenAPI documentation).
  - Verification & Documentation:
    - [tests/test_dataset_ingestion.py](file:///Users/bineetkeshari/Desktop/IndoorAI/tests/test_dataset_ingestion.py) (25 synthetic unit and API integration tests covering security, corrupted archives, manifest validation, privacy checks, and endpoints).
    - [docs/DATASET_INGESTION_ARCHITECTURE.md](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/DATASET_INGESTION_ARCHITECTURE.md) (Comprehensive ingestion architecture, ZIP schemas, defense-in-depth security, and API documentation).







