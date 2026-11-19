# IndoorAI — Implementation Status & Task Tracker

This document tracks progress across all 30 implementation tasks defined in the [Development Roadmap](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/DEVELOPMENT_ROADMAP.md).

**Summary Progress:**
- **Completed:** 30 / 30 tasks (100.0%)
- **In Progress:** 0 / 30 tasks
- **Not Started:** 0 / 30 tasks

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
| **Task 19** | **Multimodal Dataset Explorer & Visualization Dashboard** | **Phase 4** | **COMPLETED** | **2026-11-17** |
| **Task 20** | **Multimodal Data Preprocessing & Temporal Alignment Pipeline** | **Phase 5** | **COMPLETED** | **2026-11-17** |
| **Task 21** | **Pedestrian Dead Reckoning (PDR) Engine** | **Phase 6** | **COMPLETED** | **2026-11-17** |
| **Task 22** | **PDR Calibration, Ground-Truth Validation & Drift Analysis** | **Phase 6** | **COMPLETED** | **2026-11-18** |
| **Task 23** | **Site-Specific Wi-Fi Radio Map & Fingerprint Localization** | **Phase 6** | **COMPLETED** | **2026-11-18** |
| **Task 24** | **Wi-Fi + PDR Sensor Fusion using an Extended Kalman Filter** | **Phase 7** | **COMPLETED** | **2026-11-18** |
| **Task 25** | **Computer Vision-Based Indoor Landmark Detection** | **Phase 7** | **COMPLETED** | **2026-11-18** |
| **Task 26** | **Indoor Map Generation & Spatial Representation** | **Phase 7** | **COMPLETED** | **2026-11-18** |
| **Task 27** | **Map-Constrained Multimodal Localization using a Particle Filter** | **Phase 7** | **COMPLETED** | **2026-11-18** |
| **Task 28** | **Indoor Navigation & Intelligent Route Planning** | **Phase 8** | **COMPLETED** | **2026-11-18** |
| **Task 29** | **Real-Time Ingestion & Inference Streaming Endpoints** | **Phase 8** | **COMPLETED** | **2026-11-18** |
| **Task 30** | **IndoorAI Final System Validation, Security & Release Readiness** | **Phase 8** | **COMPLETED** | **2026-11-19** |

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

### Task 19: Multimodal Dataset Explorer & Visualization Dashboard
- **Status:** COMPLETED
- **Completed On:** 2026-11-17
- **Deliverables Created & Enhanced:**
  - Backend Observation & Image Serving Layer:
    - [backend/app/api/routes/datasets.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/api/routes/datasets.py) (Added paged observation endpoints `GET /api/datasets/{id}/observations/{modality}` for `wifi`, `sensors`, `photos`, and `annotations`; added privacy-sanitized image streaming `GET /api/datasets/{id}/images/{filename}` with directory traversal defenses and mandatory `photo_metadata.csv` EXIF sanitization verification returning 403 Forbidden for unsanitized photos).
  - Robust Frontend API Client:
    - [frontend/api_client.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/api_client.py) (`DatasetApiClient` implementing typed responses, error mapping, timeouts, health checks, multi-part upload, quality retrieval, revalidation, deletion, paged observation streaming, image byte retrieval, and support for injected test clients).
  - Modular Visualization & Inspection Components:
    - [frontend/components/__init__.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/components/__init__.py) (Component package exports).
    - [frontend/components/overview.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/components/overview.py) (Catalog KPI metric cards, dataset summary table, active dataset selector dropdown, and visual privacy audit disclaimers).
    - [frontend/components/upload.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/components/upload.py) (Survey ZIP uploader with client size check, dry-run validation toggle, upload progress indicators, validation status badges, and error diagnostics).
    - [frontend/components/quality_view.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/components/quality_view.py) (6-domain QA inspection: File Integrity, Wi-Fi Quality, Sensor Quality, Photo Quality, Ground-Truth Annotations, and Diagnostic Logs).
    - [frontend/components/wifi_view.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/components/wifi_view.py) (Interactive Wi-Fi fingerprint explorer: sweep selector, AP frequency metrics, detected AP tables, RSSI distribution histogram, top observed BSSIDs chart, and AP visibility count timeline).
    - [frontend/components/sensor_view.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/components/sensor_view.py) (Tri-axial IMU and magnetometer explorer: stream selection for Accelerometer, Gyroscope, Magnetometer, and Rotation Vector; axis filtering for X, Y, Z, Magnitude; interactive Plotly zoom/pan chart; downsampling for high-frequency streams; and sampling frequency metrics).
    - [frontend/components/timeline_view.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/components/timeline_view.py) (Unified multimodal timeline: Gantt-style concurrency chart, all-modality overlap duration, average clock uncertainty metrics, and chronological observation feed).
    - [frontend/components/photo_view.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/components/photo_view.py) (Privacy-conscious photo viewer: client EXIF sanitization verification, visual audit status badges, photo metadata cards, and ground-truth spatial annotations table).
    - [frontend/components/comparison_view.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/components/comparison_view.py) (Multi-dataset comparative analysis: side-by-side KPI comparison across observation counts, AP density, IMU sampling rate, and modality breakdowns).
    - [frontend/components/management_view.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/components/management_view.py) (Dataset lifecycle management: server revalidation trigger, raw diagnostic error logs, and permanent deletion with explicit confirmation checkbox).
  - Main Dashboard Integration:
    - [frontend/app.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/app.py) (Added top-level sidebar workspace switcher: `Multimodal Survey Explorer (Tasks 17–19)` and `UJIIndoorLoc Benchmark & Inference (Tasks 07–12)`, preserving 100% of legacy model loading, inference, and floorplan maps while adding backend health monitoring and 9 explorer views).
  - Verification & Documentation:
    - [tests/test_multimodal_explorer.py](file:///Users/bineetkeshari/Desktop/IndoorAI/tests/test_multimodal_explorer.py) (15 comprehensive unit and API integration tests covering health check, dataset listing, detail inspection, quality reports, observation retrieval, photo serving, privacy rejections, sweep grouping, sensor sampling math, and timeline overlap calculation).
    - [docs/MULTIMODAL_DATASET_EXPLORER.md](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/MULTIMODAL_DATASET_EXPLORER.md) (Architectural specification, component roles, observation and photo endpoints, privacy invariants, large dataset handling, and roadmap status).

### Task 20: Multimodal Data Preprocessing & Temporal Alignment Pipeline
- **Status:** COMPLETED
- **Completed On:** 2026-11-17
- **Deliverables Created & Enhanced:**
  - Dedicated Preprocessing Engine (`src/multimodal_preprocessing/`):
    - [src/multimodal_preprocessing/config.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/multimodal_preprocessing/config.py) (`MultimodalPreprocessingConfig` defining RSSI bounds, deduplication policies, gap thresholds, window durations, segmentation strategies, and output settings).
    - [src/multimodal_preprocessing/validation.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/multimodal_preprocessing/validation.py) (`MultimodalPreValidator` pre-flight validation verifying required metadata files, CSV structures, and session consistency).
    - [src/multimodal_preprocessing/wifi.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/multimodal_preprocessing/wifi.py) (`WifiPreprocessor` performing BSSID normalization, RSSI gating, frequency filtering, sweep clustering, in-sweep deduplication, and sweep feature extraction).
    - [src/multimodal_preprocessing/sensors.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/multimodal_preprocessing/sensors.py) (`SensorPreprocessor` cleaning tri-axial accelerometer, gyroscope, magnetometer, and rotation vector streams; computing vector magnitude, instantaneous frequency, and jerk derivatives; and detecting sampling gaps without silent interpolation).
    - [src/multimodal_preprocessing/timestamps.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/multimodal_preprocessing/timestamps.py) (`TimestampAligner` performing uniform cross-modal window discretization, concurrent modality tracking, and boundary timing ambiguity detection).
    - [src/multimodal_preprocessing/photos.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/multimodal_preprocessing/photos.py) (`PhotoMetadataProcessor` validating image disk file existence, filtering unsanitized EXIF records, and tagging visual privacy audit status).
    - [src/multimodal_preprocessing/annotations.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/multimodal_preprocessing/annotations.py) (`AnnotationProcessor` canonicalizing architectural labels into standardized categories and recording surveyor manual provenance).
    - [src/multimodal_preprocessing/segmentation.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/multimodal_preprocessing/segmentation.py) (`SessionSegmenter` partitioning continuous survey sessions into analytical segments via fixed duration, observation counts, annotation boundaries, or temporal gaps).
    - [src/multimodal_preprocessing/reporting.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/multimodal_preprocessing/reporting.py) (`PreprocessingReport`, `PreprocessingManifest`, and SHA-256 cryptographic auditing).
    - [src/multimodal_preprocessing/pipeline.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/multimodal_preprocessing/pipeline.py) (`MultimodalPreprocessingPipeline` orchestrating pre-validation, cleaning, alignment, Parquet artifact generation, and manifest logging).
  - Backend API Schemas & Service Endpoints:
    - [backend/app/schemas/preprocessing.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/schemas/preprocessing.py) (Pydantic request/response schemas for preprocessing execution, report retrieval, and processed file listings).
    - [backend/app/services/preprocessing_service.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/services/preprocessing_service.py) (`PreprocessingService` managing execution, status checks, and processed file cataloging).
    - [backend/app/api/routes/datasets.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/api/routes/datasets.py) (Added endpoints `POST /api/datasets/{id}/preprocess`, `GET /api/datasets/{id}/preprocessing`, and `GET /api/datasets/{id}/preprocessing/files`).
  - Frontend API Client & Streamlit UI Integration:
    - [frontend/api_client.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/api_client.py) (Added methods `preprocess_dataset()`, `get_preprocessing_status()`, and `get_processed_files()`).
    - [frontend/components/preprocessing_view.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/components/preprocessing_view.py) (Interactive Preprocessing & Alignment view with configurable thresholds, run button, diagnostic KPI cards, cleaning breakdown, temporal alignment stats, segments table, and artifact viewer).
    - [frontend/app.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/app.py) (Added `10. Preprocessing & Alignment` to Multimodal Explorer navigation).
  - Testing & Verification:
    - [tests/test_multimodal_preprocessing.py](file:///Users/bineetkeshari/Desktop/IndoorAI/tests/test_multimodal_preprocessing.py) (14 unit and API integration tests covering cleaning, deduplication, gap detection, temporal alignment, photo sanitization gating, annotation canonicalization, segmentation, reproducibility, single-modality datasets, and endpoints).
    - Full repository test run passing 159 tests with 0 failures, 0 errors, and zero regressions on existing UJIIndoorLoc baselines.
  - Architectural Documentation:
    - [docs/MULTIMODAL_PREPROCESSING_ARCHITECTURE.md](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/MULTIMODAL_PREPROCESSING_ARCHITECTURE.md) (Architecture specification, cleaning rules, kinematics, window alignment, privacy gates, segmentation, and manifest schemas).

### Task 21: Pedestrian Dead Reckoning (PDR) Engine
- **Status:** COMPLETED
- **Completed On:** 2026-11-17
- **Deliverables Created & Enhanced:**
  - Dedicated PDR Engine Package (`src/pdr/`):
    - [src/pdr/config.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/pdr/config.py) (`PDRConfig` dataclass encapsulating gravity thresholds, filter frequencies, step detection parameters, Weinberg factor, and heading sources).
    - [src/pdr/coordinate_frames.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/pdr/coordinate_frames.py) (Transformations across Smartphone Body, Local Navigation ENU, Gravity-Aligned, and Horizontal Movement planes; quaternion math and Euler angles).
    - [src/pdr/filtering.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/pdr/filtering.py) (Zero-phase Butterworth lowpass and bandpass filters; streaming exponential smoothing with sampling-gap state flushes).
    - [src/pdr/gravity.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/pdr/gravity.py) (`GravityCompensator` supporting Method A: Android Rotation Vector conjugate rotation, and Method B: Low-pass accelerometer filtering with automatic fallback).
    - [src/pdr/orientation.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/pdr/orientation.py) (`OrientationEstimator` estimating roll, pitch, yaw from Rotation Vector, Gyroscope integration, or tilt-compensated Magnetometer with magnetic anomaly detection).
    - [src/pdr/heading.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/pdr/heading.py) (`HeadingEstimator` extracting per-step heading angles $\theta_k$ and propagating sensor-specific angular uncertainties).
    - [src/pdr/step_detection.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/pdr/step_detection.py) (`StepDetector` implementing Butterworth bandpass filtering, prominence-based peak extraction, and refractory duplicate rejection within 330 ms).
    - [src/pdr/step_length.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/pdr/step_length.py) (`StepLengthEstimator` implementing calibrated Weinberg bounce model $L = K \cdot (a_{\max}-a_{\min})^{0.25}$ clamped to physiological bounds $[0.30, 1.20]$ m).
    - [src/pdr/uncertainty.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/pdr/uncertainty.py) (`PDRUncertaintyPropagator` computing 2D spatial covariance accumulation $P_k = P_{k-1} + J_k Q_k J_k^T$ and 95% confidence uncertainty radius).
    - [src/pdr/trajectory.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/pdr/trajectory.py) (`TrajectoryReconstructor` accumulating 2D relative dead reckoning coordinates starting from origin $(0, 0)$).
    - [src/pdr/reporting.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/pdr/reporting.py) (`PDRReport`, `PDRManifest`, and SHA-256 cryptographic verification).
    - [src/pdr/pipeline.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/pdr/pipeline.py) (`PDRPipeline` end-to-end execution orchestrator writing 5 Parquet artifacts, report, and manifest).
    - [src/pdr/__init__.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/pdr/__init__.py) and [src/sensor_fusion/__init__.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/sensor_fusion/__init__.py) (Package exports).
  - Backend API Schemas & Service Endpoints:
    - [backend/app/schemas/pdr.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/schemas/pdr.py) (Pydantic request/response models for PDR execution, status, trajectory points, step events, and report).
    - [backend/app/services/pdr_service.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/services/pdr_service.py) (`PDRService` executing PDR pipeline, querying cached parquet artifacts, and caching reports).
    - [backend/app/api/routes/datasets.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/api/routes/datasets.py) (Added REST endpoints: `POST /api/datasets/{id}/pdr`, `GET /api/datasets/{id}/pdr`, `GET /api/datasets/{id}/pdr/trajectory`, `GET /api/datasets/{id}/pdr/steps`, `GET /api/datasets/{id}/pdr/report`).
  - Frontend UI & Visualizations:
    - [frontend/api_client.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/api_client.py) (Added PDR client methods: `run_pdr()`, `get_pdr_status()`, `get_pdr_trajectory()`, `get_pdr_steps()`).
    - [frontend/components/pdr_view.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/components/pdr_view.py) (Interactive Section 11 view: KPI metric cards, 2D Plotly trajectory with uncertainty ellipses, step events table, and heading-over-time plots).
    - [frontend/components/__init__.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/components/__init__.py) and [frontend/app.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/app.py) (Integrated `11. PDR & Movement Analysis` into sidebar navigation).
  - Testing & Verification:
    - [tests/test_pdr.py](file:///Users/bineetkeshari/Desktop/IndoorAI/tests/test_pdr.py) (9 comprehensive unit and API integration tests covering coordinate transformations, gravity filtering, walking step detection with refractory rejection, Weinberg step length estimation, 2D dead reckoning with L-turn trajectory, covariance growth, and all 5 FastAPI REST endpoints).
    - Full repository test run passing 168 tests (159 previous + 9 PDR) with 0 failures, 0 errors, and zero regressions on baseline models.
  - Architectural Documentation:
    - [docs/PDR_ENGINE_ARCHITECTURE.md](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/PDR_ENGINE_ARCHITECTURE.md) (Complete specification covering coordinate frames, gravity compensation methods, Butterworth filtering, heading sources, magnetic disturbance handling, Weinberg model, spatial covariance Jacobian math, Parquet schemas, and system boundaries).

### Task 22: PDR Calibration, Ground-Truth Validation & Drift Analysis
- **Status:** COMPLETED
- **Completed On:** 2026-11-18
- **Deliverables Created & Enhanced:**
  - Dedicated Calibration & Evaluation Package (`src/pdr/calibration/`):
    - [src/pdr/calibration/models.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/pdr/calibration/models.py) (`WalkingTrial`, `GroundTruthMeasurement`, `ReferenceWaypoint`, `TrialType`, `PhonePlacement`, `WalkingCondition`, `CalibrationProfile`, `CalibrationStatus`, `ConfidenceEllipse`, `StepDetectionMetrics`, `HeadingMetrics`, `TrajectoryDriftMetrics`, `PDREvaluationReport`).
    - [src/pdr/calibration/storage.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/pdr/calibration/storage.py) (`CalibrationStorage` managing trials, results, and registry in `data/calibration/pdr/` with default uncalibrated profile `PROF-DEFAULT`).
    - [src/pdr/calibration/step_length_calibrator.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/pdr/calibration/step_length_calibrator.py) (`WeinbergCalibrator` implementing bounded scalar optimization over $K \in [0.20, 0.80]$ with physiological clamping, weighted relative error, and separate train/validation metrics).
    - [src/pdr/calibration/step_evaluator.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/pdr/calibration/step_evaluator.py) (`StepDetectionEvaluator` computing TP, FP, FN, precision, recall, F1-score, and count error against manual counts or reference timestamps).
    - [src/pdr/calibration/heading_evaluator.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/pdr/calibration/heading_evaluator.py) (`HeadingEvaluator` implementing circular statistics, wraparound handling in $[-180^\circ, 180^\circ]$, mean/median/p95 errors, drift rate deg/min, and magnetic disturbance tracking).
    - [src/pdr/calibration/trajectory_evaluator.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/pdr/calibration/trajectory_evaluator.py) (`TrajectoryDriftEvaluator` computing distance error, endpoint Euclidean error, relative endpoint error, net displacement error, drift per meter, drift per minute, and cross-track error).
    - [src/pdr/calibration/uncertainty_evaluator.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/pdr/calibration/uncertainty_evaluator.py) (`UncertaintyAuditor` computing exact 2D Gaussian 95% confidence ellipse parameters via $\chi^2_{2, 0.95} = 5.99146$, conservative circular bound, and empirical ground-truth containment coverage via Mahalanobis distance).
    - [src/pdr/calibration/evaluator.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/pdr/calibration/evaluator.py) (`PDREvaluator` orchestrating step detection, heading, trajectory drift, and uncertainty evaluation into structured `PDREvaluationReport`).
    - [src/pdr/calibration/__init__.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/pdr/calibration/__init__.py) (Package public exports).
  - Enhanced Core PDR Kinematics (`src/pdr/uncertainty.py`, `src/pdr/trajectory.py`):
    - [src/pdr/uncertainty.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/pdr/uncertainty.py) (Added 2D Chi-square 95% confidence ellipse metrics: `semi_major_axis_m`, `semi_minor_axis_m`, `ellipse_angle_deg`, `conservative_95_radius_m`; added augmented 3-state $[x, y, \theta]^T$ covariance update `update_with_heading_drift()`).
    - [src/pdr/trajectory.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/pdr/trajectory.py) (Propagated confidence ellipse parameters into sequential trajectory points).
  - Backend API Schemas & Service Endpoints:
    - [backend/app/schemas/pdr_calibration.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/schemas/pdr_calibration.py) (Pydantic request/response schemas for trial registration, Weinberg calibration, profile retrieval, and evaluation).
    - [backend/app/services/pdr_calibration_service.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/services/pdr_calibration_service.py) (`PDRCalibrationService` managing trial storage, Weinberg calibration, profile retrieval, and evaluation reports).
    - [backend/app/api/routes/datasets.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/api/routes/datasets.py) (Added 7 REST endpoints: `POST /pdr/trials`, `GET /pdr/trials`, `GET /pdr/trials/{id}`, `POST /pdr/calibrate`, `GET /pdr/calibration`, `POST /pdr/evaluate`, `GET /pdr/evaluation`).
  - Frontend UI & Visualizations:
    - [frontend/api_client.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/api_client.py) (Added client methods for trials, calibration, profiles, and evaluation).
    - [frontend/components/pdr_calibration_view.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/components/pdr_calibration_view.py) (Interactive Section 12 dashboard view: Weinberg Stride Calibration, Walking Trials Registry, Trajectory Drift & 95% Ellipses with Plotly, and Step & Heading Validation).
    - [frontend/components/__init__.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/components/__init__.py) and [frontend/app.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/app.py) (Registered `12. PDR Calibration & Validation` in sidebar navigation).
  - Testing & Verification:
    - [tests/test_pdr_calibration.py](file:///Users/bineetkeshari/Desktop/IndoorAI/tests/test_pdr_calibration.py) (15 comprehensive unit and API integration tests covering trial serialization, Weinberg optimization, insufficient data fallback, train/val separation, step precision/recall/F1, circular wraparound, heading drift rate, endpoint and cross-track error, 2D confidence ellipse calculation, 3-state covariance update, profile resolution hierarchy, and all FastAPI calibration endpoints).
    - Full repository test run passing **183 tests** (168 previous + 15 new) with 0 failures, 0 errors, and zero regressions.
  - Architectural Documentation:
    - [docs/PDR_CALIBRATION_AND_VALIDATION.md](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/PDR_CALIBRATION_AND_VALIDATION.md) (Complete specification covering trial schemas, measurement methodology, Weinberg optimization math, circular statistics, drift metrics, 2D Chi-square confidence ellipse derivation, and system limitations).

### Task 23: Site-Specific Wi-Fi Radio Map & Fingerprint Localization
- **Status:** COMPLETED
- **Completed On:** 2026-11-18
- **Deliverables Created & Enhanced:**
  - Dedicated Site-Specific Wi-Fi Package (`src/wifi_localization/`):
    - [src/wifi_localization/models.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/wifi_localization/models.py) (`ReferenceLocation`, `ApObservation`, `SurveySweep`, `ApStatistics`, `LocationRadioProfile`, `RadioMapMetadata`, `RadioMap`, `CandidateMatch`, `LocalizationPrediction`, `EvaluationMetrics`).
    - [src/wifi_localization/normalization.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/wifi_localization/normalization.py) (Lowercase colon-separated BSSID regex normalization, $[-105, -20]$ dBm bounds validation, multi-band frequency classification [2.4GHz, 5GHz, 6GHz], in-sweep duplicate elimination via maximum RSSI).
    - [src/wifi_localization/radio_map_builder.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/wifi_localization/radio_map_builder.py) (`RadioMapBuilder` extracting temporal sweeps from flat survey observations, computing AP visibility rate $V = N/N_{\text{sweeps}}$, mean, median, sample std with Bessel correction, min/max RSSI, and compiling versioned site radio maps).
    - [src/wifi_localization/storage.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/wifi_localization/storage.py) (Atomic JSON serialization and retrieval for radio maps, reference locations, and metadata under `data/radio_maps/<site_id>/v<version>/`).
    - [src/wifi_localization/knn_localizer.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/wifi_localization/knn_localizer.py) (`KnnLocalizer` implementing union Euclidean/Manhattan distance with unobserved AP penalty $P_{\text{missing}} = -105.0$ dBm, inverse distance weighting $w_i = \frac{1}{D_i + \epsilon}$, 2D coordinate interpolation, building/floor plurality voting, and exponential confidence mapping).
    - [src/wifi_localization/probabilistic_localizer.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/wifi_localization/probabilistic_localizer.py) (`ProbabilisticLocalizer` evaluating per-AP Gaussian log-likelihood $\ln \mathcal{N}(z; \mu, \sigma^2)$ with variance floor $\sigma \ge 2.0$ dBm, asymmetric unobserved AP log-penalties, softmax posterior weights, and coordinate interpolation).
    - [src/wifi_localization/evaluator.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/wifi_localization/evaluator.py) (`WifiLocalizationEvaluator` enforcing strict session-partitioned train/test separation to eliminate temporal autocorrelation leakage from stationary repeated scans; computing building/floor/location accuracy, mean/median/p95 coordinate errors in meters, and floor confusion matrices).
    - [src/wifi_localization/__init__.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/wifi_localization/__init__.py) (Package public exports).
  - Backend API Schemas & Service Endpoints:
    - [backend/app/schemas/wifi_survey.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/schemas/wifi_survey.py) (Pydantic models for reference location registration, radio map compilation, query localization, candidate matches, and evaluation).
    - [backend/app/services/wifi_survey_service.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/services/wifi_survey_service.py) (`WifiSurveyService` managing reference locations, auto-discovering locations from survey annotations, compiling radio maps from `wifi_cleaned.parquet`, executing online queries, and orchestrating session holdout evaluations).
    - [backend/app/api/routes/datasets.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/api/routes/datasets.py) (Added 6 REST endpoints: `POST /wifi-survey/locations`, `GET /wifi-survey/locations`, `POST /wifi-survey/fingerprints/build`, `GET /wifi-survey/radio-map`, `GET /wifi-survey/radio-maps`, `POST /wifi-survey/localize`, `POST /wifi-survey/evaluate`).
  - Frontend UI & Visualizations:
    - [frontend/api_client.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/api_client.py) (Added client methods for location registration, radio map compilation, query localization, and evaluation).
    - [frontend/components/wifi_localization_view.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/components/wifi_localization_view.py) (Interactive Section 13 dashboard view: Reference Locations & Survey Registry with 2D scatter layout, Radio Map Compiler & AP Analytics with visibility histograms and top-15 RSSI bar charts, Live Fingerprint Localizer Testbed with candidate matches and 2D spatial positioning map, and Leakage-Resistant Evaluation with floor confusion matrix).
    - [frontend/components/__init__.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/components/__init__.py) and [frontend/app.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/app.py) (Registered `13. Site-Specific Wi-Fi Localization` in sidebar navigation).
  - Testing & Verification:
    - [tests/test_wifi_localization.py](file:///Users/bineetkeshari/Desktop/IndoorAI/tests/test_wifi_localization.py) (18 comprehensive unit and API integration tests covering BSSID normalization, RSSI gating, channel band classification, sweep deduplication, multi-sweep aggregation, visibility rate calculation, Weighted k-NN interpolation and voting, Probabilistic Gaussian matching, zero overlap warnings, strict session separation leakage detection, filesystem storage, and all FastAPI endpoints).
    - Full repository test run passing **201 tests** (183 previous + 18 new) with 0 failures, 0 errors, and zero regressions on baseline models.
  - Architectural Documentation:
    - [docs/SITE_SPECIFIC_WIFI_LOCALIZATION.md](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/SITE_SPECIFIC_WIFI_LOCALIZATION.md) (Comprehensive architecture specification covering decoupling from UJI benchmark, BSSID sanitization, radio map construction, weighted k-NN, probabilistic modeling, confidence estimation, leakage-resistant session holdout methodology, and REST API schemas).

### Task 24: Wi-Fi + PDR Sensor Fusion using an Extended Kalman Filter
- **Status:** COMPLETED
- **Completed On:** 2026-11-18
- **Deliverables Created & Enhanced:**
  - Dedicated Sensor Fusion Package (`src/sensor_fusion/`):
    - [src/sensor_fusion/coordinates.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/sensor_fusion/coordinates.py) (Standard metric building-local ENU frame (+X East, +Y North, heading $\theta \in [-\pi, \pi]$ counter-clockwise from +X), azimuth angle conversion functions, `Pose2D`, and `CoordinateFrameMetadata` validation).
    - [src/sensor_fusion/models.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/sensor_fusion/models.py) (Data models for `FusionStatus`, `UncertaintyType`, `ConfidenceEllipse2D` with $\chi^2_{2, 0.95} = 5.99146$, `PdrStepEvent`, `WifiMeasurementEvent`, `FusedState`, `FusionConfig`, and `FusionTrajectorySummary`).
    - [src/sensor_fusion/uncertainty.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/sensor_fusion/uncertainty.py) (`FusionUncertainty` 4-tier hierarchy: Level 1 empirical survey residuals, Level 2 candidate spatial dispersion, Level 3 confidence-scaled nominal, Level 4 conservative fallback prior; plus pre-filtering freshness, latency, and AP overlap gates).
    - [src/sensor_fusion/ekf.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/sensor_fusion/ekf.py) (`EKFEngine` maintaining 3-state vector $[x, y, \theta]^T$, non-linear PDR prediction with Jacobians $F_k$ and $G_k$, process noise $Q_k = G M G^T + Q_{base}$, linear measurement model $H = [[1, 0, 0], [0, 1, 0]]$, 2-DOF Chi-Square Mahalanobis outlier gating $d^2 \le 5.991$, Joseph form covariance update $(I - KH)P(I - KH)^T + KRK^T$, symmetrization, and eigenvalue clamping).
    - [src/sensor_fusion/pipeline.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/sensor_fusion/pipeline.py) (`FusionPipeline` orchestrating asynchronous chronological event sorting, cross-floor and latency pre-filtering, step integration, filter state execution, and parquet/JSON trajectory persistence).
    - [src/sensor_fusion/evaluation.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/sensor_fusion/evaluation.py) (`FusionEvaluator` computing trajectory metrics: mean/median/p95 error, endpoint drift per meter, drift per minute, 95% ellipse coverage, innovation diagnostics, and `generate_controlled_synthetic_trajectory` benchmark fixture).
    - [src/sensor_fusion/__init__.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/sensor_fusion/__init__.py) (Clean package public exports).
  - Backend API Schemas & Service Endpoints:
    - [backend/app/schemas/sensor_fusion.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/schemas/sensor_fusion.py) (Pydantic request/response schemas: `FusionRunRequest`, `FusionStatusResponse`, `ConfidenceEllipseSchema`, `FusedStateSchema`, `FusionTrajectoryResponse`, `FusionEvaluateRequest`, `FusionEvaluationResponse`).
    - [backend/app/services/sensor_fusion_service.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/services/sensor_fusion_service.py) (`SensorFusionService` extracting PDR steps from calibrated/raw motion records, localizing survey Wi-Fi scans against compiled radio maps with uncertainty extraction, running EKF fusion, and saving artifacts).
    - [backend/app/api/routes/datasets.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/api/routes/datasets.py) (Added 5 REST endpoints: `POST /fusion/run`, `GET /fusion/status`, `GET /fusion/trajectory`, `POST /fusion/evaluate`, `GET /fusion/evaluation`).
  - Frontend UI & Visualizations:
    - [frontend/api_client.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/api_client.py) (Added client methods: `run_fusion`, `get_fusion_status`, `get_fused_trajectory`, `evaluate_fusion`, `get_fusion_evaluation`).
    - [frontend/components/sensor_fusion_view.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/components/sensor_fusion_view.py) (Interactive Section 14 dashboard view: 2D Fused Trajectory & Error Ellipses with Plotly floorplan, Filter Parameters & Pipeline Execution controls, Innovation & Outlier Gating Diagnostics, and Ground-Truth Benchmark & Drift Evaluation).
    - [frontend/components/__init__.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/components/__init__.py) and [frontend/app.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/app.py) (Registered `14. Wi-Fi + PDR Sensor Fusion (EKF)` in sidebar navigation).
  - Testing & Verification:
    - [tests/test_sensor_fusion.py](file:///Users/bineetkeshari/Desktop/IndoorAI/tests/test_sensor_fusion.py) (16 comprehensive unit and API integration tests covering coordinate transformations, azimuth conversions, PDR step propagation, Wi-Fi observation update, Joseph form numerical stability, Chi-Square 2-DOF outlier gating, cross-floor rejection, asynchronous event interleaving, 4-tier uncertainty hierarchy, synthetic benchmark evaluation with >50% drift reduction and 100% 95% ellipse coverage, and all 5 FastAPI REST endpoints).
    - Full repository test run passing **217 tests** (201 previous + 16 new) with 0 failures, 0 errors, and zero regressions on baseline models.
  - Architectural Documentation:
    - [docs/WIFI_PDR_SENSOR_FUSION.md](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/WIFI_PDR_SENSOR_FUSION.md) (Complete architectural and mathematical specification covering kinematic motion models, state transition Jacobians, Joseph form covariance updates, Chi-Square 2-DOF outlier rejection, 4-tier uncertainty hierarchy, 95% confidence ellipses, controlled benchmark results, REST endpoints, and UI dashboard).

### Task 25: Computer Vision-Based Indoor Landmark Detection
- **Status:** COMPLETED
- **Completed On:** 2026-11-18
- **Deliverables Created & Enhanced:**
  - Dedicated Computer Vision Package (`src/computer_vision/`):
    - [src/computer_vision/models.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/computer_vision/models.py) (`LandmarkCategory` 11-class taxonomy: `DOOR`, `ROOM_ENTRANCE`, `STAIRCASE`, `ELEVATOR`, `CORRIDOR`, `INTERSECTION`, `ROOM_NUMBER_SIGN`, `DIRECTION_SIGN`, `EXIT_SIGN`, `DISTINCTIVE_STRUCTURE`, `UNKNOWN`; `VerificationStatus`, `VisualMatchStatus`, `BoundingBox`, `OcrTextRecord`, `LandmarkDetection`, `LandmarkRecord`, `VisualMatchCandidate`, `VisualMatchReport`, `CategoryEvaluationMetrics`, `VisionEvaluationReport`).
    - [src/computer_vision/image_validation.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/computer_vision/image_validation.py) (`ImagePrivacyValidator` verifying `is_exif_sanitized` from survey metadata, allowed extensions `.jpg`/`.png`/`.webp`, magic numbers, dimension bounds 32px–8192px, file size bounds 512B–25MB, and path traversal rejection).
    - [src/computer_vision/image_loader.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/computer_vision/image_loader.py) (`SafeImageLoader` loading orientation-transposed RGB/BGR NumPy arrays respecting privacy barriers).
    - [src/computer_vision/preprocessing.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/computer_vision/preprocessing.py) (`ImagePreprocessor` applying CLAHE contrast enhancement in LAB space, aspect-ratio preserving resizing, normalized bounding box cropping with margin, and visual overlay drawing).
    - [src/computer_vision/feature_extraction.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/computer_vision/feature_extraction.py) (`VisualFeatureExtractor` extracting up to 500 ORB keypoints/descriptors with FAST/Harris scoring and 512-dim normalized 3D HSV color histogram embeddings).
    - [src/computer_vision/ocr_engine.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/computer_vision/ocr_engine.py) (`SignboardOcrEngine` integrating EasyOCR for text recognition on room plates and directional signs, with OpenCV morphological stroke gradient fallback; strict policy maintaining OCR output as unverified until manual confirmation).
    - [src/computer_vision/landmark_detector.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/computer_vision/landmark_detector.py) (`LandmarkDetector` with dual-stage architecture: fine-tuned Ultralytics YOLO interface + architectural heuristic detector finding door frames, corridor perspective vanishing lines, and sign plaques; NMS deduplication at configurable IoU).
    - [src/computer_vision/landmark_classifier.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/computer_vision/landmark_classifier.py) (`LandmarkClassifier` taxonomy parsing, metadata sanitation, and strict rule: never inventing building/floor labels from appearance alone).
    - [src/computer_vision/landmark_registry.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/computer_vision/landmark_registry.py) (`LandmarkRegistry` managing persistent building landmark records under `data/landmarks/{building_id}/landmarks.json`, lifecycle `UNVERIFIED` -> `MANUALLY_VERIFIED` / `REJECTED`, and anti-duplication guards).
    - [src/computer_vision/landmark_matching.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/computer_vision/landmark_matching.py) (`VisualLandmarkMatcher` computing appearance similarity, ORB descriptor inliers, and returning `STRONG_MATCH`, `WEAK_MATCH`, or `NO_MATCH` without claiming probabilities).
    - [src/computer_vision/evaluation.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/computer_vision/evaluation.py) (`VisionEvaluator` calculating multi-class precision, recall, F1 at IoU >= 0.50, category confusion matrix, OCR CER via Levenshtein distance, visual matching precision/recall, and generating strictly partitioned synthetic benchmarks).
    - [src/computer_vision/pipeline.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/computer_vision/pipeline.py) (`VisionPipeline` orchestrating privacy validation, loading, detection, OCR, descriptor extraction, and batch survey photo processing).
    - [src/computer_vision/__init__.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/computer_vision/__init__.py) (Clean package public exports).
  - Backend API Schemas & Service Endpoints:
    - [backend/app/schemas/computer_vision.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/schemas/computer_vision.py) (Pydantic models: `BoundingBoxSchema`, `OcrTextSchema`, `LandmarkDetectionSchema`, `VisionAnalyzeRequest`, `VisionAnalyzeResponse`, `LandmarkRecordSchema`, `LandmarkVerifyRequest`, `VisualMatchCandidateSchema`, `VisualMatchRequest`, `VisualMatchResponse`, `VisionEvaluateRequest`, `VisionEvaluateResponse`).
    - [backend/app/services/computer_vision_service.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/services/computer_vision_service.py) (`ComputerVisionService` managing survey batch analysis, parquet artifact persistence, registry queries, manual verification, visual matching, and evaluation).
    - [backend/app/api/routes/datasets.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/api/routes/datasets.py) (Added 6 REST endpoints: `POST /vision/analyze`, `GET /vision/detections`, `GET /vision/landmarks`, `POST /vision/landmarks/verify`, `POST /vision/match`, `POST /vision/evaluate`).
  - Frontend UI & Visualizations:
    - [frontend/api_client.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/api_client.py) (Added client methods: `analyze_vision`, `get_vision_detections`, `list_landmarks`, `verify_landmark`, `match_survey_photo`, `evaluate_vision`).
    - [frontend/components/computer_vision_view.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/components/computer_vision_view.py) (Interactive Section 15 dashboard view with 5 tabs: Image Gallery & Detection Overlay, Pipeline Execution & Batch Analysis, Landmark Registry & Manual Verification, Visual Descriptor Matching Testbed, and Performance Evaluation & Confusion Matrix).
    - [frontend/components/__init__.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/components/__init__.py) and [frontend/app.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/app.py) (Registered `15. Computer Vision & Landmarks` in sidebar navigation).
  - Testing & Verification:
    - [tests/test_computer_vision.py](file:///Users/bineetkeshari/Desktop/IndoorAI/tests/test_computer_vision.py) (22 comprehensive unit and API integration tests covering bounding box geometry, IoU, privacy rejection of unsanitized photos, corrupt file handling, landmark detector heuristics, unverified OCR policy, registry persistence and manual verification, visual descriptor matching and NO_MATCH behavior, session-level partitioned evaluation, and all 6 FastAPI REST endpoints).
    - Full repository test run passing **239 tests** (217 previous + 22 new) with 0 failures, 0 errors, and zero regressions on baseline models.
  - Architectural Documentation:
    - [docs/COMPUTER_VISION_LANDMARK_DETECTION.md](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/COMPUTER_VISION_LANDMARK_DETECTION.md) (Complete specification covering image-processing architecture, privacy-by-design requirements, landmark taxonomy, detector heuristics, OCR pipeline, persistent registry schemas, visual matching methodology, session holdout evaluation, REST API reference, and UI dashboard).

### Task 26: Indoor Map Generation & Spatial Representation
- **Status:** COMPLETED
- **Completed On:** 2026-11-18
- **Deliverables Created & Enhanced:**
  - Dedicated Spatial Mapping Package (`src/mapping/`):
    - [src/mapping/geometry.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/mapping/geometry.py) (2D metric geometry engine: `Point2D`, `LineSegment2D` with distance and proper intersection, `BoundingBox2D`, `Polygon2D` with Shoelace area, centroid, ray-casting boundary containment, non-adjacent self-intersection detection, and rectangular room generator).
    - [src/mapping/models.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/mapping/models.py) (Indoor map domain models: `MapVerificationStatus`, `RoomCategory`, `ConnectionType`, `AccessibilityType`, `DoorType`, `Room`, `Corridor`, `Door`, `VerticalTransition`, `SpatialConnection`, `FloorMap`, `BuildingMap` in `BUILDING_LOCAL_CARTESIAN_ENU` frame).
    - [src/mapping/topology.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/mapping/topology.py) (`IndoorTopologyGraph` generating room centroid nodes, corridor waypoints/intersections, door portals, and multi-floor vertical landing nodes; BFS connected components, isolated node detection, Dijkstra multi-floor routing with wheelchair accessibility filtering).
    - [src/mapping/map_validator.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/mapping/map_validator.py) (`MapValidator` auditing coordinate frames, floor boundary containment, bowtie polygon self-intersections, corridor lengths, door referential integrity, vertical landing planar shifts $\le 8\text{m}$, and topological connectivity).
    - [src/mapping/map_storage.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/mapping/map_storage.py) (`MapStorage` managing atomic persistence under `data/indoor_maps/{building_id}/`, staging directories, immutable numbered version snapshot publishing under `versions/v{N}/`, and version history queries).
    - [src/mapping/map_export.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/mapping/map_export.py) (`MapExporter` exporting RFC 7946 GeoJSON FeatureCollections for rooms, corridors, doors, vertical transitions, Wi-Fi reference points, visual landmarks, and topology edges, plus hierarchical JSON serialization).
    - [src/mapping/map_builder.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/mapping/map_builder.py) (`MapBuilder` fluent builder API for floors, rooms, corridors, doors, vertical transitions, auto-associating Task 23 Wi-Fi reference locations and Task 25 visual landmarks, and auto-linking door/stair connections).
    - [src/mapping/evaluation.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/mapping/evaluation.py) (`MapEvaluator` computing spatial coverage, graph node/edge counts, verification ratio, accessibility ratio, and `generate_controlled_synthetic_building_map` 2-floor benchmark fixture).
    - [src/mapping/pipeline.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/mapping/pipeline.py) (`MappingPipeline` orchestrating survey ingestion, validation, atomic storage, and export).
    - [src/mapping/__init__.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/mapping/__init__.py) (Clean package public exports).
  - Backend API Schemas & Service Endpoints:
    - [backend/app/schemas/mapping.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/schemas/mapping.py) (Pydantic models: `CreateBuildingMapRequest`, `CreateFloorRequest`, `CreateRoomRequest`, `CreateCorridorRequest`, `CreateDoorRequest`, `CreateVerticalTransitionRequest`, `CreateConnectionRequest`, `ImportWifiReferencesRequest`, `ImportLandmarksRequest`, `PublishVersionRequest`, `MapValidationResponse`, `MapMetricsResponse`, `BuildingMapSummaryResponse`).
    - [backend/app/services/mapping_service.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/services/mapping_service.py) (`MappingService` managing building floorplans, entity creation, multimodal imports, validation audits, metrics evaluation, version publishing, and exports).
    - [backend/app/api/routes/datasets.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/api/routes/datasets.py) (Added 16 REST endpoints for building creation, floor/room/corridor/door registration, vertical transitions, Wi-Fi/landmark imports, validation, evaluation, version publishing, and GeoJSON/JSON exports).
  - Frontend UI & Visualizations:
    - [frontend/api_client.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/api_client.py) (Added 16 client methods for mapping operations).
    - [frontend/components/mapping_view.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/components/mapping_view.py) (Interactive Section 16 dashboard with 4 tabs: 2D Floorplan & Spatial Geometry Canvas, Topological Navigation Graph, Map Editor & Multimodal Ingestion, and Validation, Evaluation & Standards-Compliant GeoJSON/JSON Export).
    - [frontend/components/__init__.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/components/__init__.py) and [frontend/app.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/app.py) (Registered `16. Indoor Mapping & Spatial Graphs` in sidebar navigation).
  - Testing & Verification:
    - [tests/test_mapping.py](file:///Users/bineetkeshari/Desktop/IndoorAI/tests/test_mapping.py) (23 comprehensive unit and integration tests covering 2D geometry, Shoelace area, ray-casting containment, self-intersection detection, domain entity models, topological navigation graph, multi-floor Dijkstra reachability, wheelchair accessibility filtering, map validation, atomic storage, immutable version snapshots, GeoJSON/JSON exports, and all FastAPI REST endpoints).
    - Full repository test run passing **262 tests** (239 previous + 23 new) with 0 failures, 0 errors, and zero regressions on baseline models.
  - Architectural Documentation:
    - [docs/INDOOR_MAP_GENERATION_AND_SPATIAL_REPRESENTATION.md](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/INDOOR_MAP_GENERATION_AND_SPATIAL_REPRESENTATION.md) (Comprehensive specification covering 2D spatial geometry, indoor entity models, topological navigation graphs, validation rules, atomic storage and versioning, RFC 7946 GeoJSON export, REST APIs, and UI dashboard).

### Task 27: Map-Constrained Multimodal Localization using a Particle Filter
- **Status:** COMPLETED
- **Completed On:** 2026-11-18
- **Deliverables Created & Enhanced:**
  - Dedicated Map Localization Package (`src/map_localization/`):
    - [src/map_localization/models.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/map_localization/models.py) (Domain models: `FilterStatus`, `ObservationType`, `ResampleAlgorithm`, `ParticleFilterConfig`, `ParticleFilterState`, `ParticleFilterSummary`).
    - [src/map_localization/particle.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/map_localization/particle.py) (Particle state vector $s_i = (x_i, y_i, \theta_i, f_i, w_i)$, coordinate normalization, copy, distance, and heading wrap to $[-\pi, \pi]$).
    - [src/map_localization/initialization.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/map_localization/initialization.py) (`ParticleInitializer` supporting known location, Wi-Fi fix with covariance dispersion, EKF baseline state, and bounding box initialization with floor boundary containment and solid wall avoidance).
    - [src/map_localization/motion_model.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/map_localization/motion_model.py) (`PdrMotionModel` stochastic step propagation with step length and heading noise, stationary angular diffusion).
    - [src/map_localization/map_constraints.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/map_localization/map_constraints.py) (`MapConstraintEvaluator` enforcing solid room wall collision rejection, doorway passage tolerance $\le 1.2\text{m}$, outer floor boundary containment, multi-floor vertical transitions $\le 3.5\text{m}$ near stairs/elevators, and preservation of unsurveyed areas `allow_unknown_areas=True`).
    - [src/map_localization/measurement_model.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/map_localization/measurement_model.py) (`WifiMeasurementModel` 2D Gaussian likelihood updates with $2\times 2$ covariance matrix, floor mismatch penalty, and soft log-likelihood lower bounds).
    - [src/map_localization/landmark_observation.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/map_localization/landmark_observation.py) (`LandmarkMeasurementModel` visual landmark proximity updates within $r_{max} \approx 6.0\text{m}$ viewing zone without claiming exact camera position).
    - [src/map_localization/resampling.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/map_localization/resampling.py) (`Resampler` log-sum-exp weight normalization, effective sample size $N_{eff} = 1/\sum w_i^2$, $O(N)$ low-variance systematic resampling, diversity monitoring, and roughening jitter).
    - [src/map_localization/particle_filter.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/map_localization/particle_filter.py) (`MapConstrainedParticleFilter` full state lifecycle orchestration, circular mean heading, floor consensus mode, particle spread $\sigma_{xy}$, and 95% circular uncertainty).
    - [src/map_localization/evaluation.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/map_localization/evaluation.py) (`MapLocalizationEvaluator`, `TrajectoryComparisonMetrics`, `MapLocalizationEvaluationReport`, and `generate_controlled_benchmark_trajectory` multi-floor ground truth benchmark fixture).
    - [src/map_localization/pipeline.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/map_localization/pipeline.py) (`MapLocalizationPipeline` chronological event loop merging PDR, Wi-Fi, landmarks, and concurrent EKF baseline).
    - [src/map_localization/__init__.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/map_localization/__init__.py) (Clean public package exports).
  - Backend API Schemas & Service:
    - [backend/app/schemas/map_localization.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/schemas/map_localization.py) (Pydantic request/response models: `ParticleFilterRunRequest`, `ParticleFilterStateSchema`, `ParticleFilterTrajectoryResponse`, `ParticleFilterStatusResponse`, `ParticleFilterEvaluateRequest`, `ParticleFilterEvaluationResponse`).
    - [backend/app/services/map_localization_service.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/services/map_localization_service.py) (`MapLocalizationService` integrating PDR steps, Wi-Fi radio maps, visual landmarks, and building maps, with parquet and JSON trajectory persistence).
    - [backend/app/api/routes/datasets.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/api/routes/datasets.py) (Added 5 REST endpoints for map localization run, status, trajectory retrieval, evaluation, and report retrieval).
  - Frontend UI & Visualizations:
    - [frontend/api_client.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/api_client.py) (Added 5 client methods for particle filter operations).
    - [frontend/components/map_localization_view.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/components/map_localization_view.py) (Interactive Section 17 dashboard with 4 tabs: 2D Trajectory & Floorplan Canvas, Filter Controls & Execution, Particle Spread & Diagnostics, and Comparative Benchmark Evaluation).
    - [frontend/components/__init__.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/components/__init__.py) and [frontend/app.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/app.py) (Registered `17. Map-Constrained Localization` in sidebar navigation).
  - Testing & Verification:
    - [tests/test_map_localization.py](file:///Users/bineetkeshari/Desktop/IndoorAI/tests/test_map_localization.py) (21 comprehensive unit and integration tests covering particle primitives, initialization, motion propagation, wall collision rejection, doorway passage, Wi-Fi likelihoods, visual landmarks, systematic resampling, multi-floor transitions, comparative benchmark evaluation, and all FastAPI REST endpoints).
    - Full repository test run passing **283 tests** (262 previous + 21 new) with 0 failures, 0 errors, and zero regressions on baseline models.
  - Architectural Documentation:
    - [docs/MAP_CONSTRAINED_MULTIMODAL_LOCALIZATION.md](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/MAP_CONSTRAINED_MULTIMODAL_LOCALIZATION.md) (Comprehensive technical specification covering particle filter mathematical state space, building constraints and anti-fabrication principles, observation models, resampling, circular statistics, comparative benchmarking, REST API, and UI dashboard).

### Task 28: Indoor Navigation & Intelligent Route Planning
- **Status:** COMPLETED
- **Completed On:** 2026-11-18
- **Deliverables Created & Enhanced:**
  - Dedicated Indoor Navigation Package (`src/navigation/`):
    - [src/navigation/models.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/navigation/models.py) (Domain models & enums: `NavigationStatus`, `InstructionType`, `RoutePreference`, `RoutingAlgorithm`, `DestinationCategory`, `RouteCostWeights`, `Destination`, `NavigationWaypoint`, `NavigationInstruction`, `PlannedRoute`, `NavigationProgress`, `RerouteEvent`, `NavigationSession`).
    - [src/navigation/destination.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/navigation/destination.py) (`DestinationManager` indexing verified rooms, doorways/exits, elevators, staircases, visual landmarks from `BuildingMap`, category filtering, and multi-field ranked fuzzy search).
    - [src/navigation/route_graph.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/navigation/route_graph.py) (`RouteGraph` adapting `IndoorTopologyGraph`, multi-criteria cost function $C = w_d D + w_t T + w_s S + w_a A$, dynamic edge/node obstruction blocking, and scaled admissible 3D Euclidean heuristic $h(u, v) \le h^*(u, v)$).
    - [src/navigation/path_geometry.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/navigation/path_geometry.py) (`PathGeometryBuilder` waypoint creation, corridor centerline snapping, segment bearings, cumulative distances, and cross-track perpendicular distance projection).
    - [src/navigation/turn_detection.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/navigation/turn_detection.py) (`TurnDetector` mapping Cartesian ENU bearings to pedestrian directions: straight $<20^\circ$, slight $20^\circ-45^\circ$, standard turn $45^\circ-135^\circ$, sharp/U-turn $>135^\circ$, and vertical transition detection).
    - [src/navigation/instruction_generator.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/navigation/instruction_generator.py) (`InstructionGenerator` compiling human-readable turn-by-turn directives with spatial landmark and floor-transition cues).
    - [src/navigation/accessibility.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/navigation/accessibility.py) (`AccessibilityAuditor` conservative audit engine evaluating vertical transitions, door clearance, step barriers, and returning `ACCESSIBLE`, `INACCESSIBLE`, or `ACCESSIBILITY_UNKNOWN`).
    - [src/navigation/route_validator.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/navigation/route_validator.py) (`RouteValidator` checking topological continuity, map version alignment, accessibility constraints, and solid wall non-penetration audits rejecting invalid wall piercings).
    - [src/navigation/route_planner.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/navigation/route_planner.py) (`RoutePlanner` executing Dijkstra exact shortest and A* directed graph search, start/destination snapping, waypoint assembly, validation, and route compilation).
    - [src/navigation/progress_tracker.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/navigation/progress_tracker.py) (`ProgressTracker` tracking user localization fixes, waypoint advancement, remaining distance, cross-track error, completion percentage, and arrival threshold $\le 1.5\text{m}$).
    - [src/navigation/rerouting.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/navigation/rerouting.py) (`ReroutingEngine` monitoring off-route deviations $>4.0\text{m}$ or floor mismatches, applying a 5.0-second cooldown debounce and localization reliability checks before recalculating routes).
    - [src/navigation/navigation_session.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/navigation/navigation_session.py) (`NavigationSessionManager` session lifecycle tracking from `CREATED` $\rightarrow$ `ACTIVE` $\rightarrow$ `COMPLETED` / `CANCELLED` / `REROUTED`, with atomic JSON persistence).
    - [src/navigation/evaluation.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/navigation/evaluation.py) (`NavigationEvaluator`, `RouteEvaluationMetrics`, `NavigationSimulationMetrics`, `NavigationEvaluationReport` benchmarking Dijkstra vs A* node expansions, runtime, admissibility, and simulating multi-floor navigation trips).
    - [src/navigation/pipeline.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/navigation/pipeline.py) (`NavigationPipeline` coordinating destinations, planning, session management, tracking, and evaluation).
    - [src/navigation/__init__.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/navigation/__init__.py) (Clean package public exports).
  - Backend API Schemas & Service:
    - [backend/app/schemas/navigation.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/schemas/navigation.py) (Pydantic request/response schemas: `DestinationResponse`, `PlanRouteRequest`, `PlannedRouteResponse`, `StartNavigationSessionRequest`, `NavigationSessionResponse`, `ProgressUpdateRequest`, `NavigationProgressResponse`, `NavigationEvaluationResponse`).
    - [backend/app/services/navigation_service.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/services/navigation_service.py) (`NavigationService` orchestrating destination discovery, route planning, active session tracking, progress recording, and performance evaluation).
    - [backend/app/api/routes/datasets.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/api/routes/datasets.py) (Added 8 REST endpoints: `GET /api/datasets/{id}/navigation/destinations`, `POST /api/datasets/{id}/navigation/plan`, `POST /api/datasets/{id}/navigation/sessions`, `GET /api/datasets/{id}/navigation/sessions/{session_id}`, `POST /api/datasets/{id}/navigation/sessions/{session_id}/progress`, `POST /api/datasets/{id}/navigation/sessions/{session_id}/reroute`, `POST /api/datasets/{id}/navigation/sessions/{session_id}/cancel`, `POST /api/datasets/{id}/navigation/evaluate`).
  - Frontend UI & Visualizations:
    - [frontend/api_client.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/api_client.py) (Added 9 client methods for navigation operations).
    - [frontend/components/navigation_view.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/components/navigation_view.py) (Interactive Section 18 dashboard with 5 tabs: Route Planner & Multi-Floor Map Canvas, Turn-by-Turn Instruction HUD, Active Session & Simulation Tracker, Accessibility Audit & Wall Safety Verification, and Algorithmic Performance & Benchmarks).
    - [frontend/components/__init__.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/components/__init__.py) and [frontend/app.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/app.py) (Registered `18. Indoor Navigation & Route Planning` in sidebar navigation).
  - Testing & Verification:
    - [tests/test_navigation.py](file:///Users/bineetkeshari/Desktop/IndoorAI/tests/test_navigation.py) (17 comprehensive unit and integration tests covering destination indexing, Dijkstra shortest path, A* heuristic admissibility, turn detection, instruction generation, accessibility audit with wheelchair constraints, solid wall non-penetration audit, progress tracking, arrival detection, dynamic rerouting debounce, session lifecycle, and all FastAPI REST endpoints).
    - Full repository test run passing **300 tests** (283 previous + 17 new) with 0 failures, 0 errors, and zero regressions across all modules.
  - Architectural Documentation:
    - [docs/INDOOR_NAVIGATION_AND_ROUTE_PLANNING.md](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/INDOOR_NAVIGATION_AND_ROUTE_PLANNING.md) (Comprehensive technical specification covering route graph formulation, multi-criteria cost function, admissible heuristics, anti-fabrication and accessibility invariants, turn-by-turn guidance, progress tracking, debounced rerouting, REST APIs, and UI dashboard).

### Task 29: Real-Time Multimodal Ingestion & Streaming Integration
- **Status:** COMPLETED
- **Completed On:** 2026-11-18
- **Deliverables Created & Enhanced:**
  - Dedicated Backend Real-Time Ingestion Package (`backend/app/services/realtime/`):
    - [backend/app/services/realtime/event_models.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/services/realtime/event_models.py) (Domain models: `RealtimeEventType` [WIFI_SCAN, IMU_BATCH, ORIENTATION_UPDATE, PHOTO_CAPTURE, LANDMARK_OBSERVATION, ANNOTATION, HEARTBEAT, SESSION_STATUS], `RealtimeEvent`, `ServerAckMessage`, `LiveLocalizationUpdateMessage`, `LiveNavigationUpdateMessage`, `StreamStatusMessage`, `RejectionReason`, `EventProcessingStatus`).
    - [backend/app/services/realtime/event_validator.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/services/realtime/event_validator.py) (`EventValidator` checking active session, schema versions, timestamp drift, payload size bounds, Wi-Fi RSSI [-120, 0 dBm] and frequencies [2.4, 5, 6 GHz], sensor NaN/Inf rejection, and mandatory photo privacy sanitization).
    - [backend/app/services/realtime/event_buffer.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/services/realtime/event_buffer.py) (`BoundedEventBuffer` thread-safe FIFO buffer with configurable capacity [default 5000] and backpressure drop policies `DROP_OLDEST` vs `REJECT_NEW`).
    - [backend/app/services/realtime/replay_manager.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/services/realtime/replay_manager.py) (`ReplayManager` tracking strictly increasing sequence numbers, event ID indexing, and replay slice generation for seamless client reconnection).
    - [backend/app/services/realtime/connection_manager.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/services/realtime/connection_manager.py) (`WebSocketConnectionManager` multiplexing active client connections, personal messaging, and session broadcasts).
    - [backend/app/services/realtime/session_stream.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/services/realtime/session_stream.py) (`SessionStream` session state, bounded LRU deduplication cache [10,000 items], attached maps, and live telemetry).
    - [backend/app/services/realtime/event_processor.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/services/realtime/event_processor.py) (`EventProcessor` chronological event dispatcher routing to Task 23 Wi-Fi localizers, Task 21 PDR kinematics, Task 27 Particle Filter, and Task 28 Navigation tracking with anti-flapping checks).
    - [backend/app/services/realtime/realtime_pipeline.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/services/realtime/realtime_pipeline.py) (`RealtimePipeline` singleton coordinator for WebSocket loops, REST fallback batch ingestion, and synthetic simulation injection).
    - [backend/app/services/realtime/__init__.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/services/realtime/__init__.py) (Clean package public exports).
  - Backend API Schemas & Routes:
    - [backend/app/schemas/realtime.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/schemas/realtime.py) (Pydantic schemas: `RealtimeEventSchema`, `BatchEventsRequest`, `ServerAckResponse`, `BatchAckResponse`, `StreamStatusResponse`, `SessionControlRequest`, `SimulateStepRequest`, `AttachNavigationRequest`).
    - [backend/app/api/routes/realtime.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/api/routes/realtime.py) (FastAPI streaming endpoints: `GET /api/realtime/sessions/{session_id}/stream`, `POST /api/realtime/sessions/{session_id}/events/batch`, `GET /api/realtime/sessions/{session_id}/status`, `POST /api/realtime/sessions/{session_id}/control`, `GET /api/realtime/sessions/{session_id}/history`, `POST /api/realtime/sessions/{session_id}/simulate`, `POST /api/realtime/sessions/{session_id}/attach-navigation`, `GET /api/realtime/sessions`).
    - [backend/app/main.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/main.py) (Mounted `realtime_router` under `/api/realtime`).
  - Android Client Streaming Layer (`android/app/src/main/java/ai/indoor/app/realtime/`):
    - [android/app/src/main/java/ai/indoor/app/realtime/RealtimeEventModels.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/realtime/RealtimeEventModels.kt) (Typed Kotlin models: `RealtimeEventType`, `RealtimeEventDto`, `ApScanItem`, `ImuSampleItem`, `ServerAckDto`, `LiveLocalizationUpdateDto`, `LiveNavigationUpdateDto`, `StreamStatusDto`).
    - [android/app/src/main/java/ai/indoor/app/realtime/PendingEventQueue.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/realtime/PendingEventQueue.kt) (Thread-safe bounded FIFO queue [1,000 capacity] with drop-oldest backpressure and idempotent event ID preservation).
    - [android/app/src/main/java/ai/indoor/app/realtime/RealtimeWebSocketClient.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/realtime/RealtimeWebSocketClient.kt) (OkHttp WebSocket client with exponential backoff reconnect, 15-second heartbeat pinging, StateFlow/SharedFlow reactive streams).
    - [android/app/src/main/java/ai/indoor/app/realtime/RealtimeBatchFallbackClient.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/realtime/RealtimeBatchFallbackClient.kt) (HTTP/REST fallback batch uploader when WebSockets are unavailable).
    - [android/app/src/main/java/ai/indoor/app/realtime/RealtimeStreamingManager.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/main/java/ai/indoor/app/realtime/RealtimeStreamingManager.kt) (High-level manager coordinating capture usecases, queue drainage, and UI state flows).
  - Android Unit Test Suite (`android/app/src/test/java/ai/indoor/app/realtime/`):
    - [android/app/src/test/java/ai/indoor/app/realtime/RealtimeEventSerializationTest.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/test/java/ai/indoor/app/realtime/RealtimeEventSerializationTest.kt) (JSON serialization/deserialization verification for event payloads, server ACKs, localization updates, and navigation directives).
    - [android/app/src/test/java/ai/indoor/app/realtime/PendingEventQueueTest.kt](file:///Users/bineetkeshari/Desktop/IndoorAI/android/app/src/test/java/ai/indoor/app/realtime/PendingEventQueueTest.kt) (Testing FIFO ordering, bounded drop-oldest, ACK removal by ID, batch removal, and queue clear).
  - Frontend UI & Visualizations:
    - [frontend/api_client.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/api_client.py) (Added 7 client methods for real-time operations).
    - [frontend/components/realtime_view.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/components/realtime_view.py) (Interactive Section 19 dashboard with 5 tabs: Stream Telemetry & Buffer, Live Sensor & Wi-Fi Feed, Live Map Localization Canvas, Turn-by-Turn Navigation HUD, and Simulation & Replay Harness with prominent warning banner).
    - [frontend/components/__init__.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/components/__init__.py) and [frontend/app.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/app.py) (Registered `19. Real-Time IndoorAI` in sidebar navigation).
  - Testing & Verification:
    - [tests/test_realtime.py](file:///Users/bineetkeshari/Desktop/IndoorAI/tests/test_realtime.py) (15 comprehensive unit and integration tests covering event validation, NaN/Inf rejection, duplicate idempotency, bounded buffer backpressure, REST batch ingestion, WebSocket streaming, reconnection replay, session isolation, unverified landmark rejection, live navigation tracking with anti-flapping, session lifecycle control, and development simulator injection).
    - Full repository test run passing **315 tests** (300 previous + 15 new) with 0 failures, 0 errors, and zero regressions across all modules.
  - Architectural Documentation:
    - [docs/REAL_TIME_INGESTION_AND_STREAMING.md](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/REAL_TIME_INGESTION_AND_STREAMING.md) (Comprehensive technical specification covering streaming architecture, typed event schemas, WebSocket protocol, REST batch fallback, buffering and backpressure, replay synchronization, engine integration, Android client architecture, and UI dashboard).

### Task 30: IndoorAI Final System Validation, Security & Release Readiness
- **Status:** COMPLETED
- **Completed On:** 2026-11-19
- **Deliverables:**
  - Full Repository Subsystem Audit (Tasks 01–29 verified with zero omissions).
  - Cryptographic Authentication & Security Hardening:
    - [backend/app/core/security.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/core/security.py) (RFC 7519 HMAC-SHA256 JWT tokens, constant-time verification, expiration enforcement, `SessionAccessController`, role hierarchy `ADMIN`, `SURVEYOR`, `NAVIGATOR`, `VIEWER`, WebSocket auth policy closure code 1008, REST authorization dependencies).
    - [backend/app/api/routes/auth.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/api/routes/auth.py) (Token issuance `/api/auth/token`, claim verification `/api/auth/verify`, dev helper `/api/auth/dev-token`).
    - [backend/app/api/routes/realtime.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/api/routes/realtime.py) (Secured WebSocket handshake and REST streaming endpoints with session ownership validation).
  - Multi-Session Isolation & Concurrency Verification:
    - [backend/app/services/realtime/session_stream.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/services/realtime/session_stream.py) (Ownership tracking, isolated dedup caches, bounded event buffers, and resource release via `close()`).
    - [backend/app/services/realtime/realtime_pipeline.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/services/realtime/realtime_pipeline.py) (Session lifecycle registration, deregistration, and teardown via `remove_stream()`).
  - Mathematical & Uncertainty Representation Corrections:
    - [src/map_localization/models.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/map_localization/models.py) and [src/map_localization/particle_filter.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/map_localization/particle_filter.py) (Replaced simplistic circular Gaussian confidence assumption with true anisotropic spatial covariance: $var_x$, $var_y$, $cov_{xy}$, and empirical RMS particle spread $\sigma_{xy}$).
    - [backend/app/services/realtime/event_processor.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/services/realtime/event_processor.py) (Resolved `PdrStepEvent` parameter mismatch and attribute alignment for particle filter steps).
  - Reliability & Performance Benchmark Suite:
    - [src/evaluation/system_benchmark.py](file:///Users/bineetkeshari/Desktop/IndoorAI/src/evaluation/system_benchmark.py) (Comprehensive benchmarking harness measuring ingestion throughput [>100k events/s], median latency [0.006–0.009 ms], p95 latency [<0.05 ms], 5x multithreaded concurrent session scalability, and replay recovery speed).
  - Comprehensive Final Test Suite:
    - [tests/test_system_validation.py](file:///Users/bineetkeshari/Desktop/IndoorAI/tests/test_system_validation.py) (8 dedicated end-to-end integration and security tests: JWT creation, expired token rejection, tampered signature detection, cross-session 403 isolation, multithreaded session isolation, end-to-end navigation streaming, anisotropic uncertainty, and SLA compliance).
    - **Total Test Suite**: 323 passing tests (100% pass rate) across entire repository without regressions.
  - Complete Documentation Deliverables:
    - [docs/FINAL_SYSTEM_AUDIT.md](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/FINAL_SYSTEM_AUDIT.md) (Full subsystem audit, privacy verification, mathematical checks, resolved discrepancies, release sign-off).
    - [docs/FINAL_ARCHITECTURE.md](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/FINAL_ARCHITECTURE.md) (End-to-end topology, Mermaid data flow, security model, component integration).
    - [docs/PHYSICAL_DEVICE_VALIDATION_CHECKLIST.md](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/PHYSICAL_DEVICE_VALIDATION_CHECKLIST.md) (Step-by-step 8-phase physical field validation protocol with tests marked `PENDING`).
    - [docs/DEPLOYMENT_AND_SETUP.md](file:///Users/bineetkeshari/Desktop/IndoorAI/docs/DEPLOYMENT_AND_SETUP.md) (Production deployment, environment variables, Nginx reverse proxy, security checklist).
    - [.env.example](file:///Users/bineetkeshari/Desktop/IndoorAI/.env.example) (Updated with JWT signing key and auth configuration).

### Post-Release Enhancement: Multitask MLP Dashboard Integration & Complete White Theme Redesign
- **Status:** COMPLETED
- **Completed On:** 2026-11-19
- **Deliverables:**
  - Integrated pre-trained Multitask MLP (`experiments/mlp/models/multitask_mlp.pt`, 709,258 parameters) into `PredictionService` ([backend/app/services/prediction_service.py](file:///Users/bineetkeshari/Desktop/IndoorAI/backend/app/services/prediction_service.py)) on PyTorch CPU with zero silent fallback.
  - Sidebar Model Selector updated to present all 4 model options clearly:
    1. `Random Forest — Task 07 Baseline`
    2. `k-Nearest Neighbors — Task 08 Baseline`
    3. `Multitask MLP — Task 12 Deep Learning`
    4. `Hybrid RF + k-NN — Task 10 Experimental`
  - Seamless switching between all four models on the same fingerprint without re-selection.
  - Prediction interface with building, floor, continuous coordinates, uncalibrated joint softmax confidence ($P_{\max,\text{bldg}} \times P_{\max,\text{floor}}$), and explicit point regressor spatial uncertainty `None` / `N/A (Point Regressor)`.
  - Three-way independently audited benchmark table (RF vs k-NN vs Multitask MLP) with exact verified Task 09/12 metrics and native dataset coordinate units.
  - Separate experimental hybrid localizer table with $\tau = 0.70$ post-hoc gating disclosure and 4-way comparative strengths/limitations analysis.
  - Complete professional white theme redesign (`#FFFFFF` background, `#F8FAFC` sidebar, `#0F172A` text, `#2563EB` accent, `#E2E8F0` borders) across both workspaces and all Matplotlib/Plotly figures ([.streamlit/config.toml](file:///Users/bineetkeshari/Desktop/IndoorAI/.streamlit/config.toml), [frontend/app.py](file:///Users/bineetkeshari/Desktop/IndoorAI/frontend/app.py)).
  - Dedicated unit and integration test suite ([tests/test_mlp_dashboard.py](file:///Users/bineetkeshari/Desktop/IndoorAI/tests/test_mlp_dashboard.py) with 10 tests).
  - All 333 tests passing with 100% pass rate and zero regressions across the entire repository.


















