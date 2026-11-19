# IndoorAI Web Dashboard User Guide & Operational Reference

## 1. Executive Summary

The **IndoorAI Web Dashboard** (`frontend/app.py`) is the official interactive demonstration interface for the IndoorAI project. It provides real-time multi-model localization inference, Wi-Fi fingerprint exploration, signal strength analysis, 2D coordinate mapping, and evaluation against ground truth on the **UJIIndoorLoc** academic benchmark dataset, alongside the complete Multimodal Survey Explorer.

The dashboard integrates the four machine learning architectures developed and benchmarked in Tasks 07–12:
1. **Random Forest Baseline** (Task 07): 5-head multi-output ensemble (200 trees/head).
2. **k-Nearest Neighbors Baseline** (Task 08): Instance-based continuous regression using Cosine distance metric with distance weighting ($k=7$).
3. **Multitask MLP** (Task 12 Deep Learning): Deep neural network with shared representation trunk (512 $\rightarrow$ 256 $\rightarrow$ 128) and four task heads.
4. **Hybrid Localizer** (Task 10): Two-stage hierarchical exploratory prototype combining Random Forest macro-classification with localized micro-k-NN coordinate regression.

The entire dashboard is designed with a consistent, professional white theme (`#FFFFFF` background, `#F8FAFC` sidebar, `#0F172A` text, `#2563EB` primary accent, `#E2E8F0` subtle borders) across both workspaces.

---

## 2. Quickstart & Startup Instructions

### Prerequisites
- Python 3.11 virtual environment activated (`.venv`).
- Streamlit v1.67+ installed.
- Pre-trained models available in `experiments/random_forest/models/`, `experiments/knn/models/`, `experiments/mlp/models/multitask_mlp.pt`, and `experiments/hybrid/models/`.

### Starting the Application

From the project root:

```bash
# Standard interactive launch
./.venv/bin/streamlit run frontend/app.py

# Headless / custom port launch (e.g., port 8501)
./.venv/bin/streamlit run frontend/app.py --server.port 8501 --server.headless true
```

The application will be accessible at:
- **Local URL:** `http://localhost:8501`
- **Network URL:** `http://127.0.0.1:8501`

---

## 3. Application Architecture & Sections

```
+-----------------------------------------------------------------------------------------+
|                                        INDOORAI                                         |
|                 Interactive Indoor Localization & Wi-Fi Fingerprinting Engine           |
+-----------------------------------------------------------------------------------------+
|  [Sidebar Navigation & Active Model Selector (RF / k-NN / Multitask MLP / Hybrid)]      |
+-----------------------------------------------------------------------------------------+
|  Section A: System & Dataset Overview                                                   |
|  - High-level KPIs: 19,937 Train, 1,111 Val, 3 Buildings, 5 Floors, 520 WAPs, 4 Models  |
|  - Benchmark Disclosure & Offline Evaluation Notice                                     |
+-----------------------------------------------------------------------------------------+
|  Section B: Wi-Fi Fingerprint Explorer                                                  |
|  - Dual Mode: Browse 1,111 Validation Samples OR Upload Custom CSV Fingerprint          |
|  - Detected WAPs Table (RSSI dBm, Signal Quality: Strong, Medium, Weak)                 |
|  - Signal Strength Distribution Chart (Top 15 Detected WAPs)                            |
+-----------------------------------------------------------------------------------------+
|  Section C: Indoor Location Prediction                                                  |
|  - Active Model Selection & Instant Prediction across all 4 architectures               |
|  - Output Cards: Building ID, Floor ID, Longitude, Latitude, Confidence, Uncertainty    |
|  - MLP Softmax & Point Regressor Disclosures                                            |
|  - Hierarchical Routing Details (Local vs Global k-NN Fallback for Hybrid)              |
+-----------------------------------------------------------------------------------------+
|  Section D: 2D Indoor Coordinate Map                                                    |
|  - Ambient Fingerprint Scatter colored by Building (Campus Overview or Filtered)        |
|  - Red Star: Predicted Location | Green Square: Ground Truth Location                   |
|  - Dashed Error Vector with Distance Annotation in Dataset Coordinate Units             |
+-----------------------------------------------------------------------------------------+
|  Section E: Prediction Accuracy Evaluation (When Ground Truth is Available)             |
|  - Building Match Status | Floor Match Status (Delta Floors)                            |
|  - 2D Euclidean Error Decomposition in Dataset Coordinate Units                         |
+-----------------------------------------------------------------------------------------+
|  Section F: Model Benchmarking & Selection                                              |
|  - Three-way independently audited benchmark table (RF vs k-NN vs Multitask MLP)        |
|  - Comparative strengths and operational trade-offs breakdown                           |
|  - Experimental Two-Stage Hybrid table with post-hoc tau=0.70 gating disclosure         |
+-----------------------------------------------------------------------------------------+
```

---

## 4. Key Functional Capabilities

### A. System & Dataset Overview
- Real-time KPI summary across training samples, validation samples, buildings, floors, and monitored access points.
- **Mandatory Disclosure Notice:** Explicitly reminds operators that the system operates on the UJIIndoorLoc academic benchmark dataset and **does not** perform live phone tracking or GPS trilateration.

### B. Wi-Fi Fingerprint Explorer
- **Validation Dataset Browser:** Allows scrubbing through all 1,111 validation records with a numeric selector or one-click preset buttons for representative buildings and floors.
- **Custom CSV Upload:** Users can upload a CSV containing Wi-Fi access point observations. The file is validated on upload against the 520 WAP column requirement.
- **Detected WAPs Table:** Identifies which APs were received, signal level in dBm, and categorized quality:
  - 🟢 Strong: $\ge -60\text{ dBm}$
  - 🟡 Medium: $-80\text{ to } -60\text{ dBm}$
  - 🔴 Weak: $< -80\text{ dBm}$
- **Signal Distribution Chart:** Horizontal bar visualization showing top detected APs sorted by signal power.

### C. Indoor Location Prediction
- Instant multi-model inference backed by `backend/app/services/prediction_service.py`.
- Seamless switching between all four models on the same fingerprint without re-selecting.
- Computes:
  - **Predicted Building ID:** 0, 1, or 2.
  - **Predicted Floor Level:** 0, 1, 2, 3, or 4.
  - **Predicted Longitude & Latitude:** Continuous coordinates in dataset coordinate space.
  - **Model Confidence:**
    - Tree Ensembles (RF): Maximum ensemble tree vote proportion.
    - Instance-Based (k-NN): Distance-weighted neighbor consensus.
    - Deep Learning (MLP): Joint uncalibrated softmax score ($P_{\max,\text{bldg}} \times P_{\max,\text{floor}}$).
  - **Spatial Uncertainty:**
    - RF: Heuristic vote-spread proxy.
    - k-NN / Hybrid: Empirical neighbor spatial dispersion $\sigma_{\text{spatial}} = \sqrt{\sigma_{\text{lon}}^2 + \sigma_{\text{lat}}^2}$.
    - Multitask MLP: Displayed as `N/A (Point Regressor)` because continuous point regression does not produce spatial dispersion.

### D. 2D Indoor Coordinate Map
- Interactive scatter plot built with Matplotlib on a pure white canvas (`#FFFFFF`).
- Displays ambient validation points across the 3 campus buildings with clean contrast colors.
- Allows filtering to the specific predicted building or displaying all campus buildings.
- Clearly indicates Predicted Location (★) and Ground Truth (■) with a dashed error vector.
- **Axis Invariant:** Axes are explicitly labeled **Dataset Longitude (Coordinate X)** and **Dataset Latitude (Coordinate Y)**.

### E. Prediction Evaluation
- Active whenever ground truth target columns are present.
- Computes 2D Euclidean coordinate error:
  $$e_{2D} = \sqrt{(\hat{x} - x)^2 + (\hat{y} - y)^2}$$
- Categorizes error into performance bands:
  - 🟢 Excellent ($< 5.0\text{ units}$)
  - 🟡 Good ($5.0 - 10.0\text{ units}$)
  - 🟠 Moderate ($10.0 - 20.0\text{ units}$)
  - 🔴 Elevated ($> 20.0\text{ units}$)

### F. Model Benchmarking & Selection
Tabulates verified validation metrics established across Tasks 07, 08, and 12, alongside experimental prototypes from Task 10:

#### 1. Independently Audited Three-Way Baselines (Tasks 07, 08, & 12 Source of Truth)

| Model Architecture | Building Acc (%) | Floor Acc (%) | Building-Floor Acc (%) | Mean 2D Error | Median 2D Error | 95th Pct Error | Model Size | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Baseline** | **99.82%** | 88.66% | 88.48% | 11.53 units | 8.65 units | 32.74 units | ~561 MB | Verified Baseline (Task 07 / Task 09) |
| **k-Nearest Neighbors (k=7)** | **99.91%** | 84.43% | 84.43% | **10.05 units** | **7.33 units** | 29.06 units | ~17.7 MB | Verified Baseline (Task 08 / Task 09) |
| **Multitask MLP (Task 12)** | 99.82% | **90.37%** | **90.37%** | 12.18 units | 10.06 units | **28.77 units** | **~2.75 MB** | Verified Deep Learning (Task 12) |

#### 2. Experimental Research Prototype (Task 10 Error Analysis)

| Model Architecture | Building Acc (%) | Floor Acc (%) | Building-Floor Acc (%) | Mean 2D Error | Median 2D Error | 95th Pct Error | Model Size | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Two-Stage Hybrid Localizer** | 99.82% | 88.66% | 88.48% | 9.98 units | 7.29 units | 29.26 units | ~578.8 MB | Experimental Prototype ($\tau=0.70$ post-hoc) |

---

## 5. Custom CSV Upload Specifications

To upload custom Wi-Fi fingerprints, the CSV must adhere to the following schema:

1. **Required Columns (520 Access Points):**
   `WAP001`, `WAP002`, ..., `WAP520`
   - Numeric values representing RSSI in dBm (typically $-104\text{ to } -30\text{ dBm}$).
   - Unavailable/undetected APs must be represented by `100` (UJIIndoorLoc standard) or `-105.0` (noise floor).
2. **Optional Target Columns (for ground-truth evaluation):**
   - `BUILDINGID`: Integer (0, 1, or 2)
   - `FLOOR`: Integer (0, 1, 2, 3, or 4)
   - `LONGITUDE`: Float (e.g., $-7500.0$)
   - `LATITUDE`: Float (e.g., $4864900.0$)
3. **Processing Invariant:**
   Target columns are strictly segregated upon ingestion; only the 520 WAP columns are passed through the feature extraction pipeline, preventing any target leakage.

---

## 6. System Invariants & Important Disclosures

1. **No Live Tracking:** This system is an offline benchmark evaluation tool. It does not interface with mobile GPS, Wi-Fi hardware chips, or network routers.
2. **Coordinate Space:** All spatial coordinates and Euclidean errors are reported in **dataset coordinate units**, never metres. No physical metric scale conversion is fabricated.
3. **No Fabricated Floorplans:** The 2D coordinate map plots empirical data points in relative coordinate space; no architectural walls, doors, hallways, or building blueprints are fabricated.
4. **Zero Data Leakage:** Target and metadata columns are never exposed to feature extraction or inference heads.
5. **Deep Learning Softmax Semantics:** Multitask MLP confidence represents joint uncalibrated softmax output, not a Bayesian calibrated posterior probability. Continuous point regression produces coordinates without empirical dispersion bounds.

---

## 5. Custom CSV Upload Specifications

To upload custom Wi-Fi fingerprints, the CSV must adhere to the following schema:

1. **Required Columns (520 Access Points):**
   `WAP001`, `WAP002`, ..., `WAP520`
   - Numeric values representing RSSI in dBm (typically $-104\text{ to } -30\text{ dBm}$).
   - Unavailable/undetected APs must be represented by `100` (UJIIndoorLoc standard) or `-105.0` (noise floor).
2. **Optional Target Columns (for ground-truth evaluation):**
   - `BUILDINGID`: Integer (0, 1, or 2)
   - `FLOOR`: Integer (0, 1, 2, 3, or 4)
   - `LONGITUDE`: Float (e.g., $-7500.0$)
   - `LATITUDE`: Float (e.g., $4864900.0$)
3. **Processing Invariant:**
   Target columns are strictly segregated upon ingestion; only the 520 WAP columns are passed through the feature extraction pipeline, preventing any target leakage.

---

## 6. System Invariants & Important Disclosures

1. **No Live Tracking:** This system is an offline benchmark evaluation tool. It does not interface with mobile GPS, Wi-Fi hardware chips, or network routers.
2. **Coordinate Space:** All spatial coordinates and Euclidean errors are reported in **dataset coordinate units**, never metres. No physical metric scale conversion is fabricated.
3. **No Fabricated Floorplans:** The 2D coordinate map plots empirical data points in relative coordinate space; no architectural walls, doors, hallways, or building blueprints are fabricated.
4. **Zero Data Leakage:** Target and metadata columns are never exposed to feature extraction or inference heads.

