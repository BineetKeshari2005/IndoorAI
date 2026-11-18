# PDR Calibration, Ground-Truth Validation & Drift Analysis Architecture

## 1. Overview & Objective

The **PDR Calibration and Validation Framework** provides systematic empirical verification for the IndoorAI Pedestrian Dead Reckoning (PDR) engine developed in Task 21. While Task 21 established the kinematic and signal processing algorithms (gravity separation, zero-phase Butterworth filtering, Weinberg bounce model, and 2D dead reckoning), Task 22 establishes the scientific methodology to measure how accurately the engine estimates distance, direction, and relative trajectories against manually measured physical ground truth.

```
+---------------------------------------------------------------------------------------+
|                    Task 22 Calibration & Validation Pipeline                           |
|                                                                                       |
|   [Measured Walking Trials]            [PDR Reconstructed Trajectory & Steps]         |
|   (data/calibration/pdr/trials/)       (data/processed/multimodal/<id>/pdr/)          |
|                 │                                        │                            |
|                 ▼                                        ▼                            |
|   ┌──────────────────────────────────────────────────────────────┐                    |
|   │ 1. Weinberg Stride Optimization (K Estimation)               │                    |
|   │    • Objective: min sum w_i * ((D_est(K) - D_true) / D_true)^2│                   |
|   │    • Bounded Scalar Minimization over K in [0.20, 0.80]      │                    |
|   │    • Separate Training & Held-Out Validation Evaluation      │                    |
|   └──────────────────────────────┬───────────────────────────────┘                    |
|                                  │                                                    |
|                                  ▼                                                    |
|   ┌──────────────────────────────────────────────────────────────┐                    |
|   │ 2. Calibration Profile Hierarchy & Persistence               │                    |
|   │    • PROF-DEFAULT (K=0.42, UNCALIBRATED)                     │                    |
|   │    • Participant / Device / Placement Specific Profiles      │                    |
|   └──────────────────────────────┬───────────────────────────────┘                    |
|                                  │                                                    |
|         ┌────────────────────────┼────────────────────────┐                           |
|         ▼                        ▼                        ▼                           |
|   ┌───────────────┐     ┌──────────────────┐     ┌────────────────────────────────┐   |
|   │ 3. Step Eval  │     │ 4. Heading Eval  │     │ 5. Trajectory & Drift Eval     │   |
|   │  • Precision  │     │  • Wraparound    │     │  • Endpoint Error              │   |
|   │  • Recall     │     │  • Circular Mean │     │  • Relative Distance Error     │   |
|   │  • F1-Score   │     │  • Drift deg/min │     │  • Drift per Meter (m/m)       │   |
|   │  • Count Err  │     │  • Mag Anomaly   │     │  • Cross-Track Path Deviation  │   |
|   └───────┬───────┘     └────────┬─────────┘     └───────────────┬────────────────┘   |
|           │                      │                               │                    |
|           └──────────────────────┼───────────────────────────────┘                    |
|                                  ▼                                                    |
|   ┌──────────────────────────────────────────────────────────────┐                    |
|   │ 6. 2D Spatial Uncertainty & 95% Confidence Ellipses          │                    |
|   │    • Chi-Square Quantile (2 DOF): chi2(2, 0.95) = 5.99146    │                    |
|   │    • Semi-Major: a = 2.448 * sqrt(lambda_1)                  │                    |
|   │    • Semi-Minor: b = 2.448 * sqrt(lambda_2)                  │                    |
|   │    • Mahalanobis Empirical Ground-Truth Containment          │                    |
|   └──────────────────────────────┬───────────────────────────────┘                    |
|                                  ▼                                                    |
|   [Structured Evaluation Reports in data/calibration/pdr/calibration_results/]        |
+---------------------------------------------------------------------------------------+
```

> [!IMPORTANT]
> **Ground-Truth Invariants**:
> 1. Ground-truth distances, headings, and step counts must be manually measured using documented tools (laser rangefinders, surveying tapes, architectural floorplans).
> 2. PDR estimates are **never** used as their own ground truth.
> 3. Synthetic test fixtures are strictly segregated from actual human walking trials via the `is_synthetic` flag and are never averaged into empirical accuracy figures.
> 4. Absolute building coordinates are not claimed: evaluations measure relative displacement, trajectory shape fidelity, and dead reckoning drift.

---

## 2. Calibration Dataset Format & Storage Architecture

Ground-truth walking trials and calibration profiles are maintained in an isolated directory structure outside the UJIIndoorLoc benchmark and survey datasets:

```text
data/
└── calibration/
    └── pdr/
        ├── trials/
        │   └── <trial_id>/
        │       ├── trial_metadata.json
        │       └── ground_truth.json
        ├── calibration_results/
        │   └── EVAL-<trial_id>-<hash>.json
        └── calibration_registry.json
```

### 2.1 `trial_metadata.json` Schema

| Field | Type | Description |
|---|---|---|
| `trial_id` | String | Unique trial identifier (e.g. `TRIAL-20261118-0001`) |
| `dataset_id` | String | Foreign key to ingested survey dataset in `data/processed/multimodal/` |
| `session_id` | String | Survey session identifier |
| `participant_id` | String | Pseudonymous participant identifier (e.g. `PARTICIPANT-01`) |
| `device_model` | String | Hardware model (e.g. `Pixel 7 Pro`) |
| `phone_placement` | Enum | `HAND_HELD`, `POCKET`, `BAG`, `TEXTING`, `EAR`, `UNKNOWN` |
| `walking_condition` | Enum | `NORMAL_PACE`, `SLOW_WALK`, `BRISK_WALK`, `STAIRS_EXCLUDED`, `SHUFFLE` |
| `trial_type` | Enum | `STRAIGHT_LINE`, `FORWARD_RETURN`, `L_SHAPED`, `MULTIPLE_TURNS`, `SHORT_DISTANCE`, `LONG_DISTANCE`, `STATIONARY` |
| `start_timestamp_ms`| Int64 | Epoch millisecond at start of walking trial |
| `end_timestamp_ms` | Int64 | Epoch millisecond at completion of walking trial |
| `is_synthetic` | Boolean | True for unit test fixtures, False for real human measurements |
| `trial_notes` | String | Documented physical environment, floor type, footwear notes |
| `created_at_iso` | String | UTC creation timestamp |

### 2.2 `ground_truth.json` Schema

| Field | Type | Description |
|---|---|---|
| `measured_distance_m` | Float | Measured physical distance traversed (meters) |
| `measurement_method` | String | `LASER_RANGEFINDER`, `SURVEY_TAPE`, `ARCHITECTURAL_BLUEPRINT` |
| `measurement_uncertainty_m` | Float | Physical instrument precision bound (e.g. $\pm 0.03$ m for laser) |
| `measured_start_heading_deg` | Float (opt) | Reference starting heading in $[0^\circ, 360^\circ]$ (True/Magnetic North) |
| `measured_end_heading_deg` | Float (opt) | Reference ending heading in $[0^\circ, 360^\circ]$ |
| `heading_measurement_uncertainty_deg`| Float | Heading measurement precision bound (e.g. $\pm 2.0^\circ$) |
| `ground_truth_step_count` | Int (opt) | Manually counted step count (tally counter / video audit) |
| `step_count_uncertainty` | Int | Step counting tolerance (typically 0–1 step) |
| `num_turns` | Int | Number of $90^\circ$ or $180^\circ$ directional turns in path |
| `waypoints` | List[Obj] | Intermediate $(x, y, t)$ survey checkpoints if surveyed |

---

## 3. Weinberg Step-Length Optimization Math

In Task 21, the Weinberg stride heuristic was implemented with nominal default $K = 0.42$:

$$L_k = \text{clip}\left(K \cdot \left(a_{\max, k} - a_{\min, k}\right)^{1/4}, \ L_{\min}, \ L_{\max}\right)$$

In reality, stride mechanics depend heavily on user height, leg length, walking speed, and phone placement (e.g. pocket vs handheld). Task 22 introduces a scalar optimization procedure to fit $K$ from empirical trials.

### 3.1 Mathematical Formulation
For a trial $i$ with $N_i$ accepted steps, the estimated distance as a function of $K$ is:

$$\hat{D}_i(K) = \sum_{k=1}^{N_i} \text{clip}\left(K \cdot \Delta a_{i, k}^{0.25}, \ 0.30 \text{ m}, \ 1.20 \text{ m}\right)$$

Given training trials $i \in T_{\text{train}}$ with measured distances $D_{\text{true}, i}$ and measurement instrument uncertainties $\sigma_{\text{meas}, i}$:

$$\min_{K \in [0.20, 0.80]} \ J(K) = \sum_{i \in T_{\text{train}}} \frac{1}{\sigma_{\text{meas}, i}^2} \left( \frac{\hat{D}_i(K) - D_{\text{true}, i}}{D_{\text{true}, i}} \right)^2$$

### 3.2 Solution Method & Bounding
1. **Bounded Scalar Minimization**: Solved using Brent's method / golden section search (`scipy.optimize.minimize_scalar(method='bounded')`) over $[0.20, 0.80]$ with tolerance $10^{-4}$.
2. **Train/Validation Partitioning**: When $\ge 2$ trials exist, trials are separated into training sets and held-out validation sets. Training metrics (Mean Absolute Error, Root Mean Squared Error, Mean Relative Error %) and held-out validation metrics are calculated independently.
3. **Insufficient Data Fallback**: If zero valid trials are supplied, the optimizer aborts, assigns status `INSUFFICIENT_DATA`, and retains the default $K = 0.42$.

---

## 4. Step Detection Evaluation

Step detection evaluation tests heel-strike classification against ground-truth step counts and timestamps.

### 4.1 Timestamp Window Matching
When reference step timestamps $\{t_{\text{true}, j}\}$ are available:
1. Candidate detections within tolerance window $\tau = \pm 250 \text{ ms}$ of $t_{\text{true}, j}$ are paired using greedy nearest-neighbor matching.
2. Unmatched detections are classified as **False Positives (FP)** (e.g. phone fidgeting, hand tremors).
3. Unmatched reference steps are classified as **False Negatives (FN)** (e.g. soft shuffling, carpet walking).
4. Matched steps are **True Positives (TP)**.

$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

### 4.2 Manual Count Evaluation
When only total manual count $N_{\text{true}}$ is available:
- Count Error: $e_N = \hat{N} - N_{\text{true}}$
- Relative Count Error: $\frac{|\hat{N} - N_{\text{true}}|}{N_{\text{true}}}$
- Lower-bound classification metrics are assigned based on excess / deficit counts.

---

## 5. Heading & Direction Evaluation

Smartphone heading estimation is subject to magnetic soft/hard iron distortions and gyroscope integration drift. Task 22 establishes circular statistics to evaluate direction accuracy.

### 5.1 Circular Wraparound Difference
Angular differences must respect circular topology ($359^\circ$ to $1^\circ$ is a $2^\circ$ deviation, not $358^\circ$):

$$\Delta\theta = \text{atan2}\left(\sin(\hat{\theta} - \theta_{\text{ref}}), \ \cos(\hat{\theta} - \theta_{\text{ref}})\right) \in [-\pi, \pi]$$
$$|\Delta\theta_{\text{deg}}| = \left| \frac{\Delta\theta \cdot 180^\circ}{\pi} \right| \in [0^\circ, 180^\circ]$$

### 5.2 Circular Statistical Metrics
- **Mean Absolute Error (MAE)**: $\frac{1}{N} \sum |\Delta\theta_i|$
- **Median Absolute Error**: 50th percentile of $|\Delta\theta_i|$
- **95th Percentile Error ($P_{95}$)**: 95th percentile of $|\Delta\theta_i|$
- **Circular Mean and Standard Deviation**:
  $$\bar{S} = \frac{1}{N} \sum \sin\Delta\theta_i, \quad \bar{C} = \frac{1}{N} \sum \cos\Delta\theta_i$$
  $$\bar{R} = \sqrt{\bar{S}^2 + \bar{C}^2}, \quad \sigma_\text{circ} = \sqrt{-2 \ln\bar{R}}$$
- **Heading Drift Rate**:
  $$\text{Drift Rate} = \frac{|\Delta\theta_{\text{end}} - \Delta\theta_{\text{start}}|}{\Delta t} \quad \left(^\circ/\text{sec} \text{ and } ^\circ/\text{min}\right)$$

---

## 6. Trajectory Drift & Geometric Accuracy

### 6.1 Distance and Endpoint Drift Metrics
Reconstructed PDR trajectories terminate at estimated coordinates $(\hat{x}_\text{end}, \hat{y}_\text{end})$:
- **Total Estimated Distance**: $\hat{D} = \sum_{k=1}^N L_k$
- **Absolute Distance Error**: $|\hat{D} - D_{\text{true}}|$
- **Relative Distance Error**: $\frac{|\hat{D} - D_{\text{true}}|}{D_{\text{true}}}$
- **Endpoint Euclidean Error**:
  $$e_\text{end} = \sqrt{(\hat{x}_\text{end} - x_{\text{end, true}})^2 + (\hat{y}_\text{end} - y_{\text{end, true}})^2}$$
- **Relative Endpoint Error**: $\frac{e_\text{end}}{D_{\text{true}}}$
- **Drift per Meter**: $\frac{e_\text{end}}{D_{\text{true}}} \quad (\text{m/m})$
- **Drift per Minute**: $\frac{e_\text{end}}{\Delta t_\text{min}} \quad (\text{m/min})$

### 6.2 Cross-Track Error
When intermediate reference waypoints $\mathbf{w}_1, \dots, \mathbf{w}_M$ exist, the minimum perpendicular distance from each trajectory point $\mathbf{p}_k$ to the piecewise linear corridor centerline is evaluated:

$$e_\text{cross-track} = \frac{1}{N} \sum_{k=1}^N \min_{j \in [1, M-1]} \text{dist}(\mathbf{p}_k, \ \overline{\mathbf{w}_j \mathbf{w}_{j+1}})$$

---

## 7. Spatial Uncertainty Audit & 2D Confidence Ellipses

### 7.1 Statistical Audit of Task 21 Formulation
Task 21 computed position covariance accumulation:
$$\mathbf{P}_k = \mathbf{P}_{k-1} + \mathbf{J}_k \mathbf{Q}_k \mathbf{J}_k^T$$
and reported $R_k = 1.96 \sqrt{\lambda_{\max}(\mathbf{P}_k)}$.

The statistical audit identifies two fundamental refinements:
1. **Degrees of Freedom in 2D**:
   The value $1.96$ is the critical value for a **one-dimensional** normal distribution ($\alpha = 0.05$). For a **two-dimensional** bivariate Gaussian distribution $(X, Y)$, the Mahalanobis distance squared follows a Chi-square distribution with **2 degrees of freedom**:
   $$(\mathbf{p} - \hat{\mathbf{p}})^T \mathbf{P}^{-1} (\mathbf{p} - \hat{\mathbf{p}}) \sim \chi^2_2$$
   For $95\%$ probability coverage ($1 - \alpha = 0.95$):
   $$\chi^2_{2, 0.95} = -2 \ln(1 - 0.95) = -2 \ln(0.05) \approx \mathbf{5.99146}$$
   $$\sqrt{\chi^2_{2, 0.95}} \approx \mathbf{2.4477}$$
   Using $1.96$ in 2D provides only $\approx 86.5\%$ coverage ($1 - e^{-1.96^2 / 2} = 0.8535$). Task 22 upgrades all 2D error regions to use $\chi^2_{2, 0.95} = 5.99146$.

2. **Full 2D Confidence Ellipse Parameters**:
   Given symmetric covariance matrix:
   $$\mathbf{P} = \begin{bmatrix} P_{xx} & P_{xy} \\ P_{xy} & P_{yy} \end{bmatrix}$$
   Analytical eigenvalues ($\lambda_1 \ge \lambda_2 \ge 0$):
   $$\lambda_{1, 2} = \frac{P_{xx} + P_{yy}}{2} \pm \sqrt{\left(\frac{P_{xx} - P_{yy}}{2}\right)^2 + P_{xy}^2}$$
   - **Semi-Major Axis ($a$)**: $a = \sqrt{5.99146 \cdot \lambda_1} \approx 2.4477 \sqrt{\lambda_1}$
   - **Semi-Minor Axis ($b$)**: $b = \sqrt{5.99146 \cdot \lambda_2} \approx 2.4477 \sqrt{\lambda_2}$
   - **Ellipse Orientation Angle ($\phi$)**:
     $$\phi = \frac{1}{2} \text{atan2}\left(2 P_{xy}, \ P_{xx} - P_{yy}\right)$$
   - **Conservative Circular Bound**: $R_{\text{circ, 95\%}} = a = 2.448 \sqrt{\lambda_1}$

### 7.2 Augmented 3-State Covariance Propagation Model
When heading error is autocorrelated (e.g. gyroscope bias drift without absolute compass corrections), heading errors rotate all subsequent displacements. Task 22 implements the augmented 3-state Kalman transition:

$$\mathbf{x}_k = [x_k, \ y_k, \ \theta_k]^T$$
$$\mathbf{F}_k = \begin{bmatrix} 1 & 0 & -L_k \sin\theta_{k-1} \\ 0 & 1 & L_k \cos\theta_{k-1} \\ 0 & 0 & 1 \end{bmatrix}, \quad \mathbf{G}_k = \begin{bmatrix} \cos\theta_{k-1} & 0 \\ \sin\theta_{k-1} & 0 \\ 0 & 1 \end{bmatrix}$$
$$\mathbf{P}_{3, k} = \mathbf{F}_k \mathbf{P}_{3, k-1} \mathbf{F}_k^T + \mathbf{G}_k \mathbf{Q}_k \mathbf{G}_k^T$$

---

## 8. Calibration Profile System & Hierarchical Matching

Calibration profiles are managed by `CalibrationStorage` and persisted in `data/calibration/pdr/calibration_registry.json`.

### 8.1 Profile Resolution Hierarchy
When selecting a calibration profile for a walking session:
1. `Participant + Phone Placement + Device Model` (most specific)
2. `Participant + Phone Placement`
3. `Phone Placement + Device Model`
4. `Phone Placement only`
5. `Device Model only`
6. Fallback: `PROF-DEFAULT` ($K = 0.42$, `UNCALIBRATED`)

A specialized profile is only applied if its calibration status is `CALIBRATED` or `VALIDATED`.

---

## 9. REST API Endpoints

The framework exposes seven REST endpoints in `backend/app/api/routes/datasets.py`:

```
POST /api/datasets/{dataset_id}/pdr/trials          # Register ground-truth walking trial
GET  /api/datasets/{dataset_id}/pdr/trials          # List registered trials for dataset
GET  /api/datasets/{dataset_id}/pdr/trials/{id}     # Get full trial record & waypoints
POST /api/datasets/{dataset_id}/pdr/calibrate       # Optimize Weinberg stride parameter K
GET  /api/datasets/{dataset_id}/pdr/calibration     # List all calibration profiles
POST /api/datasets/{dataset_id}/pdr/evaluate        # Run evaluation against trials
GET  /api/datasets/{dataset_id}/pdr/evaluation      # Retrieve historical evaluation reports
```

---

## 10. Dashboard Integration (Section 12)

The Streamlit dashboard includes Section **12. PDR Calibration & Validation** (`frontend/components/pdr_calibration_view.py`) with four dedicated tabs:
1. **🔬 Weinberg Stride Calibration**: Multiselect training and validation trials, run optimization, compare training MAE vs default K.
2. **📋 Walking Trials Registry**: Interactive registration form with measurement precision input and trial catalog.
3. **🗺️ Trajectory Drift & 95% Ellipses**: 2D Plotly canvas rendering estimated PDR paths against ground-truth paths with true Chi-square 95% confidence ellipses.
4. **🎯 Step & Heading Validation**: Precision, recall, F1-score cards, distance error, endpoint drift rate, and report export.
