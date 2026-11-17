# Pedestrian Dead Reckoning (PDR) Engine Architecture

## 1. Overview & Architectural Role in IndoorAI

The **Pedestrian Dead Reckoning (PDR) Engine** provides continuous, kinematically consistent relative indoor trajectory tracking for the IndoorAI platform. In modern indoor positioning systems, Wi-Fi fingerprinting provides periodic absolute position fixes with meter-level uncertainty, but Wi-Fi scans are intermittent (typically throttled to 0.1–0.5 Hz on modern Android devices) and suffer from multipath fading. 

The PDR Engine bridges these temporal gaps by tracking pedestrian movement at IMU rates (50–100 Hz). Operating exclusively on preprocessed smartphone sensor observations (accelerometer, gyroscope, magnetometer, and rotation vector) produced by Task 20, the PDR engine estimates step occurrences, step lengths, walking headings, and 2D relative trajectories:

$$\mathbf{p}_k = \mathbf{p}_{k-1} + \begin{bmatrix} L_k \cos\theta_k \\ L_k \sin\theta_k \end{bmatrix}$$

```
+-------------------------------------------------------------------------------+
|                        IndoorAI Multimodal Pipeline                            |
|                                                                               |
|   [Android Survey Export]                                                     |
|              |                                                                |
|              v                                                                |
|      [Task 18 Ingestion]                                                      |
|              |                                                                |
|              v                                                                |
|      [Task 20 Preprocessing]                                                  |
|              |                                                                |
|              +--------------------------+                                     |
|              |                          |                                     |
|              v                          v                                     |
|    [sensor_cleaned.parquet]   [wifi_cleaned.parquet]                          |
|              |                          |                                     |
|              v                          |                                     |
|     +--------------------+              |                                     |
|     |   Task 21: PDR     |              |                                     |
|     |  Relative Tracking |              |                                     |
|     +--------------------+              |                                     |
|              |                          |                                     |
|              | (Relative Trajectory,    |                                     |
|              |  Steps, Uncertainty)     |                                     |
|              |                          |                                     |
|              v                          v                                     |
|     [Future Sensor Fusion / Graph Optimization]                               |
+-------------------------------------------------------------------------------+
```

> [!IMPORTANT]
> **Relative Positioning Scope**: The PDR Engine tracks relative movement starting from an arbitrary origin $(0, 0)$ in an East-North-Up (ENU) tangent plane. It does **not** claim absolute indoor coordinates or georeferenced building coordinates. Absolute positioning requires sensor fusion with Wi-Fi landmarks and map constraints in downstream tasks.

---

## 2. Coordinate Frames & Transformations

The PDR Engine manages four distinct coordinate reference frames to accurately isolate pedestrian horizontal movement from arbitrary device orientation:

```
           Z (Up)
           ^
           |     Y (North)
           |    /
           |   /
           |  /
           | /
           +------------> X (East)
    [Local Navigation Frame: ENU]
```

### 2.1 Coordinate Definitions

1. **Smartphone Body Frame ($\mathcal{B}$)**:
   - Defined by Android sensor hardware standards:
     - $+X$: Points right along the display width.
     - $+Y$: Points up along the display height.
     - $+Z$: Points out perpendicular to the display face.
   - Sensor observations from raw IMU records reside in $\mathcal{B}$.

2. **Local Navigation Frame ($\mathcal{N}$ / ENU)**:
   - A right-handed horizontal tangent frame:
     - $+X$: Local East.
     - $+Y$: Local Magnetic / True North.
     - $+Z$: Local Up (aligned anti-parallel to gravity $\mathbf{g}$).

3. **Gravity-Aligned Frame ($\mathcal{G}$)**:
   - An intermediate frame whose vertical axis $\mathbf{z}_\mathcal{G}$ aligns with local gravity:
     $$\hat{\mathbf{g}} = \frac{\mathbf{g}}{\|\mathbf{g}\|}$$
   - Rotates the body frame such that vertical motion (heel-strike bounces) decouples cleanly from horizontal motion regardless of whether the phone is held vertically, horizontally, or tilted in hand.

4. **Horizontal Movement Plane ($\mathcal{P}_\text{horiz}$)**:
   - The 2D plane perpendicular to $\hat{\mathbf{g}}$ spanned by unit vectors $(\hat{\mathbf{e}}_x, \hat{\mathbf{e}}_y)$.
   - Trajectory reconstruction and spatial covariance propagation occur strictly within this plane.

### 2.2 Mathematical Rotations

Given a unit quaternion $\mathbf{q} = [q_w, q_x, q_y, q_z]^T$ representing device orientation from body frame to navigation frame:

$$\mathbf{R}(\mathbf{q}) = \begin{bmatrix} 
1 - 2(q_y^2 + q_z^2) & 2(q_x q_y - q_w q_z) & 2(q_x q_z + q_w q_y) \\
2(q_x q_y + q_w q_z) & 1 - 2(q_x^2 + q_z^2) & 2(q_y q_z - q_w q_x) \\
2(q_x q_z - q_w q_y) & 2(q_y q_z + q_w q_x) & 1 - 2(q_x^2 + q_y^2)
\end{bmatrix}$$

Vector rotation from Body to World is:
$$\mathbf{v}_\mathcal{W} = \mathbf{R}(\mathbf{q}) \mathbf{v}_\mathcal{B}$$

And World to Body:
$$\mathbf{v}_\mathcal{B} = \mathbf{R}(\mathbf{q})^T \mathbf{v}_\mathcal{W}$$

---

## 3. Gravity Compensation

Accurate step detection requires separating constant Earth gravity ($\approx 9.80665 \text{ m/s}^2$) from dynamic user acceleration ($\mathbf{a}_\text{user}$). The engine implements two distinct compensation strategies with automatic fallback:

### 3.1 Method A: Android Rotation Vector (Preferred)

When synthetic orientation vectors (`ROTATION_VECTOR` or `GAME_ROTATION_VECTOR`) are available from the Android OS fused sensor hub:
1. Extract unit quaternion $\mathbf{q}$ at each sample.
2. In the navigation frame, nominal Earth gravity is $\mathbf{g}_\mathcal{N} = [0, 0, 9.80665]^T$.
3. Rotate gravity into the smartphone body frame:
   $$\mathbf{g}_\mathcal{B} = \mathbf{R}(\mathbf{q})^T \mathbf{g}_\mathcal{N}$$
4. Calculate pure user dynamic linear acceleration:
   $$\mathbf{a}_\text{linear, }\mathcal{B} = \mathbf{a}_\text{total, }\mathcal{B} - \mathbf{g}_\mathcal{B}$$
5. Project dynamic acceleration onto the vertical gravity vector:
   $$a_\text{vertical} = \mathbf{a}_\text{linear, }\mathcal{B} \cdot \left(-\frac{\mathbf{g}_\mathcal{B}}{\|\mathbf{g}_\mathcal{B}\|}\right)$$

### 3.2 Method B: Low-Pass Accelerometer Filter (Fallback)

When rotation vectors are absent or corrupted:
1. Apply a zero-phase forward-backward Butterworth low-pass filter ($f_c = 0.5 \text{ Hz}$, Order 2) to each accelerometer channel:
   $$\mathbf{g}_\mathcal{B}(t) = \text{LowPass}_{0.5\text{Hz}}(\mathbf{a}_\text{total, }\mathcal{B}(t))$$
2. Scale estimated gravity vector to nominal magnitude ($9.80665 \text{ m/s}^2$):
   $$\hat{\mathbf{g}}_\mathcal{B} = 9.80665 \cdot \frac{\mathbf{g}_\mathcal{B}}{\|\mathbf{g}_\mathcal{B}\|}$$
3. Isolate linear acceleration and vertical component:
   $$\mathbf{a}_\text{linear, }\mathcal{B} = \mathbf{a}_\text{total, }\mathcal{B} - \hat{\mathbf{g}}_\mathcal{B}$$
   $$a_\text{vertical} = \mathbf{a}_\text{linear, }\mathcal{B} \cdot \hat{\mathbf{g}}_\mathcal{B}$$

---

## 4. Signal Filtering & Discontinuity Handling

Human walking induces cyclical accelerations in the 0.5–3.5 Hz band. Raw smartphone IMU data contains high-frequency hand tremors, engine vibration, and digitizer noise.

```
Raw Acceleration  ──> [2nd-Order Butterworth Bandpass] ──> Filtered Step Signal
                      (Passband: 0.5 Hz - 3.5 Hz)
                      (Zero-phase filtfilt)
```

### 4.1 Filter Pipeline
- **Step Signal Filter**: Zero-phase Butterworth bandpass ($f_\text{low} = 0.5 \text{ Hz}$, $f_\text{high} = 3.5 \text{ Hz}$, order 2). Zero-phase forward-backward filtering ensures peak locations in time are preserved without group-delay latency.
- **Online Exponential Smoothing**: For streaming attitude/gravity updates, an exponential filter with gap reset logic is provided:
  $$y_k = \alpha y_{k-1} + (1 - \alpha) x_k, \quad \alpha = \exp(-2\pi f_c \Delta t)$$

### 4.2 Gap-Preserving Resets (Zero Silent Interpolation)
In accordance with IndoorAI core data integrity principles, if a sensor sampling gap exceeds `sensor_gap_threshold_ms` (500 ms):
- Filter states are flushed and reset directly to current observation values.
- Step detection windows break at gaps; no steps are interpolated across missing data intervals.
- The `is_gap` flag is propagated into downstream records.

---

## 5. Orientation & Heading Estimation

The user's horizontal movement heading $\theta_k \in [-\pi, \pi]$ relative to North is resolved using multi-tiered estimation:

### 5.1 Heading Sources & Priority
1. **Android Rotation Vector**: Provides fused 3D attitude with continuous internal gyro bias compensation.
2. **Tilt-Compensated Magnetometer**: When rotation vectors are absent, pitch ($\phi$) and roll ($\gamma$) derived from the gravity vector level the magnetometer observations:
   $$B_{x,\text{level}} = B_x \cos\phi + B_z \sin\phi$$
   $$B_{y,\text{level}} = B_x \sin\gamma \sin\phi + B_y \cos\gamma - B_z \sin\gamma \cos\phi$$
   $$\psi_\text{mag} = \text{atan2}(-B_{y,\text{level}}, B_{x,\text{level}})$$
3. **Gyroscope Yaw Integration**: If magnetic disturbances are detected, heading updates integrate rate of turn around the gravity vector:
   $$\theta_k = \theta_{k-1} + \omega_{z,\text{grav}} \Delta t$$

### 5.2 Magnetic Disturbance Detection
Indoor buildings contain structural steel, reinforced concrete, and electrical machinery that warp Earth's magnetic field. The engine continuously validates magnetic readings:
- Total field anomaly: $\|\mathbf{B}\| \notin [25, 65] \ \mu\text{T}$ triggers disturbance warning.
- Attitude divergence: If $|\psi_\text{mag} - \psi_\text{gyro}| > 45^\circ$, the source is flagged as `DISTURBED` and heading confidence drops to 0.50.

---

## 6. Step Detection Engine

Step detection operates on vertical dynamic acceleration $a_\text{vertical}$ using a four-stage signal processing chain:

```
[a_vertical] ──> [Bandpass Filter] ──> [Peak Detection] ──> [Refractory Filter] ──> [Accepted Steps]
                  (0.5 - 3.5 Hz)       (Prominence >= 0.4)   (Interval >= 330 ms)
```

1. **Candidate Peak Identification**:
   - Detect local maxima with dynamic thresholding:
     $$\text{Peak Value} \ge 0.5 \cdot a_\text{thresh} \quad (0.40 \text{ m/s}^2)$$
     $$\text{Peak Prominence} \ge \Pi_\text{min} \quad (0.40 \text{ m/s}^2)$$
   - Candidate peaks represent tentative foot strikes.

2. **Refractory Duplicate Rejection**:
   - Biomechanical constraints enforce that human walking cadence rarely exceeds 3.0 steps/sec.
   - Any candidate peak occurring within $\tau_\text{refractory} = 330 \text{ ms}$ of the preceding accepted peak is classified as bounce noise and rejected.
   - If a candidate peak in the refractory window has amplitude $> 1.2 \times$ the accepted peak, it replaces the prior peak as the true heel strike.

3. **Step Metrics & Confidence Scoring**:
   - For each accepted step $k$, the algorithm searches backward up to 500 ms for the preceding local acceleration trough ($a_{\min, k}$).
   - Dynamic acceleration bounce range:
     $$\Delta a_k = a_{\max, k} - a_{\min, k}$$
   - Cadence regularity: Step interval $\Delta t_k = t_k - t_{k-1}$ is evaluated against human physiological norms ($330 \text{ ms} \le \Delta t_k \le 1500 \text{ ms}$). Regular cadences earn confidence $\ge 0.90$.

---

## 7. Weinberg Step Length Estimation

Step length varies dynamically with walking speed and bounce amplitude. Rather than using an unrealistic fixed stride length, the PDR engine implements the empirically validated **Weinberg bounce model**:

$$L_k = K \cdot \left(a_{\max, k} - a_{\min, k}\right)^{1/4} = K \cdot \Delta a_k^{0.25}$$

### 7.1 Model Parameters & Bounding
- **Calibration Factor ($K$)**: Configurable gain, default $K = 0.42$ for smartphone hand-held mode.
- **Physiological Clamping**: Step length is strictly clamped to valid human stride bounds:
  $$L_k = \text{clip}\left(L_k, \ 0.30 \text{ m}, \ 1.20 \text{ m}\right)$$
- **Step Length Uncertainty**: Assigned based on dynamic bounce regularity and sensor confidence:
  $$\sigma_{L, k} = \max\left(0.05 \text{ m}, \ 0.10 \cdot L_k\right)$$

---

## 8. Trajectory Reconstruction & Relative Dead Reckoning

Trajectories are reconstructed step-by-step in the horizontal movement plane $(\hat{\mathbf{e}}_x, \hat{\mathbf{e}}_y)$:

```
           (x_k, y_k)
               ^
              /
   L_k       /
            /   theta_k
           +---------------> East
     (x_k-1, y_k-1)
```

### 8.1 Dead Reckoning Kinematics
For step $k$ occurring at timestamp $t_k$:
$$\Delta x_k = L_k \cdot \cos\theta_k$$
$$\Delta y_k = L_k \cdot \sin\theta_k$$
$$x_k = x_{k-1} + \Delta x_k$$
$$y_k = y_{k-1} + \Delta y_k$$

- **Origin Anchor**: Every trajectory begins at point `PDR-00000` with coordinates $(0.0, 0.0)$, zero distance, and zero uncertainty.

---

## 9. Spatial Covariance & Uncertainty Propagation

PDR is inherently subject to dead reckoning drift: step length miscalibration and heading bias cause positional error to accumulate over time. The PDR Engine tracks this uncertainty rigorously using first-order Kalman-style spatial covariance propagation:

### 9.1 Step Error Covariance ($Q_k$)
Errors in step length and heading angle are assumed uncorrelated over individual steps:
$$\mathbf{Q}_k = \begin{bmatrix} \sigma_{L, k}^2 & 0 \\ 0 & \sigma_{\theta, k}^2 \end{bmatrix}$$

- $\sigma_{L, k}$: Step length standard deviation (typically $\approx 0.06$–$0.12$ m).
- $\sigma_{\theta, k}$: Heading uncertainty based on sensor source (Rotation Vector: $\approx 0.08$ rad / $4.5^\circ$; Magnetometer: $\approx 0.20$ rad / $11.5^\circ$).

### 9.2 First-Order Taylor Series Jacobian ($J_k$)
The kinematic transformation vector is:
$$\mathbf{f}(L_k, \theta_k) = \begin{bmatrix} L_k \cos\theta_k \\ L_k \sin\theta_k \end{bmatrix}$$

The Jacobian with respect to the state inputs $[L_k, \theta_k]^T$ is:
$$\mathbf{J}_k = \begin{bmatrix}
\frac{\partial \Delta x}{\partial L} & \frac{\partial \Delta x}{\partial \theta} \\
\frac{\partial \Delta y}{\partial L} & \frac{\partial \Delta y}{\partial \theta}
\end{bmatrix} = \begin{bmatrix}
\cos\theta_k & -L_k \sin\theta_k \\
\sin\theta_k & L_k \cos\theta_k
\end{bmatrix}$$

### 9.3 2D Spatial Covariance Accumulation ($P_k$)
$$\mathbf{P}_k = \mathbf{P}_{k-1} + \mathbf{J}_k \mathbf{Q}_k \mathbf{J}_k^T$$

Expanding the matrix multiplication yields:
$$\mathbf{\Sigma}_{\Delta k} = \begin{bmatrix}
\sigma_L^2 \cos^2\theta_k + L_k^2 \sigma_\theta^2 \sin^2\theta_k & (\sigma_L^2 - L_k^2 \sigma_\theta^2) \sin\theta_k \cos\theta_k \\
(\sigma_L^2 - L_k^2 \sigma_\theta^2) \sin\theta_k \cos\theta_k & \sigma_L^2 \sin^2\theta_k + L_k^2 \sigma_\theta^2 \cos^2\theta_k
\end{bmatrix}$$

### 9.4 95% Confidence Uncertainty Radius ($R_k$)
From the eigenvalues $\lambda_{\max}$ of $\mathbf{P}_k$:
$$R_k = 1.96 \cdot \sqrt{\lambda_{\max}(\mathbf{P}_k)}$$
The resulting uncertainty radius $R_k$ grows monotonically with traveled distance, accurately bounding position confidence until external absolute fixes (Wi-Fi) are fused.

---

## 10. Output Artifact Schema & Storage Layout

When executed for a given dataset, the PDR engine writes all computed artifacts to:
`data/processed/multimodal/<dataset_id>/pdr/`

```
data/processed/multimodal/<dataset_id>/pdr/
├── gravity_compensated.parquet     # Gravity-separated 3D acceleration stream
├── orientation_estimates.parquet    # Roll, pitch, yaw attitude estimates
├── detected_steps.parquet          # Filtered heel-strike step events
├── step_features.parquet           # Weinberg step lengths & kinematics
├── relative_trajectory.parquet     # 2D relative dead reckoning path & covariances
├── pdr_report.json                 # Comprehensive metrics & movement KPIs
└── pdr_manifest.json               # SHA-256 checksums and record counts
```

### 10.1 `relative_trajectory.parquet` Schema

| Column | Type | Description |
|---|---|---|
| `trajectory_point_id` | String | Unique ID (`PDR-00000`, `PDR-00001`, etc.) |
| `session_id` | String | Associated survey session identifier |
| `timestamp_ms` | Int64 | Epoch millisecond timestamp of position fix |
| `step_id` | String | Associated step event ID (`ORIGIN` for start) |
| `rel_x_m` | Float64 | Relative East position coordinate (meters) |
| `rel_y_m` | Float64 | Relative North position coordinate (meters) |
| `cum_distance_m` | Float64 | Cumulative path distance traversed (meters) |
| `heading_deg` | Float64 | Trajectory heading at this fix ($0^\circ$–$360^\circ$) |
| `step_length_m` | Float64 | Length of preceding step (meters) |
| `var_x` | Float64 | Position $X$ spatial variance ($P_{xx}$) |
| `var_y` | Float64 | Position $Y$ spatial variance ($P_{yy}$) |
| `cov_xy` | Float64 | Cross-covariance ($P_{xy}$) |
| `pos_uncertainty_radius_m` | Float64 | 95% circular error probable radius |
| `confidence` | Float64 | Aggregate kinematic confidence ($0.0$–$1.0$) |

---

## 11. REST API & UI Integration

The PDR engine is accessible via five FastAPI REST endpoints in `backend/app/api/routes/datasets.py`:

```
POST /api/datasets/{dataset_id}/pdr             # Execute PDR pipeline with parameters
GET  /api/datasets/{dataset_id}/pdr             # Query computation status and summary
GET  /api/datasets/{dataset_id}/pdr/trajectory  # Retrieve 2D relative trajectory coordinates
GET  /api/datasets/{dataset_id}/pdr/steps       # Retrieve detected step events and kinematics
GET  /api/datasets/{dataset_id}/pdr/report      # Retrieve detailed movement report and stats
```

The Streamlit dashboard includes Section **11. PDR & Movement Analysis** (`frontend/components/pdr_view.py`), providing:
1. High-level KPIs: Total steps, cumulative distance, net displacement, mean cadence, average step length, and final drift radius.
2. Interactive 2D Plotly path visualization: Scatter plot of relative coordinates $(X, Y)$ with uncertainty error ellipses.
3. Cadence and heading time-series charts.
4. Granular step event inspector table with export capability.

---

## 12. Architectural Limitations & Downstream Roadmap

1. **Dead Reckoning Drift**:
   Without external reference points, angular gyroscope drift ($\approx 0.5^\circ$–$2^\circ$ per minute) causes the reconstructed trajectory to slowly rotate.
2. **Device Orientation Mode**:
   The engine assumes a relatively stable device carriage (e.g. handheld while walking or chest pocket). Extreme swinging (e.g., in hand while running) requires dynamic carriage mode detection.
3. **Map and Landmark Fusion (Next Tasks)**:
   Task 21 establishes pure relative dead reckoning. Downstream tasks (Task 22+) will incorporate Wi-Fi fingerprint proximity anchors, building floorplans, and map-matching particles to constrain PDR drift to corridors and valid walking paths.
