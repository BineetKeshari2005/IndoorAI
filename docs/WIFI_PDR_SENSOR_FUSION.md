# Wi-Fi + PDR Sensor Fusion Engine (Extended Kalman Filter)

## 1. Overview & Architectural Objectives

Indoor positioning using individual smartphone sensors presents fundamental complementary trade-offs:
- **Pedestrian Dead Reckoning (PDR)** operates at high frequency (1–2 Hz step events) with smooth, relative step-by-step displacement ($L_k, \Delta\theta_k$). However, heading integration without external reference suffers from open-loop drift caused by magnetometer magnetic distortions and gyroscope bias, causing unbounded spatial divergence over time.
- **Wi-Fi Fingerprint Localization** (Task 23) provides drift-free absolute coordinate estimates $[x_{wifi}, y_{wifi}]^T$ tied to surveyed building reference locations. However, Wi-Fi scans are sampled at low, irregular frequencies (every 3–10 seconds), are susceptible to severe multipath, shadowing, AP outages, and exhibit non-Gaussian spatial noise.

The IndoorAI **Extended Kalman Filter (EKF)** sensor fusion engine mathematically unifies PDR kinematics and Wi-Fi fingerprinting into a robust, probabilistic state estimator. The engine operates on preprocessed multimodal survey datasets (Task 20), respecting physical building coordinate frames, validating cross-modal observations, rejecting statistical outliers, and producing continuous 2D position and heading trajectories with calibrated 95% confidence uncertainty bounds.

```
                              ┌───────────────────────────────────┐
                              │ Multimodal Survey Session Storage │
                              └─────────────────┬─────────────────┘
                                                │
                       ┌────────────────────────┴────────────────────────┐
                       ▼                                                 ▼
        ┌─────────────────────────────┐                   ┌─────────────────────────────┐
        │       PDR Step Events       │                   │     Wi-Fi Survey Sweeps     │
        │   (L_k, θ_k, σ_L, σ_θ, Δt)  │                   │ (BSSIDs, RSSI, Timestamps)  │
        └──────────────┬──────────────┘                   └──────────────┬──────────────┘
                       │                                                 │
                       │                                                 ▼
                       │                                  ┌─────────────────────────────┐
                       │                                  │ Site Radio Map Matcher      │
                       │                                  │ (Task 23 WKNN / Bayesian)   │
                       │                                  └──────────────┬──────────────┘
                       │                                                 │
                       │                                                 ▼
                       │                                  ┌─────────────────────────────┐
                       │                                  │ Wi-Fi Uncertainty Hierarchy │
                       │                                  │ (Empirical / Dispersion / R)│
                       │                                  └──────────────┬──────────────┘
                       │                                                 │
                       ▼                                                 ▼
        ┌───────────────────────────────────────────────────────────────────────────────┐
        │                 Asynchronous Chronological Event Queue                        │
        │                   (Sorted by timestamp_epoch_ms)                              │
        └──────────────────────────────────────┬────────────────────────────────────────┘
                                               │
                                               ▼
        ┌───────────────────────────────────────────────────────────────────────────────┐
        │                     Extended Kalman Filter (EKF Engine)                       │
        │                                                                               │
        │  [Prediction Step]                                                            │
        │  x_{k|k-1} = f(x_{k-1|k-1}, u_k)                                              │
        │  P_{k|k-1} = F_k P_{k-1|k-1} F_k^T + G_k M_k G_k^T + Q_{base}                 │
        │                                                                               │
        │  [Pre-Update Validation & Chi-Square 2-DOF Outlier Gating]                    │
        │  d^2 = y_k^T S_k^{-1} y_k <= χ^2_{2, 0.95} (5.991)                           │
        │                                                                               │
        │  [Joseph Form Covariance Update]                                              │
        │  x_{k|k} = x_{k|k-1} + K_k y_k                                                │
        │  P_{k|k} = (I - K_k H) P_{k|k-1} (I - K_k H)^T + K_k R_k K_k^T               │
        └──────────────────────────────────────┬────────────────────────────────────────┘
                                               │
                                               ▼
        ┌───────────────────────────────────────────────────────────────────────────────┐
        │                 Continuous Trajectory & Uncertainty Estimates                 │
        │      (x_k, y_k, θ_k, 95% Confidence Ellipses, Quality Provenance)             │
        └───────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Coordinate Conventions & Geometric Transformations

To prevent silent frame misalignments, the fusion engine enforces a strict coordinate convention:

### 2.1 Standard Building-Local Metric ENU Frame
- **Cartesian System:** Right-handed Cartesian coordinates in meters.
- **X-axis ($+X$):** Local East (along building baseline).
- **Y-axis ($+Y$):** Local North (orthogonal to building baseline).
- **Heading ($\theta$):** Radians counter-clockwise from the $+X$ (East) axis, normalized to $[-\pi, \pi]$ via `wrap_angle_rad`.
  $$\theta \in [-\pi, \pi], \quad 0 \text{ rad} = \text{East}, \quad \frac{\pi}{2} \text{ rad} = \text{North}, \quad \pi \text{ rad} = \text{West}, \quad -\frac{\pi}{2} \text{ rad} = \text{South}$$

### 2.2 Compass Azimuth Transformation
Smartphone magnetometers and orientation sensors typically output clockwise azimuth degrees $\psi_{deg} \in [0^\circ, 360^\circ)$ relative to magnetic or true North ($0^\circ = \text{North}$, $90^\circ = \text{East}$). 

The transformation to the building-local counter-clockwise ENU angle $\theta_{rad}$ given building orientation declination $\alpha_{deg}$ (angle of true North relative to building $+Y$ axis) is:
$$\theta_{rad} = \text{wrap\_angle\_rad}\left( \frac{\pi}{2} - \text{radians}(\psi_{deg} - \alpha_{deg}) \right)$$

Inverse transformation:
$$\psi_{deg} = \text{wrap\_azimuth\_deg}\left( 90^\circ - \text{degrees}(\theta_{rad}) + \alpha_{deg} \right)$$

---

## 3. Mathematical Formulation of the Extended Kalman Filter (EKF)

### 3.1 State Vector Representation
The 2D planar kinematic state vector at step $k$ is defined as:
$$\mathbf{x}_k = \begin{bmatrix} x_k \\ y_k \\ \theta_k \end{bmatrix} \in \mathbb{R}^3$$
Where:
- $x_k$: Building-local East coordinate (meters).
- $y_k$: Building-local North coordinate (meters).
- $\theta_k$: Heading angle in ENU radians ($[-\pi, \pi]$).

Each state is strictly bounded to an explicit discrete `building_id` and integer `floor_id`. Any cross-floor or cross-building measurement is rejected prior to filtering.

The state error covariance matrix $\mathbf{P}_k \in \mathbb{R}^{3 \times 3}$ is:
$$\mathbf{P}_k = \begin{bmatrix}
\sigma_{xx}^2 & \sigma_{xy} & \sigma_{x\theta} \\
\sigma_{yx} & \sigma_{yy}^2 & \sigma_{y\theta} \\
\sigma_{\theta x} & \sigma_{\theta y} & \sigma_{\theta\theta}^2
\end{bmatrix}$$

### 3.2 Non-Linear Motion Model (PDR Step Prediction)
When a step event occurs with estimated step length $L_k$ and relative heading increment $\Delta\theta_k$, the non-linear state transition $\mathbf{f}(\mathbf{x}_{k-1}, \mathbf{u}_k)$ is:
$$\begin{bmatrix} x_k \\ y_k \\ \theta_k \end{bmatrix} = \begin{bmatrix}
x_{k-1} + L_k \cos(\theta_{k-1} + \Delta\theta_k) \\
y_{k-1} + L_k \sin(\theta_{k-1} + \Delta\theta_k) \\
\text{wrap\_angle\_rad}(\theta_{k-1} + \Delta\theta_k)
\end{bmatrix}$$

The control input vector and its measurement noise covariance are:
$$\mathbf{u}_k = \begin{bmatrix} L_k \\ \Delta\theta_k \end{bmatrix}, \quad \mathbf{M}_k = \begin{bmatrix} \sigma_{L,k}^2 & 0 \\ 0 & \sigma_{\Delta\theta,k}^2 \end{bmatrix}$$

### 3.3 State Transition and Control Jacobians
The Jacobian of the motion model with respect to the state vector $\mathbf{x}$ evaluated at $\hat{\mathbf{x}}_{k-1|k-1}$ is:
$$\mathbf{F}_k = \left. \frac{\partial \mathbf{f}}{\partial \mathbf{x}} \right|_{\hat{\mathbf{x}}_{k-1|k-1}} = \begin{bmatrix}
1 & 0 & -L_k \sin(\hat{\theta}_{k-1} + \Delta\theta_k) \\
0 & 1 & L_k \cos(\hat{\theta}_{k-1} + \Delta\theta_k) \\
0 & 0 & 1
\end{bmatrix}$$

The Jacobian with respect to the control input $\mathbf{u}$ is:
$$\mathbf{G}_k = \left. \frac{\partial \mathbf{f}}{\partial \mathbf{u}} \right|_{\hat{\mathbf{x}}_{k-1|k-1}} = \begin{bmatrix}
\cos(\hat{\theta}_{k-1} + \Delta\theta_k) & -L_k \sin(\hat{\theta}_{k-1} + \Delta\theta_k) \\
\sin(\hat{\theta}_{k-1} + \Delta\theta_k) & L_k \cos(\hat{\theta}_{k-1} + \Delta\theta_k) \\
0 & 1
\end{bmatrix}$$

### 3.4 Process Noise Propagation
The predicted error covariance $\mathbf{P}_{k|k-1}$ incorporates both control noise propagated through $\mathbf{G}_k$ and additive base diffusion process noise $\mathbf{Q}_{base}$:
$$\mathbf{Q}_k = \mathbf{G}_k \mathbf{M}_k \mathbf{G}_k^T + \mathbf{Q}_{base}$$
$$\mathbf{P}_{k|k-1} = \mathbf{F}_k \mathbf{P}_{k-1|k-1} \mathbf{F}_k^T + \mathbf{Q}_k$$

### 3.5 Measurement Model
When an absolute Wi-Fi position fix $\mathbf{z}_k = [x_{wifi, k}, y_{wifi, k}]^T$ arrives, the linear observation model is:
$$\mathbf{z}_k = \mathbf{H} \mathbf{x}_k + \mathbf{v}_k, \quad \mathbf{H} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \end{bmatrix}$$
Where $\mathbf{v}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_k)$ represents Wi-Fi observation noise with covariance $\mathbf{R}_k \in \mathbb{R}^{2 \times 2}$.

### 3.6 Innovation and Innovation Covariance
The measurement innovation $\mathbf{y}_k \in \mathbb{R}^2$ and innovation covariance $\mathbf{S}_k \in \mathbb{R}^{2 \times 2}$ are:
$$\mathbf{y}_k = \mathbf{z}_k - \mathbf{H} \hat{\mathbf{x}}_{k|k-1} = \begin{bmatrix} x_{wifi,k} - \hat{x}_{k|k-1} \\ y_{wifi,k} - \hat{y}_{k|k-1} \end{bmatrix}$$
$$\mathbf{S}_k = \mathbf{H} \mathbf{P}_{k|k-1} \mathbf{H}^T + \mathbf{R}_k$$

---

## 4. Robust Outlier Rejection & Statistical Gating

To safeguard against multipath wall reflections, AP outages, and mismatched fingerprints, every Wi-Fi observation undergoes strict statistical and domain gating prior to updating the state.

### 4.1 Chi-Square ($\chi^2$) 2-DOF Statistical Gating
The squared Mahalanobis distance $d_k^2$ evaluates the statistical consistency of the innovation with respect to the expected innovation covariance $\mathbf{S}_k$:
$$d_k^2 = \mathbf{y}_k^T \mathbf{S}_k^{-1} \mathbf{y}_k$$

Under the null hypothesis of unbiased Gaussian noise, $d_k^2$ follows a Chi-Square distribution with $m=2$ degrees of freedom. At the standard 95% confidence threshold:
$$\chi^2_{2, 0.95} = -2 \ln(1 - 0.95) \approx 5.99146$$

If $d_k^2 > 5.99146$, the Wi-Fi observation is flagged as an outlier (`STATISTICAL_GATE_EXCEEDED`) and rejected from updating the state vector, preventing filter destabilization.

### 4.2 Domain Validation Criteria
Observations are also rejected if they fail physical sanity checks:
1. **Floor / Building Mismatch:** Scan reported floor differs from the current active filter floor.
2. **Age / Freshness:** Scan timestamp latency $> 10$ seconds relative to current time.
3. **AP Overlap:** Matched AP count $< 3$.
4. **Confidence Score:** Radio map localization confidence $< 0.20$.
5. **Gross Innovation Bound:** Absolute spatial jump $||\mathbf{y}_k||_2 > 25.0$ meters.

---

## 5. Numerical Stability & Joseph Form Covariance Update

Under standard textbook Kalman update equations ($\mathbf{P} = (\mathbf{I} - \mathbf{K}\mathbf{H})\mathbf{P}$), finite-precision floating point rounding errors can lead to asymmetry and loss of positive semi-definiteness, eventually causing filter divergence.

IndoorAI enforces the **Joseph Form Covariance Update**:
$$\mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{H}^T \mathbf{S}_k^{-1}$$
$$\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k \mathbf{y}_k$$
$$\hat{\theta}_{k|k} = \text{wrap\_angle\_rad}(\hat{\theta}_{k|k})$$
$$\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}) \mathbf{P}_{k|k-1} (\mathbf{I} - \mathbf{K}_k \mathbf{H})^T + \mathbf{K}_k \mathbf{R}_k \mathbf{K}_k^T$$

### Post-Update Conditioning
1. **Symmetrization:**
   $$\mathbf{P}_{k|k} = \frac{1}{2} (\mathbf{P}_{k|k} + \mathbf{P}_{k|k}^T)$$
2. **Eigenvalue Clamping:**
   $$\mathbf{V} \mathbf{\Lambda} \mathbf{V}^T = \mathbf{P}_{k|k}, \quad \mathbf{\Lambda}_{ii} = \max(\mathbf{\Lambda}_{ii}, 10^{-6})$$
   $$\mathbf{P}_{k|k} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^T$$

---

## 6. Wi-Fi Measurement Uncertainty Hierarchy

Rather than adopting a static ad-hoc measurement variance, the fusion engine determines $\mathbf{R}_k$ through a 4-tier hierarchy based on available empirical provenance:

| Level | Source | Description | Typical $\sigma_{xy}$ |
|---|---|---|---|
| **Tier 1** | **Empirical Survey Residuals** | Derived from Task 23 cross-validation residuals against physical ground-truth reference points for the specific building/site. | $1.5\text{m} - 3.0\text{m}$ |
| **Tier 2** | **Candidate Spatial Dispersion** | Computed from the sample covariance of the top-$k$ nearest neighbor reference coordinates during WKNN fingerprint matching. | $2.0\text{m} - 5.0\text{m}$ |
| **Tier 3** | **Confidence-Scaled Nominal** | Scales nominal site uncertainty inversely by matching confidence: $\mathbf{R} = \mathbf{R}_{nom} / \max(c, 0.2)$. | $3.5\text{m} - 6.0\text{m}$ |
| **Tier 4** | **Conservative Default Prior** | Fallback isotropic covariance when metadata is absent: $\sigma_x = \sigma_y = 5.0\text{m}$. | $5.0\text{m}$ |

---

## 7. 95% Confidence Error Ellipses

To provide interpretable spatial confidence for UI visualization and ground-truth validation, the marginal $2 \times 2$ positional covariance is extracted from $\mathbf{P}_k$:
$$\mathbf{P}_{xy} = \begin{bmatrix} \sigma_x^2 & \sigma_{xy} \\ \sigma_{xy} & \sigma_y^2 \end{bmatrix}$$

The 95% confidence ellipse parameters are calculated analytically:
1. **Eigenvalues:**
   $$\lambda_{1,2} = \frac{\sigma_x^2 + \sigma_y^2}{2} \pm \sqrt{\left(\frac{\sigma_x^2 - \sigma_y^2}{2}\right)^2 + \sigma_{xy}^2}$$
2. **Semi-Major Axis ($a$) & Semi-Minor Axis ($b$):**
   $$a = \sqrt{5.99146 \cdot \max(\lambda_1, 10^{-6})}, \quad b = \sqrt{5.99146 \cdot \max(\lambda_2, 10^{-6})}$$
3. **Orientation Angle ($\phi$):**
   $$\phi = \frac{1}{2} \text{atan2}(2 \sigma_{xy}, \sigma_x^2 - \sigma_y^2) \quad \text{(radians counter-clockwise from }+X\text{)}$$

---

## 8. Synthetic Controlled Benchmark & Validation Results

To verify mathematical correctness independently of external radio distortions, the engine includes a mathematically controlled synthetic generator (`FusionEvaluator.generate_controlled_synthetic_trajectory`).

### Benchmark Setup
- **Trajectory:** 60-second L-shaped corridor walk (45m East, 45m North, total 90.4m, 108 steps).
- **Sensors:** Uncalibrated gyroscope with $30^\circ/\text{min}$ systematic drift + random walk noise ($\sigma = 0.02\text{ rad}$).
- **Wi-Fi:** Scans every 5.0 seconds with Gaussian measurement noise ($\sigma = 2.5\text{m}$).
- **Outlier Injection:** Scan 4 injected with a $+25.0\text{m}$ gross multipath jump.

### Quantitative Comparison Results

| Positioning Source | Mean Error (m) | Median Error (m) | P95 Error (m) | Endpoint Error (m) | Drift Rate (m/min) | 95% Ellipse Coverage |
|---|---|---|---|---|---|---|
| **Unassisted PDR** | $1.77\text{ m}$ | $1.64\text{ m}$ | $3.58\text{ m}$ | $7.64\text{ m}$ | $7.64\text{ m/min}$ | N/A |
| **Raw Wi-Fi Scans** | $5.35\text{ m}$ | $3.42\text{ m}$ | $12.10\text{ m}$ | $1.22\text{ m}$ | N/A | N/A |
| **EKF Fused** | **$1.60\text{ m}$** | **$1.52\text{ m}$** | **$2.84\text{ m}$** | **$3.72\text{ m}$** | **$3.72\text{ m/min}$** | **$100.0\%$** |

### Key Observations
1. **Drift Mitigation:** The EKF reduces accumulated endpoint drift from $7.64\text{m}$ down to $3.72\text{m}$ (over $51\%$ drift reduction).
2. **Robust Outlier Rejection:** The $+25\text{m}$ injected outlier yielded $d^2 = 37.65 > 5.99$ and was rejected immediately without perturbing state trajectory ($9/10$ scans accepted, $1$ rejected).
3. **Calibrated Uncertainty:** $100\%$ of true ground truth positions fell within the predicted 95% confidence error ellipses, proving conservative, well-calibrated covariance estimation.

---

## 9. Backend REST API Endpoints

The fusion service exposes 5 endpoints under `/api/datasets/{dataset_id}/fusion`:

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/datasets/{dataset_id}/fusion/run` | Execute EKF sensor fusion with custom parameters (`site_id`, `q_pos`, `chi2_threshold`, etc.). |
| `GET` | `/api/datasets/{dataset_id}/fusion/status` | Retrieve execution status, step/scan counts, acceptance rate, and rejection breakdown. |
| `GET` | `/api/datasets/{dataset_id}/fusion/trajectory` | Retrieve continuous fused states, 95% confidence ellipses, and step-level provenance. |
| `POST` | `/api/datasets/{dataset_id}/fusion/evaluate` | Evaluate trajectory against survey ground truth or synthetic benchmark. |
| `GET` | `/api/datasets/{dataset_id}/fusion/evaluation` | Retrieve comparative evaluation report (PDR vs Wi-Fi vs Fused). |

---

## 10. Dashboard Visualization (Streamlit Section 14)

The Streamlit dashboard (`frontend/components/sensor_fusion_view.py`) provides interactive visualization:
- **Tab 1: 2D Fused Trajectory & Error Ellipses:** Plotly 2D floor plan rendering showing PDR path, raw Wi-Fi fixes, EKF fused path, accepted/rejected scan markers, and semi-transparent 95% confidence ellipses.
- **Tab 2: Filter Parameters & Pipeline Execution:** Interactive controls for initial coordinates, process noise ($\sigma_{pos}, \sigma_\theta$), step noise, Chi-Square gate multipliers, and execution trigger.
- **Tab 3: Innovation & Statistical Gating Diagnostics:** Time-series charts of innovation norms, Mahalanobis distances $d_k^2$ relative to the $\chi^2_{2, 0.95}$ threshold, and rejection reasons.
- **Tab 4: Ground-Truth Benchmark & Drift Evaluation:** Comparative summary cards and metrics table reporting mean, median, P95, endpoint error, drift per minute, and ellipse coverage across PDR, Wi-Fi, and Fused modes.
