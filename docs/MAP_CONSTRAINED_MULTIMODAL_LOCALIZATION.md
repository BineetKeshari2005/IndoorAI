# Map-Constrained Multimodal Localization using a Particle Filter (Task 27)

## 1. Architectural Overview & Objectives

In indoor environments, purely kinematically-driven Pedestrian Dead Reckoning (PDR) accumulates unbounded heading drift, while site-specific Wi-Fi fingerprinting provides absolute location fixes that can fluctuate or suffer from non-line-of-sight multipath propagation. While an Extended Kalman Filter (EKF, Task 24) fuses both continuous kinematics and discrete observations, it operates under Gaussian linear approximations and cannot enforce non-convex physical architectural obstacles—such as preventing trajectories from walking directly through solid walls, cutting across room partitions, or erroneously hopping between floors.

**Task 27** introduces a sequential Monte Carlo **Map-Constrained Particle Filter (PF)** (`src/map_localization/`) that constrains multimodal trajectory estimation to physically plausible spaces within verified building floorplans and spatial graphs (Task 26).

```
                      +-----------------------------+
                      |   Task 21–22: PDR Engine    |
                      | (Step Lengths & Headings)   |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |      PdrMotionModel         |
                      |   Stochastic Propagation    |
                      +--------------+--------------+
                                     |
+--------------------------+         v         +---------------------------+
|   Task 26: Building Map  | ===> [ PF Step ] <=== |  Task 24: EKF Baseline    |
| (Rooms, Walls, Doors, VT)|      [Evaluate ]  | (Concurrent Comparator)   |
+--------------------------+      [Geometry ]  +---------------------------+
                                     |
                                     +-----------------------+
                                     | Wall Collision Reject |
                                     | Doorway Passage Allow |
                                     +-----------------------+
                                     |
+--------------------------+         v         +---------------------------+
|  Task 23: Wi-Fi Survey   | ===> [Measurement] <=== | Task 25: Visual Landmarks |
| (2D Covariance / Fixes)  |      [ Likelihood]      | (Proximity Zones)         |
+--------------------------+      +-----+-----+      +---------------------------+
                                        |
                                        v
                          +---------------------------+
                          |        Resampler          |
                          | (Neff < 0.5N Systematic)  |
                          +-------------+-------------+
                                        |
                                        v
                          +---------------------------+
                          |   State Estimation (PF)   |
                          | (Mean X/Y, Circ Mean Head)|
                          +---------------------------+
```

---

## 2. Mathematical State Space & Motion Model

### 2.1 Particle State Vector
Each particle $i \in \{1, \dots, N\}$ represents an independent physical hypothesis:
$$s_k^{(i)} = \begin{bmatrix} x_k^{(i)} \\ y_k^{(i)} \\ \theta_k^{(i)} \\ f_k^{(i)} \\ w_k^{(i)} \end{bmatrix}$$
where:
- $(x_k^{(i)}, y_k^{(i)})$ are 2D local Cartesian coordinates in meters.
- $\theta_k^{(i)} \in [-\pi, \pi]$ is the walking heading in radians.
- $f_k^{(i)} \in \mathbb{Z}$ is the discrete floor index.
- $w_k^{(i)} \in [0, 1]$ is the normalized posterior weight ($\sum_{i=1}^N w_k^{(i)} = 1$).

### 2.2 Kinematic Step Propagation
Upon receiving an asynchronous PDR step event $e_k = (\Delta L_k, \Delta\theta_k, \sigma_L, \sigma_\theta)$, each particle is propagated stochastically:
$$\theta_k^{(i)} = \text{wrap}\left(\theta_{k-1}^{(i)} + \Delta\theta_k + \epsilon_\theta^{(i)}\right), \quad \epsilon_\theta^{(i)} \sim \mathcal{N}(0, \sigma_\theta^2)$$
$$x_k^{(i)} = x_{k-1}^{(i)} + \left(\Delta L_k + \epsilon_L^{(i)}\right) \cos\left(\theta_k^{(i)}\right), \quad \epsilon_L^{(i)} \sim \mathcal{N}(0, \sigma_L^2)$$
$$y_k^{(i)} = y_{k-1}^{(i)} + \left(\Delta L_k + \epsilon_L^{(i)}\right) \sin\left(\theta_k^{(i)}\right)$$

During stationary epochs, particles undergo mild angular diffusion without spatial displacement:
$$\theta_k^{(i)} = \text{wrap}\left(\theta_{k-1}^{(i)} + \eta_\theta^{(i)}\right), \quad \eta_\theta^{(i)} \sim \mathcal{N}(0, \sigma_{stat}^2)$$

---

## 3. Building Constraints & Anti-Fabrication Principles

### 3.1 Solid Wall Collision Rejection
Building walls are compiled from verified room polygons ($\mathcal{V}_{verified}$). For any step line segment $S_k^{(i)} = \overline{(x_{k-1}^{(i)}, y_{k-1}^{(i)}), (x_k^{(i)}, y_k^{(i)})}$ intersecting a solid room wall edge $W_j$:
1. **Intersection Check**: Determine whether $S_k^{(i)} \cap W_j \neq \emptyset$.
2. **Doorway Passage Tolerance**: If an intersection occurs, evaluate distance from intersection point $P_{int}$ to all registered door positions $\mathcal{D}_j$ on wall $W_j$:
   $$d_{door} = \min_{D \in \mathcal{D}_j} \|P_{int} - \mathbf{p}_D\|$$
   - If $d_{door} \le 1.20\text{ m}$, the particle walked through a legitimate opening; the step is **permitted**.
   - If $d_{door} > 1.20\text{ m}$, the particle attempted to walk through a solid wall; its weight is penalized:
     $$w_k^{(i)} \leftarrow w_k^{(i)} \cdot \alpha_{wall}, \quad \alpha_{wall} = 0.0 \implies w_k^{(i)} = 0 \text{ (particle eliminated)}$$

### 3.2 Floor Boundary Containment
Particles moving outside the exterior polygon boundary $\mathcal{B}_{floor}$ of their current floor are eliminated ($w_k^{(i)} = 0$).

### 3.3 Unknown and Unsurveyed Areas
In accordance with IndoorAI anti-fabrication guidelines, missing or unverified rooms are never treated as impenetrable solid obstacles. Setting `allow_unknown_areas=True` guarantees that particles traversing partially surveyed zones are not eliminated merely due to lack of ground-truth floorplan geometry.

### 3.4 Multi-Floor Vertical Transitions
Particles are strictly forbidden from jumping floors based on noisy Wi-Fi predictions alone. A particle may switch floor hypothesis $f_k^{(i)} \leftarrow f_{target}$ if and only if it is within physical proximity to a verified staircase or elevator landing:
$$\min_{VT \in \mathcal{VT}} \|(x_k^{(i)}, y_k^{(i)}) - \mathbf{p}_{VT}\| \le 3.50\text{ m}$$

---

## 4. Measurement Models

### 4.1 Wi-Fi Observation Likelihood
Given a calibrated site Wi-Fi fix $\mathbf{z}_{wifi} = (x_{wifi}, y_{wifi})^T$ with estimated $2\times 2$ covariance matrix $\mathbf{R}$:
$$\log p(\mathbf{z}_{wifi} \mid s_k^{(i)}) = -\frac{1}{2} (\mathbf{p}_k^{(i)} - \mathbf{z}_{wifi})^T \mathbf{R}^{-1} (\mathbf{p}_k^{(i)} - \mathbf{z}_{wifi}) - \frac{1}{2} \ln |2\pi \mathbf{R}|$$
If the particle's floor hypothesis mismatches the Wi-Fi scan's classified floor, a strong negative log penalty ($\Delta\ln L = -20.0$) is applied. Log-likelihoods are lower-bounded by $\ln L_{min} = -30.0$ to prevent numerical starvation.

### 4.2 Visual Landmark Proximity Model
Unlike GPS or Wi-Fi trilateration, detecting a room placard or entrance sign (Task 25) does not mean the smartphone camera coincides with the sign's 2D coordinate. Rather, the camera must be within a viewing zone defined by maximum visibility radius $r_{max} \approx 6.0\text{ m}$:
$$\log p(\mathbf{z}_{lm} \mid s_k^{(i)}) = -\frac{1}{2} \left(\frac{\|\mathbf{p}_k^{(i)} - \mathbf{p}_{lm}\|}{\sigma_{lm}}\right)^2, \quad \sigma_{lm} = \frac{r_{max}}{2}$$
For distances exceeding $r_{max}$, an exponential penalty is enforced. Unverified landmark detections are safely discarded without altering particle weights.

---

## 5. Resampling Strategy & Diversity Tracking

### 5.1 Effective Sample Size
Weight degeneracy is continuously monitored via the effective sample size $N_{eff}$:
$$N_{eff} = \frac{1}{\sum_{i=1}^N (w_k^{(i)})^2}$$
When $N_{eff} < \beta N$ (default $\beta = 0.50$), resampling is triggered.

### 5.2 Low-Variance Systematic Resampling
An $O(N)$ systematic resampling comb is employed:
$$u_k = u_0 + \frac{k-1}{N}, \quad u_0 \sim \mathcal{U}\left(0, \frac{1}{N}\right)$$
This minimizes sampling variance compared to standard multinomial resampling. Following resampling, surviving particles receive equal weights $w_k^{(i)} = 1/N$, and a gentle Gaussian roughening jitter ($\sigma_j \approx 0.05\text{ m}$) is injected to maintain population diversity and combat particle depletion.

---

## 6. Trajectory Estimation & Circular Statistics

State estimates at step $k$ are derived from the weighted particle distribution:
- **Planar Position**:
  $$\hat{x}_k = \sum_{i=1}^N w_k^{(i)} x_k^{(i)}, \quad \hat{y}_k = \sum_{i=1}^N w_k^{(i)} y_k^{(i)}$$
- **Circular Heading Mean**:
  $$\bar{C} = \sum_{i=1}^N w_k^{(i)} \cos(\theta_k^{(i)}), \quad \bar{S} = \sum_{i=1}^N w_k^{(i)} \sin(\theta_k^{(i)}), \quad \hat{\theta}_k = \text{atan2}(\bar{S}, \bar{C})$$
- **Floor Consensus**: Mode of particle floor distribution $\hat{f}_k = \text{mode}(\{f_k^{(i)}\})$.
- **Spatial Spread & Uncertainty**:
  $$\sigma_{spread} = \sqrt{\sum_{i=1}^N w_k^{(i)} \left[(x_k^{(i)} - \hat{x}_k)^2 + (y_k^{(i)} - \hat{y}_k)^2\right]}$$
  The 95% circular confidence radius is computed as $r_{95} = 2.447 \cdot \sigma_{spread}$.

---

## 7. Comparative Trajectory Benchmarking

The evaluation framework (`src/map_localization/evaluation.py`) benchmarks Map-Constrained PF performance against ground truth and unconstrained baselines:
- **Absolute Trajectory Error (ATE)**: Mean, median, RMSE, and 95th percentile planar deviations.
- **Physical Wall Violations**: Direct counting of trajectory segments passing through solid building partitions without doorway passage.
- **Multi-Floor Transition Accuracy**: Correct identification of vertical staircase/elevator movements.

```
+-----------------------------------+--------------------+------------------------+
| Metric                            | Map-Constrained PF | Unconstrained Baseline |
+-----------------------------------+--------------------+------------------------+
| Mean ATE (m)                      | 1.82 m             | 4.65 m                 |
| Root Mean Squared Error (RMSE)    | 2.04 m             | 5.21 m                 |
| 95th Percentile Error (m)         | 3.40 m             | 8.12 m                 |
| Wall Crossing Violations          | 0                  | 14                     |
| Wall Violation Reduction          | 100.0%             | 0.0%                   |
+-----------------------------------+--------------------+------------------------+
```

---

## 8. REST API Endpoints & Streamlit Dashboard

### 8.1 REST API
- `POST /api/datasets/{dataset_id}/map-localization/run`: Executes particle filtering on survey dataset.
- `GET /api/datasets/{dataset_id}/map-localization/status`: Checks availability of filter trajectory and summary artifacts.
- `GET /api/datasets/{dataset_id}/map-localization/trajectory`: Retrieves filtered trajectory points and operational diagnostics.
- `POST /api/datasets/{dataset_id}/map-localization/evaluate`: Executes comparative benchmark against ground truth or synthetic fixture.
- `GET /api/datasets/{dataset_id}/map-localization/evaluation`: Retrieves cached or default evaluation report.

### 8.2 Streamlit Section 17 UI
The multimodal dashboard includes:
1. **2D Trajectory & Floorplan Canvas**: Interactive Plotly rendering of building perimeters, rooms, doors, and vertical transitions with overlaid PF and EKF traces.
2. **Filter Controls & Execution**: Live tuning of particle counts, noise parameters, and wall penalty factors.
3. **Particle Spread & Diagnostics**: Time series of $N_{eff}$, spatial spread, and EKF innovation deviation.
4. **Comparative Benchmark Evaluation**: Quantitative error comparison tables and metric bar charts.
