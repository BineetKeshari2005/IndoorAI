# IndoorAI — Task 28: Indoor Navigation & Intelligent Route Planning

## Overview

Task 28 establishes a production-grade, map-constrained **Indoor Navigation and Intelligent Route Planning Engine** for the IndoorAI platform. Built on top of the verified spatial representations established in Task 26 and the particle-filter localization engine from Task 27, this module generates topologically valid, multi-floor paths, computes pedestrian turn-by-turn guidance, continuously tracks trip progress, and dynamically recalculates routes when deviations occur.

---

## Architectural Principles & Anti-Fabrication Safeguards

1. **Topological Grounding**: Routes are planned strictly over nodes and edges derived from verified indoor entities (`Room`, `Corridor`, `Door`, `VerticalTransition`) in `BuildingMap`. No paths or transit connections are hallucinated through unverified or unknown building areas.
2. **Solid Wall Collision Avoidance**: Every planned line segment is audited against room boundary polygons. If a candidate path crosses a room boundary away from a verified doorway portal (tolerance: $\le 1.2\text{m}$), it is immediately rejected with `WALL_COLLISION_DETECTED`.
3. **Accessibility Conservatism**: When accessibility data is missing or incomplete, the system designates accessibility as `ACCESSIBILITY_UNKNOWN` rather than falsely claiming compliance. Staircases are audited and penalized or strictly forbidden under `RoutePreference.WHEELCHAIR_ACCESSIBLE`.
4. **Localization Reliability Awareness**: During transient signal dropouts (`LOW` confidence or `LOST` tracking status), the guidance directives become conservative, and dynamic rerouting is suppressed to prevent flapping.
5. **Debounced Dynamic Rerouting**: Real-time cross-track tracking detects deviations exceeding $4.0\text{m}$ or sudden floor mismatches. To prevent oscillatory reroutes, a 5-second cooldown debounce window is enforced.

---

## Core Navigation Architecture

```
                                    +------------------------------+
                                    |         BuildingMap          |
                                    | (Rooms, Doors, Corridors, VT)|
                                    +--------------+---------------+
                                                   |
                         +-------------------------+-------------------------+
                         |                                                   |
                         v                                                   v
           +---------------------------+                       +---------------------------+
           |    DestinationManager     |                       |        RouteGraph         |
           | - Verified Room Targets   |                       | - Cost Function C(D,T,S,A)|
           | - Portals & Exits         |                       | - Admissible Multi-Floor h|
           | - Vertical Transitions    |                       | - Dynamic Edge/Node Blocks|
           +-------------+-------------+                       +-------------+-------------+
                         |                                                   |
                         +-------------------------+-------------------------+
                                                   |
                                                   v
                                     +---------------------------+
                                     |       RoutePlanner        |
                                     | - Dijkstra (Exact Optimal)|
                                     | - A* (Heuristic Guided)   |
                                     | - Coordinate Snapping     |
                                     +-------------+-------------+
                                                   |
                                                   v
                                     +---------------------------+
                                     |      RouteValidator       |
                                     | - Wall Collision Check    |
                                     | - Continuity & Map Version|
                                     | - Accessibility Audit     |
                                     +-------------+-------------+
                                                   |
                                                   v
                                     +---------------------------+
                                     |   InstructionGenerator    |
                                     | - TurnDetector (Pedestrian|
                                     |   Cartesian convention)   |
                                     | - Turn-by-Turn Directives |
                                     +-------------+-------------+
                                                   |
                                                   v
                                     +---------------------------+
                                     | NavigationSessionManager  |
                                     | - Real-time Progress (1.5m|
                                     |   arrival detection)      |
                                     | - ReroutingEngine (4.0m   |
                                     |   threshold, 5s cooldown) |
                                     | - JSON Storage/Lifecycle  |
                                     +---------------------------+
```

---

## Mathematical Formulation

### 1. Composite Edge Cost Function
Edge traversal cost $C(e)$ models distance, walking velocity, physical stairs exertion, and accessibility barriers:

$$C(e) = w_d \cdot D(e) + w_t \cdot T(e) + w_s \cdot S(e) + w_a \cdot A(e)$$

Where:
- $D(e) = \text{distance in meters}$
- $T(e) = \frac{D(e)}{v_{\text{walk}}} + \tau_{\text{vertical}}$ (with normal walking speed $v_{\text{walk}} = 1.2\text{ m/s}$ and nominal elevator wait $\tau = 25.0\text{s}$)
- $S(e) = \text{staircase penalty weight}$
- $A(e) = \text{accessibility barrier penalty}$

### 2. Multi-Floor Admissible Heuristic
For $A^*$ graph search, heuristic $h(u, v)$ provides a conservative lower bound on the remaining path cost:

$$h(u, v) = \alpha \cdot \sqrt{(x_u - x_v)^2 + (y_u - y_v)^2} + \beta \cdot |f_u - f_v|$$

Because topological edges may have compressed representation relative to coordinate Euclidean distances, scaling factor $\alpha = 0.01$ and $\beta = 0.5$ ensure that $h(u, v) \le h^*(u, v)$ across all graph topologies, preserving strict admissibility.

### 3. Pedestrian Turn Classification
In local East-North-Up (ENU) coordinates ($+X$ = East, $+Y$ = North), angles are measured counter-clockwise from East ($0^\circ$). In pedestrian navigation, clockwise heading changes correspond to Right turns, while counter-clockwise changes correspond to Left turns.

Given incoming bearing $\theta_1$ and outgoing bearing $\theta_2$:
$$\Delta\theta_{\text{raw}} = \theta_2 - \theta_1$$
$$\Delta\theta = ((\Delta\theta_{\text{raw}} + 180^\circ) \pmod{360^\circ}) - 180^\circ$$
$$\Delta\theta_{\text{turn}} = -\Delta\theta$$

Turn classification thresholds:
- $|\Delta\theta_{\text{turn}}| \le 20^\circ$: `CONTINUE_STRAIGHT`
- $20^\circ < \Delta\theta_{\text{turn}} \le 45^\circ$: `SLIGHT_RIGHT`
- $-45^\circ \le \Delta\theta_{\text{turn}} < -20^\circ$: `SLIGHT_LEFT`
- $45^\circ < \Delta\theta_{\text{turn}} \le 135^\circ$: `TURN_RIGHT`
- $-135^\circ \le \Delta\theta_{\text{turn}} < -45^\circ$: `TURN_LEFT`
- $135^\circ < \Delta\theta_{\text{turn}} \le 160^\circ$: `SHARP_RIGHT`
- $-160^\circ \le \Delta\theta_{\text{turn}} < -135^\circ$: `SHARP_LEFT`
- $|\Delta\theta_{\text{turn}}| > 160^\circ$: `UTURN`

### 4. Cross-Track Perpendicular Error
For pedestrian position $P(x, y)$ along active path segment between waypoints $A$ and $B$:

$$d_{\perp} = \frac{|(B_y - A_y)x - (B_x - A_x)y + B_x A_y - B_y A_x|}{\sqrt{(B_y - A_y)^2 + (B_x - A_x)^2}}$$

If the pedestrian's projection falls outside the segment segment endpoints $[A, B]$, the distance defaults to the nearest endpoint:
$$d_{\perp} = \min(\|P - A\|, \|P - B\|)$$

---

## REST API Specification

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/datasets/{id}/navigation/destinations` | List or query verified indoor destinations with floor and category filters |
| `POST` | `/api/datasets/{id}/navigation/routes` | Compute Dijkstra or A* indoor route with wall audit and instructions |
| `GET` | `/api/datasets/{id}/navigation/routes/{route_id}` | Retrieve cached route by ID |
| `POST` | `/api/datasets/{id}/navigation/sessions` | Initialize active navigation session |
| `GET` | `/api/datasets/{id}/navigation/sessions/{session_id}` | Retrieve active session state and progress history |
| `POST` | `/api/datasets/{id}/navigation/sessions/{session_id}/progress` | Ingest real-time localization fix, advance waypoints, trigger debounced rerouting |
| `POST` | `/api/datasets/{id}/navigation/sessions/{session_id}/complete` | Finalize active navigation session |
| `POST` | `/api/datasets/{id}/navigation/evaluate` | Run comparative Dijkstra vs. A* benchmark and simulated trips |
| `GET` | `/api/datasets/{id}/navigation/evaluation` | Fetch default synthetic multi-floor evaluation report |

---

## Streamlit UI Integration (Section 18)

Located at `frontend/components/navigation_view.py` and registered under `"18. Indoor Navigation & Route Planning"`:
1. **Destination Directory**: Filter destinations by building, floor, category, and free-text search.
2. **Route Planning & Algorithms**: Compare Dijkstra vs. A*, configure preferences (`SHORTEST`, `WHEELCHAIR_ACCESSIBLE`, `AVOID_STAIRS`, `FEWEST_TURNS`), and inspect route summaries.
3. **2D Interactive Canvas**: Plotly overlay of room footprints, doors, vertical transitions, route polyline, waypoints, and user pose.
4. **Turn-by-Turn Guidance & Simulation**: Card sequence of guidance instructions with active step highlight, live trip stepper, and simulated +6m deviation test button.
5. **Rerouting Diagnostics & Evaluation Benchmark**: Empirical verification of admissibility ($h(u, v) \le h^*(u, v)$), node expansion speedup, and simulated reroute resilience.
