# IndoorAI — Android IMU & Magnetometer Streamer Specification

## 1. Executive Summary & Physics Foundations

Task 15 implements **real-time, user-controlled inertial and magnetic sensor collection** within the IndoorAI Android application. The module streams raw and filtered signals from the smartphone's:
1. **Tri-Axial Accelerometer (`TYPE_ACCELEROMETER`)**: Linear dynamic motion + gravity vector in $\text{m/s}^2$.
2. **Tri-Axial Gyroscope (`TYPE_GYROSCOPE`)**: Angular velocity in $\text{rad/s}$.
3. **Tri-Axial Magnetometer (`TYPE_MAGNETIC_FIELD`)**: Ambient geomagnetic field flux density in microteslas ($\mu\text{T}$).
4. **Sensor Fusion Foundation (`SensorFusionManager`)**: Device orientation Euler angles (Yaw/Azimuth, Pitch, Roll) and low-pass filtered gravity estimation.

```
                           +-----------------------------------------------+
                           |          SensorEventListener (Hardware)       |
                           |   (Accelerometer, Gyroscope, Magnetometer)    |
                           +-----------------------------------------------+
                                                  | Raw events: (x, y, z, ts_ns)
                                                  v
                           +-----------------------------------------------+
                           |             Physics & Signal Filters          |
                           |   - LowPassGravityFilter: g[t] = α g + (1-α)a |
                           |   - Dynamic Linear Accel: a_lin = a - g       |
                           |   - SensorFusionManager: Cross-product Triad  |
                           +-----------------------------------------------+
                                                  |
                                                  v
                           +-----------------------------------------------+
                           |               SensorProvider                  |
                           |  (AndroidSensorProvider vs MockSensorProvider)|
                           +-----------------------------------------------+
                                                  |
                                                  v
                           +-----------------------------------------------+
                           |      StateFlow<ImuSample> & ViewModel         |
                           |   - Sampling Profiles: UI, Game, Fastest      |
                           |   - Session Integration: SES-XXXXX            |
                           |   - Local Export: RFC-4180 CSV & JSON         |
                           +-----------------------------------------------+
```

---

## 2. Supported Sensor Types, Units & Physical Semantics

| Sensor Type | Android Type Constant | Measurement Unit | Physical Interpretation & Invariant |
| :--- | :--- | :--- | :--- |
| **Accelerometer** | `Sensor.TYPE_ACCELEROMETER` | $\text{m/s}^2$ | Measures proper acceleration applied to the device across X (lateral), Y (longitudinal), and Z (vertical). When at rest on a horizontal surface, $Z \approx +9.81\,\text{m/s}^2$. |
| **Gyroscope** | `Sensor.TYPE_GYROSCOPE` | $\text{rad/s}$ | Measures angular rate of rotation around X (pitch rate), Y (roll rate), and Z (yaw rate). Positive values indicate counter-clockwise rotation according to the right-hand rule. |
| **Magnetometer** | `Sensor.TYPE_MAGNETIC_FIELD` | $\mu\text{T}$ (microteslas) | Measures total ambient geomagnetic flux density across X, Y, and Z axes. The nominal Earth geomagnetic field magnitude ranges from $25\,\mu\text{T}$ to $65\,\mu\text{T}$. |
| **Rotation Vector** | `Sensor.TYPE_ROTATION_VECTOR` | Unitless (Quaternions) | Hardware or software fused device orientation relative to the East-North-Up (ENU) coordinate frame. |

### Critical Anti-Drift Physics Invariants
> [!WARNING]
> **Strict Scientific Invariant:**
> 1. **No IMU-Only Position Integration:** Integrating raw acceleration twice ($\iint a \, dt^2$) leads to **quadratic error drift $O(t^2)$**, causing unbounded position divergence within seconds.
> 2. **Gyroscope Bias Drift:** Integrating raw angular velocity ($\int \omega \, dt$) suffers from continuous bias drift and lacks an absolute heading reference.
> 3. **Indoor Magnetic Distortion:** Indoor steel structural columns, reinforced concrete rebar, electrical conduits, and computer equipment produce strong **hard-iron and soft-iron distortions**. Raw magnetometer readings cannot be treated as calibrated indoor headings without spatial mapping.
> 4. **Future PDR Roadmap:** Accurate Dead Reckoning requires step detection algorithms, step-length estimation models, and multi-sensor fusion (Task 16+), never raw double integration.

---

## 3. Sampling Rates & Battery Conservation

The application exposes three configurable sampling profiles:

| Sampling Profile | Android Delay Constant | Microsecond Delay | Nominal Frequency | Primary Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **UI (Default)** | `SENSOR_DELAY_UI` | $\approx 60,000\,\mu\text{s}$ | $\approx 16\,\text{Hz}$ | Battery-efficient visual monitoring and coarse motion detection. |
| **Game** | `SENSOR_DELAY_GAME` | $\approx 20,000\,\mu\text{s}$ | $\approx 50\,\text{Hz}$ | Real-time Pedestrian Dead Reckoning (PDR) step counting and gait cadence analysis. |
| **Fastest** | `SENSOR_DELAY_FASTEST` | $0\,\mu\text{s}$ | Hardware max ($\ge 100\,\text{Hz}$) | Short-duration high-resolution impact or vibration analysis. High battery consumption. |

### Battery Conservation Invariants
1. **Zero Background Collection:** Listeners are registered strictly in foreground screens or active user-controlled recording sessions. In `MainActivity.onStop()`, all sensor listeners are immediately unregistered.
2. **Actual Hardware Timestamps:** The application records the actual `event.timestamp` (nanoseconds since boot) generated by the hardware FIFO rather than assuming a synthetic fixed-rate interval.

---

## 4. Signal Processing & Sensor Fusion Foundation

### Low-Pass Gravity Separation Filter
To isolate dynamic body acceleration ($a_{\text{linear}}$) from static Earth gravity ($g$):

$$g_t = \alpha \cdot g_{t-1} + (1 - \alpha) \cdot a_t$$

$$a_{\text{linear},t} = a_t - g_t$$

- **Smoothing Factor ($\alpha$):** $0.80$ (balances rapid response with steady-state gravity rejection).
- **Initialization:** The first incoming acceleration sample initializes $g_0 = a_0$.
- **Reset:** Calling `gravityFilter.reset()` re-initializes on new capture sessions.

### Orientation Estimation (Triad Vector Fusion)
The `SensorFusionManager` computes device orientation using the normalized cross-product of gravity and geomagnetic vectors:

1. **East Vector ($H = E \times G$):** $H = \text{Geomagnetic} \times \text{Gravity}$, normalized to unit length.
2. **North Vector ($M = G \times H$):** $M = \text{Gravity} \times H$, normalized to unit length.
3. **Euler Angles:**
   - **Yaw / Azimuth ($\psi$):** $\psi = \text{atan2}(H_x, M_x)$ in degrees.
   - **Pitch ($\theta$):** $\theta = \text{atan2}(-M_y, \sqrt{M_x^2 + G_y^2 + G_z^2})$ in degrees.
   - **Roll ($\phi$):** $\phi = \text{atan2}(-G_y, G_z)$ in degrees.
   - **Compass Heading:** Normalized to $0^\circ \text{ to } 360^\circ$.

---

## 5. Local Storage & Telemetry Export Specifications

### CSV Export Format (RFC 4180 Compliant)
```csv
session_id,sensor_type,x,y,z,magnitude,timestamp_ns,accuracy,sampling_rate,is_mock
```

#### Example CSV Output
```csv
session_id,sensor_type,x,y,z,magnitude,timestamp_ns,accuracy,sampling_rate,is_mock
SES-001,ACCELEROMETER,0.052,0.124,9.805,9.806,123456789123000,HIGH,UI (~16 Hz),false
SES-001,GYROSCOPE,0.001,-0.002,0.000,0.002,123456789123000,HIGH,UI (~16 Hz),false
SES-001,MAGNETIC_FIELD,22.400,-5.800,41.200,47.228,123456789123000,HIGH,UI (~16 Hz),false
```

### JSON Export Format
```json
{
  "sessionId": "SES-001",
  "readings": [
    {
      "sensorType": "ACCELEROMETER",
      "x": 0.052,
      "y": 0.124,
      "z": 9.805,
      "timestampNs": 123456789123000,
      "accuracy": "HIGH",
      "sessionId": "SES-001",
      "samplingRate": "UI (~16 Hz)",
      "isMock": false
    }
  ],
  "samplingRate": "UI (~16 Hz)",
  "isMock": false
}
```

---

## 6. Physical Android Device Testing Guide

To test IMU and Magnetometer streaming on a physical Android handset:
1. **Deploy to Device:** Connect the physical phone via USB and run the application from Android Studio.
2. **Navigate to Screen D (Sensors):**
   - Verify that the header displays `HARDWARE ACTIVE`.
   - Inspect the **Hardware Availability** badges: Accelerometer, Gyroscope, and Magnetometer should reflect the phone's physical hardware.
3. **Test Sampling:**
   - Tap **"Start"** to initiate streaming.
   - Verify that Accelerometer X/Y/Z readings react in real-time to handheld rotation, with total magnitude staying close to $9.81\,\text{m/s}^2$ when resting.
   - Tilt the device forward/backward and confirm that the **Orientation Card** updates Pitch, Roll, and Compass Heading.
4. **Test Sampling Rate Switch:**
   - Tap the **Sampling Rate Profile** dropdown and select `Game (~50 Hz)`.
   - Confirm that the refresh frequency increases noticeably.
5. **Test Session Recording & Export:**
   - Record a 10-second motion sample.
   - Tap **"Export CSV"** and **"Export JSON"** to inspect the buffered records, nanosecond timestamps, and accuracy ratings.
6. **Test Capture Session Screen:**
   - Navigate to **Screen F (Capture Session)**.
   - Toggle **"Stream IMU Sensors"** ON and tap **"Start Session"**.
   - Verify that the IMU sample counter increments continuously during the active session.
