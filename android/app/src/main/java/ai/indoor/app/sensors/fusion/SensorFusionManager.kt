package ai.indoor.app.sensors.fusion

import kotlin.math.atan2
import kotlin.math.sqrt

/**
 * Low-pass filtered gravity estimation and linear dynamic acceleration calculation.
 *
 * Filter equation:
 *   g[t] = α * g[t-1] + (1 - α) * a[t]
 *   a_linear[t] = a[t] - g[t]
 *
 * where α ∈ [0, 1] determines smoothing (default α = 0.8).
 */
data class GravityEstimate(
    val gx: Float,
    val gy: Float,
    val gz: Float,
    val linearAccelX: Float,
    val linearAccelY: Float,
    val linearAccelZ: Float
) {
    val linearMagnitude: Float
        get() = sqrt(linearAccelX * linearAccelX + linearAccelY * linearAccelY + linearAccelZ * linearAccelZ)
}

/**
 * Filter separating static gravity from dynamic body acceleration.
 */
class LowPassGravityFilter(
    private val alpha: Float = 0.8f
) {
    private var gx = 0f
    private var gy = 0f
    private var gz = 9.81f
    private var initialized = false

    fun update(rawX: Float, rawY: Float, rawZ: Float): GravityEstimate {
        if (!initialized) {
            gx = rawX
            gy = rawY
            gz = rawZ
            initialized = true
        } else {
            gx = alpha * gx + (1f - alpha) * rawX
            gy = alpha * gy + (1f - alpha) * rawY
            gz = alpha * gz + (1f - alpha) * rawZ
        }
        val linX = rawX - gx
        val linY = rawY - gy
        val linZ = rawZ - gz
        return GravityEstimate(gx, gy, gz, linX, linY, linZ)
    }

    fun reset() {
        initialized = false
        gx = 0f
        gy = 0f
        gz = 9.81f
    }
}

/**
 * Estimated device spatial orientation.
 *
 * @param azimuthDeg Yaw / Heading angle in degrees (-180° to +180°, where 0° points North)
 * @param pitchDeg Pitch angle in degrees (-90° to +90°, tilt forward/backward)
 * @param rollDeg Roll angle in degrees (-180° to +180°, tilt left/right)
 * @param isValid Whether orientation was reliably calculated
 */
data class OrientationEstimate(
    val azimuthDeg: Float,
    val pitchDeg: Float,
    val rollDeg: Float,
    val isValid: Boolean = true
) {
    val compassHeadingDeg: Float
        get() = (azimuthDeg + 360f) % 360f
}

/**
 * Sensor fusion foundation combining Accelerometer and Magnetometer readings
 * to estimate device orientation matrix and Euler angles.
 *
 * IMPORTANT ARCHITECTURAL DISTINCTION:
 * 1. Raw measurements: Unfiltered high-frequency readings ($a$, $\omega$, $B$).
 * 2. Filtered measurements: Low-pass gravity and high-pass body motion.
 * 3. Estimated orientation: Yaw/Pitch/Roll relative to Earth's gravitational and magnetic poles.
 * 4. Future PDR: Dead-reckoning requires step detection, step-length modeling, and heading integration.
 *
 * IMU data ALONE cannot provide accurate indoor positioning due to sensor bias drift.
 */
object SensorFusionManager {

    /**
     * Computes orientation Euler angles from gravity and geomagnetic flux vectors.
     * Implements cross-product triad calculation compatible with Android SensorManager.
     */
    fun computeOrientation(
        gravity: FloatArray,
        geomagnetic: FloatArray
    ): OrientationEstimate? {
        if (gravity.size < 3 || geomagnetic.size < 3) return null

        val ax = gravity[0]
        val ay = gravity[1]
        val az = gravity[2]

        val ex = geomagnetic[0]
        val ey = geomagnetic[1]
        val ez = geomagnetic[2]

        // H = E x G (Vector cross product: points East)
        var hx = ey * az - ez * ay
        var hy = ez * ax - ex * az
        var hz = ex * ay - ey * ax
        val normH = sqrt(hx * hx + hy * hy + hz * hz)

        if (normH < 0.1f) {
            // Device is in free fall or magnetic field is parallel to gravity
            return null
        }

        val invH = 1.0f / normH
        hx *= invH
        hy *= invH
        hz *= invH

        // Normalize G (points Down)
        val normG = sqrt(ax * ax + ay * ay + az * az)
        if (normG < 0.1f) return null
        val invG = 1.0f / normG
        val gx = ax * invG
        val gy = ay * invG
        val gz = az * invG

        // M = G x H (Points North)
        val mx = gy * hz - gz * hy
        val my = gz * hx - gx * hz
        val mz = gx * hy - gy * hx

        // Rotation matrix elements:
        // R = [ Hx, Hy, Hz ] (row 0)
        //     [ Mx, My, Mz ] (row 1)
        //     [ Gx, Gy, Gz ] (row 2)

        // Azimuth (yaw): rotation about Z
        val azimuthRad = atan2(hx, mx)
        // Pitch: rotation about X
        val pitchRad = atan2(-my, sqrt(mx * mx + gy * gy + gz * gz))
        // Roll: rotation about Y
        val rollRad = atan2(-gy, gz)

        val azDeg = Math.toDegrees(azimuthRad.toDouble()).toFloat()
        val pitchDeg = Math.toDegrees(pitchRad.toDouble()).toFloat()
        val rollDeg = Math.toDegrees(rollRad.toDouble()).toFloat()

        return OrientationEstimate(
            azimuthDeg = azDeg,
            pitchDeg = pitchDeg,
            rollDeg = rollDeg,
            isValid = true
        )
    }
}
