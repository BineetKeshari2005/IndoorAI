package ai.indoor.app.domain.models

import kotlin.math.sqrt

/**
 * Composite multimodal sample combining 3-axis accelerometer, gyroscope,
 * magnetometer, low-pass gravity estimates, and orientation angles.
 */
data class ImuSample(
    // Raw Accelerometer (m/s²)
    val ax: Float = 0f,
    val ay: Float = 0f,
    val az: Float = 9.81f,
    val accelTimestampNs: Long = 0L,
    val accelAccuracy: SensorAccuracy = SensorAccuracy.HIGH,

    // Raw Gyroscope (rad/s)
    val gx: Float = 0f,
    val gy: Float = 0f,
    val gz: Float = 0f,
    val gyroTimestampNs: Long = 0L,
    val gyroAccuracy: SensorAccuracy = SensorAccuracy.HIGH,

    // Raw Magnetometer (µT)
    val mx: Float = 0f,
    val my: Float = 0f,
    val mz: Float = 0f,
    val magTimestampNs: Long = 0L,
    val magAccuracy: SensorAccuracy = SensorAccuracy.HIGH,

    // Low-Pass Gravity Estimation & Dynamic Linear Acceleration
    val gravityX: Float = 0f,
    val gravityY: Float = 0f,
    val gravityZ: Float = 9.81f,
    val linearAccelX: Float = 0f,
    val linearAccelY: Float = 0f,
    val linearAccelZ: Float = 0f,

    // Estimated Device Orientation (Degrees)
    val azimuthDeg: Float = 0f,
    val pitchDeg: Float = 0f,
    val rollDeg: Float = 0f,
    val hasOrientation: Boolean = false,

    val timestamp: Long = System.currentTimeMillis(),
    val isMock: Boolean = false
) {
    /**
     * Total raw acceleration magnitude in m/s² (including Earth's gravity vector ~9.81 m/s²).
     */
    val accelerationMagnitude: Float
        get() = sqrt(ax * ax + ay * ay + az * az)

    /**
     * Dynamic linear acceleration magnitude in m/s² (gravity subtracted).
     */
    val linearAccelerationMagnitude: Float
        get() = sqrt(linearAccelX * linearAccelX + linearAccelY * linearAccelY + linearAccelZ * linearAccelZ)

    /**
     * Angular velocity magnitude in rad/s.
     */
    val angularSpeedMagnitude: Float
        get() = sqrt(gx * gx + gy * gy + gz * gz)

    /**
     * Total geomagnetic flux density magnitude in µT (typical Earth field 25–65 µT).
     */
    val magneticFieldMagnitude: Float
        get() = sqrt(mx * mx + my * my + mz * mz)
}
