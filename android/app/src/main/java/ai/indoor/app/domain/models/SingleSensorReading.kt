package ai.indoor.app.domain.models

import kotlin.math.sqrt

/**
 * An individual hardware or synthetic sensor event record.
 *
 * Preserves the exact nanosecond hardware timestamp without premature conversion
 * into wall-clock time, ensuring accurate differential physics ($\Delta t$).
 *
 * @param sensorType Type of the emitting sensor
 * @param x Calibrated reading on X axis
 * @param y Calibrated reading on Y axis
 * @param z Calibrated reading on Z axis
 * @param timestampNs Hardware sensor event timestamp in nanoseconds
 * @param accuracy Sensor reporting accuracy
 * @param sessionId Optional active capture session identifier
 * @param samplingRate Active sampling profile label
 * @param isMock Flag indicating synthetic demonstration origin
 */
data class SingleSensorReading(
    val sensorType: SensorType,
    val x: Float,
    val y: Float,
    val z: Float,
    val timestampNs: Long,
    val accuracy: SensorAccuracy = SensorAccuracy.HIGH,
    val sessionId: String? = null,
    val samplingRate: String = "UI",
    val isMock: Boolean = false
) {
    val magnitude: Float
        get() = sqrt(x * x + y * y + z * z)
}
