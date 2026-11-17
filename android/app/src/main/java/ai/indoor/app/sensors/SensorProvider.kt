package ai.indoor.app.sensors

import ai.indoor.app.domain.models.CapturedSensorBatch
import ai.indoor.app.domain.models.ImuSample
import ai.indoor.app.domain.models.SensorSamplingRate
import ai.indoor.app.domain.models.SingleSensorReading
import kotlinx.coroutines.flow.StateFlow

/**
 * Sensor hardware availability matrix.
 */
data class SensorAvailability(
    val hasAccelerometer: Boolean = false,
    val hasGyroscope: Boolean = false,
    val hasMagnetometer: Boolean = false,
    val hasRotationVector: Boolean = false,
    val isMock: Boolean = false
) {
    val hasAnySensor: Boolean
        get() = hasAccelerometer || hasGyroscope || hasMagnetometer

    val allCoreSensorsPresent: Boolean
        get() = hasAccelerometer && hasGyroscope && hasMagnetometer

    val sensorSummary: String
        get() = buildString {
            append("Accel: ${if (hasAccelerometer) "OK" else "Missing"}, ")
            append("Gyro: ${if (hasGyroscope) "OK" else "Missing"}, ")
            append("Mag: ${if (hasMagnetometer) "OK" else "Missing"}, ")
            append("RotVec: ${if (hasRotationVector) "OK" else "Missing"}")
        }
}

/**
 * Abstract boundary for inertial, magnetic, and orientation sensor monitoring.
 */
interface SensorProvider {

    /**
     * Hardware presence and capabilities.
     */
    val availability: SensorAvailability

    /**
     * Observable stream of latest composite 3-axis motion, magnetic, and orientation readings.
     */
    val sensorReadings: StateFlow<ImuSample>

    /**
     * Whether provider is currently listening to hardware events.
     */
    val isListening: StateFlow<Boolean>

    /**
     * Configured sampling rate profile.
     */
    val samplingRate: StateFlow<SensorSamplingRate>

    /**
     * Configures the desired sampling rate profile (UI, Game, Fastest).
     */
    fun setSamplingRate(rate: SensorSamplingRate)

    /**
     * Starts listening to sensor updates, optionally associating with a session ID.
     */
    fun startListening(sessionId: String? = null)

    /**
     * Stops listening and unregisters all hardware event listeners.
     */
    fun stopListening()

    /**
     * Returns recorded individual sensor readings buffer.
     */
    val collectedReadings: List<SingleSensorReading>

    /**
     * Clears buffered individual sensor readings.
     */
    fun clearCollectedReadings()

    /**
     * Exports buffered sensor readings to RFC-4180 CSV string.
     */
    fun exportReadingsCsv(): String {
        return CapturedSensorBatch.exportReadingsToCsv(collectedReadings)
    }

    /**
     * Exports buffered sensor readings to JSON string.
     */
    fun exportReadingsJson(): String {
        val currentSessionId = collectedReadings.firstOrNull()?.sessionId ?: "UNASSIGNED"
        val batch = CapturedSensorBatch(
            sessionId = currentSessionId,
            readings = collectedReadings,
            samplingRate = samplingRate.value.label,
            isMock = availability.isMock
        )
        return batch.toJson()
    }
}
