package ai.indoor.app.domain.usecases

import ai.indoor.app.domain.models.ImuSample
import ai.indoor.app.sensors.SensorAvailability
import ai.indoor.app.sensors.SensorProvider
import kotlinx.coroutines.flow.StateFlow

/**
 * Use case managing motion & magnetic sensor streaming.
 */
class ObserveSensorsUseCase(
    private val sensorProvider: SensorProvider
) {
    val availability: SensorAvailability
        get() = sensorProvider.availability

    val readings: StateFlow<ImuSample>
        get() = sensorProvider.sensorReadings

    val isListening: StateFlow<Boolean>
        get() = sensorProvider.isListening

    fun start() = sensorProvider.startListening()

    fun stop() = sensorProvider.stopListening()
}

