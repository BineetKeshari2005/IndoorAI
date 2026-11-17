package ai.indoor.app.viewmodel

import androidx.lifecycle.ViewModel
import ai.indoor.app.domain.models.ImuSample
import ai.indoor.app.domain.models.SensorSamplingRate
import ai.indoor.app.sensors.MockSensorProvider
import ai.indoor.app.sensors.SensorAvailability
import ai.indoor.app.sensors.SensorProvider
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * ViewModel managing motion, orientation, and magnetic sensor monitoring,
 * sampling configurations, and local telemetry exports.
 */
class SensorViewModel(
    private var provider: SensorProvider = MockSensorProvider()
) : ViewModel() {

    val availability: SensorAvailability
        get() = provider.availability

    val readings: StateFlow<ImuSample>
        get() = provider.sensorReadings

    val isListening: StateFlow<Boolean>
        get() = provider.isListening

    val samplingRate: StateFlow<SensorSamplingRate>
        get() = provider.samplingRate

    private val _exportPreview = MutableStateFlow<String?>(null)
    val exportPreview: StateFlow<String?> = _exportPreview.asStateFlow()

    private val _exportFormat = MutableStateFlow<String?>(null)
    val exportFormat: StateFlow<String?> = _exportFormat.asStateFlow()

    fun setProvider(newProvider: SensorProvider) {
        if (provider.isListening.value) {
            provider.stopListening()
        }
        provider = newProvider
    }

    fun setSamplingRate(rate: SensorSamplingRate) {
        provider.setSamplingRate(rate)
    }

    fun toggleListening(sessionId: String? = null) {
        if (provider.isListening.value) {
            provider.stopListening()
        } else {
            provider.startListening(sessionId)
        }
    }

    fun startListening(sessionId: String? = null) {
        if (!provider.isListening.value) {
            provider.startListening(sessionId)
        }
    }

    fun stopListening() {
        if (provider.isListening.value) {
            provider.stopListening()
        }
    }

    fun getRecordedCount(): Int = provider.collectedReadings.size

    fun exportAsCsv(): String {
        val csv = provider.exportReadingsCsv()
        _exportPreview.value = csv
        _exportFormat.value = "CSV"
        return csv
    }

    fun exportAsJson(): String {
        val json = provider.exportReadingsJson()
        _exportPreview.value = json
        _exportFormat.value = "JSON"
        return json
    }

    fun clearExportPreview() {
        _exportPreview.value = null
        _exportFormat.value = null
    }

    fun clearCollectedReadings() {
        provider.clearCollectedReadings()
        clearExportPreview()
    }

    override fun onCleared() {
        super.onCleared()
        provider.stopListening()
    }
}
