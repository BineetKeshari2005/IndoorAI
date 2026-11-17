package ai.indoor.app.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import ai.indoor.app.domain.models.CaptureSession
import ai.indoor.app.domain.usecases.ManageCaptureSessionUseCase
import ai.indoor.app.sensors.SensorProvider
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

/**
 * ViewModel managing active recording sessions, elapsed duration timer,
 * real-time IMU sensor collection integration, and telemetry sync.
 */
class SessionViewModel(
    private val useCase: ManageCaptureSessionUseCase,
    private val sensorProvider: SensorProvider? = null
) : ViewModel() {

    val currentSession: StateFlow<CaptureSession?> = useCase.currentSession

    private val _elapsedSeconds = MutableStateFlow(0L)
    val elapsedSeconds: StateFlow<Long> = _elapsedSeconds.asStateFlow()

    private val _syncMessage = MutableStateFlow<String?>(null)
    val syncMessage: StateFlow<String?> = _syncMessage.asStateFlow()

    private val _streamSensors = MutableStateFlow(true)
    val streamSensors: StateFlow<Boolean> = _streamSensors.asStateFlow()

    private var timerJob: Job? = null

    fun toggleSensorStreaming(enabled: Boolean) {
        _streamSensors.value = enabled
        if (currentSession.value?.isRunning == true) {
            if (enabled) {
                sensorProvider?.startListening(currentSession.value?.sessionId)
            } else {
                sensorProvider?.stopListening()
            }
        }
    }

    fun startSession() {
        val session = useCase.startSession()
        _elapsedSeconds.value = 0L
        _syncMessage.value = null

        if (_streamSensors.value) {
            sensorProvider?.clearCollectedReadings()
            sensorProvider?.startListening(session.sessionId)
        }

        timerJob?.cancel()
        timerJob = viewModelScope.launch {
            while (isActive) {
                delay(1000)
                _elapsedSeconds.value += 1
                if (_streamSensors.value && sensorProvider != null) {
                    val count = sensorProvider.collectedReadings.size
                    useCase.updateImuSampleCount(count)
                }
            }
        }
    }

    fun stopSession() {
        timerJob?.cancel()
        timerJob = null
        sensorProvider?.stopListening()
        useCase.stopSession()
    }

    fun addMockObservation() {
        useCase.recordWifiScan()
        useCase.recordImuSample()
    }

    fun addPhotoObservation() {
        useCase.recordPhoto()
    }

    fun syncWithBackend() {
        viewModelScope.launch {
            val result = useCase.syncSession()
            _syncMessage.value = result.getOrNull() ?: "Sync failed: ${result.exceptionOrNull()?.message}"
        }
    }

    override fun onCleared() {
        super.onCleared()
        timerJob?.cancel()
        sensorProvider?.stopListening()
    }
}
