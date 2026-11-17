package ai.indoor.app.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import ai.indoor.app.domain.models.AnnotationType
import ai.indoor.app.domain.models.ManualLocationLabel
import ai.indoor.app.domain.models.SensorType
import ai.indoor.app.domain.models.SingleSensorReading
import ai.indoor.app.domain.models.SurveySession
import ai.indoor.app.domain.models.SurveySessionStatus
import ai.indoor.app.domain.models.WifiFingerprint
import ai.indoor.app.domain.timeline.SessionTimelineItem
import ai.indoor.app.domain.usecases.ManageCaptureSessionUseCase
import ai.indoor.app.sensors.SensorProvider
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * ViewModel managing multimodal survey sessions, timeline aggregation,
 * lifecycle state machines, and dataset ZIP exports.
 */
class SessionViewModel(
    val useCase: ManageCaptureSessionUseCase,
    private val sensorProvider: SensorProvider? = null
) : ViewModel() {

    val currentSession: StateFlow<SurveySession?> = useCase.currentSession
    val allSessions: StateFlow<List<SurveySession>> = useCase.allSessions
    val timeline: StateFlow<List<SessionTimelineItem>> = useCase.timeline

    private val _elapsedSeconds = MutableStateFlow(0L)
    val elapsedSeconds: StateFlow<Long> = _elapsedSeconds.asStateFlow()

    private val _streamSensors = MutableStateFlow(true)
    val streamSensors: StateFlow<Boolean> = _streamSensors.asStateFlow()

    private val _statusMessage = MutableStateFlow<String?>(null)
    val statusMessage: StateFlow<String?> = _statusMessage.asStateFlow()

    private val _exportedZipFile = MutableStateFlow<File?>(null)
    val exportedZipFile: StateFlow<File?> = _exportedZipFile.asStateFlow()

    private val _isExporting = MutableStateFlow(false)
    val isExporting: StateFlow<Boolean> = _isExporting.asStateFlow()

    private var timerJob: Job? = null
    private var lastRecordedSensorCount = 0

    init {
        // Initialize timer from existing active session if any
        currentSession.value?.let { session ->
            if (session.status == SurveySessionStatus.ACTIVE) {
                startTimer()
            }
        }
    }

    fun toggleSensorStreaming(enabled: Boolean) {
        _streamSensors.value = enabled
        val session = currentSession.value
        if (session != null && session.status == SurveySessionStatus.ACTIVE) {
            if (enabled) {
                sensorProvider?.startListening(session.sessionId)
            } else {
                sensorProvider?.stopListening()
            }
        }
    }

    fun createSession(
        name: String = "Survey Session",
        building: String = "",
        floor: String = "",
        room: String = "",
        notes: String = ""
    ): SurveySession {
        val label = if (building.isNotBlank() || floor.isNotBlank() || room.isNotBlank()) {
            ManualLocationLabel(building, floor, room, notes)
        } else null

        val session = useCase.createSession(
            name = name,
            buildingId = building,
            floorId = floor,
            roomId = room,
            manualLabel = label
        )
        _statusMessage.value = "Created session: ${session.sessionName} (${session.sessionId})"
        return session
    }

    fun startSession(sessionId: String? = null) {
        val session = useCase.startSession(sessionId)
        _elapsedSeconds.value = session.durationSeconds
        _statusMessage.value = "Started session ${session.sessionId}"

        if (_streamSensors.value && sensorProvider != null) {
            sensorProvider.clearCollectedReadings()
            lastRecordedSensorCount = 0
            sensorProvider.startListening(session.sessionId)
        }

        startTimer()
    }

    fun pauseSession() {
        val session = useCase.pauseSession()
        timerJob?.cancel()
        timerJob = null
        sensorProvider?.stopListening()
        if (session != null) {
            _statusMessage.value = "Session paused (${session.sessionId})"
        }
    }

    fun resumeSession() {
        val session = useCase.resumeSession()
        if (session != null) {
            _elapsedSeconds.value = session.durationSeconds
            _statusMessage.value = "Session resumed (${session.sessionId})"
            if (_streamSensors.value && sensorProvider != null) {
                sensorProvider.startListening(session.sessionId)
            }
            startTimer()
        }
    }

    fun completeSession() {
        timerJob?.cancel()
        timerJob = null
        sensorProvider?.stopListening()
        drainSensorBuffer()

        val session = useCase.completeSession()
        if (session != null) {
            _statusMessage.value = "Session completed (${session.sessionId}). Ready for export."
        }
    }

    /**
     * Legacy alias for completeSession.
     */
    fun stopSession() = completeSession()

    fun cancelSession() {
        timerJob?.cancel()
        timerJob = null
        sensorProvider?.stopListening()

        val session = useCase.cancelSession()
        if (session != null) {
            _statusMessage.value = "Session cancelled (${session.sessionId})"
        }
    }

    fun deleteSession(sessionId: String) {
        val currentId = currentSession.value?.sessionId
        if (currentId == sessionId) {
            timerJob?.cancel()
            timerJob = null
            sensorProvider?.stopListening()
        }
        val success = useCase.deleteSession(sessionId)
        _statusMessage.value = if (success) "Session deleted" else "Failed deleting session"
    }

    fun addAnnotation(
        type: AnnotationType,
        description: String,
        building: String = "",
        floor: String = "",
        room: String = "",
        relatedId: String? = null,
        isMock: Boolean = false
    ) {
        val ann = useCase.recordAnnotation(type, description, building, floor, room, relatedId, isMock)
        if (ann != null) {
            _statusMessage.value = "Annotation added: ${ann.type.displayName}"
        }
    }

    fun addMockObservation() {
        val session = currentSession.value ?: return
        if (session.status != SurveySessionStatus.ACTIVE) return

        // 1. Record synthetic Wi-Fi scan
        val mockWifi = WifiFingerprint.createMockFingerprint(session.sessionId)
        useCase.recordWifiScan(mockWifi)

        // 2. Record synthetic IMU sample
        val mockReading = SingleSensorReading(
            sensorType = SensorType.ACCELEROMETER,
            x = 0.12f,
            y = 0.45f,
            z = 9.81f,
            timestampNs = System.nanoTime(),
            sessionId = session.sessionId,
            isMock = true
        )
        useCase.recordSensorReading(mockReading)
        _statusMessage.value = "Added mock Wi-Fi + IMU observation packet"
    }

    fun addPhotoObservation() {
        val session = currentSession.value ?: return
        val mockPhoto = ai.indoor.app.domain.models.CapturedPhotoMetadata(
            photoId = "IMG-MOCK-" + java.util.UUID.randomUUID().toString().take(6),
            sessionId = session.sessionId,
            timestampMs = System.currentTimeMillis(),
            filename = "mock_photo.jpg",
            filePath = "/mock/path.jpg",
            landmarkType = ai.indoor.app.domain.models.LandmarkType.CORRIDOR,
            isExifSanitized = true,
            isMock = true
        )
        useCase.recordPhoto(mockPhoto)
    }

    fun exportDatasetZip(sessionId: String? = currentSession.value?.sessionId) {
        _isExporting.value = true
        _statusMessage.value = "Validating and creating dataset ZIP..."

        viewModelScope.launch(Dispatchers.IO) {
            val result = useCase.exportSessionZip(sessionId)
            withContext(Dispatchers.Main) {
                _isExporting.value = false
                if (result.isSuccess) {
                    val zipFile = result.getOrNull()
                    _exportedZipFile.value = zipFile
                    _statusMessage.value = "Export successful: ${zipFile?.name} (${(zipFile?.length() ?: 0) / 1024} KB)"
                } else {
                    _statusMessage.value = "Export failed: ${result.exceptionOrNull()?.message}"
                }
            }
        }
    }

    fun syncWithBackend() {
        viewModelScope.launch {
            val result = useCase.syncSession()
            _statusMessage.value = result.getOrNull() ?: "Sync failed: ${result.exceptionOrNull()?.message}"
        }
    }

    fun clearExportedZip() {
        _exportedZipFile.value = null
    }

    fun clearStatusMessage() {
        _statusMessage.value = null
    }

    private fun startTimer() {
        timerJob?.cancel()
        timerJob = viewModelScope.launch {
            while (isActive) {
                delay(1000)
                _elapsedSeconds.value += 1
                drainSensorBuffer()
            }
        }
    }

    private fun drainSensorBuffer() {
        if (_streamSensors.value && sensorProvider != null) {
            val readings = sensorProvider.collectedReadings
            val count = readings.size
            if (count > lastRecordedSensorCount) {
                for (i in lastRecordedSensorCount until count) {
                    useCase.recordSensorReading(readings[i])
                }
                lastRecordedSensorCount = count
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        timerJob?.cancel()
        sensorProvider?.stopListening()
    }
}
