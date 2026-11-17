package ai.indoor.app.domain.usecases

import ai.indoor.app.data.repository.LocalizationRepository
import ai.indoor.app.domain.models.CaptureSession
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.UUID

/**
 * Use case managing active recording session lifecycle and telemetry accounting.
 */
class ManageCaptureSessionUseCase(
    private val repository: LocalizationRepository
) {
    private val _currentSession = MutableStateFlow<CaptureSession?>(null)
    val currentSession: StateFlow<CaptureSession?> = _currentSession.asStateFlow()

    fun startSession(): CaptureSession {
        val session = CaptureSession(
            sessionId = "SES-" + UUID.randomUUID().toString().take(8).uppercase(),
            startTimeMs = System.currentTimeMillis(),
            isRunning = true,
            uploadStatus = CaptureSession.UploadStatus.IDLE
        )
        _currentSession.value = session
        return session
    }

    fun stopSession(): CaptureSession? {
        val session = _currentSession.value ?: return null
        val stopped = session.copy(
            endTimeMs = System.currentTimeMillis(),
            isRunning = false,
            uploadStatus = CaptureSession.UploadStatus.BUFFERED_LOCALLY
        )
        _currentSession.value = stopped
        return stopped
    }

    fun recordWifiScan() {
        val session = _currentSession.value ?: return
        if (session.isRunning) {
            _currentSession.value = session.copy(wifiScanCount = session.wifiScanCount + 1)
        }
    }

    fun recordImuSample() {
        val session = _currentSession.value ?: return
        if (session.isRunning) {
            _currentSession.value = session.copy(imuSampleCount = session.imuSampleCount + 1)
        }
    }

    fun updateImuSampleCount(count: Int) {
        val session = _currentSession.value ?: return
        if (session.isRunning) {
            _currentSession.value = session.copy(imuSampleCount = count)
        }
    }

    fun recordPhoto() {
        val session = _currentSession.value ?: return
        if (session.isRunning) {
            _currentSession.value = session.copy(photoCount = session.photoCount + 1)
        }
    }

    suspend fun syncSession(): Result<String> {
        val session = _currentSession.value ?: return Result.failure(IllegalStateException("No session active"))
        _currentSession.value = session.copy(uploadStatus = CaptureSession.UploadStatus.UPLOADING)
        val result = repository.syncSessionTelemetry(session)
        if (result.isSuccess) {
            _currentSession.value = session.copy(uploadStatus = CaptureSession.UploadStatus.SYNCED)
        } else {
            _currentSession.value = session.copy(uploadStatus = CaptureSession.UploadStatus.FAILED)
        }
        return result
    }
}
