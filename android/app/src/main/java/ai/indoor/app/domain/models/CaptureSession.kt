package ai.indoor.app.domain.models

/**
 * State and telemetry accounting for an explicit, user-initiated data capture session.
 */
data class CaptureSession(
    val sessionId: String,
    val startTimeMs: Long,
    val endTimeMs: Long? = null,
    val isRunning: Boolean = false,
    val wifiScanCount: Int = 0,
    val imuSampleCount: Int = 0,
    val photoCount: Int = 0,
    val uploadStatus: UploadStatus = UploadStatus.IDLE
) {
    val durationSeconds: Long
        get() = if (isRunning) {
            (System.currentTimeMillis() - startTimeMs) / 1000
        } else if (endTimeMs != null) {
            (endTimeMs - startTimeMs) / 1000
        } else {
            0
        }

    val formattedDuration: String
        get() {
            val totalSec = durationSeconds
            val mins = totalSec / 60
            val secs = totalSec % 60
            return "%02d:%02d".format(mins, secs)
        }

    enum class UploadStatus {
        IDLE, BUFFERED_LOCALLY, UPLOADING, SYNCED, FAILED
    }
}

