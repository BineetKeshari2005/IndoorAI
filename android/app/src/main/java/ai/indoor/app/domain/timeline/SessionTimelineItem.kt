package ai.indoor.app.domain.timeline

import ai.indoor.app.domain.models.SensorType
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Types of observations capable of presentation in the chronological session timeline.
 */
enum class TimelineObservationType(val displayName: String) {
    WIFI_SCAN("Wi-Fi Scan"),
    ACCELEROMETER("Accelerometer"),
    GYROSCOPE("Gyroscope"),
    MAGNETOMETER("Magnetometer"),
    ORIENTATION("Orientation Vector"),
    PHOTO_CAPTURE("Landmark Photo"),
    ANNOTATION("Manual Annotation")
}

/**
 * Single chronological item on the unified session observation timeline.
 */
data class SessionTimelineItem(
    val id: String,
    val sessionId: String,
    val type: TimelineObservationType,
    val normalizedTimestampMs: Long?,
    val originalTimestamp: Long,
    val timeSourceDisplay: String,
    val formattedTime: String,
    val title: String,
    val shortDescription: String,
    val isMock: Boolean,
    val isTimingCertain: Boolean,
    val uncertaintyDisplay: String,
    val detailKeyValues: Map<String, String>
) {
    companion object {
        fun formatTimestamp(epochMs: Long?): String {
            if (epochMs == null || epochMs <= 0) return "Time Unavailable"
            val sdf = SimpleDateFormat("HH:mm:ss.SSS", Locale.US)
            return sdf.format(Date(epochMs))
        }
    }
}
