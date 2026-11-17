package ai.indoor.app.domain.models

import ai.indoor.app.domain.timestamp.ClockAnchor
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import java.util.UUID

/**
 * Unified multimodal survey session container.
 *
 * Encapsulates lifecycle state, clock anchor, manual ground-truth labels,
 * telemetry counts, and data completeness metrics.
 */
data class SurveySession(
    val sessionId: String = "SES-" + UUID.randomUUID().toString().take(8).uppercase(),
    val sessionName: String = "Survey Session",
    val createdAtMs: Long = System.currentTimeMillis(),
    val startTimeMs: Long? = null,
    val endTimeMs: Long? = null,
    val status: SurveySessionStatus = SurveySessionStatus.CREATED,
    val buildingId: String = "",
    val floorId: String = "",
    val roomId: String = "",
    val manualLocationLabel: ManualLocationLabel? = null,
    val clockAnchor: ClockAnchor = ClockAnchor.create(),
    // Telemetry and observation counts
    val wifiCount: Int = 0,
    val sensorCount: Int = 0,
    val photoCount: Int = 0,
    val annotationCount: Int = 0,
    // Source provenance accounting
    val realObservationCount: Int = 0,
    val mockObservationCount: Int = 0,
    // Interruption and recovery accounting
    val isInterrupted: Boolean = false,
    val interruptionNote: String? = null
) {
    val totalObservations: Int
        get() = wifiCount + sensorCount + photoCount + annotationCount

    val isRunning: Boolean
        get() = status == SurveySessionStatus.ACTIVE

    val wifiScanCount: Int
        get() = wifiCount

    val imuSampleCount: Int
        get() = sensorCount

    val hasRealData: Boolean
        get() = realObservationCount > 0

    val hasMockData: Boolean
        get() = mockObservationCount > 0

    val dataSourceType: DataSourceType
        get() = when {
            hasRealData && hasMockData -> DataSourceType.HYBRID
            hasMockData -> DataSourceType.MOCK
            else -> DataSourceType.REAL
        }

    val durationSeconds: Long
        get() = when {
            startTimeMs == null -> 0L
            status == SurveySessionStatus.ACTIVE -> (System.currentTimeMillis() - startTimeMs) / 1000
            endTimeMs != null -> (endTimeMs - startTimeMs) / 1000
            else -> 0L
        }

    val formattedDuration: String
        get() {
            val totalSec = durationSeconds
            val mins = totalSec / 60
            val secs = totalSec % 60
            return "%02d:%02d".format(mins, secs)
        }

    val displayLocation: String
        get() = buildString {
            if (buildingId.isNotBlank()) append("Building: $buildingId")
            if (floorId.isNotBlank()) {
                if (isNotEmpty()) append(", ")
                append("Floor: $floorId")
            }
            if (roomId.isNotBlank()) {
                if (isNotEmpty()) append(", ")
                append("Room: $roomId")
            }
            if (isEmpty()) {
                manualLocationLabel?.let { append(it.formattedDisplay) }
                    ?: append("Unlabeled Survey Area")
            }
        }

    /**
     * Converts session metadata into structured JSON for dataset export (`session.json`).
     */
    fun toJson(): String {
        return gson.toJson(this)
    }

    companion object {
        private val gson: Gson by lazy {
            GsonBuilder().setPrettyPrinting().create()
        }

        fun fromJson(json: String): SurveySession {
            return gson.fromJson(json, SurveySession::class.java)
        }
    }
}
