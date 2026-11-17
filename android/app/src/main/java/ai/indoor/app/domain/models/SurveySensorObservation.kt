package ai.indoor.app.domain.models

import ai.indoor.app.domain.timestamp.TimestampRecord
import java.util.UUID

/**
 * Unified motion or magnetic sensor observation belonging to a survey session.
 */
data class SurveySensorObservation(
    val observationId: String = "IMU-" + UUID.randomUUID().toString().take(8).uppercase(),
    val sessionId: String,
    val sensorType: SensorType,
    val x: Float,
    val y: Float,
    val z: Float,
    val magnitude: Float,
    val accuracy: SensorAccuracy,
    val samplingRate: String,
    val timestamp: TimestampRecord,
    val isMock: Boolean
) {
    fun toCsvRow(): String {
        return listOf(
            escapeCsv(observationId),
            escapeCsv(sessionId),
            timestamp.originalTimestamp.toString(),
            timestamp.timeSource.name,
            timestamp.normalizedEpochMs?.toString() ?: "",
            escapeCsv(timestamp.normalizedIso ?: ""),
            timestamp.syncMethod.name,
            timestamp.uncertaintyMs?.toString() ?: "",
            sensorType.name,
            x.toString(),
            y.toString(),
            z.toString(),
            magnitude.toString(),
            accuracy.name,
            escapeCsv(samplingRate),
            isMock.toString()
        ).joinToString(",")
    }

    companion object {
        const val CSV_HEADER =
            "observation_id,session_id,original_timestamp_ns,time_source,normalized_epoch_ms,normalized_iso,sync_method,uncertainty_ms,sensor_type,x,y,z,magnitude,accuracy,sampling_rate,is_mock"

        private fun escapeCsv(value: String): String {
            return if (value.contains(",") || value.contains("\"") || value.contains("\n") || value.contains("\r")) {
                "\"" + value.replace("\"", "\"\"") + "\""
            } else {
                value
            }
        }
    }
}
