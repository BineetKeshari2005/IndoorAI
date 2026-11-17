package ai.indoor.app.domain.models

import com.google.gson.Gson
import com.google.gson.GsonBuilder

/**
 * An aggregation of captured inertial and magnetic sensor events for a session,
 * supporting structured JSON and RFC-4180 CSV export.
 */
data class CapturedSensorBatch(
    val sessionId: String,
    val readings: List<SingleSensorReading>,
    val samplingRate: String = "UI",
    val isMock: Boolean = false
) {
    fun toJson(): String {
        return gson.toJson(this)
    }

    fun toCsv(includeHeader: Boolean = true): String {
        return buildString {
            if (includeHeader) {
                appendLine(CSV_HEADER)
            }
            for (reading in readings) {
                appendReadingRow(reading)
            }
        }
    }

    companion object {
        const val CSV_HEADER =
            "session_id,sensor_type,x,y,z,magnitude,timestamp_ns,accuracy,sampling_rate,is_mock"

        private val gson: Gson by lazy {
            GsonBuilder().setPrettyPrinting().create()
        }

        fun toBatchJson(batches: List<CapturedSensorBatch>): String {
            return gson.toJson(batches)
        }

        fun toBatchCsv(batches: List<CapturedSensorBatch>): String {
            return buildString {
                appendLine(CSV_HEADER)
                for (batch in batches) {
                    for (reading in batch.readings) {
                        appendReadingRow(reading)
                    }
                }
            }
        }

        fun exportReadingsToCsv(readings: List<SingleSensorReading>): String {
            return buildString {
                appendLine(CSV_HEADER)
                for (reading in readings) {
                    appendReadingRow(reading)
                }
            }
        }

        private fun StringBuilder.appendReadingRow(r: SingleSensorReading) {
            val sid = r.sessionId ?: "UNASSIGNED"
            val type = r.sensorType.name
            val acc = r.accuracy.name
            val rate = r.samplingRate
            appendLine("$sid,$type,${r.x},${r.y},${r.z},${r.magnitude},${r.timestampNs},$acc,$rate,${r.isMock}")
        }
    }
}
