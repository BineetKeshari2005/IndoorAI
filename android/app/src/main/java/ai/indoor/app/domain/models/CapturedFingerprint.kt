package ai.indoor.app.domain.models

import com.google.gson.Gson
import com.google.gson.GsonBuilder
import java.util.UUID

/**
 * Local fingerprint record saved for empirical dataset collection or calibration.
 *
 * Couples raw ambient 802.11 beacon measurements with explicit user-entered
 * ground-truth labels and minimal device telemetry.
 */
data class CapturedFingerprint(
    val captureId: String = UUID.randomUUID().toString(),
    val sessionId: String = UUID.randomUUID().toString(),
    val timestamp: Long = System.currentTimeMillis(),
    val fingerprint: WifiFingerprint,
    val manualLabel: ManualLocationLabel? = null,
    val deviceMetadata: DeviceMetadata = DeviceMetadata(),
    val isMock: Boolean = fingerprint.isMock
) {
    /**
     * Serializes this captured observation into formatted JSON.
     */
    fun toJson(): String {
        val gson = GsonBuilder().setPrettyPrinting().create()
        return gson.toJson(this)
    }

    /**
     * Serializes this captured observation into standard CSV rows.
     */
    fun toCsv(includeHeader: Boolean = true): String {
        return buildString {
            if (includeHeader) {
                appendLine(CSV_HEADER)
            }
            appendRows(this@CapturedFingerprint)
        }
    }

    companion object {
        const val CSV_HEADER =
            "capture_id,session_id,timestamp_epoch_ms,is_fresh,is_mock,bssid,ssid,rssi_dbm,frequency_mhz,channel,band,manual_building,manual_floor,manual_room,manual_notes"

        private val gson: Gson by lazy {
            GsonBuilder().setPrettyPrinting().create()
        }

        fun toBatchJson(captures: List<CapturedFingerprint>): String {
            return gson.toJson(captures)
        }

        fun toBatchCsv(captures: List<CapturedFingerprint>): String {
            return buildString {
                appendLine(CSV_HEADER)
                for (capture in captures) {
                    appendRows(capture)
                }
            }
        }

        private fun StringBuilder.appendRows(capture: CapturedFingerprint) {
            val bldg = escapeCsv(capture.manualLabel?.building.orEmpty())
            val flr = escapeCsv(capture.manualLabel?.floor.orEmpty())
            val rm = escapeCsv(capture.manualLabel?.room.orEmpty())
            val notes = escapeCsv(capture.manualLabel?.notes.orEmpty())

            if (capture.fingerprint.accessPoints.isEmpty()) {
                appendLine(
                    "${capture.captureId},${capture.sessionId},${capture.timestamp},${capture.fingerprint.isFresh},${capture.isMock}," +
                            "\"\",\"\",0,0,0,\"\",$bldg,$flr,$rm,$notes"
                )
            } else {
                for (ap in capture.fingerprint.accessPoints) {
                    val bssidEsc = escapeCsv(ap.bssid)
                    val ssidEsc = escapeCsv(ap.ssid)
                    val bandEsc = escapeCsv(ap.channelBand)
                    appendLine(
                        "${capture.captureId},${capture.sessionId},${ap.timestamp},${ap.isFresh},${capture.isMock}," +
                                "$bssidEsc,$ssidEsc,${ap.rssi},${ap.frequency},${ap.channel},$bandEsc,$bldg,$flr,$rm,$notes"
                    )
                }
            }
        }

        private fun escapeCsv(value: String): String {
            return if (value.contains(",") || value.contains("\"") || value.contains("\n") || value.contains("\r")) {
                "\"" + value.replace("\"", "\"\"") + "\""
            } else {
                value
            }
        }
    }
}
