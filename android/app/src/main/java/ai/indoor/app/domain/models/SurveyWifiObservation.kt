package ai.indoor.app.domain.models

import ai.indoor.app.domain.timestamp.TimestampRecord
import java.util.UUID

/**
 * Single access point signal observation within a Wi-Fi sweep.
 */
data class SurveyWifiAccessPoint(
    val bssid: String,
    val ssid: String,
    val rssi: Int,
    val frequency: Int,
    val channel: Int,
    val channelBand: String,
    val isFresh: Boolean
)

/**
 * Unified Wi-Fi scan sweep observation belonging to a survey session.
 */
data class SurveyWifiObservation(
    val observationId: String = "WIFI-" + UUID.randomUUID().toString().take(8).uppercase(),
    val sessionId: String,
    val timestamp: TimestampRecord,
    val accessPoints: List<SurveyWifiAccessPoint>,
    val isFresh: Boolean,
    val isMock: Boolean,
    val manualLabel: ManualLocationLabel? = null
) {
    /**
     * Converts all detected access points into standard CSV rows.
     */
    fun toCsvRows(): List<String> {
        val bldg = escapeCsv(manualLabel?.building.orEmpty())
        val flr = escapeCsv(manualLabel?.floor.orEmpty())
        val rm = escapeCsv(manualLabel?.room.orEmpty())

        if (accessPoints.isEmpty()) {
            return listOf(
                listOf(
                    escapeCsv(observationId),
                    escapeCsv(sessionId),
                    timestamp.originalTimestamp.toString(),
                    timestamp.timeSource.name,
                    timestamp.normalizedEpochMs?.toString() ?: "",
                    escapeCsv(timestamp.normalizedIso ?: ""),
                    timestamp.syncMethod.name,
                    timestamp.uncertaintyMs?.toString() ?: "",
                    isFresh.toString(),
                    isMock.toString(),
                    "", "", "0", "0", "0", "",
                    bldg, flr, rm
                ).joinToString(",")
            )
        }

        return accessPoints.map { ap ->
            listOf(
                escapeCsv(observationId),
                escapeCsv(sessionId),
                timestamp.originalTimestamp.toString(),
                timestamp.timeSource.name,
                timestamp.normalizedEpochMs?.toString() ?: "",
                escapeCsv(timestamp.normalizedIso ?: ""),
                timestamp.syncMethod.name,
                timestamp.uncertaintyMs?.toString() ?: "",
                isFresh.toString(),
                isMock.toString(),
                escapeCsv(ap.bssid),
                escapeCsv(ap.ssid),
                ap.rssi.toString(),
                ap.frequency.toString(),
                ap.channel.toString(),
                escapeCsv(ap.channelBand),
                bldg, flr, rm
            ).joinToString(",")
        }
    }

    companion object {
        const val CSV_HEADER =
            "observation_id,session_id,original_timestamp,time_source,normalized_epoch_ms,normalized_iso,sync_method,uncertainty_ms,is_fresh,is_mock,bssid,ssid,rssi_dbm,frequency_mhz,channel,channel_band,building,floor,room"

        private fun escapeCsv(value: String): String {
            return if (value.contains(",") || value.contains("\"") || value.contains("\n") || value.contains("\r")) {
                "\"" + value.replace("\"", "\"\"") + "\""
            } else {
                value
            }
        }
    }
}
