package ai.indoor.app.data.models

import com.google.gson.annotations.SerializedName

/**
 * Data Transfer Object for an observed Wi-Fi access point in an API request/response.
 */
data class WifiScanResultDto(
    @SerializedName("bssid") val bssid: String,
    @SerializedName("ssid") val ssid: String,
    @SerializedName("rssi") val rssi: Int,
    @SerializedName("frequency") val frequency: Int,
    @SerializedName("timestamp") val timestamp: Long
)

