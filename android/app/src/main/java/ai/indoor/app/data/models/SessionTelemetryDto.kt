package ai.indoor.app.data.models

import com.google.gson.annotations.SerializedName

/**
 * Multi-frame session batch upload payload for `/api/v1/telemetry/session`.
 */
data class SessionTelemetryDto(
    @SerializedName("session_id") val sessionId: String,
    @SerializedName("device_id") val deviceId: String,
    @SerializedName("start_time_ms") val startTimeMs: Long,
    @SerializedName("end_time_ms") val endTimeMs: Long,
    @SerializedName("wifi_scans") val wifiScans: List<List<WifiScanResultDto>> = emptyList(),
    @SerializedName("imu_stream") val imuStream: List<ImuObservationDto> = emptyList(),
    @SerializedName("photo_count") val photoCount: Int = 0
)

data class SessionUploadResponseDto(
    @SerializedName("session_id") val sessionId: String,
    @SerializedName("status") val status: String,
    @SerializedName("records_ingested") val recordsIngested: Int,
    @SerializedName("message") val message: String
)

