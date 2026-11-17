package ai.indoor.app.data.models

import com.google.gson.annotations.SerializedName

/**
 * Payload sent to backend `/api/v1/predict` endpoint for online indoor localization.
 */
data class LocalizationRequestDto(
    @SerializedName("device_id") val deviceId: String,
    @SerializedName("timestamp") val timestamp: Long = System.currentTimeMillis(),
    @SerializedName("model_preference") val modelPreference: String = "multitask_mlp",
    @SerializedName("wifi_fingerprint") val wifiFingerprint: List<WifiScanResultDto> = emptyList(),
    @SerializedName("imu_sample") val imuSample: ImuObservationDto? = null
)

