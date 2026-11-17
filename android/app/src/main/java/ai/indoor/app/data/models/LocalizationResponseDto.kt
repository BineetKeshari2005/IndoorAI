package ai.indoor.app.data.models

import com.google.gson.annotations.SerializedName

/**
 * Spatial positioning prediction returned by backend `/api/v1/predict`.
 */
data class LocalizationResponseDto(
    @SerializedName("building_id") val buildingId: Int,
    @SerializedName("floor") val floor: Int,
    @SerializedName("building_floor_index") val buildingFloorIndex: Int,
    @SerializedName("longitude") val longitude: Double,
    @SerializedName("latitude") val latitude: Double,
    @SerializedName("confidence") val confidence: Float,
    @SerializedName("model_used") val modelUsed: String,
    @SerializedName("inference_latency_ms") val inferenceLatencyMs: Long,
    @SerializedName("timestamp") val timestamp: Long
)

