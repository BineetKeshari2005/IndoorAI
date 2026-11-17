package ai.indoor.app.data.models

import com.google.gson.annotations.SerializedName

/**
 * Health check response from backend `/api/v1/health` or `/health`.
 */
data class HealthResponseDto(
    @SerializedName("status") val status: String,
    @SerializedName("version") val version: String? = null,
    @SerializedName("service") val service: String? = null,
    @SerializedName("timestamp") val timestamp: Long? = null
)

