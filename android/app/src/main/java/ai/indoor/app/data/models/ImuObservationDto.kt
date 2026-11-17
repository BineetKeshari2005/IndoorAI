package ai.indoor.app.data.models

import com.google.gson.annotations.SerializedName

/**
 * Data Transfer Object for 3-axis motion and magnetic measurements.
 */
data class ImuObservationDto(
    @SerializedName("ax") val ax: Float,
    @SerializedName("ay") val ay: Float,
    @SerializedName("az") val az: Float,
    @SerializedName("gx") val gx: Float,
    @SerializedName("gy") val gy: Float,
    @SerializedName("gz") val gz: Float,
    @SerializedName("mx") val mx: Float,
    @SerializedName("my") val my: Float,
    @SerializedName("mz") val mz: Float,
    @SerializedName("timestamp") val timestamp: Long
)

