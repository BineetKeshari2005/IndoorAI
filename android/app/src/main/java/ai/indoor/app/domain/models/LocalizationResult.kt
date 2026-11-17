package ai.indoor.app.domain.models

/**
 * Spatial location estimate produced by the indoor positioning pipeline.
 */
data class LocalizationResult(
    val buildingId: Int,
    val floor: Int,
    val buildingFloorIndex: Int,
    val longitude: Double,
    val latitude: Double,
    val confidence: Float,
    val modelType: String = "Multitask MLP",
    val latencyMs: Long = 0,
    val timestamp: Long = System.currentTimeMillis()
)

