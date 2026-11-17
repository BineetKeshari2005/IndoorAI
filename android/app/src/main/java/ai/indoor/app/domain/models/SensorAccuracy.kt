package ai.indoor.app.domain.models

/**
 * Android SensorEventListener accuracy status code mapping.
 */
enum class SensorAccuracy(val code: Int, val label: String) {
    UNRELIABLE(0, "Unreliable"),
    LOW(1, "Low"),
    MEDIUM(2, "Medium"),
    HIGH(3, "High");

    companion object {
        fun fromCode(code: Int): SensorAccuracy = when (code) {
            1 -> LOW
            2 -> MEDIUM
            3 -> HIGH
            else -> UNRELIABLE
        }
    }
}
