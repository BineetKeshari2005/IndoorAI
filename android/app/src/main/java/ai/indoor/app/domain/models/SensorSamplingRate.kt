package ai.indoor.app.domain.models

/**
 * Configurable Android SensorManager sampling rate profiles.
 *
 * SENSOR_DELAY_FASTEST = 0
 * SENSOR_DELAY_GAME = 1 (~50 Hz / 20,000 µs)
 * SENSOR_DELAY_UI = 2 (~16 Hz / 60,000 µs)
 * SENSOR_DELAY_NORMAL = 3 (~5 Hz / 200,000 µs)
 */
enum class SensorSamplingRate(
    val delayCode: Int,
    val label: String,
    val rateDescription: String
) {
    UI(2, "UI (~16 Hz)", "~16 Hz (Battery Efficient)"),
    GAME(1, "Game (~50 Hz)", "~50 Hz (PDR Tracking)"),
    FASTEST(0, "Fastest (~100+ Hz)", "Hardware Max (High Power)");
}
