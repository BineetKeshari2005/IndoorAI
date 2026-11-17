package ai.indoor.app.domain.models

/**
 * Minimal non-identifying device telemetry necessary for calibration comparability.
 *
 * NOTE: Strictly avoids personal identifiers (no IMEI, no MAC address, no Google Account).
 */
data class DeviceMetadata(
    val manufacturer: String = "Generic",
    val model: String = "Device",
    val sdkInt: Int = 34,
    val appVersion: String = "1.0.0"
) {
    companion object {
        fun current(): DeviceMetadata {
            return try {
                DeviceMetadata(
                    manufacturer = android.os.Build.MANUFACTURER ?: "Generic",
                    model = android.os.Build.MODEL ?: "Device",
                    sdkInt = android.os.Build.VERSION.SDK_INT,
                    appVersion = "1.0.0"
                )
            } catch (_: Throwable) {
                DeviceMetadata()
            }
        }
    }
}
