package ai.indoor.app.domain.models

/**
 * Supported smartphone hardware inertial and magnetic sensor types.
 */
enum class SensorType(val displayName: String, val unit: String) {
    ACCELEROMETER("Accelerometer", "m/s²"),
    GYROSCOPE("Gyroscope", "rad/s"),
    MAGNETIC_FIELD("Magnetometer", "µT"),
    ROTATION_VECTOR("Rotation Vector", "unitless")
}
