package ai.indoor.app.domain.timestamp

/**
 * Underlying clock or system hardware domain of the original timestamp.
 */
enum class TimeSource(val description: String) {
    /**
     * Nanoseconds since device boot (e.g., Android SensorEvent.timestamp).
     */
    CLOCK_BOOTTIME_NANOS("Nanoseconds elapsed since hardware boot"),

    /**
     * Microseconds since device boot (e.g., Android ScanResult.timestamp on API 17+).
     */
    CLOCK_BOOTTIME_MICROS("Microseconds elapsed since hardware boot"),

    /**
     * Standard UNIX epoch milliseconds (wall-clock UTC).
     */
    CLOCK_REALTIME_MILLIS("Wall-clock milliseconds since UNIX epoch (UTC)"),

    /**
     * Uncalibrated or synthetic clock source.
     */
    UNKNOWN("Unknown or uncalibrated time source")
}

/**
 * Strategy utilized to map an observation's original timestamp into a unified timeline epoch.
 */
enum class SyncMethod(val description: String) {
    /**
     * Mathematically converted using a session ClockAnchor offset.
     */
    BOOTTIME_ANCHOR_OFFSET("Offset relative to synchronized session ClockAnchor"),

    /**
     * Direct wall-clock reading recorded at observation emission.
     */
    WALLCLOCK_DIRECT("Direct wall-clock acquisition"),

    /**
     * Derived from Wi-Fi scan cache age and receiver broadcast latency.
     */
    SCAN_AGE_CONVERTED("Calculated from Wi-Fi scan age"),

    /**
     * Timestamp could not be synchronized reliably; original preserved as raw value.
     */
    UNSYNCHRONIZED("Unsynchronized or invalid original timestamp")
}
