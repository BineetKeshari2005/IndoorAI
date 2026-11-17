package ai.indoor.app.domain.timestamp

import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone

/**
 * An explicit clock anchor pairing monotonic hardware boot-time nanoseconds with
 * standard wall-clock epoch milliseconds.
 *
 * Established at session creation or start, this anchor enables mathematical mapping
 * of high-resolution sensor timestamps (`SensorEvent.timestamp`) to wall-clock time
 * while preserving the raw sensor nanoseconds.
 */
data class ClockAnchor(
    val wallClockEpochMs: Long,
    val bootTimeNs: Long,
    val anchorIsoUtc: String
) {
    companion object {
        fun create(clock: SystemClockProvider = SystemClockProvider.Default): ClockAnchor {
            val epochMs = clock.currentTimeMillis()
            val bootNs = clock.elapsedRealtimeNanos()
            val sdf = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.US).apply {
                timeZone = TimeZone.getTimeZone("UTC")
            }
            return ClockAnchor(
                wallClockEpochMs = epochMs,
                bootTimeNs = bootNs,
                anchorIsoUtc = sdf.format(Date(epochMs))
            )
        }
    }
}
