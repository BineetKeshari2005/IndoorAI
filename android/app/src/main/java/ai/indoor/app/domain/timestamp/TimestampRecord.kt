package ai.indoor.app.domain.timestamp

import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone

/**
 * Normalized timestamp record pairing the raw original measurement with its
 * wall-clock approximation, synchronization strategy, and uncertainty bound.
 *
 * Requirements enforced:
 * 1. Preserves every original timestamp without destructive rounding.
 * 2. Explicitly records original clock domain (boottime nanos, realtime millis, etc.).
 * 3. Stores normalized UTC epoch milliseconds and ISO-8601 representation when valid.
 * 4. Documents estimated uncertainty (±ms) rather than claiming perfect sync.
 * 5. Marks invalid or missing timestamps safely as unavailable.
 */
data class TimestampRecord(
    val originalTimestamp: Long,
    val timeSource: TimeSource,
    val normalizedEpochMs: Long?,
    val normalizedIso: String?,
    val syncMethod: SyncMethod,
    val uncertaintyMs: Float?,
    val isReliable: Boolean
) {
    val uncertaintyDisplay: String
        get() = if (uncertaintyMs != null) "±%.1f ms".format(Locale.US, uncertaintyMs) else "Uncertain"

    companion object {
        fun formatIso(epochMs: Long): String {
            val sdf = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.US).apply {
                timeZone = TimeZone.getTimeZone("UTC")
            }
            return sdf.format(Date(epochMs))
        }

        fun unavailable(original: Long, source: TimeSource = TimeSource.UNKNOWN): TimestampRecord {
            return TimestampRecord(
                originalTimestamp = original,
                timeSource = source,
                normalizedEpochMs = null,
                normalizedIso = null,
                syncMethod = SyncMethod.UNSYNCHRONIZED,
                uncertaintyMs = null,
                isReliable = false
            )
        }
    }
}
