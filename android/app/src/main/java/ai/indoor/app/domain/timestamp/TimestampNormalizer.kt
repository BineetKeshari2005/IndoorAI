package ai.indoor.app.domain.timestamp

/**
 * Timestamp normalization and synchronization engine.
 *
 * Translates multimodal sensor, Wi-Fi, camera, and manual annotation timestamps
 * into a coherent, wall-clock aligned timeline while strictly preserving original values
 * and quantifying timing uncertainty.
 */
class TimestampNormalizer(
    var anchor: ClockAnchor = ClockAnchor.create()
) {

    /**
     * Normalizes a hardware or synthetic sensor event timestamp.
     *
     * @param timestampNs Raw nanoseconds elapsed since device boot (SensorEvent.timestamp).
     * @param isMock Whether the event originates from a synthetic provider.
     */
    fun normalizeSensorTimestamp(timestampNs: Long, isMock: Boolean = false): TimestampRecord {
        if (timestampNs <= 0L) {
            return TimestampRecord.unavailable(timestampNs, TimeSource.CLOCK_BOOTTIME_NANOS)
        }

        val deltaNs = timestampNs - anchor.bootTimeNs
        val deltaMs = deltaNs / 1_000_000L
        val normalizedEpochMs = anchor.wallClockEpochMs + deltaMs

        // Realistic uncertainty bounds: hardware sensors experience ~2-5ms kernel scheduling jitter
        val uncertainty = if (isMock) 0.5f else 3.5f

        return TimestampRecord(
            originalTimestamp = timestampNs,
            timeSource = TimeSource.CLOCK_BOOTTIME_NANOS,
            normalizedEpochMs = normalizedEpochMs,
            normalizedIso = TimestampRecord.formatIso(normalizedEpochMs),
            syncMethod = SyncMethod.BOOTTIME_ANCHOR_OFFSET,
            uncertaintyMs = uncertainty,
            isReliable = true
        )
    }

    /**
     * Normalizes a direct wall-clock timestamp (used by cameras, manual annotations, session start/stops).
     *
     * @param epochMs Milliseconds since UNIX epoch.
     * @param isMock Whether the record originates from mock test generation.
     */
    fun normalizeWallClockTimestamp(epochMs: Long, isMock: Boolean = false): TimestampRecord {
        if (epochMs <= 0L) {
            return TimestampRecord.unavailable(epochMs, TimeSource.CLOCK_REALTIME_MILLIS)
        }

        val uncertainty = if (isMock) 0.5f else 1.5f

        return TimestampRecord(
            originalTimestamp = epochMs,
            timeSource = TimeSource.CLOCK_REALTIME_MILLIS,
            normalizedEpochMs = epochMs,
            normalizedIso = TimestampRecord.formatIso(epochMs),
            syncMethod = SyncMethod.WALLCLOCK_DIRECT,
            uncertaintyMs = uncertainty,
            isReliable = true
        )
    }

    /**
     * Normalizes a Wi-Fi scan result timestamp.
     *
     * Handles both microseconds-since-boot (standard Android API 17+ ScanResult.timestamp)
     * and wall-clock epoch milliseconds (fallback or synthetic scans).
     *
     * @param rawTimestamp Raw scan timestamp in microseconds or milliseconds.
     * @param isFresh Whether the scan was fresh from the Wi-Fi radio or cached from a throttled request.
     * @param isMock Whether the scan was synthetically generated.
     */
    fun normalizeWifiScanTimestamp(
        rawTimestamp: Long,
        isFresh: Boolean,
        isMock: Boolean = false
    ): TimestampRecord {
        if (rawTimestamp <= 0L) {
            return TimestampRecord.unavailable(rawTimestamp, TimeSource.UNKNOWN)
        }

        // Differentiate epoch ms (> 1 trillion, e.g. 1716000000000) from boot-time micros
        val isEpochMs = rawTimestamp > 1_000_000_000_000L

        val (normalizedEpochMs, source, method) = if (isEpochMs) {
            Triple(rawTimestamp, TimeSource.CLOCK_REALTIME_MILLIS, SyncMethod.WALLCLOCK_DIRECT)
        } else {
            // Convert boottime microseconds to nanoseconds and apply ClockAnchor offset
            val scanNs = rawTimestamp * 1_000L
            val deltaNs = scanNs - anchor.bootTimeNs
            val deltaMs = deltaNs / 1_000_000L
            Triple(
                anchor.wallClockEpochMs + deltaMs,
                TimeSource.CLOCK_BOOTTIME_MICROS,
                SyncMethod.SCAN_AGE_CONVERTED
            )
        }

        // Wi-Fi scan timing uncertainty:
        // Fresh scans have radio sweep duration jitter (~20-50ms)
        // Cached/throttled scans can be tens of seconds old
        val uncertainty = when {
            isMock -> 1.0f
            isFresh -> 30.0f
            else -> 250.0f
        }

        return TimestampRecord(
            originalTimestamp = rawTimestamp,
            timeSource = source,
            normalizedEpochMs = normalizedEpochMs,
            normalizedIso = TimestampRecord.formatIso(normalizedEpochMs),
            syncMethod = method,
            uncertaintyMs = uncertainty,
            isReliable = true
        )
    }
}
