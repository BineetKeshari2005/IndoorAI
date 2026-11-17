package ai.indoor.app.domain.timestamp

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class TimestampNormalizerTest {

    private val fakeClock = object : SystemClockProvider {
        var currentTime = 1716000000000L // epoch ms
        var elapsedNanos = 50_000_000_000L // 50 seconds since boot

        override fun currentTimeMillis(): Long = currentTime
        override fun elapsedRealtimeNanos(): Long = elapsedNanos
    }

    @Test
    fun testClockAnchorCreation() {
        val anchor = ClockAnchor.create(fakeClock)
        assertEquals(1716000000000L, anchor.wallClockEpochMs)
        assertEquals(50_000_000_000L, anchor.bootTimeNs)
        assertTrue(anchor.anchorIsoUtc.contains("2024-05-18"))
    }

    @Test
    fun testSensorTimestampNormalizationFutureAndPast() {
        val anchor = ClockAnchor.create(fakeClock)
        val normalizer = TimestampNormalizer(anchor)

        // 1 second after anchor: 51_000_000_000 ns
        val sensorT1 = 51_000_000_000L
        val record1 = normalizer.normalizeSensorTimestamp(sensorT1, isMock = false)

        assertTrue(record1.isReliable)
        assertEquals(1716000001000L, record1.normalizedEpochMs)
        assertEquals(SyncMethod.BOOTTIME_ANCHOR_OFFSET, record1.syncMethod)
        assertEquals(TimeSource.CLOCK_BOOTTIME_NANOS, record1.timeSource)
        assertEquals(3.5f, record1.uncertaintyMs ?: 0f, 0.01f)

        // 500 ms before anchor: 49_500_000_000 ns
        val sensorT0 = 49_500_000_000L
        val record0 = normalizer.normalizeSensorTimestamp(sensorT0, isMock = false)

        assertTrue(record0.isReliable)
        assertEquals(1716000000000L - 500L, record0.normalizedEpochMs)
    }

    @Test
    fun testInvalidOrZeroTimestampHandledSafely() {
        val normalizer = TimestampNormalizer(ClockAnchor.create(fakeClock))

        val zeroRecord = normalizer.normalizeSensorTimestamp(0L)
        assertFalse(zeroRecord.isReliable)
        assertNull(zeroRecord.normalizedEpochMs)
        assertEquals(SyncMethod.UNSYNCHRONIZED, zeroRecord.syncMethod)

        val negativeRecord = normalizer.normalizeSensorTimestamp(-12345L)
        assertFalse(negativeRecord.isReliable)
        assertNull(negativeRecord.normalizedEpochMs)

        val zeroWallClock = normalizer.normalizeWallClockTimestamp(0L)
        assertFalse(zeroWallClock.isReliable)
        assertNull(zeroWallClock.normalizedEpochMs)
    }

    @Test
    fun testWifiScanTimestampMicrosecondsAndEpoch() {
        val anchor = ClockAnchor.create(fakeClock)
        val normalizer = TimestampNormalizer(anchor)

        // Boottime microseconds: 50_000_000 µs = 50 seconds
        val scanMicros = 50_000_000L
        val freshRecord = normalizer.normalizeWifiScanTimestamp(scanMicros, isFresh = true, isMock = false)

        assertTrue(freshRecord.isReliable)
        assertEquals(1716000000000L, freshRecord.normalizedEpochMs)
        assertEquals(TimeSource.CLOCK_BOOTTIME_MICROS, freshRecord.timeSource)
        assertEquals(SyncMethod.SCAN_AGE_CONVERTED, freshRecord.syncMethod)
        assertEquals(30.0f, freshRecord.uncertaintyMs ?: 0f, 0.01f)

        // Throttled / cached scan has higher uncertainty
        val cachedRecord = normalizer.normalizeWifiScanTimestamp(scanMicros, isFresh = false, isMock = false)
        assertEquals(250.0f, cachedRecord.uncertaintyMs ?: 0f, 0.01f)

        // Wall-clock epoch milliseconds scan
        val epochScan = 1716000005000L
        val epochRecord = normalizer.normalizeWifiScanTimestamp(epochScan, isFresh = true, isMock = false)
        assertEquals(1716000005000L, epochRecord.normalizedEpochMs)
        assertEquals(TimeSource.CLOCK_REALTIME_MILLIS, epochRecord.timeSource)
        assertEquals(SyncMethod.WALLCLOCK_DIRECT, epochRecord.syncMethod)
    }

    @Test
    fun testMockTimestampsHaveLowerUncertainty() {
        val normalizer = TimestampNormalizer(ClockAnchor.create(fakeClock))

        val mockSensor = normalizer.normalizeSensorTimestamp(50_000_000_000L, isMock = true)
        assertEquals(0.5f, mockSensor.uncertaintyMs ?: 0f, 0.01f)

        val mockWall = normalizer.normalizeWallClockTimestamp(1716000000000L, isMock = true)
        assertEquals(0.5f, mockWall.uncertaintyMs ?: 0f, 0.01f)
    }
}
