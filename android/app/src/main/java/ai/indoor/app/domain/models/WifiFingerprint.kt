package ai.indoor.app.domain.models

import java.util.UUID

/**
 * An aggregation of multiple Wi-Fi access points observed simultaneously at an epoch.
 *
 * @param accessPoints List of observed access point beacons
 * @param timestamp Epoch timestamp in milliseconds
 * @param scanDurationMs Measured duration of radio sweep
 * @param isFresh Whether results reflect a fresh radio sweep or OS cache fallback
 * @param isMock Whether this fingerprint originated from a synthetic benchmark provider
 * @param scanId Unique UUID identifying this scan event
 */
data class WifiFingerprint(
    val accessPoints: List<AccessPoint>,
    val timestamp: Long = System.currentTimeMillis(),
    val scanDurationMs: Long = 0,
    val isFresh: Boolean = true,
    val isMock: Boolean = false,
    val scanId: String = UUID.randomUUID().toString()
) {
    val count: Int get() = accessPoints.size

    val freshCount: Int get() = accessPoints.count { it.isFresh }

    val cachedCount: Int get() = accessPoints.count { !it.isFresh }

    val strongestAp: AccessPoint? get() = accessPoints.maxByOrNull { it.rssi }

    val meanRssi: Double get() = if (accessPoints.isEmpty()) 0.0 else accessPoints.map { it.rssi }.average()

    val frequencyBands: Set<String> get() = accessPoints.map { it.channelBand }.toSet()
}
