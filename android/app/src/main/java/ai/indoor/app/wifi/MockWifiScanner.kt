package ai.indoor.app.wifi

import ai.indoor.app.domain.models.AccessPoint
import ai.indoor.app.domain.models.WifiFingerprint
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlin.random.Random

/**
 * Synthetic demonstration provider for running on Android Studio Emulators,
 * testing environments, or devices without active Wi-Fi hardware.
 *
 * All data is strictly flagged as `isMock = true` to prevent data fabrication.
 */
class MockWifiScanner(
    private val simulateThrottling: Boolean = false
) : WifiScanner {

    private val _scanState = MutableStateFlow<WifiScanState>(WifiScanState.Idle)
    override val scanState: StateFlow<WifiScanState> = _scanState.asStateFlow()

    override val isMockProvider: Boolean = true

    private var currentAps: List<AccessPoint> = generateMockAccessPoints()
    private var scanCount = 0

    override suspend fun triggerScan(): Result<WifiFingerprint> {
        _scanState.value = WifiScanState.Scanning

        // Simulate realistic 1.2s radio scan sweep latency
        delay(1200)
        scanCount++

        val isFresh = if (simulateThrottling && scanCount > 1) false else true

        // Add minor signal jitter (±2 dBm) to simulate multipath noise
        currentAps = currentAps.map { ap ->
            val jitter = Random.nextInt(-2, 3)
            val newRssi = (ap.rssi + jitter).coerceIn(-100, -35)
            ap.copy(
                rssi = newRssi,
                timestamp = System.currentTimeMillis(),
                isFresh = isFresh
            )
        }.sortedByDescending { it.rssi }

        val fingerprint = WifiFingerprint(
            accessPoints = currentAps,
            timestamp = System.currentTimeMillis(),
            scanDurationMs = 1200,
            isFresh = isFresh,
            isMock = true
        )

        if (isFresh) {
            _scanState.value = WifiScanState.Success(fingerprint, isFresh = true)
        } else {
            _scanState.value = WifiScanState.Throttled(
                "Simulated OS rate throttling: cached results returned.",
                fingerprint
            )
        }

        return Result.success(fingerprint)
    }

    override fun getCachedScanResults(): WifiFingerprint {
        return WifiFingerprint(
            accessPoints = currentAps,
            isFresh = false,
            isMock = true
        )
    }

    companion object {
        fun generateMockAccessPoints(): List<AccessPoint> {
            val now = System.currentTimeMillis()
            return listOf(
                AccessPoint("00:14:bf:b1:97:8a", "eduroam", -48, 5180, AccessPoint.frequencyToChannel(5180), now, isFresh = true, isMock = true),
                AccessPoint("00:14:bf:b1:97:8b", "Campus-Staff", -52, 5240, AccessPoint.frequencyToChannel(5240), now, isFresh = true, isMock = true),
                AccessPoint("00:14:bf:b1:97:80", "Campus-Guest", -55, 2412, AccessPoint.frequencyToChannel(2412), now, isFresh = true, isMock = true),
                AccessPoint("70:69:79:8c:12:4a", "TI-Lab-AP04", -64, 5200, AccessPoint.frequencyToChannel(5200), now, isFresh = true, isMock = true),
                AccessPoint("70:69:79:8c:12:4b", "TI-Lab-IoT", -67, 2437, AccessPoint.frequencyToChannel(2437), now, isFresh = true, isMock = true),
                AccessPoint("00:26:cb:91:32:01", "Library-Floor2-East", -71, 5180, AccessPoint.frequencyToChannel(5180), now, isFresh = true, isMock = true),
                AccessPoint("00:26:cb:91:32:02", "Library-Public", -74, 2462, AccessPoint.frequencyToChannel(2462), now, isFresh = true, isMock = true),
                AccessPoint("44:ad:d9:15:fe:88", "Engineering-North", -79, 5260, AccessPoint.frequencyToChannel(5260), now, isFresh = true, isMock = true),
                AccessPoint("44:ad:d9:15:fe:89", "Faculty-Private", -83, 2412, AccessPoint.frequencyToChannel(2412), now, isFresh = true, isMock = true),
                AccessPoint("2c:30:33:04:19:bc", "Hallway-Repeater-09", -88, 2437, AccessPoint.frequencyToChannel(2437), now, isFresh = true, isMock = true),
                AccessPoint("2c:30:33:04:19:bd", "Hallway-Sensor-Net", -92, 2462, AccessPoint.frequencyToChannel(2462), now, isFresh = true, isMock = true)
            )
        }
    }
}
