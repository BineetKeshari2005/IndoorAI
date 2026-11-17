package ai.indoor.app.wifi

import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class MockWifiScannerTest {

    private lateinit var scanner: MockWifiScanner

    @Before
    fun setUp() {
        scanner = MockWifiScanner()
    }

    @Test
    fun testInitialScanStateIsIdle() {
        assertEquals(WifiScanState.Idle, scanner.scanState.value)
        assertTrue(scanner.isMockProvider)
    }

    @Test
    fun testTriggerScanProducesValidFingerprint() = runTest {
        val result = scanner.triggerScan()

        assertTrue(result.isSuccess)
        val fingerprint = result.getOrNull()
        assertNotNull(fingerprint)

        assertTrue(fingerprint!!.count > 0)
        assertTrue(fingerprint.isMock)

        val state = scanner.scanState.value
        assertTrue(state is WifiScanState.Success)
        assertEquals(fingerprint, (state as WifiScanState.Success).fingerprint)
    }

    @Test
    fun testAccessPointSignalLevelAndFrequencyBands() = runTest {
        val result = scanner.triggerScan()
        val aps = result.getOrNull()!!.accessPoints

        for (ap in aps) {
            assertTrue("RSSI should be in valid dBm range", ap.rssi in -105..-30)
            assertTrue("BSSID should not be blank", ap.bssid.isNotBlank())
            assertTrue("SSID should not be blank", ap.ssid.isNotBlank())
            assertTrue(
                "Channel band should be recognized",
                ap.channelBand == "2.4 GHz" || ap.channelBand == "5.0 GHz" || ap.channelBand == "6.0 GHz (Wi-Fi 6E)"
            )
            assertNotNull(ap.signalLevel)
        }
    }

    @Test
    fun testCachedResultsAreAvailableWithoutScanning() {
        val cached = scanner.getCachedScanResults()
        assertTrue(cached.count > 0)
        assertTrue(cached.isMock)
    }
}

