package ai.indoor.app.domain.adapter

import ai.indoor.app.domain.models.AccessPoint
import ai.indoor.app.domain.models.WifiFingerprint
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class WifiFingerprintModelAdapterTest {

    @Test
    fun testUncalibratedRealFingerprintReturnsUncalibratedResult() {
        val adapter = WifiFingerprintModelAdapter(calibrationMapping = null)
        assertFalse(adapter.isCalibrated)

        val realAp1 = AccessPoint(
            bssid = "00:14:bf:b1:97:8a",
            ssid = "Campus-Net",
            rssi = -55,
            frequency = 2412,
            isFresh = true,
            isMock = false
        )
        val realAp2 = AccessPoint(
            bssid = "00:14:bf:b1:97:8b",
            ssid = "Campus-Net-5G",
            rssi = -68,
            frequency = 5180,
            isFresh = true,
            isMock = false
        )

        val realFp = WifiFingerprint(listOf(realAp1, realAp2), isFresh = true, isMock = false)
        val result = adapter.adaptForInference(realFp)

        assertTrue(result is ModelAdaptationResult.Uncalibrated)
        val uncalibrated = result as ModelAdaptationResult.Uncalibrated
        assertEquals(2, uncalibrated.observedBssidCount)
        assertTrue(uncalibrated.reason.contains("WAP001–WAP520"))
        assertTrue(uncalibrated.reason.contains("calibration"))
    }

    @Test
    fun testMockFingerprintReturnsDemoSynthetic() {
        val adapter = WifiFingerprintModelAdapter()
        val mockAp = AccessPoint(
            bssid = "00:14:bf:b1:97:8a",
            ssid = "Demo",
            rssi = -50,
            frequency = 2412,
            isMock = true
        )
        val mockFp = WifiFingerprint(listOf(mockAp), isMock = true)
        val result = adapter.adaptForInference(mockFp)

        assertTrue(result is ModelAdaptationResult.DemoSynthetic)
    }

    @Test
    fun testCalibratedSurveyMappingProducesCorrect520FeatureVector() {
        // Simulated calibrated site survey mapping 2 BSSIDs to UJI WAP slots:
        // "aa:bb:cc:dd:ee:01" -> WAP001 (index 0)
        // "aa:bb:cc:dd:ee:05" -> WAP005 (index 4)
        val mapping = mapOf(
            "aa:bb:cc:dd:ee:01" to "WAP001",
            "aa:bb:cc:dd:ee:05" to "WAP005"
        )
        val adapter = WifiFingerprintModelAdapter(calibrationMapping = mapping)
        assertTrue(adapter.isCalibrated)

        val ap1 = AccessPoint(bssid = "AA:BB:CC:DD:EE:01", ssid = "AP1", rssi = -52, frequency = 2412, isMock = false)
        val ap2 = AccessPoint(bssid = "AA:BB:CC:DD:EE:05", ssid = "AP5", rssi = -74, frequency = 5180, isMock = false)
        val apUnmapped = AccessPoint(bssid = "99:99:99:99:99:99", ssid = "Other", rssi = -80, frequency = 2412, isMock = false)

        val fp = WifiFingerprint(listOf(ap1, ap2, apUnmapped), isMock = false)
        val result = adapter.adaptForInference(fp)

        assertTrue(result is ModelAdaptationResult.Calibrated)
        val calibrated = result as ModelAdaptationResult.Calibrated

        assertEquals(2, calibrated.matchedWapCount)
        assertEquals(520, calibrated.totalExpectedWaps)
        assertEquals(520, calibrated.wapFeatureVector.size)

        // Matched WAP001 (index 0) has RSSI -52.0
        assertEquals(-52.0f, calibrated.wapFeatureVector[0], 0.001f)
        // Matched WAP005 (index 4) has RSSI -74.0
        assertEquals(-74.0f, calibrated.wapFeatureVector[4], 0.001f)
        // Unmatched WAPs have default noise floor -105.0 dBm
        assertEquals(-105.0f, calibrated.wapFeatureVector[1], 0.001f)
        assertEquals(-105.0f, calibrated.wapFeatureVector[2], 0.001f)
        assertEquals(-105.0f, calibrated.wapFeatureVector[519], 0.001f)
    }

    @Test
    fun testNoArbitraryMappingWhenMappingMissing() {
        val adapter = WifiFingerprintModelAdapter(calibrationMapping = emptyMap())
        assertFalse(adapter.isCalibrated)

        val ap = AccessPoint("00:11:22:33:44:55", "TestNet", -60, 2412, isMock = false)
        val fp = WifiFingerprint(listOf(ap), isMock = false)

        val result = adapter.adaptForInference(fp)
        assertTrue("Must NOT produce calibrated vector without explicit survey map", result !is ModelAdaptationResult.Calibrated)
        assertTrue(result is ModelAdaptationResult.Uncalibrated)
    }
}
