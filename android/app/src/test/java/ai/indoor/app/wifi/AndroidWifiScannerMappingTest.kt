package ai.indoor.app.wifi

import ai.indoor.app.domain.models.AccessPoint
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class AndroidWifiScannerMappingTest {

    @Test
    fun testFrequencyToChannelCalculation24Ghz() {
        assertEquals(1, AccessPoint.frequencyToChannel(2412))
        assertEquals(6, AccessPoint.frequencyToChannel(2437))
        assertEquals(11, AccessPoint.frequencyToChannel(2462))
        assertEquals(13, AccessPoint.frequencyToChannel(2472))
        assertEquals(14, AccessPoint.frequencyToChannel(2484))
    }

    @Test
    fun testFrequencyToChannelCalculation5Ghz() {
        assertEquals(36, AccessPoint.frequencyToChannel(5180))
        assertEquals(40, AccessPoint.frequencyToChannel(5200))
        assertEquals(44, AccessPoint.frequencyToChannel(5220))
        assertEquals(48, AccessPoint.frequencyToChannel(5240))
        assertEquals(52, AccessPoint.frequencyToChannel(5260))
        assertEquals(100, AccessPoint.frequencyToChannel(5500))
        assertEquals(149, AccessPoint.frequencyToChannel(5745))
        assertEquals(165, AccessPoint.frequencyToChannel(5825))
    }

    @Test
    fun testFrequencyToChannelCalculation6Ghz() {
        assertEquals(1, AccessPoint.frequencyToChannel(5955))
        assertTrue(AccessPoint.frequencyToChannel(6115) > 0)
    }

    @Test
    fun testFrequencyToChannelUnknownFrequency() {
        assertEquals(0, AccessPoint.frequencyToChannel(900))
        assertEquals(0, AccessPoint.frequencyToChannel(1800))
        assertEquals(0, AccessPoint.frequencyToChannel(0))
    }

    @Test
    fun testChannelBandIdentification() {
        val ap24 = AccessPoint("00:11:22:33:44:55", "Net24", -60, 2437)
        assertEquals("2.4 GHz", ap24.channelBand)

        val ap5 = AccessPoint("00:11:22:33:44:56", "Net5", -65, 5180)
        assertEquals("5.0 GHz", ap5.channelBand)

        val ap6 = AccessPoint("00:11:22:33:44:57", "Net6", -70, 6000)
        assertEquals("6.0 GHz (Wi-Fi 6E)", ap6.channelBand)

        val apUnknown = AccessPoint("00:11:22:33:44:58", "NetOther", -80, 915)
        assertEquals("Unknown Band", apUnknown.channelBand)
    }

    @Test
    fun testSignalLevelCategorization() {
        val excellent = AccessPoint("00:11:22:33:44:55", "Net", -50, 2412)
        assertEquals(AccessPoint.SignalLevel.EXCELLENT, excellent.signalLevel)

        val good = AccessPoint("00:11:22:33:44:55", "Net", -70, 2412)
        assertEquals(AccessPoint.SignalLevel.GOOD, good.signalLevel)

        val fair = AccessPoint("00:11:22:33:44:55", "Net", -80, 2412)
        assertEquals(AccessPoint.SignalLevel.FAIR, fair.signalLevel)

        val poor = AccessPoint("00:11:22:33:44:55", "Net", -90, 2412)
        assertEquals(AccessPoint.SignalLevel.POOR, poor.signalLevel)
    }

    @Test
    fun testWallClockTimestampCalculation() {
        val calculated = AndroidWifiScanner.calculateWallClockTimestamp(1000000L)
        assertTrue("Wall clock timestamp should be reasonable epoch time", calculated > 1700000000000L)
    }
}
