package ai.indoor.app.domain.models

/**
 * Domain representation of an observed Wi-Fi Access Point beacon.
 *
 * @param bssid Basic Service Set Identifier (hardware MAC address or redacted proxy)
 * @param ssid Service Set Identifier (human-readable network name)
 * @param rssi Received Signal Strength Indicator in dBm (typically -30 to -100 dBm)
 * @param frequency Channel frequency in MHz (e.g. 2412 for 2.4 GHz, 5180 for 5 GHz)
 * @param channel Derivable channel number (e.g. 1-14 for 2.4 GHz, 36-165 for 5 GHz)
 * @param timestamp Epoch timestamp in milliseconds when signal was received
 * @param isFresh Whether this observation is from a fresh radio sweep (true) or cached/throttled (false)
 * @param isMock Whether this observation originated from a synthetic benchmark provider
 */
data class AccessPoint(
    val bssid: String,
    val ssid: String,
    val rssi: Int,
    val frequency: Int,
    val channel: Int = frequencyToChannel(frequency),
    val timestamp: Long = System.currentTimeMillis(),
    val isFresh: Boolean = true,
    val isMock: Boolean = false
) {
    /**
     * Identifies the Wi-Fi frequency band.
     */
    val channelBand: String
        get() = when {
            frequency in 2400..2500 -> "2.4 GHz"
            frequency in 4900..5900 -> "5.0 GHz"
            frequency in 5925..7125 -> "6.0 GHz (Wi-Fi 6E)"
            else -> "Unknown Band"
        }

    /**
     * Categorizes signal quality for visual meters.
     */
    val signalLevel: SignalLevel
        get() = when {
            rssi >= -65 -> SignalLevel.EXCELLENT
            rssi >= -75 -> SignalLevel.GOOD
            rssi >= -85 -> SignalLevel.FAIR
            else -> SignalLevel.POOR
        }

    enum class SignalLevel {
        EXCELLENT, GOOD, FAIR, POOR
    }

    companion object {
        /**
         * Reliably calculates the 802.11 channel number from radio frequency in MHz.
         */
        fun frequencyToChannel(freqMhz: Int): Int {
            return when {
                freqMhz == 2484 -> 14
                freqMhz in 2412..2472 -> (freqMhz - 2407) / 5
                freqMhz in 5000..5895 -> (freqMhz - 5000) / 5
                freqMhz in 5925..7125 -> (freqMhz - 5950) / 5
                else -> 0
            }
        }
    }
}
