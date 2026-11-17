package ai.indoor.app.wifi

import ai.indoor.app.domain.models.WifiFingerprint
import kotlinx.coroutines.flow.StateFlow

/**
 * State of Wi-Fi scanning hardware, Android OS scheduler, and system prerequisites.
 */
sealed class WifiScanState {
    object Idle : WifiScanState()
    object Scanning : WifiScanState()
    data class Success(val fingerprint: WifiFingerprint, val isFresh: Boolean = true) : WifiScanState()
    data class Throttled(val message: String, val cachedFingerprint: WifiFingerprint?) : WifiScanState()
    data class PermissionRequired(val missingPermissions: List<String>) : WifiScanState()
    data class LocationServicesRequired(val message: String) : WifiScanState()
    data class WifiDisabled(val message: String) : WifiScanState()
    data class Error(val message: String, val cause: Throwable? = null) : WifiScanState()
}

/**
 * Abstract boundary for Wi-Fi fingerprint scanning.
 */
interface WifiScanner {

    /**
     * Observable scan state and observed access points.
     */
    val scanState: StateFlow<WifiScanState>

    /**
     * Whether this provider uses actual hardware or synthetic demonstration data.
     */
    val isMockProvider: Boolean

    /**
     * Initiates a new Wi-Fi scan subject to OS permissions and throttling.
     */
    suspend fun triggerScan(): Result<WifiFingerprint>

    /**
     * Retrieves the most recent scan results without triggering an active radio scan.
     */
    fun getCachedScanResults(): WifiFingerprint
}
