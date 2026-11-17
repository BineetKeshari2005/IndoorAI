package ai.indoor.app.wifi

import ai.indoor.app.domain.models.AccessPoint
import ai.indoor.app.domain.models.WifiFingerprint
import android.Manifest
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.location.LocationManager
import android.net.wifi.ScanResult
import android.net.wifi.WifiManager
import android.os.Build
import android.os.SystemClock
import androidx.core.content.ContextCompat
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Production implementation of Wi-Fi scanning interacting directly with Android's WifiManager.
 *
 * Implements strict handling of Android 9/10/11/12/13/14 scanning limitations:
 * 1. OS Throttling: Android restricts foreground apps to 4 scans every 2 minutes.
 * 2. Location Requirement: Requires ACCESS_FINE_LOCATION and active GPS/Location Services.
 * 3. Freshness Detection: Evaluates WifiManager.EXTRA_RESULTS_UPDATED broadcast boolean.
 * 4. Deduplication: Eliminates duplicate BSSIDs by retaining the strongest RSSI observation.
 * 5. BSSID Redaction: Flags '02:00:00:00:00:00' privacy redaction gracefully.
 */
class AndroidWifiScanner(
    private val context: Context
) : WifiScanner {

    private val wifiManager: WifiManager? =
        context.applicationContext.getSystemService(Context.WIFI_SERVICE) as? WifiManager

    private val _scanState = MutableStateFlow<WifiScanState>(WifiScanState.Idle)
    override val scanState: StateFlow<WifiScanState> = _scanState.asStateFlow()

    override val isMockProvider: Boolean = false

    private var cachedResults: List<AccessPoint> = emptyList()
    private val isReceiverRegistered = AtomicBoolean(false)

    private val wifiScanReceiver = object : BroadcastReceiver() {
        override fun onReceive(c: Context?, intent: Intent?) {
            if (intent?.action == WifiManager.SCAN_RESULTS_AVAILABLE_ACTION) {
                val updated = intent.getBooleanExtra(WifiManager.EXTRA_RESULTS_UPDATED, false)
                handleScanCompleted(isFresh = updated)
            }
        }
    }

    private fun hasLocationPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.ACCESS_FINE_LOCATION
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun isLocationServicesEnabled(): Boolean {
        val lm = context.getSystemService(Context.LOCATION_SERVICE) as? LocationManager ?: return false
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            lm.isLocationEnabled
        } else {
            lm.isProviderEnabled(LocationManager.GPS_PROVIDER) ||
                    lm.isProviderEnabled(LocationManager.NETWORK_PROVIDER)
        }
    }

    private fun registerReceiverSafely() {
        if (!isReceiverRegistered.get()) {
            val intentFilter = IntentFilter(WifiManager.SCAN_RESULTS_AVAILABLE_ACTION)
            try {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                    context.registerReceiver(wifiScanReceiver, intentFilter, Context.RECEIVER_EXPORTED)
                } else {
                    context.registerReceiver(wifiScanReceiver, intentFilter)
                }
                isReceiverRegistered.set(true)
            } catch (_: Exception) {
                try {
                    context.registerReceiver(wifiScanReceiver, intentFilter)
                    isReceiverRegistered.set(true)
                } catch (_: Exception) {}
            }
        }
    }

    fun unregisterReceiver() {
        if (isReceiverRegistered.get()) {
            try {
                context.unregisterReceiver(wifiScanReceiver)
            } catch (_: Exception) {}
            isReceiverRegistered.set(false)
        }
    }

    override suspend fun triggerScan(): Result<WifiFingerprint> {
        if (wifiManager == null) {
            val err = "Wi-Fi hardware service is unavailable on this device."
            _scanState.value = WifiScanState.Error(err)
            return Result.failure(IllegalStateException(err))
        }

        if (!hasLocationPermission()) {
            val missing = listOf(Manifest.permission.ACCESS_FINE_LOCATION)
            _scanState.value = WifiScanState.PermissionRequired(missing)
            return Result.failure(SecurityException("ACCESS_FINE_LOCATION permission required by Android to scan Wi-Fi beacons."))
        }

        if (!wifiManager.isWifiEnabled) {
            val msg = "Wi-Fi radio is turned off. Please enable Wi-Fi in Android Quick Settings."
            _scanState.value = WifiScanState.WifiDisabled(msg)
            return Result.failure(IllegalStateException(msg))
        }

        if (!isLocationServicesEnabled()) {
            val msg = "System Location Services (GPS) are turned off. Android requires active location services to return Wi-Fi scan results."
            _scanState.value = WifiScanState.LocationServicesRequired(msg)
            return Result.failure(IllegalStateException(msg))
        }

        registerReceiverSafely()
        _scanState.value = WifiScanState.Scanning

        @Suppress("DEPRECATION")
        val success = try {
            wifiManager.startScan()
        } catch (e: SecurityException) {
            _scanState.value = WifiScanState.Error("SecurityException initiating scan: ${e.message}", e)
            return Result.failure(e)
        }

        return if (success) {
            // Asynchronous radio sweep in progress; BroadcastReceiver will handle completion
            Result.success(getCachedScanResults())
        } else {
            // Android OS throttled the scan request (foreground limit of 4 scans per 2 minutes exceeded)
            val msg = "Wi-Fi scan throttled by Android OS (Foreground limit: 4 scans per 2 minutes). Cached results returned."
            val cached = getCachedScanResults()
            _scanState.value = WifiScanState.Throttled(msg, cached)
            Result.success(cached)
        }
    }

    private fun handleScanCompleted(isFresh: Boolean) {
        if (!hasLocationPermission() || wifiManager == null) return

        try {
            val rawResults: List<ScanResult> = wifiManager.scanResults ?: emptyList()
            val parsedAps = mapAndDeduplicateScanResults(rawResults, isFresh = isFresh)

            cachedResults = parsedAps
            val fingerprint = WifiFingerprint(
                accessPoints = parsedAps,
                timestamp = System.currentTimeMillis(),
                isFresh = isFresh,
                isMock = false
            )

            if (isFresh) {
                _scanState.value = WifiScanState.Success(fingerprint, isFresh = true)
            } else {
                _scanState.value = WifiScanState.Throttled(
                    "Cached results returned; Android OS throttled fresh radio sweep.",
                    fingerprint
                )
            }
        } catch (e: SecurityException) {
            _scanState.value = WifiScanState.Error("Security exception reading scan results: ${e.message}", e)
        } catch (e: Exception) {
            _scanState.value = WifiScanState.Error("Failed to parse Wi-Fi scan results: ${e.message}", e)
        }
    }

    override fun getCachedScanResults(): WifiFingerprint {
        return WifiFingerprint(
            accessPoints = cachedResults,
            isFresh = false,
            isMock = false
        )
    }

    companion object {
        /**
         * Maps raw Android ScanResults to domain AccessPoint models, deduplicating
         * multiple observations of identical BSSIDs by keeping the strongest RSSI.
         */
        fun mapAndDeduplicateScanResults(
            rawResults: List<ScanResult>,
            isFresh: Boolean
        ): List<AccessPoint> {
            return rawResults
                .filter { !it.BSSID.isNullOrBlank() }
                .groupBy { it.BSSID.lowercase() }
                .map { (_, group) ->
                    val best = group.maxByOrNull { it.level } ?: group.first()
                    val wallClockMs = calculateWallClockTimestamp(best.timestamp)
                    val bssidStr = if (best.BSSID == "02:00:00:00:00:00") "<Redacted BSSID>" else best.BSSID
                    val ssidStr = if (best.SSID.isNullOrEmpty()) "<Hidden SSID>" else best.SSID

                    AccessPoint(
                        bssid = bssidStr,
                        ssid = ssidStr,
                        rssi = best.level,
                        frequency = best.frequency,
                        channel = AccessPoint.frequencyToChannel(best.frequency),
                        timestamp = wallClockMs,
                        isFresh = isFresh,
                        isMock = false
                    )
                }
                .sortedByDescending { it.rssi }
        }

        /**
         * Converts microsecond elapsed-since-boot timestamp to epoch wall-clock milliseconds.
         */
        fun calculateWallClockTimestamp(microsecondTimestamp: Long): Long {
            return try {
                val bootElapsedMs = SystemClock.elapsedRealtime()
                val srElapsedMs = microsecondTimestamp / 1000L
                if (srElapsedMs in 1..bootElapsedMs) {
                    System.currentTimeMillis() - (bootElapsedMs - srElapsedMs)
                } else {
                    System.currentTimeMillis()
                }
            } catch (_: Throwable) {
                System.currentTimeMillis()
            }
        }
    }
}
