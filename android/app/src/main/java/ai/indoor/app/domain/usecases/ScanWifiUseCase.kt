package ai.indoor.app.domain.usecases

import ai.indoor.app.domain.models.WifiFingerprint
import ai.indoor.app.wifi.WifiScanner

/**
 * Use case orchestrating Wi-Fi fingerprint scans.
 */
class ScanWifiUseCase(
    private val wifiScanner: WifiScanner
) {
    suspend operator fun invoke(): Result<WifiFingerprint> {
        return wifiScanner.triggerScan()
    }
}

