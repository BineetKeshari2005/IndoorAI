package ai.indoor.app.data.repository

import ai.indoor.app.domain.models.CaptureSession
import ai.indoor.app.domain.models.LocalizationResult
import ai.indoor.app.domain.models.WifiFingerprint

/**
 * Clean architecture boundary for indoor localization operations.
 */
interface LocalizationRepository {

    /**
     * Checks whether the configured backend server is reachable.
     */
    suspend fun checkServerHealth(): Result<Boolean>

    /**
     * Requests location estimation from a Wi-Fi fingerprint.
     */
    suspend fun estimateLocation(fingerprint: WifiFingerprint): Result<LocalizationResult>

    /**
     * Uploads completed session telemetry to backend or local cache.
     */
    suspend fun syncSessionTelemetry(session: CaptureSession): Result<String>

    /**
     * Updates the active backend base URL.
     */
    fun updateBaseUrl(newUrl: String)

    /**
     * Gets the currently configured base URL.
     */
    fun getBaseUrl(): String
}

