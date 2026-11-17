package ai.indoor.app.domain.export

import com.google.gson.Gson
import com.google.gson.GsonBuilder

/**
 * Manifest file entry recording filename, file size in bytes, and SHA-256 cryptographic checksum.
 */
data class ManifestFileEntry(
    val filename: String,
    val sizeBytes: Long,
    val sha256: String
)

/**
 * Observation count breakdown within the exported session bundle.
 */
data class ExportRecordCounts(
    val wifiCount: Int,
    val imuCount: Int,
    val photoCount: Int,
    val annotationCount: Int,
    val totalObservations: Int
)

/**
 * Provenance count breakdown distinguishing real hardware and synthetic mock telemetry.
 */
data class ExportSourceCounts(
    val realObservationCount: Int,
    val mockObservationCount: Int,
    val containsRealData: Boolean,
    val containsMockData: Boolean
)

/**
 * Audit manifest packaged as `manifest.json` inside the exported session ZIP.
 */
data class ExportManifest(
    val exportVersion: String = "1.0.0",
    val sessionId: String,
    val exportedAtMs: Long,
    val exportedAtIso: String,
    val recordCounts: ExportRecordCounts,
    val sourceCounts: ExportSourceCounts,
    val files: List<ManifestFileEntry>,
    val validationStatus: String = "VALIDATED"
) {
    fun toJson(): String {
        val gson = GsonBuilder().setPrettyPrinting().create()
        return gson.toJson(this)
    }

    companion object {
        fun fromJson(json: String): ExportManifest {
            return Gson().fromJson(json, ExportManifest::class.java)
        }
    }
}

/**
 * Result bundle emitted by pre-export dataset validation checks.
 */
data class ExportValidationResult(
    val isValid: Boolean,
    val errors: List<String> = emptyList(),
    val warnings: List<String> = emptyList()
) {
    val summary: String
        get() = if (isValid) {
            "Validation Passed (${warnings.size} warnings)"
        } else {
            "Validation Failed: ${errors.joinToString("; ")}"
        }
}
