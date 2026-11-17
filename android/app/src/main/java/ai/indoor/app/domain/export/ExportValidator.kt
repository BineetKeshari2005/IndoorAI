package ai.indoor.app.domain.export

import ai.indoor.app.domain.models.SurveyAnnotation
import ai.indoor.app.domain.models.SurveyPhotoObservation
import ai.indoor.app.domain.models.SurveySensorObservation
import ai.indoor.app.domain.models.SurveySession
import ai.indoor.app.domain.models.SurveyWifiObservation
import java.io.File

/**
 * Validates survey session data prior to generating the multimodal export ZIP archive.
 */
object ExportValidator {

    fun validate(
        session: SurveySession,
        wifiScans: List<SurveyWifiObservation>,
        sensorReadings: List<SurveySensorObservation>,
        photos: List<SurveyPhotoObservation>,
        annotations: List<SurveyAnnotation>
    ): ExportValidationResult {
        val errors = mutableListOf<String>()
        val warnings = mutableListOf<String>()

        // 1. Session ID validation
        if (session.sessionId.isBlank()) {
            errors.add("Session has an empty or invalid sessionId")
        }

        // 2. Parent session reference consistency
        wifiScans.forEachIndexed { idx, w ->
            if (w.sessionId != session.sessionId) {
                errors.add("Wi-Fi scan at index $idx has mismatched sessionId '${w.sessionId}' (expected '${session.sessionId}')")
            }
            if (w.observationId.isBlank()) {
                errors.add("Wi-Fi scan at index $idx has an empty observationId")
            }
        }

        sensorReadings.forEachIndexed { idx, s ->
            if (s.sessionId != session.sessionId) {
                errors.add("Sensor reading at index $idx has mismatched sessionId '${s.sessionId}' (expected '${session.sessionId}')")
            }
            if (s.observationId.isBlank()) {
                errors.add("Sensor reading at index $idx has an empty observationId")
            }
        }

        photos.forEachIndexed { idx, p ->
            if (p.sessionId != session.sessionId) {
                errors.add("Photo at index $idx has mismatched sessionId '${p.sessionId}' (expected '${session.sessionId}')")
            }
            if (p.photoId.isBlank()) {
                errors.add("Photo at index $idx has an empty photoId")
            }
        }

        annotations.forEachIndexed { idx, a ->
            if (a.sessionId != session.sessionId) {
                errors.add("Annotation at index $idx has mismatched sessionId '${a.sessionId}' (expected '${session.sessionId}')")
            }
            if (a.annotationId.isBlank()) {
                errors.add("Annotation at index $idx has an empty annotationId")
            }
        }

        // 3. Unique Observation ID checks
        val wifiIds = wifiScans.map { it.observationId }
        if (wifiIds.size != wifiIds.distinct().size) {
            errors.add("Duplicate Wi-Fi observation IDs detected")
        }

        val sensorIds = sensorReadings.map { it.observationId }
        if (sensorIds.size != sensorIds.distinct().size) {
            errors.add("Duplicate Sensor observation IDs detected")
        }

        val photoIds = photos.map { it.photoId }
        if (photoIds.size != photoIds.distinct().size) {
            errors.add("Duplicate Photo IDs detected")
        }

        val annotationIds = annotations.map { it.annotationId }
        if (annotationIds.size != annotationIds.distinct().size) {
            errors.add("Duplicate Annotation IDs detected")
        }

        // 4. Photo file existence and mandatory sanitization verification
        photos.forEach { p ->
            val file = File(p.filePath)
            if (!file.exists() || file.length() == 0L) {
                errors.add("Photo file missing or empty on disk: '${p.filePath}' (ID: ${p.photoId})")
            }

            // CRITICAL PRIVACY REQUIREMENT: Reject unsanitized images
            if (!p.isExifSanitized) {
                errors.add("Unsanitized image rejected: '${p.filename}' (ID: ${p.photoId}) is not confirmed EXIF sanitized")
            }
        }

        // 5. Timestamp validation
        if (session.startTimeMs != null && session.endTimeMs != null) {
            if (session.endTimeMs < session.startTimeMs) {
                errors.add("Session endTimeMs (${session.endTimeMs}) is earlier than startTimeMs (${session.startTimeMs})")
            }
        }

        // 6. Record count consistency warnings
        val totalActual = wifiScans.size + sensorReadings.size + photos.size + annotations.size
        if (session.totalObservations != totalActual) {
            warnings.add("Session header metadata count (${session.totalObservations}) differs from actual observation journal count ($totalActual)")
        }

        return ExportValidationResult(
            isValid = errors.isEmpty(),
            errors = errors,
            warnings = warnings
        )
    }
}
