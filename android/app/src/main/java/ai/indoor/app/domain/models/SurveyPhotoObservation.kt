package ai.indoor.app.domain.models

import ai.indoor.app.domain.timestamp.TimestampRecord

/**
 * Unified landmark photo observation record referencing a local sanitized image file.
 *
 * Enforces:
 * 1. Image binary data is NEVER embedded directly into session records or database tables.
 * 2. References confirmed sanitized image file paths.
 * 3. Records on-device face blur and EXIF sanitization provenance.
 */
data class SurveyPhotoObservation(
    val photoId: String,
    val sessionId: String,
    val timestamp: TimestampRecord,
    val filename: String,
    val filePath: String,
    val fileSizeBytes: Long,
    val imageWidth: Int,
    val imageHeight: Int,
    val building: String = "",
    val floor: String = "",
    val room: String = "",
    val landmarkType: LandmarkType,
    val description: String = "",
    val notes: String = "",
    val isExifSanitized: Boolean,
    val facesDetectedCount: Int,
    val faceBlurApplied: Boolean,
    val faceDetectionStatus: FaceDetectionStatus,
    val isMock: Boolean
) {
    val displayLocation: String
        get() = buildString {
            if (building.isNotBlank()) append(building)
            if (floor.isNotBlank()) {
                if (isNotEmpty()) append(", Floor ")
                append(floor)
            }
            if (room.isNotBlank()) {
                if (isNotEmpty()) append(", Room ")
                append(room)
            }
            if (isEmpty()) append("Unlabeled Location")
        }

    fun toCsvRow(): String {
        return listOf(
            escapeCsv(photoId),
            escapeCsv(sessionId),
            timestamp.originalTimestamp.toString(),
            timestamp.timeSource.name,
            timestamp.normalizedEpochMs?.toString() ?: "",
            escapeCsv(timestamp.normalizedIso ?: ""),
            timestamp.syncMethod.name,
            timestamp.uncertaintyMs?.toString() ?: "",
            escapeCsv(filename),
            escapeCsv(filePath),
            fileSizeBytes.toString(),
            imageWidth.toString(),
            imageHeight.toString(),
            escapeCsv(building),
            escapeCsv(floor),
            escapeCsv(room),
            escapeCsv(landmarkType.code),
            escapeCsv(landmarkType.displayName),
            escapeCsv(description),
            escapeCsv(notes),
            isExifSanitized.toString(),
            facesDetectedCount.toString(),
            faceBlurApplied.toString(),
            faceDetectionStatus.name,
            isMock.toString()
        ).joinToString(",")
    }

    companion object {
        const val CSV_HEADER =
            "photo_id,session_id,original_timestamp,time_source,normalized_epoch_ms,normalized_iso,sync_method,uncertainty_ms,filename,file_path,file_size_bytes,image_width,image_height,building,floor,room,landmark_type_code,landmark_type_name,description,notes,is_exif_sanitized,faces_detected_count,face_blur_applied,face_detection_status,is_mock"

        private fun escapeCsv(value: String): String {
            return if (value.contains(",") || value.contains("\"") || value.contains("\n") || value.contains("\r")) {
                "\"" + value.replace("\"", "\"\"") + "\""
            } else {
                value
            }
        }
    }
}
