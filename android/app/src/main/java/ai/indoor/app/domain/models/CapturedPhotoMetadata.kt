package ai.indoor.app.domain.models

import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone

/**
 * Metadata record for a user-captured, privacy-sanitized indoor landmark photograph.
 *
 * All photos are strictly associated with a survey session and stored in
 * application-private scoped storage without external cloud transmission.
 */
data class CapturedPhotoMetadata(
    val photoId: String,
    val sessionId: String,
    val timestampMs: Long,
    val filename: String,
    val filePath: String,
    val building: String = "",
    val floor: String = "",
    val room: String = "",
    val landmarkType: LandmarkType = LandmarkType.OTHER,
    val description: String = "",
    val notes: String = "",
    val isExifSanitized: Boolean = true,
    val facesDetectedCount: Int = 0,
    val faceBlurApplied: Boolean = false,
    val faceDetectionStatus: FaceDetectionStatus = FaceDetectionStatus.NO_FACES_DETECTED,
    val fileSizeBytes: Long = 0L,
    val imageWidth: Int = 0,
    val imageHeight: Int = 0,
    val isMock: Boolean = false
) {
    val timestampIso: String
        get() {
            val sdf = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.US)
            sdf.timeZone = TimeZone.getTimeZone("UTC")
            return sdf.format(Date(timestampMs))
        }

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

    /**
     * Converts to JSON snippet.
     */
    fun toJson(indent: String = "  "): String {
        return buildString {
            append("$indent{\n")
            append("$indent  \"photo_id\": \"$photoId\",\n")
            append("$indent  \"session_id\": \"$sessionId\",\n")
            append("$indent  \"timestamp_ms\": $timestampMs,\n")
            append("$indent  \"timestamp_iso\": \"$timestampIso\",\n")
            append("$indent  \"filename\": \"${escapeJson(filename)}\",\n")
            append("$indent  \"file_path\": \"${escapeJson(filePath)}\",\n")
            append("$indent  \"building\": \"${escapeJson(building)}\",\n")
            append("$indent  \"floor\": \"${escapeJson(floor)}\",\n")
            append("$indent  \"room\": \"${escapeJson(room)}\",\n")
            append("$indent  \"landmark_type\": \"${landmarkType.code}\",\n")
            append("$indent  \"landmark_display\": \"${escapeJson(landmarkType.displayName)}\",\n")
            append("$indent  \"description\": \"${escapeJson(description)}\",\n")
            append("$indent  \"notes\": \"${escapeJson(notes)}\",\n")
            append("$indent  \"is_exif_sanitized\": $isExifSanitized,\n")
            append("$indent  \"faces_detected_count\": $facesDetectedCount,\n")
            append("$indent  \"face_blur_applied\": $faceBlurApplied,\n")
            append("$indent  \"face_detection_status\": \"${faceDetectionStatus.name}\",\n")
            append("$indent  \"file_size_bytes\": $fileSizeBytes,\n")
            append("$indent  \"image_width\": $imageWidth,\n")
            append("$indent  \"image_height\": $imageHeight,\n")
            append("$indent  \"is_mock\": $isMock\n")
            append("$indent}")
        }
    }

    /**
     * Converts to a standard RFC 4180 CSV row.
     */
    fun toCsvRow(): String {
        return listOf(
            escapeCsv(photoId),
            escapeCsv(sessionId),
            timestampMs.toString(),
            escapeCsv(timestampIso),
            escapeCsv(filename),
            escapeCsv(building),
            escapeCsv(floor),
            escapeCsv(room),
            escapeCsv(landmarkType.code),
            escapeCsv(description),
            escapeCsv(notes),
            isExifSanitized.toString(),
            facesDetectedCount.toString(),
            faceBlurApplied.toString(),
            escapeCsv(faceDetectionStatus.name),
            fileSizeBytes.toString(),
            imageWidth.toString(),
            imageHeight.toString(),
            isMock.toString()
        ).joinToString(",")
    }

    companion object {
        const val CSV_HEADER = "photo_id,session_id,timestamp_ms,timestamp_iso,filename,building,floor,room,landmark_type,description,notes,exif_sanitized,faces_detected,face_blur_applied,face_status,file_size_bytes,width,height,is_mock"

        private fun escapeJson(value: String): String {
            return value.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t")
        }

        private fun escapeCsv(value: String): String {
            val needsQuotes = value.contains(",") || value.contains("\"") || value.contains("\n") || value.contains("\r")
            return if (needsQuotes) {
                "\"" + value.replace("\"", "\"\"") + "\""
            } else {
                value
            }
        }
    }
}

/**
 * Batch container for captured photos, supporting bulk export to JSON and CSV.
 */
data class CapturedPhotoBatch(
    val photos: List<CapturedPhotoMetadata>
) {
    fun toJson(): String {
        return buildString {
            append("{\n")
            append("  \"batch_size\": ${photos.size},\n")
            append("  \"exported_at_ms\": ${System.currentTimeMillis()},\n")
            append("  \"photos\": [\n")
            photos.forEachIndexed { index, photo ->
                append(photo.toJson("    "))
                if (index < photos.size - 1) append(",")
                append("\n")
            }
            append("  ]\n")
            append("}")
        }
    }

    fun toCsv(): String {
        return buildString {
            append(CapturedPhotoMetadata.CSV_HEADER)
            append("\n")
            photos.forEach { photo ->
                append(photo.toCsvRow())
                append("\n")
            }
        }
    }
}
