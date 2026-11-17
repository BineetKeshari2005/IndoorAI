package ai.indoor.app.domain.models

import ai.indoor.app.domain.timestamp.TimestampRecord
import java.util.UUID

/**
 * Categorization for manual ground-truth annotations and architectural event tags.
 */
enum class AnnotationType(val displayName: String) {
    CORRIDOR("Corridor"),
    DOOR("Door"),
    ROOM_ENTRANCE("Room Entrance"),
    STAIRCASE("Staircase"),
    ELEVATOR("Elevator"),
    LANDMARK("Landmark"),
    REFERENCE_POINT("Known Reference Point"),
    LOCATION_CORRECTION("Manual Location Correction"),
    NOTE("General Surveyor Note")
}

/**
 * Surveyor-entered ground-truth annotation recorded during an active or paused survey session.
 *
 * Explicitly segregated from automated model inferences to prevent data leakage.
 */
data class SurveyAnnotation(
    val annotationId: String = "ANN-" + UUID.randomUUID().toString().take(8).uppercase(),
    val sessionId: String,
    val timestamp: TimestampRecord,
    val type: AnnotationType,
    val description: String,
    val building: String = "",
    val floor: String = "",
    val room: String = "",
    val relatedObservationId: String? = null,
    val isMock: Boolean = false
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
            escapeCsv(annotationId),
            escapeCsv(sessionId),
            timestamp.originalTimestamp.toString(),
            timestamp.timeSource.name,
            timestamp.normalizedEpochMs?.toString() ?: "",
            escapeCsv(timestamp.normalizedIso ?: ""),
            timestamp.syncMethod.name,
            timestamp.uncertaintyMs?.toString() ?: "",
            escapeCsv(type.name),
            escapeCsv(type.displayName),
            escapeCsv(description),
            escapeCsv(building),
            escapeCsv(floor),
            escapeCsv(room),
            escapeCsv(relatedObservationId ?: ""),
            isMock.toString()
        ).joinToString(",")
    }

    companion object {
        const val CSV_HEADER =
            "annotation_id,session_id,original_timestamp,time_source,normalized_epoch_ms,normalized_iso,sync_method,uncertainty_ms,type_code,type_name,description,building,floor,room,related_observation_id,is_mock"

        private fun escapeCsv(value: String): String {
            return if (value.contains(",") || value.contains("\"") || value.contains("\n") || value.contains("\r")) {
                "\"" + value.replace("\"", "\"\"") + "\""
            } else {
                value
            }
        }
    }
}
