package ai.indoor.app.domain.models

/**
 * Ground-truth location annotations entered manually by the user during survey capture.
 *
 * IMPORTANT: These manual labels are strictly separated from sensor readings and
 * model predictions to ensure rigorous ground-truth integrity without data leakage.
 *
 * @param building Human-readable building name or identifier (e.g. "TI-Engineering")
 * @param floor Floor number or designation (e.g. "2" or "Ground")
 * @param room Specific room, door, or corridor landmark (e.g. "Room 204", "East Entrance")
 * @param notes Optional surveyor notes (e.g. "Near window", "Walking south")
 */
data class ManualLocationLabel(
    val building: String = "",
    val floor: String = "",
    val room: String = "",
    val notes: String = ""
) {
    val isBlank: Boolean
        get() = building.isBlank() && floor.isBlank() && room.isBlank()

    val isPresent: Boolean
        get() = !isBlank

    val formattedDisplay: String
        get() = buildString {
            if (building.isNotBlank()) append("Building: $building")
            if (floor.isNotBlank()) {
                if (isNotEmpty()) append(", ")
                append("Floor: $floor")
            }
            if (room.isNotBlank()) {
                if (isNotEmpty()) append(", ")
                append("Room: $room")
            }
            if (notes.isNotBlank()) {
                if (isNotEmpty()) append(" — ")
                append(notes)
            }
            if (isEmpty()) append("Unlabeled Observation")
        }
}
