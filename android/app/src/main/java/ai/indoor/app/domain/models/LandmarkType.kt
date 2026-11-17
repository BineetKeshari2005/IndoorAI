package ai.indoor.app.domain.models

/**
 * Categorization of indoor physical features and architectural landmarks.
 *
 * Used by surveyors during ground-truth visual data collection to label
 * visual transitions, structural boundaries, and signage.
 */
enum class LandmarkType(val displayName: String, val code: String) {
    CORRIDOR("Corridor", "CORRIDOR"),
    DOOR("Door", "DOOR"),
    STAIRCASE("Staircase", "STAIRCASE"),
    ELEVATOR("Elevator", "ELEVATOR"),
    ROOM_SIGN("Room sign", "ROOM_SIGN"),
    ENTRANCE("Entrance", "ENTRANCE"),
    EXIT("Exit", "EXIT"),
    LOBBY("Lobby", "LOBBY"),
    OTHER("Other", "OTHER");

    companion object {
        fun fromString(value: String): LandmarkType {
            return entries.firstOrNull {
                it.name.equals(value, ignoreCase = true) ||
                it.displayName.equals(value, ignoreCase = true) ||
                it.code.equals(value, ignoreCase = true)
            } ?: OTHER
        }
    }
}
