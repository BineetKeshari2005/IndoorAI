package ai.indoor.app.domain.models

/**
 * Execution status for on-device privacy face detection.
 */
enum class FaceDetectionStatus(val description: String) {
    NOT_ATTEMPTED("Detection not executed"),
    SUCCEEDED("Faces detected"),
    NO_FACES_DETECTED("No faces detected in scene"),
    UNAVAILABLE("On-device face detector unavailable"),
    FAILED("Face detection encountered an error")
}
