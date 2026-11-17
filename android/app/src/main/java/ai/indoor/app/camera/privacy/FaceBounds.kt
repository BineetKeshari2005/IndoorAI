package ai.indoor.app.camera.privacy

/**
 * Geometric bounds of an on-device detected face in image coordinates.
 */
data class FaceBounds(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
    val confidence: Float = 1.0f
) {
    val width: Float get() = (right - left).coerceAtLeast(0f)
    val height: Float get() = (bottom - top).coerceAtLeast(0f)
}
