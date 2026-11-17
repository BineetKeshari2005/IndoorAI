package ai.indoor.app.camera.privacy

import android.graphics.Bitmap
import ai.indoor.app.domain.models.FaceDetectionStatus

/**
 * Result bundle returned by on-device face detection.
 */
data class FaceDetectionResult(
    val status: FaceDetectionStatus,
    val faces: List<FaceBounds>,
    val errorMessage: String? = null
)

/**
 * Abstraction for purely client-side, on-device face detection.
 *
 * Guaranteed to execute offline on the handset with zero network or cloud interaction.
 */
interface OnDeviceFaceDetector {
    fun isAvailable(): Boolean
    fun detectFaces(bitmap: Bitmap): FaceDetectionResult
}
