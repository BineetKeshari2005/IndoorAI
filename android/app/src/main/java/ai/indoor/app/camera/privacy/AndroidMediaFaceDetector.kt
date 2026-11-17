package ai.indoor.app.camera.privacy

import android.graphics.Bitmap
import android.graphics.PointF
import android.media.FaceDetector
import ai.indoor.app.domain.models.FaceDetectionStatus

/**
 * Native Android framework face detector utilizing android.media.FaceDetector.
 *
 * Runs 100% locally and offline without external ML dependencies or Google Play Services.
 */
class AndroidMediaFaceDetector(
    private val maxFaces: Int = 10
) : OnDeviceFaceDetector {

    override fun isAvailable(): Boolean = true

    override fun detectFaces(bitmap: Bitmap): FaceDetectionResult {
        return try {
            // android.media.FaceDetector requires Bitmap.Config.RGB_565 and even width
            val evenWidth = if (bitmap.width % 2 == 0) bitmap.width else bitmap.width - 1
            if (evenWidth <= 0 || bitmap.height <= 0) {
                return FaceDetectionResult(
                    status = FaceDetectionStatus.FAILED,
                    faces = emptyList(),
                    errorMessage = "Invalid image dimensions for face detection: ${bitmap.width}x${bitmap.height}"
                )
            }

            val rgb565Bitmap = if (bitmap.config != Bitmap.Config.RGB_565 || bitmap.width != evenWidth) {
                val scaled = Bitmap.createScaledBitmap(bitmap, evenWidth, bitmap.height, false)
                val converted = scaled.copy(Bitmap.Config.RGB_565, false)
                if (scaled != bitmap && scaled != converted) scaled.recycle()
                converted
            } else {
                bitmap
            }

            val detector = FaceDetector(evenWidth, rgb565Bitmap.height, maxFaces)
            val detectedFaces = arrayOfNulls<FaceDetector.Face>(maxFaces)
            val count = detector.findFaces(rgb565Bitmap, detectedFaces)

            if (rgb565Bitmap != bitmap) {
                rgb565Bitmap.recycle()
            }

            if (count > 0) {
                val bounds = mutableListOf<FaceBounds>()
                val midPoint = PointF()
                for (i in 0 until count) {
                    val face = detectedFaces[i] ?: continue
                    face.getMidPoint(midPoint)
                    val eyesDist = face.eyesDistance()
                    val confidence = face.confidence()

                    // Estimate bounding box around eyes midpoint
                    val faceHalfWidth = eyesDist * 1.15f
                    val faceHalfHeight = eyesDist * 1.45f

                    val left = (midPoint.x - faceHalfWidth).coerceAtLeast(0f)
                    val top = (midPoint.y - faceHalfHeight * 0.85f).coerceAtLeast(0f)
                    val right = (midPoint.x + faceHalfWidth).coerceAtMost(bitmap.width.toFloat())
                    val bottom = (midPoint.y + faceHalfHeight * 1.35f).coerceAtMost(bitmap.height.toFloat())

                    bounds.add(FaceBounds(left, top, right, bottom, confidence))
                }
                FaceDetectionResult(FaceDetectionStatus.SUCCEEDED, bounds)
            } else {
                FaceDetectionResult(FaceDetectionStatus.NO_FACES_DETECTED, emptyList())
            }
        } catch (e: Throwable) {
            FaceDetectionResult(FaceDetectionStatus.FAILED, emptyList(), e.message ?: "Unknown error")
        }
    }
}
