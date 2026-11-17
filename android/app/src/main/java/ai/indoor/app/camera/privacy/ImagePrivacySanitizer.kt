package ai.indoor.app.camera.privacy

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import androidx.exifinterface.media.ExifInterface
import ai.indoor.app.domain.models.FaceDetectionStatus
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

/**
 * Result bundle for an image sanitization run.
 */
data class SanitizationResult(
    val success: Boolean,
    val sanitizedFile: File?,
    val isExifSanitized: Boolean,
    val facesDetectedCount: Int,
    val faceBlurApplied: Boolean,
    val faceDetectionStatus: FaceDetectionStatus,
    val imageWidth: Int = 0,
    val imageHeight: Int = 0,
    val fileSizeBytes: Long = 0L,
    val errorMessage: String? = null
)

/**
 * Privacy sanitization engine for indoor survey photography.
 *
 * Responsibilities:
 * 1. Strips all EXIF GPS coordinates and geolocation tags.
 * 2. Removes identifying device metadata (make, model, serials, software tags).
 * 3. Preserves upright image orientation by rotating pixel buffer directly.
 * 4. Executes on-device bystander face detection and optional pixelation blur.
 * 5. Re-encodes cleanly to a new file, ensuring original raw file is never overwritten unless successful.
 */
class ImagePrivacySanitizer(
    private val faceDetector: OnDeviceFaceDetector = AndroidMediaFaceDetector()
) {

    /**
     * Sanitizes a raw captured image file into an app-private sanitized destination file.
     *
     * @param rawFile The raw captured image file from the camera.
     * @param destinationFile The target destination file for the sanitized image.
     * @param applyFaceBlur Whether to apply on-device blur if faces are detected.
     * @return SanitizationResult describing operation success, sanitization flags, and dimensions.
     */
    fun sanitize(
        rawFile: File,
        destinationFile: File,
        applyFaceBlur: Boolean = true
    ): SanitizationResult {
        if (!rawFile.exists() || rawFile.length() == 0L) {
            return SanitizationResult(
                success = false,
                sanitizedFile = null,
                isExifSanitized = false,
                facesDetectedCount = 0,
                faceBlurApplied = false,
                faceDetectionStatus = FaceDetectionStatus.FAILED,
                errorMessage = "Source image file does not exist or is empty: ${rawFile.absolutePath}"
            )
        }

        // Temporary file for atomic write safety
        val tempOutputFile = File(destinationFile.parentFile ?: rawFile.parentFile, "${destinationFile.name}.tmp")

        var rawBitmap: Bitmap? = null
        var orientedBitmap: Bitmap? = null
        var finalBitmap: Bitmap? = null

        try {
            // 1. Read EXIF orientation prior to decoding
            val orientation = try {
                val rawExif = ExifInterface(rawFile.absolutePath)
                rawExif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
            } catch (_: Exception) {
                ExifInterface.ORIENTATION_NORMAL
            }

            val rotationDegrees = when (orientation) {
                ExifInterface.ORIENTATION_ROTATE_90 -> 90f
                ExifInterface.ORIENTATION_ROTATE_180 -> 180f
                ExifInterface.ORIENTATION_ROTATE_270 -> 270f
                else -> 0f
            }

            // 2. Decode raw image bitmap
            rawBitmap = BitmapFactory.decodeFile(rawFile.absolutePath)
                ?: return SanitizationResult(
                    success = false,
                    sanitizedFile = null,
                    isExifSanitized = false,
                    facesDetectedCount = 0,
                    faceBlurApplied = false,
                    faceDetectionStatus = FaceDetectionStatus.FAILED,
                    errorMessage = "Failed to decode raw image into Bitmap"
                )

            // 3. Normalize pixel orientation physically
            orientedBitmap = if (rotationDegrees != 0f) {
                val matrix = Matrix().apply { postRotate(rotationDegrees) }
                val rotated = Bitmap.createBitmap(rawBitmap, 0, 0, rawBitmap.width, rawBitmap.height, matrix, true)
                if (rotated != rawBitmap) rawBitmap.recycle()
                rotated
            } else {
                rawBitmap
            }

            // 4. On-device face detection
            val faceResult = try {
                faceDetector.detectFaces(orientedBitmap)
            } catch (e: Throwable) {
                FaceDetectionResult(
                    status = FaceDetectionStatus.FAILED,
                    faces = emptyList(),
                    errorMessage = e.message
                )
            }

            // 5. Apply privacy face blur if requested
            val facesCount = faceResult.faces.size
            val shouldBlur = applyFaceBlur && facesCount > 0

            finalBitmap = if (shouldBlur) {
                FaceBlurApplicator.applyFaceObfuscation(
                    source = orientedBitmap,
                    faces = faceResult.faces,
                    style = FaceBlurApplicator.ObfuscationStyle.PIXELATE
                )
            } else {
                orientedBitmap
            }

            // 6. Write cleanly re-encoded JPEG to temporary file
            tempOutputFile.parentFile?.mkdirs()
            FileOutputStream(tempOutputFile).use { out ->
                val compressed = finalBitmap.compress(Bitmap.CompressFormat.JPEG, 92, out)
                if (!compressed) {
                    throw IOException("Bitmap compression to JPEG failed")
                }
                out.flush()
            }

            // 7. Strip any residual EXIF tags and set orientation to normal
            stripExifMetadata(tempOutputFile)

            // 8. Atomic rename of temp file to destination
            if (destinationFile.exists()) {
                destinationFile.delete()
            }
            val renamed = tempOutputFile.renameTo(destinationFile)
            if (!renamed) {
                // Fallback copy if renameTo fails across file systems
                tempOutputFile.copyTo(destinationFile, overwrite = true)
                tempOutputFile.delete()
            }

            val finalWidth = finalBitmap.width
            val finalHeight = finalBitmap.height
            val finalSize = destinationFile.length()

            return SanitizationResult(
                success = true,
                sanitizedFile = destinationFile,
                isExifSanitized = true,
                facesDetectedCount = facesCount,
                faceBlurApplied = shouldBlur,
                faceDetectionStatus = faceResult.status,
                imageWidth = finalWidth,
                imageHeight = finalHeight,
                fileSizeBytes = finalSize
            )

        } catch (e: Throwable) {
            if (tempOutputFile.exists()) {
                tempOutputFile.delete()
            }
            return SanitizationResult(
                success = false,
                sanitizedFile = null,
                isExifSanitized = false,
                facesDetectedCount = 0,
                faceBlurApplied = false,
                faceDetectionStatus = FaceDetectionStatus.FAILED,
                errorMessage = "Sanitization failed: ${e.message}"
            )
        } finally {
            // Memory cleanup
            if (rawBitmap != null && !rawBitmap.isRecycled && rawBitmap != orientedBitmap) {
                rawBitmap.recycle()
            }
            if (orientedBitmap != null && !orientedBitmap.isRecycled && orientedBitmap != finalBitmap) {
                orientedBitmap.recycle()
            }
            if (finalBitmap != null && !finalBitmap.isRecycled) {
                finalBitmap.recycle()
            }
        }
    }

    /**
     * Strips GPS tags and identifying attributes from an image file using ExifInterface.
     */
    fun stripExifMetadata(file: File) {
        try {
            val exif = ExifInterface(file.absolutePath)

            // Strip GPS attributes
            val gpsTags = listOf(
                ExifInterface.TAG_GPS_LATITUDE,
                ExifInterface.TAG_GPS_LONGITUDE,
                ExifInterface.TAG_GPS_LATITUDE_REF,
                ExifInterface.TAG_GPS_LONGITUDE_REF,
                ExifInterface.TAG_GPS_ALTITUDE,
                ExifInterface.TAG_GPS_ALTITUDE_REF,
                ExifInterface.TAG_GPS_TIMESTAMP,
                ExifInterface.TAG_GPS_DATESTAMP,
                ExifInterface.TAG_GPS_PROCESSING_METHOD,
                ExifInterface.TAG_GPS_AREA_INFORMATION
            )
            for (tag in gpsTags) {
                exif.setAttribute(tag, null)
            }

            // Strip hardware / software identifying attributes
            val deviceTags = listOf(
                ExifInterface.TAG_MAKE,
                ExifInterface.TAG_MODEL,
                ExifInterface.TAG_SOFTWARE,
                ExifInterface.TAG_ARTIST,
                ExifInterface.TAG_USER_COMMENT,
                ExifInterface.TAG_DEVICE_SETTING_DESCRIPTION
            )
            for (tag in deviceTags) {
                exif.setAttribute(tag, null)
            }

            // Orientation was normalized into pixels, set to normal
            exif.setAttribute(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_NORMAL.toString()
            )

            exif.saveAttributes()
        } catch (_: Exception) {
            // If ExifInterface is unavailable or throws, re-encoded JPEG is already stripped
        }
    }
}
