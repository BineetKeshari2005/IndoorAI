package ai.indoor.app.camera.storage

import android.content.Context
import ai.indoor.app.domain.models.CapturedPhotoBatch
import ai.indoor.app.domain.models.CapturedPhotoMetadata
import ai.indoor.app.domain.models.FaceDetectionStatus
import ai.indoor.app.domain.models.LandmarkType
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

/**
 * Storage manager for captured landmark photographs and associated metadata.
 *
 * Saves images and metadata exclusively inside application-private storage
 * (`context.filesDir/survey_photos/`) with zero cloud synchronization requirements.
 */
class PhotoStorageManager(
    private val context: Context
) {
    private val storageDir: File by lazy {
        File(context.filesDir, "survey_photos").apply { mkdirs() }
    }

    private val indexJsonFile: File by lazy {
        File(storageDir, "survey_photos_index.json")
    }

    private val indexCsvFile: File by lazy {
        File(storageDir, "survey_photos_index.csv")
    }

    private val memoryCache = mutableListOf<CapturedPhotoMetadata>()

    init {
        loadIndex()
    }

    @Synchronized
    fun getPhotosDirectory(): File = storageDir

    @Synchronized
    fun getAllPhotos(): List<CapturedPhotoMetadata> {
        return memoryCache.toList()
    }

    @Synchronized
    fun getPhotosForSession(sessionId: String): List<CapturedPhotoMetadata> {
        return memoryCache.filter { it.sessionId == sessionId }
    }

    @Synchronized
    fun savePhoto(metadata: CapturedPhotoMetadata) {
        memoryCache.removeAll { it.photoId == metadata.photoId }
        memoryCache.add(0, metadata) // newest first
        persistIndex()
    }

    @Synchronized
    fun deletePhoto(photoId: String): Boolean {
        val item = memoryCache.firstOrNull { it.photoId == photoId } ?: return false
        memoryCache.remove(item)

        // Delete physical image file
        try {
            val file = File(item.filePath)
            if (file.exists()) {
                file.delete()
            }
        } catch (_: Exception) {}

        persistIndex()
        return true
    }

    @Synchronized
    fun clearAll() {
        memoryCache.forEach { item ->
            try {
                val file = File(item.filePath)
                if (file.exists()) file.delete()
            } catch (_: Exception) {}
        }
        memoryCache.clear()
        persistIndex()
    }

    @Synchronized
    fun exportBatchJson(sessionId: String? = null): String {
        val targets = if (sessionId != null) {
            memoryCache.filter { it.sessionId == sessionId }
        } else {
            memoryCache
        }
        return CapturedPhotoBatch(targets).toJson()
    }

    @Synchronized
    fun exportBatchCsv(sessionId: String? = null): String {
        val targets = if (sessionId != null) {
            memoryCache.filter { it.sessionId == sessionId }
        } else {
            memoryCache
        }
        return CapturedPhotoBatch(targets).toCsv()
    }

    @Synchronized
    private fun persistIndex() {
        try {
            // Write JSON Index
            val jsonContent = CapturedPhotoBatch(memoryCache).toJson()
            FileOutputStream(indexJsonFile).use { out ->
                out.write(jsonContent.toByteArray(Charsets.UTF_8))
                out.flush()
            }

            // Write CSV Index
            val csvContent = CapturedPhotoBatch(memoryCache).toCsv()
            FileOutputStream(indexCsvFile).use { out ->
                out.write(csvContent.toByteArray(Charsets.UTF_8))
                out.flush()
            }
        } catch (_: IOException) {
            // Log/handle gracefully in edge cases (e.g. disk full)
        }
    }

    @Synchronized
    private fun loadIndex() {
        memoryCache.clear()
        if (!indexJsonFile.exists() || indexJsonFile.length() == 0L) {
            return
        }

        try {
            val text = indexJsonFile.readText(Charsets.UTF_8)
            val root = JSONObject(text)
            val array = root.optJSONArray("photos") ?: JSONArray()
            for (i in 0 until array.length()) {
                val obj = array.getJSONObject(i)
                val photo = CapturedPhotoMetadata(
                    photoId = obj.optString("photo_id"),
                    sessionId = obj.optString("session_id"),
                    timestampMs = obj.optLong("timestamp_ms"),
                    filename = obj.optString("filename"),
                    filePath = obj.optString("file_path"),
                    building = obj.optString("building"),
                    floor = obj.optString("floor"),
                    room = obj.optString("room"),
                    landmarkType = LandmarkType.fromString(obj.optString("landmark_type")),
                    description = obj.optString("description"),
                    notes = obj.optString("notes"),
                    isExifSanitized = obj.optBoolean("is_exif_sanitized", true),
                    facesDetectedCount = obj.optInt("faces_detected_count", 0),
                    faceBlurApplied = obj.optBoolean("face_blur_applied", false),
                    faceDetectionStatus = try {
                        FaceDetectionStatus.valueOf(obj.optString("face_detection_status"))
                    } catch (_: Exception) {
                        FaceDetectionStatus.NO_FACES_DETECTED
                    },
                    fileSizeBytes = obj.optLong("file_size_bytes", 0L),
                    imageWidth = obj.optInt("image_width", 0),
                    imageHeight = obj.optInt("image_height", 0),
                    isMock = obj.optBoolean("is_mock", false)
                )
                memoryCache.add(photo)
            }
        } catch (_: Exception) {
            // Failed reading corrupted index file
        }
    }
}
