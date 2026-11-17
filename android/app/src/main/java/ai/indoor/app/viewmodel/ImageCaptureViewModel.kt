package ai.indoor.app.viewmodel

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.camera.view.PreviewView
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import ai.indoor.app.camera.CameraManager
import ai.indoor.app.camera.ImageCaptureCallback
import ai.indoor.app.camera.privacy.AndroidMediaFaceDetector
import ai.indoor.app.camera.privacy.FaceBlurApplicator
import ai.indoor.app.camera.privacy.FaceBounds
import ai.indoor.app.camera.privacy.ImagePrivacySanitizer
import ai.indoor.app.camera.privacy.OnDeviceFaceDetector
import ai.indoor.app.camera.storage.PhotoStorageManager
import ai.indoor.app.domain.models.CapturedPhotoMetadata
import ai.indoor.app.domain.models.FaceDetectionStatus
import ai.indoor.app.domain.models.LandmarkType
import ai.indoor.app.domain.usecases.ManageCaptureSessionUseCase
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID

/**
 * Camera preview status. Preview NEVER starts automatically on app launch.
 */
enum class PreviewState {
    INACTIVE,
    STARTING,
    ACTIVE,
    PERMISSION_REQUIRED,
    UNAVAILABLE,
    ERROR
}

/**
 * Capture and review flow state machine.
 */
sealed class CaptureFlowState {
    data object Idle : CaptureFlowState()
    data object Capturing : CaptureFlowState()
    data class Reviewing(
        val rawTempFile: File,
        val previewBitmap: Bitmap?,
        val detectedFacesCount: Int,
        val detectedFaces: List<FaceBounds>,
        val faceStatus: FaceDetectionStatus,
        val isBlurEnabled: Boolean,
        val draft: LandmarkMetadataDraft
    ) : CaptureFlowState()
    data object Sanitizing : CaptureFlowState()
    data class Saved(val photo: CapturedPhotoMetadata) : CaptureFlowState()
    data class Error(val message: String) : CaptureFlowState()
}

/**
 * Editable landmark annotations draft.
 */
data class LandmarkMetadataDraft(
    val building: String = "",
    val floor: String = "",
    val room: String = "",
    val landmarkType: LandmarkType = LandmarkType.CORRIDOR,
    val description: String = "",
    val notes: String = ""
)

/**
 * ViewModel managing user-initiated landmark photography, on-device face blur,
 * EXIF sanitization, app-scoped storage, and survey session accounting.
 */
class ImageCaptureViewModel(
    private val cameraManager: CameraManager? = null,
    private val storageManager: PhotoStorageManager? = null,
    private val sanitizer: ImagePrivacySanitizer = ImagePrivacySanitizer(),
    private val faceDetector: OnDeviceFaceDetector = AndroidMediaFaceDetector(),
    private val sessionUseCase: ManageCaptureSessionUseCase? = null
) : ViewModel() {

    private val _previewState = MutableStateFlow(PreviewState.INACTIVE)
    val previewState: StateFlow<PreviewState> = _previewState.asStateFlow()

    private val _captureFlowState = MutableStateFlow<CaptureFlowState>(CaptureFlowState.Idle)
    val captureFlowState: StateFlow<CaptureFlowState> = _captureFlowState.asStateFlow()

    private val _capturedPhotos = MutableStateFlow<List<CapturedPhotoMetadata>>(emptyList())
    val capturedPhotos: StateFlow<List<CapturedPhotoMetadata>> = _capturedPhotos.asStateFlow()

    private val _currentDraft = MutableStateFlow(LandmarkMetadataDraft())
    val currentDraft: StateFlow<LandmarkMetadataDraft> = _currentDraft.asStateFlow()

    private val _faceBlurEnabled = MutableStateFlow(true)
    val faceBlurEnabled: StateFlow<Boolean> = _faceBlurEnabled.asStateFlow()

    private val _statusMessage = MutableStateFlow<String?>(null)
    val statusMessage: StateFlow<String?> = _statusMessage.asStateFlow()

    init {
        refreshPhotos()
    }

    fun refreshPhotos() {
        val photos = storageManager?.getAllPhotos() ?: emptyList()
        _capturedPhotos.value = photos
    }

    /**
     * User explicitly requests camera preview.
     */
    fun startCameraPreview(lifecycleOwner: LifecycleOwner, previewView: PreviewView, hasPermission: Boolean) {
        if (!hasPermission) {
            _previewState.value = PreviewState.PERMISSION_REQUIRED
            return
        }

        if (cameraManager == null || !cameraManager.isCameraAvailable()) {
            _previewState.value = PreviewState.UNAVAILABLE
            return
        }

        _previewState.value = PreviewState.STARTING
        try {
            cameraManager.startPreview(lifecycleOwner, previewView)
            _previewState.value = PreviewState.ACTIVE
        } catch (e: Exception) {
            _previewState.value = PreviewState.ERROR
            _statusMessage.value = "Failed to start camera: ${e.message}"
        }
    }

    /**
     * User explicitly stops camera preview.
     */
    fun stopCameraPreview() {
        cameraManager?.stopPreview()
        _previewState.value = PreviewState.INACTIVE
    }

    /**
     * Captures an image frame on demand.
     */
    fun takePhoto() {
        val manager = cameraManager
        if (manager == null) {
            _statusMessage.value = "Camera manager is not configured"
            return
        }

        _captureFlowState.value = CaptureFlowState.Capturing

        manager.capturePhoto(object : ImageCaptureCallback {
            override fun onPhotoCaptured(file: File) {
                processRawPhotoForReview(file)
            }

            override fun onError(exception: Throwable) {
                _captureFlowState.value = CaptureFlowState.Error("Photo capture failed: ${exception.message}")
            }
        })
    }

    private fun processRawPhotoForReview(rawFile: File) {
        viewModelScope.launch(Dispatchers.Default) {
            try {
                // Decode preview bitmap
                val rawBitmap = BitmapFactory.decodeFile(rawFile.absolutePath)
                val faceResult = faceDetector.detectFaces(rawBitmap)
                val shouldBlur = _faceBlurEnabled.value && faceResult.faces.isNotEmpty()

                val displayBitmap = if (shouldBlur) {
                    FaceBlurApplicator.applyFaceObfuscation(rawBitmap, faceResult.faces)
                } else {
                    rawBitmap
                }

                _captureFlowState.value = CaptureFlowState.Reviewing(
                    rawTempFile = rawFile,
                    previewBitmap = displayBitmap,
                    detectedFacesCount = faceResult.faces.size,
                    detectedFaces = faceResult.faces,
                    faceStatus = faceResult.status,
                    isBlurEnabled = _faceBlurEnabled.value,
                    draft = _currentDraft.value
                )
            } catch (e: Throwable) {
                _captureFlowState.value = CaptureFlowState.Error("Failed to prepare review: ${e.message}")
            }
        }
    }

    /**
     * Surveyor retakes photo, discarding the temporary capture.
     */
    fun retakePhoto() {
        val current = _captureFlowState.value
        if (current is CaptureFlowState.Reviewing) {
            try {
                if (current.rawTempFile.exists()) {
                    current.rawTempFile.delete()
                }
            } catch (_: Exception) {}
        }
        _captureFlowState.value = CaptureFlowState.Idle
    }

    /**
     * Updates editable landmark draft annotations.
     */
    fun updateDraft(
        building: String = _currentDraft.value.building,
        floor: String = _currentDraft.value.floor,
        room: String = _currentDraft.value.room,
        landmarkType: LandmarkType = _currentDraft.value.landmarkType,
        description: String = _currentDraft.value.description,
        notes: String = _currentDraft.value.notes
    ) {
        val updated = LandmarkMetadataDraft(
            building = building,
            floor = floor,
            room = room,
            landmarkType = landmarkType,
            description = description,
            notes = notes
        )
        _currentDraft.value = updated

        val currentFlow = _captureFlowState.value
        if (currentFlow is CaptureFlowState.Reviewing) {
            _captureFlowState.value = currentFlow.copy(draft = updated)
        }
    }

    /**
     * Toggles face blur on/off during review.
     */
    fun toggleFaceBlur(enabled: Boolean) {
        _faceBlurEnabled.value = enabled
        val current = _captureFlowState.value
        if (current is CaptureFlowState.Reviewing) {
            viewModelScope.launch(Dispatchers.Default) {
                val rawBitmap = BitmapFactory.decodeFile(current.rawTempFile.absolutePath)
                val displayBitmap = if (enabled && current.detectedFaces.isNotEmpty()) {
                    FaceBlurApplicator.applyFaceObfuscation(rawBitmap, current.detectedFaces)
                } else {
                    rawBitmap
                }
                _captureFlowState.value = current.copy(
                    isBlurEnabled = enabled,
                    previewBitmap = displayBitmap
                )
            }
        }
    }

    /**
     * Confirms and saves the landmark photograph:
     * 1. Sanitizes EXIF and strips GPS coordinates.
     * 2. Applies on-device face blur if enabled.
     * 3. Saves to application-private storage directory.
     * 4. Associates with active session and increments photo counter.
     */
    fun confirmAndSavePhoto(activeSessionId: String? = null) {
        val current = _captureFlowState.value as? CaptureFlowState.Reviewing ?: return
        _captureFlowState.value = CaptureFlowState.Sanitizing

        viewModelScope.launch(Dispatchers.IO) {
            try {
                val storage = storageManager
                val destDir = storage?.getPhotosDirectory() ?: current.rawTempFile.parentFile ?: current.rawTempFile
                val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
                val photoId = "IMG-" + UUID.randomUUID().toString().take(8).uppercase()
                val destFilename = "PHOTO_${timeStamp}_sanitized.jpg"
                val destFile = File(destDir, destFilename)

                // Run privacy sanitization
                val result = sanitizer.sanitize(
                    rawFile = current.rawTempFile,
                    destinationFile = destFile,
                    applyFaceBlur = current.isBlurEnabled
                )

                if (!result.success || result.sanitizedFile == null) {
                    _captureFlowState.value = CaptureFlowState.Error(
                        result.errorMessage ?: "Sanitization failed: unable to process image safely"
                    )
                    return@launch
                }

                // Delete raw temporary capture file
                try {
                    if (current.rawTempFile.exists()) {
                        current.rawTempFile.delete()
                    }
                } catch (_: Exception) {}

                val assignedSessionId = activeSessionId
                    ?: sessionUseCase?.currentSession?.value?.sessionId
                    ?: "SES-STANDALONE"

                val metadata = CapturedPhotoMetadata(
                    photoId = photoId,
                    sessionId = assignedSessionId,
                    timestampMs = System.currentTimeMillis(),
                    filename = destFilename,
                    filePath = destFile.absolutePath,
                    building = current.draft.building,
                    floor = current.draft.floor,
                    room = current.draft.room,
                    landmarkType = current.draft.landmarkType,
                    description = current.draft.description,
                    notes = current.draft.notes,
                    isExifSanitized = result.isExifSanitized,
                    facesDetectedCount = result.facesDetectedCount,
                    faceBlurApplied = result.faceBlurApplied,
                    faceDetectionStatus = result.faceDetectionStatus,
                    fileSizeBytes = result.fileSizeBytes,
                    imageWidth = result.imageWidth,
                    imageHeight = result.imageHeight,
                    isMock = current.rawTempFile.name.contains("MOCK")
                )

                // Save metadata record to persistent storage index
                storage?.savePhoto(metadata)

                // Increment session photo telemetry
                sessionUseCase?.recordPhoto()

                withContext(Dispatchers.Main) {
                    refreshPhotos()
                    _captureFlowState.value = CaptureFlowState.Saved(metadata)
                    _statusMessage.value = "Photo saved successfully (${metadata.displayLocation})"
                }
            } catch (e: Throwable) {
                withContext(Dispatchers.Main) {
                    _captureFlowState.value = CaptureFlowState.Error("Failed saving photo: ${e.message}")
                }
            }
        }
    }

    /**
     * Resets capture state back to Idle for the next capture.
     */
    fun resetCaptureState() {
        _captureFlowState.value = CaptureFlowState.Idle
    }

    /**
     * Deletes a captured photo and its metadata.
     */
    fun deletePhoto(photoId: String) {
        storageManager?.deletePhoto(photoId)
        refreshPhotos()
        _statusMessage.value = "Photo deleted"
    }

    fun exportJson(sessionId: String? = null): String {
        return storageManager?.exportBatchJson(sessionId) ?: "{\n  \"photos\": []\n}"
    }

    fun exportCsv(sessionId: String? = null): String {
        return storageManager?.exportBatchCsv(sessionId) ?: CapturedPhotoMetadata.CSV_HEADER
    }

    fun clearStatusMessage() {
        _statusMessage.value = null
    }

    override fun onCleared() {
        super.onCleared()
        cameraManager?.stopPreview()
    }
}
