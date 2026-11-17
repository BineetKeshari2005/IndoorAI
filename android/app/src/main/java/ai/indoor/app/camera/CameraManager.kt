package ai.indoor.app.camera

import androidx.camera.view.PreviewView
import androidx.lifecycle.LifecycleOwner
import java.io.File

/**
 * Camera capture callback interface.
 */
interface ImageCaptureCallback {
    fun onPhotoCaptured(file: File)
    fun onError(exception: Throwable)
}

/**
 * Abstraction for camera operations, decoupling UI and CameraX hardware execution.
 */
interface CameraManager {
    /**
     * Checks if physical or virtual camera hardware is available.
     */
    fun isCameraAvailable(): Boolean

    /**
     * Explicitly starts the live camera preview.
     * STRICT REQUIREMENT: Must only be invoked after explicit user action.
     */
    fun startPreview(lifecycleOwner: LifecycleOwner, previewView: PreviewView)

    /**
     * Stops the camera preview and releases camera resources.
     */
    fun stopPreview()

    /**
     * Captures a single image frame on demand.
     */
    fun capturePhoto(callback: ImageCaptureCallback)
}
