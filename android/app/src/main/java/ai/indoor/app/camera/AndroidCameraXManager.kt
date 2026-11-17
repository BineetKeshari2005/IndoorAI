package ai.indoor.app.camera

import android.content.Context
import android.content.pm.PackageManager
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Concrete CameraX implementation providing on-demand preview and user-initiated shutter capture.
 */
class AndroidCameraXManager(
    private val context: Context
) : CameraManager {

    private var cameraProvider: ProcessCameraProvider? = null
    private var imageCapture: ImageCapture? = null
    private var preview: Preview? = null

    override fun isCameraAvailable(): Boolean {
        return context.packageManager.hasSystemFeature(PackageManager.FEATURE_CAMERA_ANY)
    }

    override fun startPreview(lifecycleOwner: LifecycleOwner, previewView: PreviewView) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        val mainExecutor = ContextCompat.getMainExecutor(context)

        cameraProviderFuture.addListener({
            try {
                val provider = cameraProviderFuture.get()
                cameraProvider = provider

                // Unbind any previous use cases
                provider.unbindAll()

                // Configure Preview
                val newPreview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }
                preview = newPreview

                // Configure ImageCapture for low-latency manual snapshot
                val newImageCapture = ImageCapture.Builder()
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                    .build()
                imageCapture = newImageCapture

                // Select back camera by default
                val cameraSelector = if (provider.hasCamera(CameraSelector.DEFAULT_BACK_CAMERA)) {
                    CameraSelector.DEFAULT_BACK_CAMERA
                } else if (provider.hasCamera(CameraSelector.DEFAULT_FRONT_CAMERA)) {
                    CameraSelector.DEFAULT_FRONT_CAMERA
                } else {
                    return@addListener
                }

                // Bind to lifecycle
                provider.bindToLifecycle(
                    lifecycleOwner,
                    cameraSelector,
                    newPreview,
                    newImageCapture
                )
            } catch (e: Exception) {
                // Handled gracefully if camera is in use or unavailable
            }
        }, mainExecutor)
    }

    override fun stopPreview() {
        try {
            cameraProvider?.unbindAll()
            cameraProvider = null
            imageCapture = null
            preview = null
        } catch (_: Exception) {}
    }

    override fun capturePhoto(callback: ImageCaptureCallback) {
        val capture = imageCapture
        if (capture == null) {
            callback.onError(IllegalStateException("Camera capture is not initialized or preview is stopped"))
            return
        }

        val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val tempFile = try {
            File.createTempFile("RAW_${timeStamp}_", ".jpg", context.cacheDir)
        } catch (e: Exception) {
            callback.onError(e)
            return
        }

        val outputOptions = ImageCapture.OutputFileOptions.Builder(tempFile).build()
        val mainExecutor = ContextCompat.getMainExecutor(context)

        capture.takePicture(
            outputOptions,
            mainExecutor,
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {
                    callback.onPhotoCaptured(tempFile)
                }

                override fun onError(exception: ImageCaptureException) {
                    callback.onError(exception)
                }
            }
        )
    }
}
