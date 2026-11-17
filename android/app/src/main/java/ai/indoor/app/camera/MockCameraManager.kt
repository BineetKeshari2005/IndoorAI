package ai.indoor.app.camera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import androidx.camera.view.PreviewView
import androidx.lifecycle.LifecycleOwner
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Mock camera manager that generates synthetic landmark images for emulators,
 * testing environments, and devices without physical cameras.
 */
class MockCameraManager(
    private val context: Context
) : CameraManager {

    private var isPreviewActive = false

    override fun isCameraAvailable(): Boolean = true

    override fun startPreview(lifecycleOwner: LifecycleOwner, previewView: PreviewView) {
        isPreviewActive = true
    }

    override fun stopPreview() {
        isPreviewActive = false
    }

    override fun capturePhoto(callback: ImageCaptureCallback) {
        try {
            val width = 640
            val height = 480
            val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(bitmap)

            // Draw corridor background (walls and floor)
            val bgPaint = Paint().apply { color = Color.rgb(235, 238, 242) }
            canvas.drawRect(0f, 0f, width.toFloat(), height.toFloat(), bgPaint)

            // Floor
            val floorPaint = Paint().apply { color = Color.rgb(190, 195, 205) }
            canvas.drawRect(0f, height * 0.6f, width.toFloat(), height.toFloat(), floorPaint)

            // Doorway
            val doorPaint = Paint().apply { color = Color.rgb(80, 100, 120) }
            val doorRect = RectF(width * 0.35f, height * 0.25f, width * 0.65f, height * 0.85f)
            canvas.drawRoundRect(doorRect, 8f, 8f, doorPaint)

            // Door Sign
            val signPaint = Paint().apply { color = Color.rgb(255, 255, 255) }
            val signRect = RectF(width * 0.38f, height * 0.15f, width * 0.62f, height * 0.22f)
            canvas.drawRoundRect(signRect, 6f, 6f, signPaint)

            // Text on sign
            val textPaint = Paint().apply {
                color = Color.rgb(30, 40, 60)
                textSize = 22f
                isFakeBoldText = true
                textAlign = Paint.Align.CENTER
            }
            canvas.drawText("ROOM TI-204", width * 0.5f, height * 0.20f, textPaint)

            // Watermark text
            val subTextPaint = Paint().apply {
                color = Color.rgb(100, 110, 125)
                textSize = 16f
                textAlign = Paint.Align.CENTER
            }
            canvas.drawText("IndoorAI Mock Landmark Frame", width * 0.5f, height * 0.95f, subTextPaint)

            // Save to temp file
            val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val tempFile = File.createTempFile("MOCK_RAW_${timeStamp}_", ".jpg", context.cacheDir)
            FileOutputStream(tempFile).use { out ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 90, out)
                out.flush()
            }
            bitmap.recycle()

            callback.onPhotoCaptured(tempFile)
        } catch (e: Throwable) {
            callback.onError(e)
        }
    }
}
