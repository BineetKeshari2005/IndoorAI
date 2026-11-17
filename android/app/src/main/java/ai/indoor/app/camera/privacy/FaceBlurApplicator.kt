package ai.indoor.app.camera.privacy

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF

/**
 * Applies privacy obfuscation (pixelation or tint) over detected facial regions.
 *
 * All operations execute locally in-memory on the provided bitmap.
 */
object FaceBlurApplicator {

    enum class ObfuscationStyle {
        PIXELATE,
        SOLID_TINT
    }

    /**
     * Renders privacy obfuscation over each face boundary.
     * Returns a new mutable Bitmap copy with the obfuscation applied.
     */
    fun applyFaceObfuscation(
        source: Bitmap,
        faces: List<FaceBounds>,
        style: ObfuscationStyle = ObfuscationStyle.PIXELATE,
        pixelBlockSize: Int = 24
    ): Bitmap {
        if (faces.isEmpty()) return source

        val output = source.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(output)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG)

        for (face in faces) {
            val rect = RectF(
                face.left.coerceIn(0f, output.width.toFloat()),
                face.top.coerceIn(0f, output.height.toFloat()),
                face.right.coerceIn(0f, output.width.toFloat()),
                face.bottom.coerceIn(0f, output.height.toFloat())
            )

            if (rect.width() <= 2f || rect.height() <= 2f) continue

            when (style) {
                ObfuscationStyle.PIXELATE -> pixelateRegion(output, rect, pixelBlockSize)
                ObfuscationStyle.SOLID_TINT -> {
                    paint.color = Color.argb(220, 64, 64, 64)
                    paint.style = Paint.Style.FILL
                    canvas.drawRoundRect(rect, 16f, 16f, paint)
                }
            }
        }

        return output
    }

    private fun pixelateRegion(bitmap: Bitmap, rect: RectF, blockSize: Int) {
        val startX = rect.left.toInt().coerceIn(0, bitmap.width - 1)
        val endX = rect.right.toInt().coerceIn(0, bitmap.width)
        val startY = rect.top.toInt().coerceIn(0, bitmap.height - 1)
        val endY = rect.bottom.toInt().coerceIn(0, bitmap.height)

        for (y in startY until endY step blockSize) {
            for (x in startX until endX step blockSize) {
                val blockW = minOf(blockSize, endX - x)
                val blockH = minOf(blockSize, endY - y)
                if (blockW <= 0 || blockH <= 0) continue

                var totalR = 0L
                var totalG = 0L
                var totalB = 0L
                var count = 0

                for (by in 0 until blockH) {
                    for (bx in 0 until blockW) {
                        val px = bitmap.getPixel(x + bx, y + by)
                        totalR += (px shr 16) and 0xFF
                        totalG += (px shr 8) and 0xFF
                        totalB += px and 0xFF
                        count++
                    }
                }

                if (count > 0) {
                    val avgColor = Color.rgb(
                        (totalR / count).toInt(),
                        (totalG / count).toInt(),
                        (totalB / count).toInt()
                    )
                    for (by in 0 until blockH) {
                        for (bx in 0 until blockW) {
                            bitmap.setPixel(x + bx, y + by, avgColor)
                        }
                    }
                }
            }
        }
    }
}
