package ai.indoor.app.camera.privacy

import ai.indoor.app.domain.models.FaceDetectionStatus
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class ImagePrivacySanitizerTest {

    @get:Rule
    val tempFolder = TemporaryFolder()

    @Test
    fun testFaceBoundsCalculations() {
        val bounds = FaceBounds(left = 100f, top = 150f, right = 300f, bottom = 450f, confidence = 0.85f)
        assertEquals(200f, bounds.width, 0.001f)
        assertEquals(300f, bounds.height, 0.001f)
        assertEquals(0.85f, bounds.confidence, 0.001f)
    }

    @Test
    fun testFaceBoundsZeroOrInvertedDimensions() {
        val bounds = FaceBounds(left = 300f, top = 400f, right = 200f, bottom = 300f)
        // coerceAtLeast(0f) ensures no negative dimensions
        assertEquals(0f, bounds.width, 0.001f)
        assertEquals(0f, bounds.height, 0.001f)
    }

    @Test
    fun testSanitizerFailsGracefullyWhenFileDoesNotExist() {
        val sanitizer = ImagePrivacySanitizer()
        val nonExistentFile = File(tempFolder.root, "does_not_exist.jpg")
        val destFile = File(tempFolder.root, "sanitized.jpg")

        val result = sanitizer.sanitize(nonExistentFile, destFile)

        assertFalse("Sanitization should fail on missing file", result.success)
        assertFalse("Destination file should not be reported", result.sanitizedFile != null)
        assertFalse("Destination file must not exist on disk", destFile.exists())
        assertEquals(FaceDetectionStatus.FAILED, result.faceDetectionStatus)
        assertNotNull("Error message must be present", result.errorMessage)
        assertTrue("Error message explains file missing", result.errorMessage!!.contains("does not exist"))
    }

    @Test
    fun testSanitizerFailsGracefullyWhenFileIsEmpty() {
        val sanitizer = ImagePrivacySanitizer()
        val emptyFile = tempFolder.newFile("empty.jpg")
        val destFile = File(tempFolder.root, "sanitized.jpg")

        val result = sanitizer.sanitize(emptyFile, destFile)

        assertFalse("Sanitization should fail on 0-byte file", result.success)
        assertFalse("Destination file must not exist on disk", destFile.exists())
        assertNotNull(result.errorMessage)
    }

    @Test
    fun testFaceDetectionStatusesDescriptions() {
        assertEquals("Faces detected", FaceDetectionStatus.SUCCEEDED.description)
        assertEquals("No faces detected in scene", FaceDetectionStatus.NO_FACES_DETECTED.description)
        assertEquals("On-device face detector unavailable", FaceDetectionStatus.UNAVAILABLE.description)
        assertEquals("Face detection encountered an error", FaceDetectionStatus.FAILED.description)
        assertEquals("Detection not executed", FaceDetectionStatus.NOT_ATTEMPTED.description)
    }
}
