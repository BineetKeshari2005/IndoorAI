package ai.indoor.app.camera

import ai.indoor.app.domain.models.CapturedPhotoBatch
import ai.indoor.app.domain.models.CapturedPhotoMetadata
import ai.indoor.app.domain.models.FaceDetectionStatus
import ai.indoor.app.domain.models.LandmarkType
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class CapturedPhotoExportTest {

    private fun createSampleMetadata(
        id: String = "IMG-12345678",
        landmarkType: LandmarkType = LandmarkType.CORRIDOR,
        description: String = "Corridor sign TI-204"
    ): CapturedPhotoMetadata {
        return CapturedPhotoMetadata(
            photoId = id,
            sessionId = "SES-ABC12345",
            timestampMs = 1716000000000L,
            filename = "PHOTO_${id}_sanitized.jpg",
            filePath = "/data/user/0/ai.indoor.app/files/survey_photos/PHOTO_${id}_sanitized.jpg",
            building = "TI-Engineering",
            floor = "2",
            room = "Room 204",
            landmarkType = landmarkType,
            description = description,
            notes = "Surveyor walking east",
            isExifSanitized = true,
            facesDetectedCount = 1,
            faceBlurApplied = true,
            faceDetectionStatus = FaceDetectionStatus.SUCCEEDED,
            fileSizeBytes = 204800L,
            imageWidth = 1920,
            imageHeight = 1080,
            isMock = false
        )
    }

    @Test
    fun testJsonSerializationContainsAllFields() {
        val meta = createSampleMetadata()
        val json = meta.toJson()

        assertTrue("JSON should contain photo_id", json.contains("\"photo_id\": \"IMG-12345678\""))
        assertTrue("JSON should contain session_id", json.contains("\"session_id\": \"SES-ABC12345\""))
        assertTrue("JSON should contain building", json.contains("\"building\": \"TI-Engineering\""))
        assertTrue("JSON should contain floor", json.contains("\"floor\": \"2\""))
        assertTrue("JSON should contain room", json.contains("\"room\": \"Room 204\""))
        assertTrue("JSON should contain landmark_type", json.contains("\"landmark_type\": \"CORRIDOR\""))
        assertTrue("JSON should contain landmark_display", json.contains("\"landmark_display\": \"Corridor\""))
        assertTrue("JSON should contain is_exif_sanitized", json.contains("\"is_exif_sanitized\": true"))
        assertTrue("JSON should contain faces_detected_count", json.contains("\"faces_detected_count\": 1"))
        assertTrue("JSON should contain face_blur_applied", json.contains("\"face_blur_applied\": true"))
        assertTrue("JSON should contain face_detection_status", json.contains("\"face_detection_status\": \"SUCCEEDED\""))
        assertTrue("JSON should contain image dimensions", json.contains("\"image_width\": 1920") && json.contains("\"image_height\": 1080"))
    }

    @Test
    fun testJsonSpecialCharactersEscaping() {
        val meta = createSampleMetadata(
            description = "Quote \"Test\" and \n newline and \t tab and \\ backslash"
        )
        val json = meta.toJson()
        assertTrue("Quotes should be escaped", json.contains("\\\"Test\\\""))
        assertTrue("Newlines should be escaped", json.contains("\\n"))
        assertTrue("Tabs should be escaped", json.contains("\\t"))
        assertTrue("Backslashes should be escaped", json.contains("\\\\"))
    }

    @Test
    fun testCsvHeaderAndRowFormatting() {
        val meta = createSampleMetadata()
        val row = meta.toCsvRow()

        val expectedHeader = "photo_id,session_id,timestamp_ms,timestamp_iso,filename,building,floor,room,landmark_type,description,notes,exif_sanitized,faces_detected,face_blur_applied,face_status,file_size_bytes,width,height,is_mock"
        assertEquals(expectedHeader, CapturedPhotoMetadata.CSV_HEADER)

        val columns = row.split(",")
        assertEquals("IMG-12345678", columns[0])
        assertEquals("SES-ABC12345", columns[1])
        assertEquals("1716000000000", columns[2])
        assertEquals("TI-Engineering", columns[5])
        assertEquals("2", columns[6])
        assertEquals("Room 204", columns[7])
        assertEquals("CORRIDOR", columns[8])
        assertEquals("true", columns[11]) // exif_sanitized
        assertEquals("1", columns[12]) // faces_detected
        assertEquals("true", columns[13]) // face_blur_applied
        assertEquals("SUCCEEDED", columns[14]) // face_status
    }

    @Test
    fun testCsvEscapingWithCommasAndQuotes() {
        val meta = createSampleMetadata(
            description = "Comma, separated, \"and quoted\" description"
        )
        val row = meta.toCsvRow()
        assertTrue("Field with commas and quotes must be enclosed in quotes", row.contains("\"Comma, separated, \"\"and quoted\"\" description\""))
    }

    @Test
    fun testBatchExportJsonAndCsv() {
        val p1 = createSampleMetadata("IMG-001", LandmarkType.DOOR, "North door")
        val p2 = createSampleMetadata("IMG-002", LandmarkType.ELEVATOR, "Central elevator")
        val batch = CapturedPhotoBatch(listOf(p1, p2))

        val batchJson = batch.toJson()
        assertTrue("Batch JSON should have batch_size 2", batchJson.contains("\"batch_size\": 2"))
        assertTrue("Batch JSON should contain IMG-001", batchJson.contains("IMG-001"))
        assertTrue("Batch JSON should contain IMG-002", batchJson.contains("IMG-002"))

        val batchCsv = batch.toCsv()
        val lines = batchCsv.trim().split("\n")
        assertEquals("Header + 2 data rows", 3, lines.size)
        assertEquals(CapturedPhotoMetadata.CSV_HEADER, lines[0])
        assertTrue("First row contains IMG-001", lines[1].contains("IMG-001"))
        assertTrue("Second row contains IMG-002", lines[2].contains("IMG-002"))
    }

    @Test
    fun testDisplayLocationFormatting() {
        val metaFull = createSampleMetadata()
        assertEquals("TI-Engineering, Floor 2, Room Room 204", metaFull.displayLocation)

        val metaNoBuilding = metaFull.copy(building = "", floor = "3", room = "TI-301")
        assertEquals("Floor 3, Room TI-301", metaNoBuilding.displayLocation)

        val metaBlank = metaFull.copy(building = "", floor = "", room = "")
        assertEquals("Unlabeled Location", metaBlank.displayLocation)
    }

    @Test
    fun testLandmarkTypeFromStringFallback() {
        assertEquals(LandmarkType.CORRIDOR, LandmarkType.fromString("corridor"))
        assertEquals(LandmarkType.CORRIDOR, LandmarkType.fromString("CORRIDOR"))
        assertEquals(LandmarkType.ROOM_SIGN, LandmarkType.fromString("Room sign"))
        assertEquals(LandmarkType.ROOM_SIGN, LandmarkType.fromString("ROOM_SIGN"))
        assertEquals(LandmarkType.ELEVATOR, LandmarkType.fromString("Elevator"))
        assertEquals(LandmarkType.OTHER, LandmarkType.fromString("NonExistentLandmark"))
    }
}
