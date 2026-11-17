package ai.indoor.app.domain.export

import ai.indoor.app.domain.models.AnnotationType
import ai.indoor.app.domain.models.LandmarkType
import ai.indoor.app.domain.models.SensorAccuracy
import ai.indoor.app.domain.models.SensorType
import ai.indoor.app.domain.models.SurveyAnnotation
import ai.indoor.app.domain.models.SurveyPhotoObservation
import ai.indoor.app.domain.models.SurveySensorObservation
import ai.indoor.app.domain.models.SurveySession
import ai.indoor.app.domain.models.SurveySessionStatus
import ai.indoor.app.domain.models.SurveyWifiAccessPoint
import ai.indoor.app.domain.models.SurveyWifiObservation
import ai.indoor.app.domain.timestamp.SyncMethod
import ai.indoor.app.domain.timestamp.TimeSource
import ai.indoor.app.domain.timestamp.TimestampRecord
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.util.zip.ZipFile

class DatasetExportAndValidationTest {

    @get:Rule
    val tempFolder = TemporaryFolder()

    private fun createSampleTimestamp(epochMs: Long = 1716000000000L): TimestampRecord {
        return TimestampRecord(
            originalTimestamp = epochMs,
            timeSource = TimeSource.CLOCK_REALTIME_MILLIS,
            normalizedEpochMs = epochMs,
            normalizedIso = "2024-05-18T02:40:00.000Z",
            syncMethod = SyncMethod.WALLCLOCK_DIRECT,
            uncertaintyMs = 1.0f,
            isReliable = true
        )
    }

    @Test
    fun testValidationFailsWhenImageIsUnsanitized() {
        val session = SurveySession(sessionId = "SES-VAL01")
        val imageFile = tempFolder.newFile("photo_unsafe.jpg")
        imageFile.writeText("sample-pixels")

        val photo = SurveyPhotoObservation(
            photoId = "IMG-UNSAFE",
            sessionId = "SES-VAL01",
            timestamp = createSampleTimestamp(),
            filename = "photo_unsafe.jpg",
            filePath = imageFile.absolutePath,
            fileSizeBytes = imageFile.length(),
            imageWidth = 640,
            imageHeight = 480,
            landmarkType = LandmarkType.CORRIDOR,
            isExifSanitized = false, // UNSANITIZED!
            facesDetectedCount = 0,
            faceBlurApplied = false,
            faceDetectionStatus = ai.indoor.app.domain.models.FaceDetectionStatus.NOT_ATTEMPTED,
            isMock = false
        )

        val result = ExportValidator.validate(
            session = session,
            wifiScans = emptyList(),
            sensorReadings = emptyList(),
            photos = listOf(photo),
            annotations = emptyList()
        )

        assertFalse("Export validation must fail for unsanitized images", result.isValid)
        assertTrue(result.errors.any { it.contains("Unsanitized image rejected") })
    }

    @Test
    fun testValidationFailsWhenImageFileIsMissing() {
        val session = SurveySession(sessionId = "SES-VAL02")
        val missingFile = File(tempFolder.root, "does_not_exist.jpg")

        val photo = SurveyPhotoObservation(
            photoId = "IMG-MISSING",
            sessionId = "SES-VAL02",
            timestamp = createSampleTimestamp(),
            filename = "does_not_exist.jpg",
            filePath = missingFile.absolutePath,
            fileSizeBytes = 100L,
            imageWidth = 640,
            imageHeight = 480,
            landmarkType = LandmarkType.CORRIDOR,
            isExifSanitized = true,
            facesDetectedCount = 0,
            faceBlurApplied = false,
            faceDetectionStatus = ai.indoor.app.domain.models.FaceDetectionStatus.NO_FACES_DETECTED,
            isMock = false
        )

        val result = ExportValidator.validate(
            session = session,
            wifiScans = emptyList(),
            sensorReadings = emptyList(),
            photos = listOf(photo),
            annotations = emptyList()
        )

        assertFalse("Export validation must fail when image file is missing", result.isValid)
        assertTrue(result.errors.any { it.contains("missing or empty") })
    }

    @Test
    fun testValidationFailsOnDuplicateObservationIds() {
        val session = SurveySession(sessionId = "SES-VAL03")
        val w1 = SurveyWifiObservation("WIFI-DUP", "SES-VAL03", createSampleTimestamp(), emptyList(), true, false)
        val w2 = SurveyWifiObservation("WIFI-DUP", "SES-VAL03", createSampleTimestamp(), emptyList(), true, false)

        val result = ExportValidator.validate(session, listOf(w1, w2), emptyList(), emptyList(), emptyList())
        assertFalse(result.isValid)
        assertTrue(result.errors.any { it.contains("Duplicate Wi-Fi observation IDs") })
    }

    @Test
    fun testFullZipExportAndManifestIntegrity() {
        val exportDir = tempFolder.newFolder("exports")
        val exporter = DatasetExporter(exportDir)

        val session = SurveySession(
            sessionId = "SES-FULL01",
            sessionName = "Full Multimodal Export Test",
            buildingId = "TI-Main",
            floorId = "2",
            status = SurveySessionStatus.COMPLETED
        )

        val wifi = listOf(
            SurveyWifiObservation(
                observationId = "WIFI-101",
                sessionId = "SES-FULL01",
                timestamp = createSampleTimestamp(),
                accessPoints = listOf(SurveyWifiAccessPoint("AA:BB:CC:DD:EE:FF", "eduroam", -70, 5180, 36, "5GHz", true)),
                isFresh = true,
                isMock = false
            )
        )

        val imu = listOf(
            SurveySensorObservation(
                observationId = "IMU-101",
                sessionId = "SES-FULL01",
                sensorType = SensorType.ACCELEROMETER,
                x = 0.05f,
                y = 0.12f,
                z = 9.81f,
                magnitude = 9.811f,
                accuracy = SensorAccuracy.HIGH,
                samplingRate = "UI",
                timestamp = createSampleTimestamp(),
                isMock = false
            )
        )

        val imageFile = tempFolder.newFile("photo_clean.jpg")
        imageFile.writeText("image-binary-simulation")

        val photos = listOf(
            SurveyPhotoObservation(
                photoId = "IMG-101",
                sessionId = "SES-FULL01",
                timestamp = createSampleTimestamp(),
                filename = "photo_clean.jpg",
                filePath = imageFile.absolutePath,
                fileSizeBytes = imageFile.length(),
                imageWidth = 1920,
                imageHeight = 1080,
                landmarkType = LandmarkType.ROOM_SIGN,
                description = "Door Sign 204",
                isExifSanitized = true,
                facesDetectedCount = 0,
                faceBlurApplied = false,
                faceDetectionStatus = ai.indoor.app.domain.models.FaceDetectionStatus.NO_FACES_DETECTED,
                isMock = false
            )
        )

        val annotations = listOf(
            SurveyAnnotation(
                annotationId = "ANN-101",
                sessionId = "SES-FULL01",
                timestamp = createSampleTimestamp(),
                type = AnnotationType.ROOM_ENTRANCE,
                description = "Room 204 entrance door",
                isMock = false
            )
        )

        // Perform Export
        val result = exporter.exportSessionZip(session, wifi, imu, photos, annotations)
        assertTrue("Export must succeed", result.isSuccess)

        val zipFile = result.getOrNull()!!
        assertTrue("Generated ZIP must exist", zipFile.exists())
        assertEquals("indoorai_session_SES-FULL01.zip", zipFile.name)

        // Inspect ZIP internal contents
        val zip = ZipFile(zipFile)
        val entries = zip.entries().toList().map { it.name }

        assertTrue("ZIP must contain session.json", entries.contains("session.json"))
        assertTrue("ZIP must contain wifi_observations.csv", entries.contains("wifi_observations.csv"))
        assertTrue("ZIP must contain imu_observations.csv", entries.contains("imu_observations.csv"))
        assertTrue("ZIP must contain photo_metadata.csv", entries.contains("photo_metadata.csv"))
        assertTrue("ZIP must contain annotations.csv", entries.contains("annotations.csv"))
        assertTrue("ZIP must contain images/photo_clean.jpg", entries.contains("images/photo_clean.jpg"))
        assertTrue("ZIP must contain manifest.json", entries.contains("manifest.json"))

        // Verify manifest contents and SHA-256
        val manifestEntry = zip.getEntry("manifest.json")
        val manifestJson = zip.getInputStream(manifestEntry).bufferedReader().readText()
        val manifest = ExportManifest.fromJson(manifestJson)

        assertEquals("SES-FULL01", manifest.sessionId)
        assertEquals(1, manifest.recordCounts.wifiCount)
        assertEquals(1, manifest.recordCounts.imuCount)
        assertEquals(1, manifest.recordCounts.photoCount)
        assertEquals(1, manifest.recordCounts.annotationCount)
        assertEquals(4, manifest.recordCounts.totalObservations)
        assertEquals("VALIDATED", manifest.validationStatus)

        // Check image entry SHA-256 in manifest matches source file
        val imageSha = DatasetExporter.calculateSha256(imageFile)
        val manifestImgEntry = manifest.files.firstOrNull { it.filename == "images/photo_clean.jpg" }
        assertNotNull(manifestImgEntry)
        assertEquals(imageSha, manifestImgEntry!!.sha256)

        zip.close()
    }
}
