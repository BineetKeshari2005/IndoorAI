package ai.indoor.app.domain.storage

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
import ai.indoor.app.domain.timestamp.ClockAnchor
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
import org.mockito.Mockito.mock
import android.content.Context
import java.io.File

class SessionPersistenceTest {

    @get:Rule
    val tempFolder = TemporaryFolder()

    private fun createPersistenceManager(dir: File = tempFolder.root): SessionPersistenceManager {
        val mockContext = mock(Context::class.java)
        return SessionPersistenceManager(mockContext, baseDir = dir)
    }

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
    fun testSaveAndRetrieveSession() {
        val manager = createPersistenceManager()
        val session = SurveySession(
            sessionId = "SES-SAVED01",
            sessionName = "North Wing Survey",
            buildingId = "TI-Eng",
            floorId = "2"
        )

        manager.saveSession(session)

        val retrieved = manager.getSession("SES-SAVED01")
        assertNotNull(retrieved)
        assertEquals("North Wing Survey", retrieved!!.sessionName)
        assertEquals("TI-Eng", retrieved.buildingId)
        assertEquals(1, manager.getAllSessions().size)
    }

    @Test
    fun testInterruptedSessionRecoveryUponRestart() {
        val baseDir = tempFolder.newFolder("sessions_recovery")

        // 1. First run: session is ACTIVE when app terminates
        val manager1 = createPersistenceManager(baseDir)
        val activeSession = SurveySession(
            sessionId = "SES-CRASH01",
            sessionName = "Interrupted Survey",
            status = SurveySessionStatus.ACTIVE
        )
        manager1.saveSession(activeSession)

        // Verify it was active on disk
        assertEquals(SurveySessionStatus.ACTIVE, manager1.getSession("SES-CRASH01")!!.status)

        // 2. Second run: simulate app restart with a fresh manager instance
        val manager2 = createPersistenceManager(baseDir)
        val recovered = manager2.getSession("SES-CRASH01")

        assertNotNull(recovered)
        // Strict requirement: Never silently mark interrupted session as completed
        assertEquals(SurveySessionStatus.PAUSED, recovered!!.status)
        assertTrue(recovered.isInterrupted)
        assertNotNull(recovered.interruptionNote)
    }

    @Test
    fun testAppendAndRetrieveObservations() {
        val manager = createPersistenceManager()
        val session = SurveySession(sessionId = "SES-OBS01", status = SurveySessionStatus.ACTIVE)
        manager.saveSession(session)

        // 1. Append Wi-Fi observation
        val wifi = SurveyWifiObservation(
            observationId = "WIFI-001",
            sessionId = "SES-OBS01",
            timestamp = createSampleTimestamp(),
            accessPoints = listOf(
                SurveyWifiAccessPoint("00:11:22:33:44:55", "eduroam", -65, 2412, 1, "2.4GHz", true)
            ),
            isFresh = true,
            isMock = false
        )
        manager.recordWifiObservation(wifi)

        // 2. Append Sensor observation
        val sensor = SurveySensorObservation(
            observationId = "IMU-001",
            sessionId = "SES-OBS01",
            sensorType = SensorType.ACCELEROMETER,
            x = 0.1f,
            y = 0.2f,
            z = 9.8f,
            magnitude = 9.805f,
            accuracy = SensorAccuracy.HIGH,
            samplingRate = "UI",
            timestamp = createSampleTimestamp(),
            isMock = false
        )
        manager.recordSensorObservation(sensor)

        // 3. Append Annotation
        val ann = SurveyAnnotation(
            annotationId = "ANN-001",
            sessionId = "SES-OBS01",
            timestamp = createSampleTimestamp(),
            type = AnnotationType.DOOR,
            description = "Office TI-204"
        )
        manager.recordAnnotation(ann)

        // Verify retrieval
        val retrievedWifi = manager.getWifiObservations("SES-OBS01")
        assertEquals(1, retrievedWifi.size)
        assertEquals("WIFI-001", retrievedWifi[0].observationId)

        val retrievedSensor = manager.getSensorObservations("SES-OBS01")
        assertEquals(1, retrievedSensor.size)
        assertEquals("IMU-001", retrievedSensor[0].observationId)

        val retrievedAnn = manager.getAnnotations("SES-OBS01")
        assertEquals(1, retrievedAnn.size)
        assertEquals("ANN-001", retrievedAnn[0].annotationId)

        // Verify session counters were incremented
        val updatedSession = manager.getSession("SES-OBS01")!!
        assertEquals(1, updatedSession.wifiCount)
        assertEquals(1, updatedSession.sensorCount)
        assertEquals(1, updatedSession.annotationCount)
    }

    @Test
    fun testDeleteSessionCleansJournalsAndImageFiles() {
        val manager = createPersistenceManager()
        val session = SurveySession(sessionId = "SES-DEL01")
        manager.saveSession(session)

        // Create a real physical image file
        val imageFile = tempFolder.newFile("photo_sample.jpg")
        imageFile.writeText("fake-image-bytes")
        assertTrue(imageFile.exists())

        val photoObs = SurveyPhotoObservation(
            photoId = "IMG-001",
            sessionId = "SES-DEL01",
            timestamp = createSampleTimestamp(),
            filename = "photo_sample.jpg",
            filePath = imageFile.absolutePath,
            fileSizeBytes = imageFile.length(),
            imageWidth = 640,
            imageHeight = 480,
            landmarkType = LandmarkType.DOOR,
            isExifSanitized = true,
            facesDetectedCount = 0,
            faceBlurApplied = false,
            faceDetectionStatus = ai.indoor.app.domain.models.FaceDetectionStatus.NO_FACES_DETECTED,
            isMock = false
        )
        manager.recordPhotoObservation(photoObs)

        // Delete session
        val deleted = manager.deleteSession("SES-DEL01")
        assertTrue(deleted)

        // Session directory deleted
        assertFalse(File(tempFolder.root, "SES-DEL01").exists())

        // Physical image file deleted (no orphaned images left)
        assertFalse("Associated image file must be deleted upon session removal", imageFile.exists())
        assertEquals(0, manager.getAllSessions().size)
    }
}
