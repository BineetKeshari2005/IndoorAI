package ai.indoor.app.domain.timeline

import ai.indoor.app.domain.models.AnnotationType
import ai.indoor.app.domain.models.LandmarkType
import ai.indoor.app.domain.models.SensorAccuracy
import ai.indoor.app.domain.models.SensorType
import ai.indoor.app.domain.models.SurveyAnnotation
import ai.indoor.app.domain.models.SurveyPhotoObservation
import ai.indoor.app.domain.models.SurveySensorObservation
import ai.indoor.app.domain.models.SurveyWifiObservation
import ai.indoor.app.domain.timestamp.SyncMethod
import ai.indoor.app.domain.timestamp.TimeSource
import ai.indoor.app.domain.timestamp.TimestampRecord
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ObservationTimelineTest {

    private fun makeTimestamp(epochMs: Long?, uncertainty: Float? = 2.0f): TimestampRecord {
        return TimestampRecord(
            originalTimestamp = epochMs ?: 0L,
            timeSource = TimeSource.CLOCK_REALTIME_MILLIS,
            normalizedEpochMs = epochMs,
            normalizedIso = if (epochMs != null) "2024-05-18T02:40:00.000Z" else null,
            syncMethod = if (epochMs != null) SyncMethod.WALLCLOCK_DIRECT else SyncMethod.UNSYNCHRONIZED,
            uncertaintyMs = uncertainty,
            isReliable = epochMs != null
        )
    }

    @Test
    fun testTimelineSortingChronologicalAscending() {
        val baseTime = 1716000000000L

        // Observation 3: Photo at T + 3000ms
        val photo = SurveyPhotoObservation(
            photoId = "IMG-03",
            sessionId = "SES-TL01",
            timestamp = makeTimestamp(baseTime + 3000L),
            filename = "img_03.jpg",
            filePath = "/fake/path/img_03.jpg",
            fileSizeBytes = 1000L,
            imageWidth = 640,
            imageHeight = 480,
            landmarkType = LandmarkType.CORRIDOR,
            isExifSanitized = true,
            facesDetectedCount = 0,
            faceBlurApplied = false,
            faceDetectionStatus = ai.indoor.app.domain.models.FaceDetectionStatus.NO_FACES_DETECTED,
            isMock = false
        )

        // Observation 1: Sensor at T + 1000ms
        val sensor = SurveySensorObservation(
            observationId = "IMU-01",
            sessionId = "SES-TL01",
            sensorType = SensorType.ACCELEROMETER,
            x = 0f,
            y = 0f,
            z = 9.8f,
            magnitude = 9.8f,
            accuracy = SensorAccuracy.HIGH,
            samplingRate = "UI",
            timestamp = makeTimestamp(baseTime + 1000L),
            isMock = false
        )

        // Observation 2: Wi-Fi at T + 2000ms
        val wifi = SurveyWifiObservation(
            observationId = "WIFI-02",
            sessionId = "SES-TL01",
            timestamp = makeTimestamp(baseTime + 2000L),
            accessPoints = emptyList(),
            isFresh = true,
            isMock = false
        )

        // Observation 4: Annotation at T + 4000ms
        val ann = SurveyAnnotation(
            annotationId = "ANN-04",
            sessionId = "SES-TL01",
            timestamp = makeTimestamp(baseTime + 4000L),
            type = AnnotationType.DOOR,
            description = "Door 204",
            isMock = false
        )

        val timeline = ObservationTimelineBuilder.buildTimeline(
            wifiScans = listOf(wifi),
            sensorReadings = listOf(sensor),
            photos = listOf(photo),
            annotations = listOf(ann)
        )

        assertEquals(4, timeline.size)
        assertEquals("IMU-01", timeline[0].id)
        assertEquals(TimelineObservationType.ACCELEROMETER, timeline[0].type)

        assertEquals("WIFI-02", timeline[1].id)
        assertEquals(TimelineObservationType.WIFI_SCAN, timeline[1].type)

        assertEquals("IMG-03", timeline[2].id)
        assertEquals(TimelineObservationType.PHOTO_CAPTURE, timeline[2].type)

        assertEquals("ANN-04", timeline[3].id)
        assertEquals(TimelineObservationType.ANNOTATION, timeline[3].type)
    }

    @Test
    fun testTimelineDistinguishesRealAndMock() {
        val sensorReal = SurveySensorObservation(
            observationId = "IMU-REAL",
            sessionId = "SES-TL02",
            sensorType = SensorType.GYROSCOPE,
            x = 0f, y = 0f, z = 0f, magnitude = 0f,
            accuracy = SensorAccuracy.HIGH, samplingRate = "UI",
            timestamp = makeTimestamp(1000L),
            isMock = false
        )

        val sensorMock = SurveySensorObservation(
            observationId = "IMU-MOCK",
            sessionId = "SES-TL02",
            sensorType = SensorType.GYROSCOPE,
            x = 0f, y = 0f, z = 0f, magnitude = 0f,
            accuracy = SensorAccuracy.HIGH, samplingRate = "UI",
            timestamp = makeTimestamp(2000L),
            isMock = true
        )

        val timeline = ObservationTimelineBuilder.buildTimeline(
            wifiScans = emptyList(),
            sensorReadings = listOf(sensorReal, sensorMock),
            photos = emptyList(),
            annotations = emptyList()
        )

        assertEquals(2, timeline.size)
        assertFalse("Real sensor reading must have isMock = false", timeline[0].isMock)
        assertTrue("Mock sensor reading must have isMock = true", timeline[1].isMock)
    }

    @Test
    fun testTimelineHandlesUncertainTimestamp() {
        val unanchoredWifi = SurveyWifiObservation(
            observationId = "WIFI-UNCERTAIN",
            sessionId = "SES-TL03",
            timestamp = makeTimestamp(null, uncertainty = null),
            accessPoints = emptyList(),
            isFresh = false,
            isMock = false
        )

        val timeline = ObservationTimelineBuilder.buildTimeline(
            wifiScans = listOf(unanchoredWifi),
            sensorReadings = emptyList(),
            photos = emptyList(),
            annotations = emptyList()
        )

        assertEquals(1, timeline.size)
        assertFalse(timeline[0].isTimingCertain)
        assertEquals("Time Unavailable", timeline[0].formattedTime)
        assertEquals("Uncertain", timeline[0].uncertaintyDisplay)
    }
}
