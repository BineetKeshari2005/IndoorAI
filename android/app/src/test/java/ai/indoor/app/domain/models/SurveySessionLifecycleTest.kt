package ai.indoor.app.domain.models

import ai.indoor.app.domain.timestamp.ClockAnchor
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class SurveySessionLifecycleTest {

    @Test
    fun testSessionCreationDefaults() {
        val session = SurveySession(
            sessionId = "SES-TEST001",
            sessionName = "Floor 2 Survey",
            buildingId = "TI-Engineering",
            floorId = "2",
            roomId = "204"
        )

        assertEquals("SES-TEST001", session.sessionId)
        assertEquals("Floor 2 Survey", session.sessionName)
        assertEquals(SurveySessionStatus.CREATED, session.status)
        assertNull(session.startTimeMs)
        assertNull(session.endTimeMs)
        assertEquals(0, session.totalObservations)
        assertFalse(session.hasRealData)
        assertFalse(session.hasMockData)
        assertEquals(DataSourceType.REAL, session.dataSourceType)
    }

    @Test
    fun testDataSourceTypeClassification() {
        val realSession = SurveySession(realObservationCount = 10, mockObservationCount = 0)
        assertEquals(DataSourceType.REAL, realSession.dataSourceType)

        val mockSession = SurveySession(realObservationCount = 0, mockObservationCount = 5)
        assertEquals(DataSourceType.MOCK, mockSession.dataSourceType)

        val hybridSession = SurveySession(realObservationCount = 8, mockObservationCount = 4)
        assertEquals(DataSourceType.HYBRID, hybridSession.dataSourceType)
    }

    @Test
    fun testFormattedDurationCalculations() {
        val now = System.currentTimeMillis()
        val completedSession = SurveySession(
            startTimeMs = now - 125_000L, // 2 mins 5 secs
            endTimeMs = now,
            status = SurveySessionStatus.COMPLETED
        )

        assertEquals(125L, completedSession.durationSeconds)
        assertEquals("02:05", completedSession.formattedDuration)

        val createdSession = SurveySession(status = SurveySessionStatus.CREATED)
        assertEquals(0L, createdSession.durationSeconds)
        assertEquals("00:00", createdSession.formattedDuration)
    }

    @Test
    fun testSessionJsonSerializationRoundTrip() {
        val original = SurveySession(
            sessionId = "SES-12345678",
            sessionName = "Corridor Benchmark",
            buildingId = "TI-Main",
            floorId = "3",
            roomId = "Room 304",
            wifiCount = 5,
            sensorCount = 120,
            photoCount = 2,
            annotationCount = 1,
            realObservationCount = 128,
            mockObservationCount = 0,
            clockAnchor = ClockAnchor(1716000000000L, 50_000_000_000L, "2024-05-18T02:40:00.000Z")
        )

        val json = original.toJson()
        assertTrue(json.contains("\"sessionId\": \"SES-12345678\""))
        assertTrue(json.contains("\"buildingId\": \"TI-Main\""))

        val parsed = SurveySession.fromJson(json)
        assertEquals(original.sessionId, parsed.sessionId)
        assertEquals(original.sessionName, parsed.sessionName)
        assertEquals(original.wifiCount, parsed.wifiCount)
        assertEquals(original.sensorCount, parsed.sensorCount)
        assertEquals(original.clockAnchor.wallClockEpochMs, parsed.clockAnchor.wallClockEpochMs)
    }
}
