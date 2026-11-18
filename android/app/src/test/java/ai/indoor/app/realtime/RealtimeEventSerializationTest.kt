package ai.indoor.app.realtime

import com.google.gson.Gson
import com.google.gson.JsonParser
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class RealtimeEventSerializationTest {

    private val gson = Gson()

    @Test
    fun testRealtimeEventDtoSerialization() {
        val event = RealtimeEventDto(
            eventId = "EV-12345",
            sessionId = "SESSION-TEST-01",
            eventType = RealtimeEventType.WIFI_SCAN,
            deviceId = "PIXEL_8_PRO",
            clientTimestampMonotonic = 123456789L,
            clientTimestampUtc = 1700000000000L,
            payloadSchemaVersion = "1.0.0",
            payload = mapOf(
                "bssid_scans" to listOf(
                    mapOf("bssid" to "aa:bb:cc:dd:ee:ff", "rssi_dbm" to -65.5)
                )
            )
        )

        val json = gson.toJson(event)
        assertNotNull(json)

        val parsed = JsonParser.parseString(json).asJsonObject
        assertEquals("EV-12345", parsed.get("event_id").asString)
        assertEquals("SESSION-TEST-01", parsed.get("session_id").asString)
        assertEquals("WIFI_SCAN", parsed.get("event_type").asString)
        assertEquals("PIXEL_8_PRO", parsed.get("device_id").asString)
        assertEquals(123456789L, parsed.get("client_timestamp_monotonic").asLong)
        assertEquals(1700000000000L, parsed.get("client_timestamp_utc").asLong)
        assertEquals("1.0.0", parsed.get("payload_schema_version").asString)
        assertTrue(parsed.has("payload"))
    }

    @Test
    fun testServerAckDtoDeserialization() {
        val ackJson = """
            {
                "type": "ACK",
                "event_id": "EV-12345",
                "session_id": "SESSION-TEST-01",
                "status": "ACCEPTED",
                "server_receive_timestamp": 1700000000500,
                "rejection_reason": "NONE",
                "message": "Event accepted and processed.",
                "sequence_number": 42
            }
        """.trimIndent()

        val ack = gson.fromJson(ackJson, ServerAckDto::class.java)
        assertEquals("ACK", ack.type)
        assertEquals("EV-12345", ack.eventId)
        assertEquals("ACCEPTED", ack.status)
        assertEquals(1700000000500L, ack.serverReceiveTimestamp)
        assertEquals("NONE", ack.rejectionReason)
        assertEquals(42L, ack.sequenceNumber)
    }

    @Test
    fun testLiveLocalizationUpdateDtoDeserialization() {
        val locJson = """
            {
                "type": "LOCALIZATION_UPDATE",
                "session_id": "SESSION-TEST-01",
                "building_id": "BLDG_ENG_01",
                "timestamp_ms": 1700000001000,
                "x_m": 12.345,
                "y_m": 24.567,
                "floor_id": 1,
                "heading_rad": 1.5708,
                "uncertainty_radius_m": 1.85,
                "confidence_level": "HIGH",
                "source": "PARTICLE_FILTER",
                "source_event_id": "EV-12345"
            }
        """.trimIndent()

        val loc = gson.fromJson(locJson, LiveLocalizationUpdateDto::class.java)
        assertEquals("LOCALIZATION_UPDATE", loc.type)
        assertEquals(12.345, loc.xM, 0.001)
        assertEquals(24.567, loc.yM, 0.001)
        assertEquals(1, loc.floorId)
        assertEquals(1.5708, loc.headingRad, 0.001)
        assertEquals(1.85, loc.uncertaintyRadiusM, 0.01)
        assertEquals("HIGH", loc.confidenceLevel)
        assertEquals("PARTICLE_FILTER", loc.source)
    }

    @Test
    fun testLiveNavigationUpdateDtoDeserialization() {
        val navJson = """
            {
                "type": "NAVIGATION_UPDATE",
                "session_id": "SESSION-TEST-01",
                "navigation_session_id": "NAV-SES-ABC123",
                "timestamp_ms": 1700000002000,
                "current_instruction": "Turn left at Robotics Lab",
                "instruction_type": "TURN_LEFT",
                "distance_to_next_waypoint_m": 5.2,
                "remaining_distance_m": 34.8,
                "cross_track_error_m": 0.45,
                "arrival_status": false,
                "reroute_occurred": false,
                "route_status": "ACTIVE",
                "target_destination": "Main Auditorium",
                "target_floor_id": 2
            }
        """.trimIndent()

        val nav = gson.fromJson(navJson, LiveNavigationUpdateDto::class.java)
        assertEquals("NAVIGATION_UPDATE", nav.type)
        assertEquals("NAV-SES-ABC123", nav.navigationSessionId)
        assertEquals("Turn left at Robotics Lab", nav.currentInstruction)
        assertEquals("TURN_LEFT", nav.instructionType)
        assertEquals(5.2, nav.distanceToNextWaypointM, 0.01)
        assertEquals(34.8, nav.remainingDistanceM, 0.01)
        assertEquals(0.45, nav.crossTrackErrorM, 0.01)
        assertEquals(false, nav.arrivalStatus)
        assertEquals("Main Auditorium", nav.targetDestination)
        assertEquals(2, nav.targetFloorId)
    }
}
