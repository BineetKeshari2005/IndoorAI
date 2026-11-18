package ai.indoor.app.realtime

import com.google.gson.annotations.SerializedName
import java.util.UUID

/**
 * Supported real-time event types matching backend Task 29 schema.
 */
enum class RealtimeEventType {
    @SerializedName("WIFI_SCAN")
    WIFI_SCAN,

    @SerializedName("IMU_BATCH")
    IMU_BATCH,

    @SerializedName("ORIENTATION_UPDATE")
    ORIENTATION_UPDATE,

    @SerializedName("PHOTO_CAPTURE")
    PHOTO_CAPTURE,

    @SerializedName("LANDMARK_OBSERVATION")
    LANDMARK_OBSERVATION,

    @SerializedName("ANNOTATION")
    ANNOTATION,

    @SerializedName("HEARTBEAT")
    HEARTBEAT,

    @SerializedName("SESSION_STATUS")
    SESSION_STATUS
}

/**
 * Status of client streaming connection.
 */
enum class RealtimeConnectionState {
    DISCONNECTED,
    CONNECTING,
    CONNECTED,
    RECONNECTING,
    FAILED
}

/**
 * Canonical real-time event model sent from Android client to backend.
 */
data class RealtimeEventDto(
    @SerializedName("event_id")
    val eventId: String = UUID.randomUUID().toString(),

    @SerializedName("session_id")
    val sessionId: String,

    @SerializedName("event_type")
    val eventType: RealtimeEventType,

    @SerializedName("device_id")
    val deviceId: String,

    @SerializedName("client_timestamp_monotonic")
    val clientTimestampMonotonic: Long? = null,

    @SerializedName("client_timestamp_utc")
    val clientTimestampUtc: Long = System.currentTimeMillis(),

    @SerializedName("payload_schema_version")
    val payloadSchemaVersion: String = "1.0.0",

    @SerializedName("payload")
    val payload: Map<String, Any?> = emptyMap()
)

/**
 * Wi-Fi AP scan entry.
 */
data class ApScanItem(
    @SerializedName("bssid")
    val bssid: String,

    @SerializedName("rssi_dbm")
    val rssiDbm: Double,

    @SerializedName("ssid")
    val ssid: String? = null,

    @SerializedName("frequency_mhz")
    val frequencyMhz: Int? = null,

    @SerializedName("channel")
    val channel: Int? = null
)

/**
 * IMU sensor observation entry.
 */
data class ImuSampleItem(
    @SerializedName("sensor_type")
    val sensorType: String,

    @SerializedName("x")
    val x: Float,

    @SerializedName("y")
    val y: Float,

    @SerializedName("z")
    val z: Float,

    @SerializedName("accuracy")
    val accuracy: Int = 3,

    @SerializedName("timestamp_epoch_ms")
    val timestampEpochMs: Long = System.currentTimeMillis()
)

/**
 * Server acknowledgement returned over WebSocket or batch REST.
 */
data class ServerAckDto(
    @SerializedName("type")
    val type: String = "ACK",

    @SerializedName("event_id")
    val eventId: String,

    @SerializedName("session_id")
    val sessionId: String,

    @SerializedName("status")
    val status: String,

    @SerializedName("server_receive_timestamp")
    val serverReceiveTimestamp: Long,

    @SerializedName("rejection_reason")
    val rejectionReason: String = "NONE",

    @SerializedName("message")
    val message: String = "",

    @SerializedName("sequence_number")
    val sequenceNumber: Long? = null
)

/**
 * Real-time 2D position update pushed by backend.
 */
data class LiveLocalizationUpdateDto(
    @SerializedName("type")
    val type: String = "LOCALIZATION_UPDATE",

    @SerializedName("session_id")
    val sessionId: String,

    @SerializedName("building_id")
    val buildingId: String = "default_building",

    @SerializedName("timestamp_ms")
    val timestampMs: Long,

    @SerializedName("x_m")
    val xM: Double,

    @SerializedName("y_m")
    val yM: Double,

    @SerializedName("floor_id")
    val floorId: Int,

    @SerializedName("heading_rad")
    val headingRad: Double,

    @SerializedName("uncertainty_radius_m")
    val uncertaintyRadiusM: Double,

    @SerializedName("confidence_level")
    val confidenceLevel: String,

    @SerializedName("source")
    val source: String,

    @SerializedName("source_event_id")
    val sourceEventId: String? = null
)

/**
 * Real-time turn-by-turn guidance update pushed by backend.
 */
data class LiveNavigationUpdateDto(
    @SerializedName("type")
    val type: String = "NAVIGATION_UPDATE",

    @SerializedName("session_id")
    val sessionId: String,

    @SerializedName("navigation_session_id")
    val navigationSessionId: String,

    @SerializedName("timestamp_ms")
    val timestampMs: Long,

    @SerializedName("current_instruction")
    val currentInstruction: String,

    @SerializedName("instruction_type")
    val instructionType: String = "CONTINUE",

    @SerializedName("distance_to_next_waypoint_m")
    val distanceToNextWaypointM: Double,

    @SerializedName("remaining_distance_m")
    val remainingDistanceM: Double,

    @SerializedName("cross_track_error_m")
    val crossTrackErrorM: Double,

    @SerializedName("arrival_status")
    val arrivalStatus: Boolean,

    @SerializedName("reroute_occurred")
    val rerouteOccurred: Boolean,

    @SerializedName("route_status")
    val routeStatus: String,

    @SerializedName("target_destination")
    val targetDestination: String = "",

    @SerializedName("target_floor_id")
    val targetFloorId: Int = 0
)

/**
 * Real-time stream status pushed by backend upon connect.
 */
data class StreamStatusDto(
    @SerializedName("type")
    val type: String = "STREAM_STATUS",

    @SerializedName("session_id")
    val sessionId: String,

    @SerializedName("is_active")
    val isActive: Boolean,

    @SerializedName("connected_clients")
    val connectedClients: Int,

    @SerializedName("events_received")
    val eventsReceived: Long,

    @SerializedName("events_processed")
    val eventsProcessed: Long,

    @SerializedName("events_rejected")
    val eventsRejected: Long,

    @SerializedName("events_dropped")
    val eventsDropped: Long,

    @SerializedName("active_navigation_session_id")
    val activeNavigationSessionId: String? = null
)
