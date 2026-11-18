package ai.indoor.app.realtime

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.util.UUID

/**
 * High-level Android real-time streaming manager coordinating capture sessions,
 * pending event queues, WebSocket streaming, and fallback batch synchronization.
 */
class RealtimeStreamingManager(
    private val pendingQueue: PendingEventQueue = PendingEventQueue(maxCapacity = 1000),
    private val wsClient: RealtimeWebSocketClient = RealtimeWebSocketClient(),
    private val batchClient: RealtimeBatchFallbackClient = RealtimeBatchFallbackClient(),
    private val scope: CoroutineScope = CoroutineScope(Dispatchers.IO)
) {
    private var activeSessionId: String? = null
    private var activeDeviceId: String = "android_device"
    private var activeBaseUrl: String? = null
    private var isStreamingActive = false

    private var drainJob: Job? = null
    private var ackCollectJob: Job? = null

    val connectionState: StateFlow<RealtimeConnectionState> = wsClient.connectionState
    val latestLocalization: StateFlow<LiveLocalizationUpdateDto?> = wsClient.latestLocalization
    val latestNavigation: StateFlow<LiveNavigationUpdateDto?> = wsClient.latestNavigation

    val pendingQueueSize: Int
        get() = pendingQueue.size

    val isStreaming: Boolean
        get() = isStreamingActive

    /**
     * Starts consented real-time streaming for an active capture session.
     */
    fun startStreaming(baseUrl: String, sessionId: String, deviceId: String) {
        if (isStreamingActive && activeSessionId == sessionId) return

        activeBaseUrl = baseUrl
        activeSessionId = sessionId
        activeDeviceId = deviceId
        isStreamingActive = true

        wsClient.connect(baseUrl = baseUrl, sessionId = sessionId)

        // Listen for incoming server ACKs
        ackCollectJob?.cancel()
        ackCollectJob = scope.launch {
            wsClient.incomingAcks.collect { ack ->
                pendingQueue.removeAcknowledged(ack.eventId)
            }
        }

        // Background drain worker for pending queue
        drainJob?.cancel()
        drainJob = scope.launch {
            while (isActive && isStreamingActive) {
                drainPendingQueue()
                delay(1000L)
            }
        }
    }

    /**
     * Enqueues and attempts to stream an individual observation event.
     */
    fun submitEvent(
        eventType: RealtimeEventType,
        payload: Map<String, Any?>,
        clientMonotonic: Long? = null
    ): String {
        val sessionId = activeSessionId ?: return ""
        val eventId = UUID.randomUUID().toString()

        val event = RealtimeEventDto(
            eventId = eventId,
            sessionId = sessionId,
            eventType = eventType,
            deviceId = activeDeviceId,
            clientTimestampMonotonic = clientMonotonic,
            clientTimestampUtc = System.currentTimeMillis(),
            payload = payload
        )

        pendingQueue.enqueue(event)

        // Try direct WebSocket delivery if currently connected
        if (wsClient.connectionState.value == RealtimeConnectionState.CONNECTED) {
            val sent = wsClient.sendEvent(event)
            if (!sent) {
                // Stays in pending queue to be drained
            }
        }

        return eventId
    }

    fun submitWifiScan(scans: List<ApScanItem>, clientMonotonic: Long? = null): String {
        val payload = mapOf(
            "bssid_scans" to scans.map {
                mapOf(
                    "bssid" to it.bssid,
                    "rssi_dbm" to it.rssiDbm,
                    "ssid" to it.ssid,
                    "frequency_mhz" to it.frequencyMhz
                )
            }
        )
        return submitEvent(RealtimeEventType.WIFI_SCAN, payload, clientMonotonic)
    }

    fun submitImuBatch(samples: List<ImuSampleItem>, clientMonotonic: Long? = null): String {
        val payload = mapOf(
            "samples" to samples.map {
                mapOf(
                    "sensor_type" to it.sensorType,
                    "x" to it.x,
                    "y" to it.y,
                    "z" to it.z,
                    "accuracy" to it.accuracy,
                    "timestamp_epoch_ms" to it.timestampEpochMs
                )
            }
        )
        return submitEvent(RealtimeEventType.IMU_BATCH, payload, clientMonotonic)
    }

    fun submitOrientation(headingRad: Double, azimuthDeg: Float, clientMonotonic: Long? = null): String {
        val payload = mapOf(
            "heading_rad" to headingRad,
            "azimuth_deg" to azimuthDeg
        )
        return submitEvent(RealtimeEventType.ORIENTATION_UPDATE, payload, clientMonotonic)
    }

    fun submitPhotoMetadata(
        photoId: String,
        filename: String,
        exifSanitized: Boolean,
        faceCount: Int = 0
    ): String {
        val payload = mapOf(
            "photo_id" to photoId,
            "filename" to filename,
            "exif_sanitized" to exifSanitized,
            "detected_faces_count" to faceCount
        )
        return submitEvent(RealtimeEventType.PHOTO_CAPTURE, payload)
    }

    fun submitLandmark(
        landmarkType: String,
        landmarkId: String? = null,
        distanceM: Double? = null,
        isVerified: Boolean = false
    ): String {
        val payload = mapOf(
            "landmark_type" to landmarkType,
            "landmark_id" to landmarkId,
            "estimated_distance_m" to distanceM,
            "is_verified" to isVerified
        )
        return submitEvent(RealtimeEventType.LANDMARK_OBSERVATION, payload)
    }

    private suspend fun drainPendingQueue() {
        if (pendingQueue.isEmpty) return

        val state = wsClient.connectionState.value
        if (state == RealtimeConnectionState.CONNECTED) {
            val batch = pendingQueue.peekBatch(20)
            for (ev in batch) {
                wsClient.sendEvent(ev)
            }
        } else if (state == RealtimeConnectionState.DISCONNECTED || state == RealtimeConnectionState.FAILED) {
            // Use REST batch fallback
            val baseUrl = activeBaseUrl ?: return
            val sessionId = activeSessionId ?: return
            val batch = pendingQueue.peekBatch(50)
            if (batch.isNotEmpty()) {
                val ackedIds = batchClient.postBatch(baseUrl, sessionId, batch)
                pendingQueue.removeBatchAcknowledged(ackedIds)
            }
        }
    }

    /**
     * Stops streaming, disconnects socket, and resets active session.
     */
    fun stopStreaming() {
        isStreamingActive = false
        drainJob?.cancel()
        ackCollectJob?.cancel()
        wsClient.disconnect()
        activeSessionId = null
    }
}
