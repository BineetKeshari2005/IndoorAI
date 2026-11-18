package ai.indoor.app.realtime

import com.google.gson.Gson
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import java.util.concurrent.TimeUnit

/**
 * WebSocket client communicating with FastAPI real-time streaming endpoint.
 *
 * Handles exponential backoff reconnection, heartbeat probes, incoming server ACKs,
 * live localization updates, and live navigation HUD updates.
 */
class RealtimeWebSocketClient(
    private val okHttpClient: OkHttpClient = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .build(),
    private val gson: Gson = Gson(),
    private val scope: CoroutineScope = CoroutineScope(Dispatchers.IO)
) {
    private var webSocket: WebSocket? = null
    private var reconnectJob: Job? = null
    private var heartbeatJob: Job? = null

    private var currentSessionId: String? = null
    private var currentBaseUrl: String? = null
    private var lastAckEventId: String? = null
    private var reconnectAttempts = 0
    private var isUserStopped = false

    private val _connectionState = MutableStateFlow(RealtimeConnectionState.DISCONNECTED)
    val connectionState: StateFlow<RealtimeConnectionState> = _connectionState.asStateFlow()

    private val _incomingAcks = MutableSharedFlow<ServerAckDto>(extraBufferCapacity = 64)
    val incomingAcks: SharedFlow<ServerAckDto> = _incomingAcks.asSharedFlow()

    private val _latestLocalization = MutableStateFlow<LiveLocalizationUpdateDto?>(null)
    val latestLocalization: StateFlow<LiveLocalizationUpdateDto?> = _latestLocalization.asStateFlow()

    private val _latestNavigation = MutableStateFlow<LiveNavigationUpdateDto?>(null)
    val latestNavigation: StateFlow<LiveNavigationUpdateDto?> = _latestNavigation.asStateFlow()

    /**
     * Connects to the real-time session stream.
     */
    fun connect(baseUrl: String, sessionId: String, lastAckId: String? = null) {
        currentBaseUrl = baseUrl
        currentSessionId = sessionId
        lastAckEventId = lastAckId
        isUserStopped = false
        reconnectAttempts = 0

        establishSocket()
    }

    private fun establishSocket() {
        val baseUrl = currentBaseUrl ?: return
        val sessionId = currentSessionId ?: return

        _connectionState.value = RealtimeConnectionState.CONNECTING

        val wsUrl = buildWebSocketUrl(baseUrl, sessionId, lastAckEventId)
        val request = Request.Builder().url(wsUrl).build()

        webSocket = okHttpClient.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(ws: WebSocket, response: Response) {
                _connectionState.value = RealtimeConnectionState.CONNECTED
                reconnectAttempts = 0
                startHeartbeat()
            }

            override fun onMessage(ws: WebSocket, text: String) {
                handleIncomingMessage(text)
            }

            override fun onClosing(ws: WebSocket, code: Int, reason: String) {
                ws.close(1000, null)
                _connectionState.value = RealtimeConnectionState.DISCONNECTED
            }

            override fun onClosed(ws: WebSocket, code: Int, reason: String) {
                _connectionState.value = RealtimeConnectionState.DISCONNECTED
                stopHeartbeat()
            }

            override fun onFailure(ws: WebSocket, t: Throwable, response: Response?) {
                stopHeartbeat()
                if (!isUserStopped) {
                    _connectionState.value = RealtimeConnectionState.RECONNECTING
                    scheduleReconnect()
                } else {
                    _connectionState.value = RealtimeConnectionState.DISCONNECTED
                }
            }
        })
    }

    private fun buildWebSocketUrl(baseUrl: String, sessionId: String, lastAckId: String?): String {
        val cleanBase = baseUrl.trimEnd('/')
            .replace("http://", "ws://")
            .replace("https://", "wss://")
        val endpoint = "$cleanBase/api/realtime/sessions/$sessionId/stream"
        return if (!lastAckId.isNullOrBlank()) {
            "$endpoint?last_ack_event_id=$lastAckId"
        } else {
            endpoint
        }
    }

    /**
     * Sends a typed event over the active WebSocket.
     */
    fun sendEvent(event: RealtimeEventDto): Boolean {
        val ws = webSocket
        if (ws != null && _connectionState.value == RealtimeConnectionState.CONNECTED) {
            val json = gson.toJson(event)
            return ws.send(json)
        }
        return false
    }

    /**
     * Sends a PING heartbeat frame.
     */
    fun sendPing(): Boolean {
        val ws = webSocket
        if (ws != null && _connectionState.value == RealtimeConnectionState.CONNECTED) {
            val ping = JsonObject().apply { addProperty("type", "PING") }
            return ws.send(ping.toString())
        }
        return false
    }

    private fun handleIncomingMessage(text: String) {
        try {
            val jsonObject = JsonParser.parseString(text).asJsonObject
            val type = jsonObject.get("type")?.asString ?: return

            when (type) {
                "ACK" -> {
                    val ack = gson.fromJson(text, ServerAckDto::class.java)
                    lastAckEventId = ack.eventId
                    scope.launch { _incomingAcks.emit(ack) }
                }
                "LOCALIZATION_UPDATE" -> {
                    val loc = gson.fromJson(text, LiveLocalizationUpdateDto::class.java)
                    _latestLocalization.value = loc
                }
                "NAVIGATION_UPDATE" -> {
                    val nav = gson.fromJson(text, LiveNavigationUpdateDto::class.java)
                    _latestNavigation.value = nav
                }
                "PONG" -> {
                    // Heartbeat acknowledged
                }
            }
        } catch (_: Exception) {
            // Safe ignore malformed frame
        }
    }

    private fun scheduleReconnect() {
        if (isUserStopped) return
        reconnectJob?.cancel()
        reconnectJob = scope.launch {
            reconnectAttempts++
            val delayMs = minOf(30000L, 1000L * (1L shl minOf(reconnectAttempts, 5)))
            delay(delayMs)
            if (!isUserStopped) {
                establishSocket()
            }
        }
    }

    private fun startHeartbeat() {
        stopHeartbeat()
        heartbeatJob = scope.launch {
            while (true) {
                delay(15000L) // Ping every 15s
                sendPing()
            }
        }
    }

    private fun stopHeartbeat() {
        heartbeatJob?.cancel()
        heartbeatJob = null
    }

    /**
     * Disconnects the socket and disables auto-reconnect.
     */
    fun disconnect() {
        isUserStopped = true
        reconnectJob?.cancel()
        stopHeartbeat()
        webSocket?.close(1000, "User disconnected")
        webSocket = null
        _connectionState.value = RealtimeConnectionState.DISCONNECTED
    }
}
