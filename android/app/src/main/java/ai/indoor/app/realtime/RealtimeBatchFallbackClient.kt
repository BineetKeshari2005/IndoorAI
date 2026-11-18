package ai.indoor.app.realtime

import com.google.gson.Gson
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.IOException

/**
 * Fallback REST client for uploading batches of buffered events when WebSocket is disconnected.
 */
class RealtimeBatchFallbackClient(
    private val okHttpClient: OkHttpClient = OkHttpClient(),
    private val gson: Gson = Gson()
) {
    private val jsonMediaType = "application/json; charset=utf-8".toMediaType()

    /**
     * Submits a batch of events via HTTP POST to /api/realtime/sessions/{sessionId}/events/batch.
     *
     * Returns:
     *   List of acknowledged event IDs that were successfully processed by server.
     */
    suspend fun postBatch(
        baseUrl: String,
        sessionId: String,
        events: List<RealtimeEventDto>
    ): List<String> = withContext(Dispatchers.IO) {
        if (events.isEmpty()) return@withContext emptyList()

        val cleanBase = baseUrl.trimEnd('/')
        val url = "$cleanBase/api/realtime/sessions/$sessionId/events/batch"

        val bodyMap = mapOf("events" to events)
        val jsonPayload = gson.toJson(bodyMap)
        val requestBody = jsonPayload.toRequestBody(jsonMediaType)

        val request = Request.Builder()
            .url(url)
            .post(requestBody)
            .build()

        val acknowledgedIds = mutableListOf<String>()

        try {
            val response = okHttpClient.newCall(request).execute()
            response.use { resp ->
                if (resp.isSuccessful) {
                    val respText = resp.body?.string() ?: return@withContext emptyList()
                    val jsonObject = JsonParser.parseString(respText).asJsonObject
                    val acksArray = jsonObject.getAsJsonArray("acknowledgements")
                    if (acksArray != null) {
                        for (element in acksArray) {
                            val ackObj = element.asJsonObject
                            val eventId = ackObj.get("event_id")?.asString
                            val status = ackObj.get("status")?.asString
                            if (eventId != null && (status == "ACCEPTED" || status == "DUPLICATE")) {
                                acknowledgedIds.add(eventId)
                            }
                        }
                    }
                }
            }
        } catch (_: IOException) {
            // Network outage handled gracefully
        }

        return@withContext acknowledgedIds
    }
}
