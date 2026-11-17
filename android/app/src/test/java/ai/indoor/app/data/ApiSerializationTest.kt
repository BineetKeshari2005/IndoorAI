package ai.indoor.app.data

import ai.indoor.app.data.models.HealthResponseDto
import ai.indoor.app.data.models.ImuObservationDto
import ai.indoor.app.data.models.LocalizationRequestDto
import ai.indoor.app.data.models.LocalizationResponseDto
import ai.indoor.app.data.models.SessionTelemetryDto
import ai.indoor.app.data.models.WifiScanResultDto
import com.google.gson.Gson
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ApiSerializationTest {

    private val gson = Gson()

    @Test
    fun testLocalizationRequestSerialization() {
        val wifiDto = WifiScanResultDto(
            bssid = "00:14:bf:b1:97:8a",
            ssid = "eduroam",
            rssi = -55,
            frequency = 5180,
            timestamp = 1700000000000L
        )

        val imuDto = ImuObservationDto(
            ax = 0.1f, ay = 0.2f, az = 9.8f,
            gx = 0.01f, gy = 0.02f, gz = 0.03f,
            mx = 20f, my = 10f, mz = 40f,
            timestamp = 1700000000000L
        )

        val request = LocalizationRequestDto(
            deviceId = "DEV-TEST-001",
            timestamp = 1700000000000L,
            modelPreference = "multitask_mlp",
            wifiFingerprint = listOf(wifiDto),
            imuSample = imuDto
        )

        val json = gson.toJson(request)
        assertTrue(json.contains("\"device_id\":\"DEV-TEST-001\""))
        assertTrue(json.contains("\"bssid\":\"00:14:bf:b1:97:8a\""))
        assertTrue(json.contains("\"rssi\":-55"))
        assertTrue(json.contains("\"model_preference\":\"multitask_mlp\""))

        val deserialized = gson.fromJson(json, LocalizationRequestDto::class.java)
        assertEquals("DEV-TEST-001", deserialized.deviceId)
        assertEquals(1, deserialized.wifiFingerprint.size)
        assertEquals(-55, deserialized.wifiFingerprint[0].rssi)
        assertNotNull(deserialized.imuSample)
        assertEquals(9.8f, deserialized.imuSample!!.az, 0.001f)
    }

    @Test
    fun testLocalizationResponseDeserialization() {
        val json = """
            {
                "building_id": 1,
                "floor": 2,
                "building_floor_index": 6,
                "longitude": -7536.62,
                "latitude": 4864934.12,
                "confidence": 0.94,
                "model_used": "multitask_mlp",
                "inference_latency_ms": 16,
                "timestamp": 1700000000500
            }
        """.trimIndent()

        val response = gson.fromJson(json, LocalizationResponseDto::class.java)
        assertEquals(1, response.buildingId)
        assertEquals(2, response.floor)
        assertEquals(6, response.buildingFloorIndex)
        assertEquals(-7536.62, response.longitude, 0.01)
        assertEquals(4864934.12, response.latitude, 0.01)
        assertEquals(0.94f, response.confidence, 0.001f)
        assertEquals("multitask_mlp", response.modelUsed)
        assertEquals(16L, response.inferenceLatencyMs)
    }

    @Test
    fun testHealthResponseDeserialization() {
        val json = """{"status":"ok","version":"0.1.0","service":"indoorai-backend"}"""
        val response = gson.fromJson(json, HealthResponseDto::class.java)
        assertEquals("ok", response.status)
        assertEquals("0.1.0", response.version)
        assertEquals("indoorai-backend", response.service)
    }

    @Test
    fun testSessionTelemetrySerialization() {
        val sessionDto = SessionTelemetryDto(
            sessionId = "SES-ABC123",
            deviceId = "DEV-999",
            startTimeMs = 1700000000000L,
            endTimeMs = 1700000060000L,
            photoCount = 2
        )

        val json = gson.toJson(sessionDto)
        assertTrue(json.contains("\"session_id\":\"SES-ABC123\""))
        assertTrue(json.contains("\"photo_count\":2"))

        val deserialized = gson.fromJson(json, SessionTelemetryDto::class.java)
        assertEquals("SES-ABC123", deserialized.sessionId)
        assertEquals(2, deserialized.photoCount)
    }
}

