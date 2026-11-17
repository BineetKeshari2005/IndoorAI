package ai.indoor.app.domain.models

import com.google.gson.JsonParser
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class CapturedSensorBatchExportTest {

    @Test
    fun testSingleReadingMagnitudeCalculation() {
        val reading = SingleSensorReading(
            sensorType = SensorType.ACCELEROMETER,
            x = 3.0f,
            y = 4.0f,
            z = 0.0f,
            timestampNs = 123456789000L,
            accuracy = SensorAccuracy.HIGH,
            sessionId = "SES-TEST"
        )

        // sqrt(3^2 + 4^2) = 5.0
        assertEquals(5.0f, reading.magnitude, 0.001f)
        assertEquals(123456789000L, reading.timestampNs)
        assertEquals("SES-TEST", reading.sessionId)
    }

    @Test
    fun testCapturedSensorBatchToJsonSerialization() {
        val r1 = SingleSensorReading(
            sensorType = SensorType.ACCELEROMETER,
            x = 0.1f, y = 0.2f, z = 9.8f,
            timestampNs = 1000000000L,
            accuracy = SensorAccuracy.HIGH,
            sessionId = "SES-001",
            samplingRate = "UI (~16 Hz)",
            isMock = false
        )
        val r2 = SingleSensorReading(
            sensorType = SensorType.GYROSCOPE,
            x = 0.01f, y = 0.02f, z = 0.03f,
            timestampNs = 1000005000L,
            accuracy = SensorAccuracy.MEDIUM,
            sessionId = "SES-001",
            samplingRate = "UI (~16 Hz)",
            isMock = false
        )

        val batch = CapturedSensorBatch(
            sessionId = "SES-001",
            readings = listOf(r1, r2),
            samplingRate = "UI (~16 Hz)",
            isMock = false
        )

        val json = batch.toJson()
        assertNotNull(json)

        val parsed = JsonParser.parseString(json).asJsonObject
        assertEquals("SES-001", parsed.get("sessionId").asString)
        assertFalse(parsed.get("isMock").asBoolean)

        val readingsArray = parsed.getAsJsonArray("readings")
        assertEquals(2, readingsArray.size())

        val firstObj = readingsArray[0].asJsonObject
        assertEquals("ACCELEROMETER", firstObj.get("sensorType").asString)
        assertEquals(1000000000L, firstObj.get("timestampNs").asLong)
    }

    @Test
    fun testCapturedSensorBatchToCsvFormatting() {
        val r = SingleSensorReading(
            sensorType = SensorType.MAGNETIC_FIELD,
            x = 22.4f, y = -5.8f, z = 41.2f,
            timestampNs = 987654321000L,
            accuracy = SensorAccuracy.HIGH,
            sessionId = "SES-MAG-01",
            samplingRate = "Game (~50 Hz)",
            isMock = false
        )

        val batch = CapturedSensorBatch(
            sessionId = "SES-MAG-01",
            readings = listOf(r),
            samplingRate = "Game (~50 Hz)",
            isMock = false
        )

        val csv = batch.toCsv(includeHeader = true)
        assertTrue(csv.startsWith(CapturedSensorBatch.CSV_HEADER))

        val lines = csv.trim().lines()
        assertEquals(2, lines.size) // Header + 1 data line

        val data = lines[1]
        assertTrue(data.contains("SES-MAG-01"))
        assertTrue(data.contains("MAGNETIC_FIELD"))
        assertTrue(data.contains("22.4"))
        assertTrue(data.contains("-5.8"))
        assertTrue(data.contains("41.2"))
        assertTrue(data.contains("987654321000"))
        assertTrue(data.contains("HIGH"))
        assertTrue(data.contains("Game (~50 Hz)"))
        assertTrue(data.contains("false"))
    }

    @Test
    fun testBatchCsvCombinesMultipleBatchesWithSingleHeader() {
        val r1 = SingleSensorReading(SensorType.ACCELEROMETER, 0f, 0f, 9.8f, 100L, SensorAccuracy.HIGH, "S1")
        val r2 = SingleSensorReading(SensorType.GYROSCOPE, 0f, 0f, 0.1f, 200L, SensorAccuracy.HIGH, "S2")

        val b1 = CapturedSensorBatch("S1", listOf(r1))
        val b2 = CapturedSensorBatch("S2", listOf(r2))

        val combinedCsv = CapturedSensorBatch.toBatchCsv(listOf(b1, b2))
        val lines = combinedCsv.trim().lines()

        assertEquals(3, lines.size) // 1 Header + 2 data rows
        assertEquals(CapturedSensorBatch.CSV_HEADER, lines[0])
        assertTrue(lines[1].contains("S1"))
        assertTrue(lines[2].contains("S2"))
    }
}
