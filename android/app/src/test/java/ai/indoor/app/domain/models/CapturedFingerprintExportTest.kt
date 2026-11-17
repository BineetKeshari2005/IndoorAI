package ai.indoor.app.domain.models

import com.google.gson.JsonParser
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class CapturedFingerprintExportTest {

    @Test
    fun testCapturedFingerprintToJsonSerialization() {
        val ap1 = AccessPoint("00:14:bf:b1:97:8a", "Eduroam", -55, 5180, 36, 1700000000000L, isFresh = true, isMock = false)
        val ap2 = AccessPoint("00:14:bf:b1:97:8b", "Staff", -68, 2412, 1, 1700000000000L, isFresh = true, isMock = false)
        val fp = WifiFingerprint(listOf(ap1, ap2), timestamp = 1700000000000L, isFresh = true, isMock = false)

        val label = ManualLocationLabel(
            building = "Engineering-North",
            floor = "2",
            room = "Lab-204",
            notes = "Survey point near south exit"
        )
        val metadata = DeviceMetadata(
            manufacturer = "Google",
            model = "Pixel 8",
            sdkInt = 34,
            appVersion = "1.0.0"
        )

        val capture = CapturedFingerprint(
            captureId = "CAP-TEST-001",
            sessionId = "SES-TEST-001",
            timestamp = 1700000000000L,
            fingerprint = fp,
            manualLabel = label,
            deviceMetadata = metadata,
            isMock = false
        )

        val jsonString = capture.toJson()
        assertNotNull(jsonString)

        val parsed = JsonParser.parseString(jsonString).asJsonObject
        assertEquals("CAP-TEST-001", parsed.get("captureId").asString)
        assertEquals("SES-TEST-001", parsed.get("sessionId").asString)
        assertFalse("Must be flagged as real data", parsed.get("isMock").asBoolean)

        val manualJson = parsed.getAsJsonObject("manualLabel")
        assertEquals("Engineering-North", manualJson.get("building").asString)
        assertEquals("2", manualJson.get("floor").asString)
        assertEquals("Lab-204", manualJson.get("room").asString)

        val apList = parsed.getAsJsonObject("fingerprint").getAsJsonArray("accessPoints")
        assertEquals(2, apList.size())
        assertEquals("00:14:bf:b1:97:8a", apList[0].asJsonObject.get("bssid").asString)
    }

    @Test
    fun testCapturedFingerprintToCsvFormatting() {
        val ap = AccessPoint("00:14:bf:b1:97:8a", "Eduroam, Secure", -52, 5180, 36, 1700000000000L, isFresh = true, isMock = false)
        val fp = WifiFingerprint(listOf(ap), timestamp = 1700000000000L, isFresh = true, isMock = false)
        val label = ManualLocationLabel("Hall \"A\"", "1", "Room 101", "Notes with, comma")

        val capture = CapturedFingerprint(
            captureId = "CAP-001",
            sessionId = "SES-001",
            timestamp = 1700000000000L,
            fingerprint = fp,
            manualLabel = label,
            isMock = false
        )

        val csv = capture.toCsv(includeHeader = true)
        assertTrue(csv.startsWith(CapturedFingerprint.CSV_HEADER))

        val lines = csv.trim().lines()
        assertEquals(2, lines.size) // Header + 1 AP row

        val dataLine = lines[1]
        assertTrue(dataLine.contains("CAP-001"))
        assertTrue(dataLine.contains("SES-001"))
        assertTrue(dataLine.contains("\"Eduroam, Secure\"")) // Escaped comma
        assertTrue(dataLine.contains("\"Hall \"\"A\"\"\"")) // Escaped quote
        assertTrue(dataLine.contains("\"Notes with, comma\""))
        assertTrue(dataLine.contains("-52"))
        assertTrue(dataLine.contains("5180"))
        assertTrue(dataLine.contains("36"))
    }

    @Test
    fun testBatchCsvSerializationHeaderOnlyOnce() {
        val ap1 = AccessPoint("00:11:22:33:44:55", "Net1", -50, 2412, 1, 1000L, isMock = false)
        val ap2 = AccessPoint("00:11:22:33:44:66", "Net2", -60, 2437, 6, 2000L, isMock = false)

        val c1 = CapturedFingerprint(
            captureId = "C1",
            sessionId = "S1",
            timestamp = 1000L,
            fingerprint = WifiFingerprint(listOf(ap1), isMock = false),
            manualLabel = ManualLocationLabel("B1", "F1", "R1")
        )
        val c2 = CapturedFingerprint(
            captureId = "C2",
            sessionId = "S1",
            timestamp = 2000L,
            fingerprint = WifiFingerprint(listOf(ap2), isMock = false),
            manualLabel = ManualLocationLabel("B1", "F1", "R2")
        )

        val batchCsv = CapturedFingerprint.toBatchCsv(listOf(c1, c2))
        val lines = batchCsv.trim().lines()

        // 1 header + 2 AP data lines = 3 lines total
        assertEquals(3, lines.size)
        assertEquals(CapturedFingerprint.CSV_HEADER, lines[0])
        assertTrue(lines[1].contains("C1"))
        assertTrue(lines[2].contains("C2"))
    }

    @Test
    fun testEmptyFingerprintCsvProducesPlaceholderRow() {
        val emptyFp = WifiFingerprint(emptyList(), isFresh = false, isMock = false)
        val capture = CapturedFingerprint(
            captureId = "C-EMPTY",
            sessionId = "S-EMPTY",
            timestamp = 1000L,
            fingerprint = emptyFp,
            manualLabel = ManualLocationLabel("Bldg-X", "0", "Basement")
        )

        val csv = capture.toCsv(includeHeader = false)
        val lines = csv.trim().lines()
        assertEquals(1, lines.size)
        assertTrue(lines[0].contains("C-EMPTY"))
        assertTrue(lines[0].contains("Bldg-X"))
        assertTrue(lines[0].contains("Basement"))
    }

    @Test
    fun testMockFlagPreservedAndNotMixed() {
        val realAp = AccessPoint("00:11:22:33:44:55", "Real", -50, 2412, isMock = false)
        val mockAp = AccessPoint("00:14:bf:b1:97:8a", "eduroam", -48, 5180, isMock = true)

        val realCap = CapturedFingerprint(fingerprint = WifiFingerprint(listOf(realAp), isMock = false), isMock = false)
        val mockCap = CapturedFingerprint(fingerprint = WifiFingerprint(listOf(mockAp), isMock = true), isMock = true)

        assertFalse(realCap.isMock)
        assertTrue(mockCap.isMock)

        val realCsv = realCap.toCsv(includeHeader = false)
        val mockCsv = mockCap.toCsv(includeHeader = false)

        assertTrue("Real CSV row must indicate is_mock=false", realCsv.contains(",false,"))
        assertTrue("Mock CSV row must indicate is_mock=true", mockCsv.contains(",true,"))
    }
}
