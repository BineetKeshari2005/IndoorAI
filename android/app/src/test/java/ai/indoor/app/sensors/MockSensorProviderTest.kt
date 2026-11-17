package ai.indoor.app.sensors

import ai.indoor.app.domain.models.CapturedSensorBatch
import ai.indoor.app.domain.models.SensorSamplingRate
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class MockSensorProviderTest {

    private val testScope = TestScope()
    private lateinit var provider: MockSensorProvider

    @Before
    fun setUp() {
        provider = MockSensorProvider(scope = testScope)
    }

    @Test
    fun testAvailabilityReportsAllSensorsPresentInMock() {
        val availability = provider.availability
        assertTrue(availability.hasAccelerometer)
        assertTrue(availability.hasGyroscope)
        assertTrue(availability.hasMagnetometer)
        assertTrue(availability.hasRotationVector)
        assertTrue(availability.isMock)
        assertTrue(availability.allCoreSensorsPresent)
        assertTrue(availability.hasAnySensor)
    }

    @Test
    fun testStartAndStopListeningTransitions() = runTest {
        assertFalse(provider.isListening.value)

        provider.startListening(sessionId = "SES-TEST-01")
        assertTrue(provider.isListening.value)

        provider.stopListening()
        assertFalse(provider.isListening.value)
    }

    @Test
    fun testSamplingRateConfiguration() {
        assertEquals(SensorSamplingRate.UI, provider.samplingRate.value)

        provider.setSamplingRate(SensorSamplingRate.GAME)
        assertEquals(SensorSamplingRate.GAME, provider.samplingRate.value)

        provider.setSamplingRate(SensorSamplingRate.FASTEST)
        assertEquals(SensorSamplingRate.FASTEST, provider.samplingRate.value)
    }

    @Test
    fun testSensorReadingMagnitudesArePhysicallyPlausible() {
        val sample = provider.sensorReadings.value

        // Earth gravity vector should approximate ~9.81 m/s^2 at rest
        assertTrue(sample.accelerationMagnitude in 9.0f..10.5f)

        // Earth magnetic field should approximate ~20 to ~80 microteslas
        assertTrue(sample.magneticFieldMagnitude in 20.0f..80.0f)

        assertTrue(sample.isMock)
    }

    @Test
    fun testBufferReadingsAndExport() {
        provider.clearCollectedReadings()
        assertEquals(0, provider.collectedReadings.size)

        provider.startListening(sessionId = "SES-EXPORT-TEST")

        val csv = provider.exportReadingsCsv()
        assertTrue(csv.contains(CapturedSensorBatch.CSV_HEADER))

        val json = provider.exportReadingsJson()
        assertNotNull(json)

        provider.stopListening()
    }
}
