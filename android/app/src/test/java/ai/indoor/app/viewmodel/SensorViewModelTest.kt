package ai.indoor.app.viewmodel

import ai.indoor.app.domain.models.CapturedSensorBatch
import ai.indoor.app.domain.models.SensorSamplingRate
import ai.indoor.app.sensors.MockSensorProvider
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class SensorViewModelTest {

    private val testDispatcher = StandardTestDispatcher()
    private val testScope = TestScope(testDispatcher)
    private lateinit var mockProvider: MockSensorProvider
    private lateinit var viewModel: SensorViewModel

    @Before
    fun setUp() {
        Dispatchers.setMain(testDispatcher)
        mockProvider = MockSensorProvider(scope = testScope)
        viewModel = SensorViewModel(provider = mockProvider)
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    @Test
    fun testInitialListeningStateIsFalse() {
        assertFalse(viewModel.isListening.value)
        assertTrue(viewModel.availability.allCoreSensorsPresent)
        assertEquals(SensorSamplingRate.UI, viewModel.samplingRate.value)
        assertNull(viewModel.exportPreview.value)
        assertNull(viewModel.exportFormat.value)
    }

    @Test
    fun testToggleListeningTogglesState() {
        assertFalse(viewModel.isListening.value)

        viewModel.toggleListening()
        assertTrue(viewModel.isListening.value)

        viewModel.toggleListening()
        assertFalse(viewModel.isListening.value)
    }

    @Test
    fun testSamplingRateChanges() {
        viewModel.setSamplingRate(SensorSamplingRate.GAME)
        assertEquals(SensorSamplingRate.GAME, viewModel.samplingRate.value)

        viewModel.setSamplingRate(SensorSamplingRate.FASTEST)
        assertEquals(SensorSamplingRate.FASTEST, viewModel.samplingRate.value)
    }

    @Test
    fun testExportCsvAndJson() {
        viewModel.startListening("SES-VM-TEST")

        val csv = viewModel.exportAsCsv()
        assertNotNull(csv)
        assertTrue(csv.contains(CapturedSensorBatch.CSV_HEADER))
        assertEquals("CSV", viewModel.exportFormat.value)
        assertEquals(csv, viewModel.exportPreview.value)

        val json = viewModel.exportAsJson()
        assertNotNull(json)
        assertEquals("JSON", viewModel.exportFormat.value)
        assertEquals(json, viewModel.exportPreview.value)

        viewModel.clearExportPreview()
        assertNull(viewModel.exportPreview.value)
        assertNull(viewModel.exportFormat.value)

        viewModel.clearCollectedReadings()
        assertEquals(0, viewModel.getRecordedCount())

        viewModel.stopListening()
    }
}
