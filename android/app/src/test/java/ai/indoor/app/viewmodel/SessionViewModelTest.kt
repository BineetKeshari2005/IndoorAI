package ai.indoor.app.viewmodel

import ai.indoor.app.data.repository.LocalizationRepository
import ai.indoor.app.domain.models.CaptureSession
import ai.indoor.app.domain.models.LocalizationResult
import ai.indoor.app.domain.models.WifiFingerprint
import ai.indoor.app.domain.usecases.ManageCaptureSessionUseCase
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
class SessionViewModelTest {

    private val testDispatcher = StandardTestDispatcher()
    private val testScope = TestScope(testDispatcher)

    private val fakeRepository = object : LocalizationRepository {
        override suspend fun checkServerHealth(): Result<Boolean> = Result.success(true)
        override suspend fun estimateLocation(fingerprint: WifiFingerprint): Result<LocalizationResult> =
            Result.success(LocalizationResult(0, 1, 1, -7500.0, 4864800.0, 0.9f))
        override suspend fun syncSessionTelemetry(session: CaptureSession): Result<String> =
            Result.success("Synced 10 records")
        override fun updateBaseUrl(newUrl: String) {}
        override fun getBaseUrl(): String = "http://10.0.2.2:8000/"
    }

    private lateinit var useCase: ManageCaptureSessionUseCase
    private lateinit var sensorProvider: MockSensorProvider
    private lateinit var viewModel: SessionViewModel

    @Before
    fun setUp() {
        Dispatchers.setMain(testDispatcher)
        useCase = ManageCaptureSessionUseCase(fakeRepository)
        sensorProvider = MockSensorProvider(scope = testScope)
        viewModel = SessionViewModel(useCase = useCase, sensorProvider = sensorProvider)
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    @Test
    fun testInitialSessionIsNull() {
        assertNull(viewModel.currentSession.value)
        assertEquals(0L, viewModel.elapsedSeconds.value)
        assertTrue(viewModel.streamSensors.value)
    }

    @Test
    fun testStartSessionCreatesActiveSessionAndStreamsSensors() {
        viewModel.startSession()

        val session = viewModel.currentSession.value
        assertNotNull(session)
        assertTrue(session!!.isRunning)
        assertEquals(0, session.wifiScanCount)
        assertEquals(0, session.photoCount)

        // Sensor provider should be listening when streamSensors is true
        assertTrue(sensorProvider.isListening.value)
    }

    @Test
    fun testTelemetryCountersIncrementCorrectly() {
        viewModel.startSession()

        viewModel.addMockObservation()
        viewModel.addMockObservation()
        viewModel.addPhotoObservation()

        val session = viewModel.currentSession.value
        assertNotNull(session)
        assertEquals(2, session!!.wifiScanCount)
        assertEquals(2, session.imuSampleCount)
        assertEquals(1, session.photoCount)
    }

    @Test
    fun testStopSessionStopsRunningStateAndSensors() {
        viewModel.startSession()
        assertTrue(viewModel.currentSession.value!!.isRunning)
        assertTrue(sensorProvider.isListening.value)

        viewModel.stopSession()
        assertFalse(viewModel.currentSession.value!!.isRunning)
        assertNotNull(viewModel.currentSession.value!!.endTimeMs)
        assertFalse(sensorProvider.isListening.value)
    }

    @Test
    fun testToggleSensorStreamingDuringActiveSession() {
        viewModel.startSession()
        assertTrue(sensorProvider.isListening.value)

        viewModel.toggleSensorStreaming(false)
        assertFalse(viewModel.streamSensors.value)
        assertFalse(sensorProvider.isListening.value)

        viewModel.toggleSensorStreaming(true)
        assertTrue(viewModel.streamSensors.value)
        assertTrue(sensorProvider.isListening.value)

        viewModel.stopSession()
    }
}
