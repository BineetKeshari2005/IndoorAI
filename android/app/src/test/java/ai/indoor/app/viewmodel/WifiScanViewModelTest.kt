package ai.indoor.app.viewmodel

import ai.indoor.app.domain.adapter.ModelAdaptationResult
import ai.indoor.app.domain.adapter.WifiFingerprintModelAdapter
import ai.indoor.app.domain.models.AccessPoint
import ai.indoor.app.domain.models.CapturedFingerprint
import ai.indoor.app.domain.models.LocalizationResult
import ai.indoor.app.domain.models.WifiFingerprint
import ai.indoor.app.wifi.MockWifiScanner
import ai.indoor.app.wifi.WifiScanState
import ai.indoor.app.wifi.WifiScanner
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
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
class WifiScanViewModelTest {

    private val testDispatcher = StandardTestDispatcher()
    private lateinit var mockScanner: MockWifiScanner
    private lateinit var viewModel: WifiScanViewModel

    @Before
    fun setUp() {
        Dispatchers.setMain(testDispatcher)
        mockScanner = MockWifiScanner()
        viewModel = WifiScanViewModel(scanner = mockScanner)
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    @Test
    fun testInitialState() {
        assertEquals(WifiScanState.Idle, viewModel.scanState.value)
        assertTrue(viewModel.isMockMode.value)
        assertNull(viewModel.lastScanTime.value)
        assertNull(viewModel.currentFingerprint.value)
        assertNull(viewModel.localizationResult.value)
        assertTrue(viewModel.capturedFingerprints.value.isEmpty())
    }

    @Test
    fun testTriggerScanUpdatesStateAndTimestamp() = runTest {
        viewModel.triggerScan()
        testDispatcher.scheduler.advanceUntilIdle()

        assertNotNull(viewModel.lastScanTime.value)
        val state = viewModel.scanState.value
        assertTrue(state is WifiScanState.Success)
        assertNotNull(viewModel.currentFingerprint.value)
    }

    @Test
    fun testManualLabelingAndSavingCapturedFingerprint() = runTest {
        viewModel.triggerScan()
        testDispatcher.scheduler.advanceUntilIdle()

        viewModel.updateManualLabels(
            building = "Engineering-North",
            floor = "3",
            room = "Lab-302",
            notes = "Test scan near doorway"
        )

        assertEquals("Engineering-North", viewModel.manualBuilding.value)
        assertEquals("3", viewModel.manualFloor.value)
        assertEquals("Lab-302", viewModel.manualRoom.value)
        assertEquals("Test scan near doorway", viewModel.manualNotes.value)

        viewModel.saveCurrentFingerprint()

        val captures = viewModel.capturedFingerprints.value
        assertEquals(1, captures.size)
        val capture = captures.first()
        assertEquals("Engineering-North", capture.manualLabel?.building)
        assertEquals("3", capture.manualLabel?.floor)
        assertEquals("Lab-302", capture.manualLabel?.room)
        assertEquals("Test scan near doorway", capture.manualLabel?.notes)
    }

    @Test
    fun testDeleteAndClearCapturedFingerprints() = runTest {
        viewModel.triggerScan()
        testDispatcher.scheduler.advanceUntilIdle()

        viewModel.saveCurrentFingerprint()
        assertEquals(1, viewModel.capturedFingerprints.value.size)

        val captureId = viewModel.capturedFingerprints.value.first().captureId
        viewModel.deleteCapture(captureId)
        assertTrue(viewModel.capturedFingerprints.value.isEmpty())

        // Save again and clear
        viewModel.saveCurrentFingerprint()
        assertEquals(1, viewModel.capturedFingerprints.value.size)
        viewModel.clearCapturedFingerprints()
        assertTrue(viewModel.capturedFingerprints.value.isEmpty())
    }

    @Test
    fun testJsonAndCsvExportGeneration() = runTest {
        viewModel.triggerScan()
        testDispatcher.scheduler.advanceUntilIdle()

        viewModel.updateManualLabels("TI-Hall", "2", "TI-204", "Near router")
        viewModel.saveCurrentFingerprint()

        val json = viewModel.exportAsJson()
        assertNotNull(json)
        assertTrue(json.contains("TI-Hall"))
        assertTrue(json.contains("TI-204"))
        assertEquals("JSON", viewModel.exportFormat.value)
        assertEquals(json, viewModel.exportPreview.value)

        val csv = viewModel.exportAsCsv()
        assertNotNull(csv)
        assertTrue(csv.contains(CapturedFingerprint.CSV_HEADER))
        assertTrue(csv.contains("TI-Hall"))
        assertEquals("CSV", viewModel.exportFormat.value)
        assertEquals(csv, viewModel.exportPreview.value)

        viewModel.clearExportPreview()
        assertNull(viewModel.exportPreview.value)
        assertNull(viewModel.exportFormat.value)
    }

    @Test
    fun testRealWorldFingerprintBlocksArbitraryWapModelInference() = runTest {
        // Real-world scanner observing non-campus BSSIDs
        val realAp = AccessPoint(
            bssid = "00:1a:2b:3c:4d:5e",
            ssid = "Office-Guest",
            rssi = -60,
            frequency = 2412,
            isFresh = true,
            isMock = false
        )
        val realScanner = object : WifiScanner {
            private val _state = MutableStateFlow<WifiScanState>(WifiScanState.Idle)
            override val scanState: StateFlow<WifiScanState> = _state
            override val isMockProvider: Boolean = false

            override suspend fun triggerScan(): Result<WifiFingerprint> {
                val fp = WifiFingerprint(listOf(realAp), isFresh = true, isMock = false)
                _state.value = WifiScanState.Success(fp, isFresh = true)
                return Result.success(fp)
            }

            override fun getCachedScanResults(): WifiFingerprint =
                WifiFingerprint(listOf(realAp), isFresh = false, isMock = false)
        }

        val vm = WifiScanViewModel(scanner = realScanner)
        vm.triggerScan()
        testDispatcher.scheduler.advanceUntilIdle()

        // Verify calibration notice is populated and live localization is blocked
        assertNotNull(vm.modelAdaptationNotice.value)
        assertTrue(vm.modelAdaptationNotice.value!!.contains("calibration"))
        assertNull(vm.localizationResult.value)

        // Attempting demo localization on real data must be refused
        vm.requestDemoLocalization()
        testDispatcher.scheduler.advanceUntilIdle()
        assertNull(vm.localizationResult.value)
        assertTrue(vm.modelAdaptationNotice.value!!.contains("Blocked"))
    }

    @Test
    fun testPermissionRequiredStateHandling() = runTest {
        val permScanner = object : WifiScanner {
            private val _state = MutableStateFlow<WifiScanState>(WifiScanState.Idle)
            override val scanState: StateFlow<WifiScanState> = _state
            override val isMockProvider: Boolean = false

            override suspend fun triggerScan(): Result<WifiFingerprint> {
                _state.value = WifiScanState.PermissionRequired(listOf("android.permission.ACCESS_FINE_LOCATION"))
                return Result.failure(SecurityException("Missing fine location"))
            }

            override fun getCachedScanResults(): WifiFingerprint = WifiFingerprint(emptyList())
        }

        val vm = WifiScanViewModel(scanner = permScanner)
        vm.triggerScan()
        testDispatcher.scheduler.advanceUntilIdle()

        val state = vm.scanState.value
        assertTrue(state is WifiScanState.PermissionRequired)
        assertEquals(listOf("android.permission.ACCESS_FINE_LOCATION"), (state as WifiScanState.PermissionRequired).missingPermissions)
    }

    @Test
    fun testLocationServicesDisabledState() = runTest {
        val locScanner = object : WifiScanner {
            private val _state = MutableStateFlow<WifiScanState>(WifiScanState.Idle)
            override val scanState: StateFlow<WifiScanState> = _state
            override val isMockProvider: Boolean = false

            override suspend fun triggerScan(): Result<WifiFingerprint> {
                _state.value = WifiScanState.LocationServicesRequired("GPS turned off")
                return Result.failure(IllegalStateException("GPS turned off"))
            }

            override fun getCachedScanResults(): WifiFingerprint = WifiFingerprint(emptyList())
        }

        val vm = WifiScanViewModel(scanner = locScanner)
        vm.triggerScan()
        testDispatcher.scheduler.advanceUntilIdle()

        val state = vm.scanState.value
        assertTrue(state is WifiScanState.LocationServicesRequired)
    }

    @Test
    fun testWifiDisabledState() = runTest {
        val disabledScanner = object : WifiScanner {
            private val _state = MutableStateFlow<WifiScanState>(WifiScanState.Idle)
            override val scanState: StateFlow<WifiScanState> = _state
            override val isMockProvider: Boolean = false

            override suspend fun triggerScan(): Result<WifiFingerprint> {
                _state.value = WifiScanState.WifiDisabled("Wi-Fi adapter off")
                return Result.failure(IllegalStateException("Wi-Fi adapter off"))
            }

            override fun getCachedScanResults(): WifiFingerprint = WifiFingerprint(emptyList())
        }

        val vm = WifiScanViewModel(scanner = disabledScanner)
        vm.triggerScan()
        testDispatcher.scheduler.advanceUntilIdle()

        val state = vm.scanState.value
        assertTrue(state is WifiScanState.WifiDisabled)
    }

    @Test
    fun testThrottledScanState() = runTest {
        val throttledScanner = object : WifiScanner {
            private val _state = MutableStateFlow<WifiScanState>(WifiScanState.Idle)
            override val scanState: StateFlow<WifiScanState> = _state
            override val isMockProvider: Boolean = false

            override suspend fun triggerScan(): Result<WifiFingerprint> {
                val cached = WifiFingerprint(emptyList(), isFresh = false)
                _state.value = WifiScanState.Throttled("Foreground rate limit: 4 scans / 2 mins", cached)
                return Result.success(cached)
            }

            override fun getCachedScanResults(): WifiFingerprint = WifiFingerprint(emptyList())
        }

        val vm = WifiScanViewModel(scanner = throttledScanner)
        vm.triggerScan()
        testDispatcher.scheduler.advanceUntilIdle()

        val state = vm.scanState.value
        assertTrue(state is WifiScanState.Throttled)
    }
}
