package ai.indoor.app.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import ai.indoor.app.data.repository.LocalizationRepository
import ai.indoor.app.domain.adapter.ModelAdaptationResult
import ai.indoor.app.domain.adapter.WifiFingerprintModelAdapter
import ai.indoor.app.domain.models.CapturedFingerprint
import ai.indoor.app.domain.models.DeviceMetadata
import ai.indoor.app.domain.models.LocalizationResult
import ai.indoor.app.domain.models.ManualLocationLabel
import ai.indoor.app.domain.models.WifiFingerprint
import ai.indoor.app.wifi.MockWifiScanner
import ai.indoor.app.wifi.WifiScanState
import ai.indoor.app.wifi.WifiScanner
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.util.UUID

/**
 * ViewModel managing Wi-Fi fingerprint scanning, ground-truth dataset capture,
 * CSV/JSON export, and UJIIndoorLoc calibration boundaries.
 */
class WifiScanViewModel(
    private var scanner: WifiScanner = MockWifiScanner(),
    private val repository: LocalizationRepository? = null,
    private val modelAdapter: WifiFingerprintModelAdapter = WifiFingerprintModelAdapter(),
    private val sessionUseCase: ai.indoor.app.domain.usecases.ManageCaptureSessionUseCase? = null
) : ViewModel() {

    val scanState: StateFlow<WifiScanState> = scanner.scanState

    private val _isMockMode = MutableStateFlow(scanner.isMockProvider)
    val isMockMode: StateFlow<Boolean> = _isMockMode.asStateFlow()

    private val _lastScanTime = MutableStateFlow<Long?>(null)
    val lastScanTime: StateFlow<Long?> = _lastScanTime.asStateFlow()

    private val _currentFingerprint = MutableStateFlow<WifiFingerprint?>(null)
    val currentFingerprint: StateFlow<WifiFingerprint?> = _currentFingerprint.asStateFlow()

    private val _capturedFingerprints = MutableStateFlow<List<CapturedFingerprint>>(emptyList())
    val capturedFingerprints: StateFlow<List<CapturedFingerprint>> = _capturedFingerprints.asStateFlow()

    private val _modelAdaptationNotice = MutableStateFlow<String?>(null)
    val modelAdaptationNotice: StateFlow<String?> = _modelAdaptationNotice.asStateFlow()

    private val _localizationResult = MutableStateFlow<LocalizationResult?>(null)
    val localizationResult: StateFlow<LocalizationResult?> = _localizationResult.asStateFlow()

    private val _isLocalizing = MutableStateFlow(false)
    val isLocalizing: StateFlow<Boolean> = _isLocalizing.asStateFlow()

    // Export preview state
    private val _exportPreview = MutableStateFlow<String?>(null)
    val exportPreview: StateFlow<String?> = _exportPreview.asStateFlow()

    private val _exportFormat = MutableStateFlow<String?>(null)
    val exportFormat: StateFlow<String?> = _exportFormat.asStateFlow()

    // Manual location labeling fields
    private val _manualBuilding = MutableStateFlow("")
    val manualBuilding: StateFlow<String> = _manualBuilding.asStateFlow()

    private val _manualFloor = MutableStateFlow("")
    val manualFloor: StateFlow<String> = _manualFloor.asStateFlow()

    private val _manualRoom = MutableStateFlow("")
    val manualRoom: StateFlow<String> = _manualRoom.asStateFlow()

    private val _manualNotes = MutableStateFlow("")
    val manualNotes: StateFlow<String> = _manualNotes.asStateFlow()

    private val sessionId: String = UUID.randomUUID().toString().take(8)

    init {
        // Observe scanner state changes
        viewModelScope.launch {
            scanner.scanState.collect { state ->
                when (state) {
                    is WifiScanState.Success -> {
                        _currentFingerprint.value = state.fingerprint
                        evaluateCalibration(state.fingerprint)
                    }
                    is WifiScanState.Throttled -> {
                        state.cachedFingerprint?.let {
                            _currentFingerprint.value = it
                            evaluateCalibration(it)
                        }
                    }
                    else -> {}
                }
            }
        }
    }

    fun setScanner(newScanner: WifiScanner) {
        scanner = newScanner
        _isMockMode.value = newScanner.isMockProvider
        _currentFingerprint.value = null
        _localizationResult.value = null
        _modelAdaptationNotice.value = null
    }

    fun triggerScan() {
        viewModelScope.launch {
            val result = scanner.triggerScan()
            _lastScanTime.value = System.currentTimeMillis()

            result.onSuccess { fp ->
                _currentFingerprint.value = fp
                evaluateCalibration(fp)
            }
        }
    }

    private fun evaluateCalibration(fingerprint: WifiFingerprint) {
        val adaptation = modelAdapter.adaptForInference(fingerprint)
        when (adaptation) {
            is ModelAdaptationResult.Uncalibrated -> {
                _modelAdaptationNotice.value = adaptation.reason
                _localizationResult.value = null
            }
            is ModelAdaptationResult.Calibrated -> {
                _modelAdaptationNotice.value = null
            }
            is ModelAdaptationResult.DemoSynthetic -> {
                _modelAdaptationNotice.value = null
            }
        }
    }

    /**
     * Triggers demo model localization ONLY when using synthetic benchmark data.
     * Real-world uncalibrated scans are strictly blocked from arbitrary WAP mapping.
     */
    fun requestDemoLocalization() {
        val fp = _currentFingerprint.value ?: return
        if (!fp.isMock) {
            _modelAdaptationNotice.value =
                "Blocked: Real Wi-Fi scans cannot be localized with UJIIndoorLoc models without an empirical survey calibration mapping."
            return
        }

        if (repository == null) return

        viewModelScope.launch {
            _isLocalizing.value = true
            val result = repository.estimateLocation(fp)
            result.onSuccess {
                _localizationResult.value = it
            }
            _isLocalizing.value = false
        }
    }

    fun updateManualLabels(building: String, floor: String, room: String, notes: String) {
        _manualBuilding.value = building
        _manualFloor.value = floor
        _manualRoom.value = room
        _manualNotes.value = notes
    }

    fun saveCurrentFingerprint() {
        val fp = _currentFingerprint.value ?: return
        val label = ManualLocationLabel(
            building = _manualBuilding.value.trim(),
            floor = _manualFloor.value.trim(),
            room = _manualRoom.value.trim(),
            notes = _manualNotes.value.trim()
        )

        val captured = CapturedFingerprint(
            captureId = UUID.randomUUID().toString(),
            sessionId = sessionUseCase?.currentSession?.value?.sessionId ?: sessionId,
            timestamp = System.currentTimeMillis(),
            fingerprint = fp,
            manualLabel = if (label.isPresent) label else null,
            deviceMetadata = DeviceMetadata.current(),
            isMock = fp.isMock
        )

        _capturedFingerprints.value = _capturedFingerprints.value + captured

        // Associate with active multimodal session if one is currently recording
        sessionUseCase?.recordWifiScan(fp, if (label.isPresent) label else null)
    }

    fun deleteCapture(captureId: String) {
        _capturedFingerprints.value = _capturedFingerprints.value.filter { it.captureId != captureId }
    }

    fun clearCapturedFingerprints() {
        _capturedFingerprints.value = emptyList()
        clearExportPreview()
    }

    fun exportAsJson(): String {
        val captures = _capturedFingerprints.value
        val json = if (captures.isEmpty()) {
            _currentFingerprint.value?.let { fp ->
                val single = CapturedFingerprint(
                    captureId = UUID.randomUUID().toString(),
                    sessionId = sessionId,
                    timestamp = System.currentTimeMillis(),
                    fingerprint = fp,
                    manualLabel = ManualLocationLabel(
                        building = _manualBuilding.value,
                        floor = _manualFloor.value,
                        room = _manualRoom.value,
                        notes = _manualNotes.value
                    ),
                    deviceMetadata = DeviceMetadata.current(),
                    isMock = fp.isMock
                )
                single.toJson()
            } ?: "[]"
        } else {
            CapturedFingerprint.toBatchJson(captures)
        }

        _exportPreview.value = json
        _exportFormat.value = "JSON"
        return json
    }

    fun exportAsCsv(): String {
        val captures = _capturedFingerprints.value
        val csv = if (captures.isEmpty()) {
            _currentFingerprint.value?.let { fp ->
                val single = CapturedFingerprint(
                    captureId = UUID.randomUUID().toString(),
                    sessionId = sessionId,
                    timestamp = System.currentTimeMillis(),
                    fingerprint = fp,
                    manualLabel = ManualLocationLabel(
                        building = _manualBuilding.value,
                        floor = _manualFloor.value,
                        room = _manualRoom.value,
                        notes = _manualNotes.value
                    ),
                    deviceMetadata = DeviceMetadata.current(),
                    isMock = fp.isMock
                )
                single.toCsv()
            } ?: CapturedFingerprint.CSV_HEADER
        } else {
            CapturedFingerprint.toBatchCsv(captures)
        }

        _exportPreview.value = csv
        _exportFormat.value = "CSV"
        return csv
    }

    fun clearExportPreview() {
        _exportPreview.value = null
        _exportFormat.value = null
    }

    fun clearResults() {
        _currentFingerprint.value = null
        _localizationResult.value = null
    }
}
