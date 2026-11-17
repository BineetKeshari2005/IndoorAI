package ai.indoor.app.domain.usecases

import ai.indoor.app.data.repository.LocalizationRepository
import ai.indoor.app.domain.export.DatasetExporter
import ai.indoor.app.domain.models.AnnotationType
import ai.indoor.app.domain.models.CaptureSession
import ai.indoor.app.domain.models.CapturedPhotoMetadata
import ai.indoor.app.domain.models.ManualLocationLabel
import ai.indoor.app.domain.models.SingleSensorReading
import ai.indoor.app.domain.models.SurveyAnnotation
import ai.indoor.app.domain.models.SurveyPhotoObservation
import ai.indoor.app.domain.models.SurveySensorObservation
import ai.indoor.app.domain.models.SurveySession
import ai.indoor.app.domain.models.SurveySessionStatus
import ai.indoor.app.domain.models.SurveyWifiAccessPoint
import ai.indoor.app.domain.models.SurveyWifiObservation
import ai.indoor.app.domain.models.WifiFingerprint
import ai.indoor.app.domain.storage.SessionPersistenceManager
import ai.indoor.app.domain.timeline.ObservationTimelineBuilder
import ai.indoor.app.domain.timeline.SessionTimelineItem
import ai.indoor.app.domain.timestamp.ClockAnchor
import ai.indoor.app.domain.timestamp.TimestampNormalizer
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.File
import java.util.UUID

/**
 * Central orchestrator managing unified multimodal survey session lifecycles,
 * timestamp normalization, persistent observation storage, and dataset export.
 */
class ManageCaptureSessionUseCase(
    private val repository: LocalizationRepository? = null,
    private val persistenceManager: SessionPersistenceManager? = null,
    private val exporter: DatasetExporter? = null,
    private val normalizer: TimestampNormalizer = TimestampNormalizer()
) {
    private val _currentSession = MutableStateFlow<SurveySession?>(null)
    val currentSession: StateFlow<SurveySession?> = _currentSession.asStateFlow()

    private val _allSessions = MutableStateFlow<List<SurveySession>>(emptyList())
    val allSessions: StateFlow<List<SurveySession>> = _allSessions.asStateFlow()

    private val _timeline = MutableStateFlow<List<SessionTimelineItem>>(emptyList())
    val timeline: StateFlow<List<SessionTimelineItem>> = _timeline.asStateFlow()

    init {
        refreshSessionList()
        // If an interrupted session was recovered by persistenceManager, surface it
        val recovered = persistenceManager?.getAllSessions()?.firstOrNull { it.isInterrupted }
            ?: persistenceManager?.getAllSessions()?.firstOrNull { it.status == SurveySessionStatus.PAUSED || it.status == SurveySessionStatus.ACTIVE }
        if (recovered != null) {
            _currentSession.value = recovered
            normalizer.anchor = recovered.clockAnchor
            refreshTimeline(recovered.sessionId)
        }
    }

    fun refreshSessionList() {
        val list = persistenceManager?.getAllSessions() ?: emptyList()
        _allSessions.value = list
    }

    fun refreshTimeline(sessionId: String? = _currentSession.value?.sessionId) {
        if (sessionId == null || persistenceManager == null) {
            _timeline.value = emptyList()
            return
        }

        val wifi = persistenceManager.getWifiObservations(sessionId)
        val imu = persistenceManager.getSensorObservations(sessionId)
        val photos = persistenceManager.getPhotoObservations(sessionId)
        val annotations = persistenceManager.getAnnotations(sessionId)

        _timeline.value = ObservationTimelineBuilder.buildTimeline(wifi, imu, photos, annotations)
    }

    /**
     * Creates a new configured session in CREATED state.
     */
    fun createSession(
        name: String = "Survey Session",
        buildingId: String = "",
        floorId: String = "",
        roomId: String = "",
        manualLabel: ManualLocationLabel? = null
    ): SurveySession {
        val anchor = ClockAnchor.create()
        normalizer.anchor = anchor

        val session = SurveySession(
            sessionId = "SES-" + UUID.randomUUID().toString().take(8).uppercase(),
            sessionName = name.ifBlank { "Survey Session" },
            createdAtMs = System.currentTimeMillis(),
            startTimeMs = null,
            endTimeMs = null,
            status = SurveySessionStatus.CREATED,
            buildingId = buildingId,
            floorId = floorId,
            roomId = roomId,
            manualLocationLabel = manualLabel,
            clockAnchor = anchor
        )

        persistenceManager?.saveSession(session)
        _currentSession.value = session
        refreshSessionList()
        refreshTimeline(session.sessionId)
        return session
    }

    /**
     * Starts an existing or newly created session, transitioning it to ACTIVE.
     */
    fun startSession(sessionId: String? = null): SurveySession {
        val target = if (sessionId != null) {
            persistenceManager?.getSession(sessionId) ?: _currentSession.value
        } else {
            _currentSession.value
        } ?: createSession()

        val anchor = target.clockAnchor
        normalizer.anchor = anchor

        val active = target.copy(
            startTimeMs = target.startTimeMs ?: System.currentTimeMillis(),
            status = SurveySessionStatus.ACTIVE,
            isInterrupted = false,
            interruptionNote = null
        )

        persistenceManager?.saveSession(active)
        _currentSession.value = active
        refreshSessionList()
        refreshTimeline(active.sessionId)
        return active
    }

    /**
     * Temporarily suspends recording, transitioning to PAUSED.
     */
    fun pauseSession(): SurveySession? {
        val session = _currentSession.value ?: return null
        if (session.status != SurveySessionStatus.ACTIVE) return session

        val paused = session.copy(status = SurveySessionStatus.PAUSED)
        persistenceManager?.saveSession(paused)
        _currentSession.value = paused
        refreshSessionList()
        return paused
    }

    /**
     * Resumes a paused or recovered session, returning to ACTIVE.
     */
    fun resumeSession(): SurveySession? {
        val session = _currentSession.value ?: return null
        if (session.status != SurveySessionStatus.PAUSED && session.status != SurveySessionStatus.CREATED) {
            return session
        }

        val active = session.copy(
            startTimeMs = session.startTimeMs ?: System.currentTimeMillis(),
            status = SurveySessionStatus.ACTIVE,
            isInterrupted = false,
            interruptionNote = null
        )
        persistenceManager?.saveSession(active)
        _currentSession.value = active
        refreshSessionList()
        return active
    }

    /**
     * Completes and seals the session.
     */
    fun completeSession(): SurveySession? {
        val session = _currentSession.value ?: return null
        val completed = session.copy(
            endTimeMs = System.currentTimeMillis(),
            status = SurveySessionStatus.COMPLETED
        )
        persistenceManager?.saveSession(completed)
        _currentSession.value = completed
        refreshSessionList()
        return completed
    }

    /**
     * Cancels an active or paused session.
     */
    fun cancelSession(): SurveySession? {
        val session = _currentSession.value ?: return null
        val cancelled = session.copy(
            endTimeMs = System.currentTimeMillis(),
            status = SurveySessionStatus.CANCELLED
        )
        persistenceManager?.saveSession(cancelled)
        _currentSession.value = cancelled
        refreshSessionList()
        return cancelled
    }

    /**
     * Deletes a session and its associated physical images completely.
     */
    fun deleteSession(sessionId: String): Boolean {
        val success = persistenceManager?.deleteSession(sessionId) ?: true
        if (_currentSession.value?.sessionId == sessionId) {
            _currentSession.value = null
            _timeline.value = emptyList()
        }
        refreshSessionList()
        return success
    }

    // --- Ingestion Handlers ---

    fun recordWifiScan(
        fingerprint: WifiFingerprint,
        manualLabel: ManualLocationLabel? = null
    ): SurveyWifiObservation? {
        val session = _currentSession.value ?: return null
        if (session.status != SurveySessionStatus.ACTIVE) return null

        val timestamp = normalizer.normalizeWifiScanTimestamp(
            rawTimestamp = fingerprint.timestamp,
            isFresh = fingerprint.isFresh,
            isMock = fingerprint.isMock
        )

        val aps = fingerprint.accessPoints.map { ap ->
            SurveyWifiAccessPoint(
                bssid = ap.bssid,
                ssid = ap.ssid,
                rssi = ap.rssi,
                frequency = ap.frequency,
                channel = ap.channel,
                channelBand = ap.channelBand,
                isFresh = ap.isFresh
            )
        }

        val obs = SurveyWifiObservation(
            sessionId = session.sessionId,
            timestamp = timestamp,
            accessPoints = aps,
            isFresh = fingerprint.isFresh,
            isMock = fingerprint.isMock,
            manualLabel = manualLabel ?: session.manualLocationLabel
        )

        persistenceManager?.recordWifiObservation(obs)
        val updated = persistenceManager?.getSession(session.sessionId)
            ?: session.copy(
                wifiCount = session.wifiCount + 1,
                realObservationCount = if (!obs.isMock) session.realObservationCount + 1 else session.realObservationCount,
                mockObservationCount = if (obs.isMock) session.mockObservationCount + 1 else session.mockObservationCount
            )
        _currentSession.value = updated
        refreshTimeline(session.sessionId)
        return obs
    }

    fun recordSensorReading(reading: SingleSensorReading): SurveySensorObservation? {
        val session = _currentSession.value ?: return null
        if (session.status != SurveySessionStatus.ACTIVE) return null

        val timestamp = normalizer.normalizeSensorTimestamp(
            timestampNs = reading.timestampNs,
            isMock = reading.isMock
        )

        val obs = SurveySensorObservation(
            sessionId = session.sessionId,
            sensorType = reading.sensorType,
            x = reading.x,
            y = reading.y,
            z = reading.z,
            magnitude = reading.magnitude,
            accuracy = reading.accuracy,
            samplingRate = reading.samplingRate,
            timestamp = timestamp,
            isMock = reading.isMock
        )

        persistenceManager?.recordSensorObservation(obs)
        val updated = persistenceManager?.getSession(session.sessionId)
            ?: session.copy(
                sensorCount = session.sensorCount + 1,
                realObservationCount = if (!obs.isMock) session.realObservationCount + 1 else session.realObservationCount,
                mockObservationCount = if (obs.isMock) session.mockObservationCount + 1 else session.mockObservationCount
            )
        _currentSession.value = updated
        // Refresh timeline periodically or selectively
        return obs
    }

    fun recordPhoto(photo: CapturedPhotoMetadata): SurveyPhotoObservation? {
        val session = _currentSession.value ?: return null
        // Photo captures can occur during active or paused reviews
        if (session.status != SurveySessionStatus.ACTIVE && session.status != SurveySessionStatus.PAUSED) {
            return null
        }

        val timestamp = normalizer.normalizeWallClockTimestamp(
            epochMs = photo.timestampMs,
            isMock = photo.isMock
        )

        val obs = SurveyPhotoObservation(
            photoId = photo.photoId,
            sessionId = session.sessionId,
            timestamp = timestamp,
            filename = photo.filename,
            filePath = photo.filePath,
            fileSizeBytes = photo.fileSizeBytes,
            imageWidth = photo.imageWidth,
            imageHeight = photo.imageHeight,
            building = photo.building.ifBlank { session.buildingId },
            floor = photo.floor.ifBlank { session.floorId },
            room = photo.room.ifBlank { session.roomId },
            landmarkType = photo.landmarkType,
            description = photo.description,
            notes = photo.notes,
            isExifSanitized = photo.isExifSanitized,
            facesDetectedCount = photo.facesDetectedCount,
            faceBlurApplied = photo.faceBlurApplied,
            faceDetectionStatus = photo.faceDetectionStatus,
            isMock = photo.isMock
        )

        persistenceManager?.recordPhotoObservation(obs)
        val updated = persistenceManager?.getSession(session.sessionId)
            ?: session.copy(
                photoCount = session.photoCount + 1,
                realObservationCount = if (!obs.isMock) session.realObservationCount + 1 else session.realObservationCount,
                mockObservationCount = if (obs.isMock) session.mockObservationCount + 1 else session.mockObservationCount
            )
        _currentSession.value = updated
        refreshTimeline(session.sessionId)
        return obs
    }

    fun recordAnnotation(
        type: AnnotationType,
        description: String,
        building: String = "",
        floor: String = "",
        room: String = "",
        relatedObservationId: String? = null,
        isMock: Boolean = false
    ): SurveyAnnotation? {
        val session = _currentSession.value ?: return null
        if (session.status != SurveySessionStatus.ACTIVE && session.status != SurveySessionStatus.PAUSED) {
            return null
        }

        val timestamp = normalizer.normalizeWallClockTimestamp(
            epochMs = System.currentTimeMillis(),
            isMock = isMock
        )

        val ann = SurveyAnnotation(
            sessionId = session.sessionId,
            timestamp = timestamp,
            type = type,
            description = description,
            building = building.ifBlank { session.buildingId },
            floor = floor.ifBlank { session.floorId },
            room = room.ifBlank { session.roomId },
            relatedObservationId = relatedObservationId,
            isMock = isMock
        )

        persistenceManager?.recordAnnotation(ann)
        val updated = persistenceManager?.getSession(session.sessionId)
            ?: session.copy(
                annotationCount = session.annotationCount + 1,
                realObservationCount = if (!ann.isMock) session.realObservationCount + 1 else session.realObservationCount,
                mockObservationCount = if (ann.isMock) session.mockObservationCount + 1 else session.mockObservationCount
            )
        _currentSession.value = updated
        refreshTimeline(session.sessionId)
        return ann
    }

    /**
     * Exports the session bundle into a self-contained ZIP archive.
     */
    fun exportSessionZip(sessionId: String? = _currentSession.value?.sessionId): Result<File> {
        val id = sessionId ?: return Result.failure(IllegalStateException("No active session selected"))
        val session = persistenceManager?.getSession(id) ?: _currentSession.value
            ?: return Result.failure(IllegalStateException("Session not found: $id"))
        val exporterInstance = exporter
            ?: return Result.failure(IllegalStateException("DatasetExporter is not configured"))

        val wifi = persistenceManager?.getWifiObservations(id) ?: emptyList()
        val imu = persistenceManager?.getSensorObservations(id) ?: emptyList()
        val photos = persistenceManager?.getPhotoObservations(id) ?: emptyList()
        val annotations = persistenceManager?.getAnnotations(id) ?: emptyList()

        return exporterInstance.exportSessionZip(session, wifi, imu, photos, annotations)
    }

    // --- Legacy Bridge for Task 13/14/15 Compatibility ---

    suspend fun syncSession(): Result<String> {
        val session = _currentSession.value ?: return Result.failure(IllegalStateException("No session active"))
        val legacy = CaptureSession(
            sessionId = session.sessionId,
            startTimeMs = session.startTimeMs ?: session.createdAtMs,
            endTimeMs = session.endTimeMs,
            isRunning = session.status == SurveySessionStatus.ACTIVE,
            wifiScanCount = session.wifiCount,
            imuSampleCount = session.sensorCount,
            photoCount = session.photoCount,
            uploadStatus = CaptureSession.UploadStatus.UPLOADING
        )
        return repository?.syncSessionTelemetry(legacy) ?: Result.success("Simulated local sync")
    }
}
