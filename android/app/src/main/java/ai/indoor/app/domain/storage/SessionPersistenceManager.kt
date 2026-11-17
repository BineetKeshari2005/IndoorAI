package ai.indoor.app.domain.storage

import android.content.Context
import ai.indoor.app.domain.models.SurveyAnnotation
import ai.indoor.app.domain.models.SurveyPhotoObservation
import ai.indoor.app.domain.models.SurveySensorObservation
import ai.indoor.app.domain.models.SurveySession
import ai.indoor.app.domain.models.SurveySessionStatus
import ai.indoor.app.domain.models.SurveyWifiObservation
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

/**
 * Robust local persistence manager for multimodal survey sessions.
 *
 * Implements:
 * 1. App-private file storage in `filesDir/survey_sessions/<sessionId>/`.
 * 2. Streaming JSONL observation journals for Wi-Fi, IMU, photos, and annotations.
 * 3. Crash / interruption recovery for active sessions upon application restart.
 * 4. Image file reference consistency and safe cleanup preventing orphaned files.
 */
class SessionPersistenceManager(
    private val context: Context,
    private val baseDir: File = File(context.filesDir, "survey_sessions")
) {

    private val gson: Gson by lazy {
        GsonBuilder().setPrettyPrinting().create()
    }

    private val compactGson: Gson by lazy {
        GsonBuilder().create()
    }

    private val sessionIndex = mutableMapOf<String, SurveySession>()

    init {
        baseDir.mkdirs()
        recoverInterruptedSessions()
    }

    /**
     * Inspects on-disk sessions upon startup. Any session found in the ACTIVE state
     * (indicating an ungraceful termination or OS kill) is converted to PAUSED with
     * an explicit interruption flag.
     */
    @Synchronized
    private fun recoverInterruptedSessions() {
        sessionIndex.clear()
        val dirs = baseDir.listFiles() ?: return
        for (dir in dirs) {
            if (!dir.isDirectory) continue
            val metaFile = File(dir, "session_metadata.json")
            if (!metaFile.exists() || metaFile.length() == 0L) continue

            try {
                val json = metaFile.readText(Charsets.UTF_8)
                var session = gson.fromJson(json, SurveySession::class.java)

                if (session.status == SurveySessionStatus.ACTIVE) {
                    session = session.copy(
                        status = SurveySessionStatus.PAUSED,
                        isInterrupted = true,
                        interruptionNote = "Session recovered after application termination; review or resume."
                    )
                    writeSessionMetadata(session)
                }

                sessionIndex[session.sessionId] = session
            } catch (_: Exception) {
                // Ignore corrupt session directories
            }
        }
    }

    @Synchronized
    fun getAllSessions(): List<SurveySession> {
        return sessionIndex.values.sortedByDescending { it.createdAtMs }
    }

    @Synchronized
    fun getSession(sessionId: String): SurveySession? {
        return sessionIndex[sessionId]
    }

    @Synchronized
    fun saveSession(session: SurveySession) {
        sessionIndex[session.sessionId] = session
        writeSessionMetadata(session)
    }

    @Synchronized
    private fun writeSessionMetadata(session: SurveySession) {
        val sessionDir = File(baseDir, session.sessionId).apply { mkdirs() }
        val metaFile = File(sessionDir, "session_metadata.json")
        val tmpFile = File(sessionDir, "session_metadata.json.tmp")

        try {
            val json = gson.toJson(session)
            FileOutputStream(tmpFile).use { out ->
                out.write(json.toByteArray(Charsets.UTF_8))
                out.flush()
            }
            if (metaFile.exists()) metaFile.delete()
            tmpFile.renameTo(metaFile)
        } catch (_: IOException) {
            if (tmpFile.exists()) tmpFile.delete()
        }
    }

    // --- Observation Appenders ---

    @Synchronized
    fun recordWifiObservation(obs: SurveyWifiObservation) {
        val sessionDir = File(baseDir, obs.sessionId).apply { mkdirs() }
        val file = File(sessionDir, "wifi_journal.jsonl")
        appendJsonLine(file, obs)

        val session = sessionIndex[obs.sessionId]
        if (session != null) {
            val updated = session.copy(
                wifiCount = session.wifiCount + 1,
                realObservationCount = if (!obs.isMock) session.realObservationCount + 1 else session.realObservationCount,
                mockObservationCount = if (obs.isMock) session.mockObservationCount + 1 else session.mockObservationCount
            )
            saveSession(updated)
        }
    }

    @Synchronized
    fun recordSensorObservation(obs: SurveySensorObservation) {
        val sessionDir = File(baseDir, obs.sessionId).apply { mkdirs() }
        val file = File(sessionDir, "imu_journal.jsonl")
        appendJsonLine(file, obs)

        val session = sessionIndex[obs.sessionId]
        if (session != null) {
            val updated = session.copy(
                sensorCount = session.sensorCount + 1,
                realObservationCount = if (!obs.isMock) session.realObservationCount + 1 else session.realObservationCount,
                mockObservationCount = if (obs.isMock) session.mockObservationCount + 1 else session.mockObservationCount
            )
            saveSession(updated)
        }
    }

    @Synchronized
    fun recordPhotoObservation(obs: SurveyPhotoObservation) {
        val sessionDir = File(baseDir, obs.sessionId).apply { mkdirs() }
        val file = File(sessionDir, "photos_journal.jsonl")
        appendJsonLine(file, obs)

        val session = sessionIndex[obs.sessionId]
        if (session != null) {
            val updated = session.copy(
                photoCount = session.photoCount + 1,
                realObservationCount = if (!obs.isMock) session.realObservationCount + 1 else session.realObservationCount,
                mockObservationCount = if (obs.isMock) session.mockObservationCount + 1 else session.mockObservationCount
            )
            saveSession(updated)
        }
    }

    @Synchronized
    fun recordAnnotation(ann: SurveyAnnotation) {
        val sessionDir = File(baseDir, ann.sessionId).apply { mkdirs() }
        val file = File(sessionDir, "annotations_journal.jsonl")
        appendJsonLine(file, ann)

        val session = sessionIndex[ann.sessionId]
        if (session != null) {
            val updated = session.copy(
                annotationCount = session.annotationCount + 1,
                realObservationCount = if (!ann.isMock) session.realObservationCount + 1 else session.realObservationCount,
                mockObservationCount = if (ann.isMock) session.mockObservationCount + 1 else session.mockObservationCount
            )
            saveSession(updated)
        }
    }

    private fun appendJsonLine(file: File, obj: Any) {
        try {
            val line = compactGson.toJson(obj) + "\n"
            FileOutputStream(file, true).use { out ->
                out.write(line.toByteArray(Charsets.UTF_8))
                out.flush()
            }
        } catch (_: IOException) {}
    }

    // --- Observation Loaders ---

    @Synchronized
    fun getWifiObservations(sessionId: String): List<SurveyWifiObservation> {
        val file = File(File(baseDir, sessionId), "wifi_journal.jsonl")
        if (!file.exists()) return emptyList()
        val list = mutableListOf<SurveyWifiObservation>()
        try {
            file.forEachLine { line ->
                if (line.isNotBlank()) {
                    list.add(compactGson.fromJson(line, SurveyWifiObservation::class.java))
                }
            }
        } catch (_: Exception) {}
        return list
    }

    @Synchronized
    fun getSensorObservations(sessionId: String): List<SurveySensorObservation> {
        val file = File(File(baseDir, sessionId), "imu_journal.jsonl")
        if (!file.exists()) return emptyList()
        val list = mutableListOf<SurveySensorObservation>()
        try {
            file.forEachLine { line ->
                if (line.isNotBlank()) {
                    list.add(compactGson.fromJson(line, SurveySensorObservation::class.java))
                }
            }
        } catch (_: Exception) {}
        return list
    }

    @Synchronized
    fun getPhotoObservations(sessionId: String): List<SurveyPhotoObservation> {
        val file = File(File(baseDir, sessionId), "photos_journal.jsonl")
        if (!file.exists()) return emptyList()
        val list = mutableListOf<SurveyPhotoObservation>()
        try {
            file.forEachLine { line ->
                if (line.isNotBlank()) {
                    list.add(compactGson.fromJson(line, SurveyPhotoObservation::class.java))
                }
            }
        } catch (_: Exception) {}
        return list
    }

    @Synchronized
    fun getAnnotations(sessionId: String): List<SurveyAnnotation> {
        val file = File(File(baseDir, sessionId), "annotations_journal.jsonl")
        if (!file.exists()) return emptyList()
        val list = mutableListOf<SurveyAnnotation>()
        try {
            file.forEachLine { line ->
                if (line.isNotBlank()) {
                    list.add(compactGson.fromJson(line, SurveyAnnotation::class.java))
                }
            }
        } catch (_: Exception) {}
        return list
    }

    /**
     * Deletes a session and its observation journals from disk.
     * Also deletes associated physical image files to ensure no orphaned files remain!
     */
    @Synchronized
    fun deleteSession(sessionId: String): Boolean {
        sessionIndex.remove(sessionId)
        val sessionDir = File(baseDir, sessionId)
        if (!sessionDir.exists()) return false

        // 1. Delete associated image files
        val photos = getPhotoObservations(sessionId)
        for (photo in photos) {
            try {
                val imgFile = File(photo.filePath)
                if (imgFile.exists()) {
                    imgFile.delete()
                }
            } catch (_: Exception) {}
        }

        // 2. Delete session folder and observation journals
        return sessionDir.deleteRecursively()
    }
}
