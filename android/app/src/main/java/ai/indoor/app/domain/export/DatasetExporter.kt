package ai.indoor.app.domain.export

import ai.indoor.app.domain.models.SurveyAnnotation
import ai.indoor.app.domain.models.SurveyPhotoObservation
import ai.indoor.app.domain.models.SurveySensorObservation
import ai.indoor.app.domain.models.SurveySession
import ai.indoor.app.domain.models.SurveyWifiObservation
import ai.indoor.app.domain.timestamp.TimestampRecord
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.security.MessageDigest
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/**
 * Packaging and export engine creating verified multimodal dataset ZIP bundles.
 */
class DatasetExporter(
    private val exportBaseDir: File
) {

    /**
     * Validates and packages a survey session into a self-contained ZIP archive.
     *
     * @return Result containing the generated ZIP File, or an error detailing validation failure.
     */
    fun exportSessionZip(
        session: SurveySession,
        wifiScans: List<SurveyWifiObservation>,
        sensorReadings: List<SurveySensorObservation>,
        photos: List<SurveyPhotoObservation>,
        annotations: List<SurveyAnnotation>
    ): Result<File> {
        // 1. Validate dataset before exporting
        val validation = ExportValidator.validate(session, wifiScans, sensorReadings, photos, annotations)
        if (!validation.isValid) {
            return Result.failure(IllegalStateException(validation.summary))
        }

        exportBaseDir.mkdirs()
        val timestamp = System.currentTimeMillis()
        val stagingDir = File(exportBaseDir, "staging_${session.sessionId}_$timestamp").apply { mkdirs() }
        val imagesDir = File(stagingDir, "images").apply { mkdirs() }
        val targetZipFile = File(exportBaseDir, "indoorai_session_${session.sessionId}.zip")

        try {
            // 2. Generate session.json
            val sessionFile = File(stagingDir, "session.json")
            sessionFile.writeText(session.toJson(), Charsets.UTF_8)

            // 3. Generate wifi_observations.csv
            val wifiFile = File(stagingDir, "wifi_observations.csv")
            FileOutputStream(wifiFile).bufferedWriter(Charsets.UTF_8).use { writer ->
                writer.appendLine(SurveyWifiObservation.CSV_HEADER)
                wifiScans.forEach { scan ->
                    scan.toCsvRows().forEach { row -> writer.appendLine(row) }
                }
            }

            // 4. Generate imu_observations.csv
            val imuFile = File(stagingDir, "imu_observations.csv")
            FileOutputStream(imuFile).bufferedWriter(Charsets.UTF_8).use { writer ->
                writer.appendLine(SurveySensorObservation.CSV_HEADER)
                sensorReadings.forEach { reading ->
                    writer.appendLine(reading.toCsvRow())
                }
            }

            // 5. Generate photo_metadata.csv
            val photoMetadataFile = File(stagingDir, "photo_metadata.csv")
            FileOutputStream(photoMetadataFile).bufferedWriter(Charsets.UTF_8).use { writer ->
                writer.appendLine(SurveyPhotoObservation.CSV_HEADER)
                photos.forEach { photo ->
                    writer.appendLine(photo.toCsvRow())
                }
            }

            // 6. Generate annotations.csv
            val annotationsFile = File(stagingDir, "annotations.csv")
            FileOutputStream(annotationsFile).bufferedWriter(Charsets.UTF_8).use { writer ->
                writer.appendLine(SurveyAnnotation.CSV_HEADER)
                annotations.forEach { ann ->
                    writer.appendLine(ann.toCsvRow())
                }
            }

            // 7. Copy sanitized image files
            val copiedImageFiles = mutableListOf<File>()
            photos.forEach { photo ->
                val sourceImg = File(photo.filePath)
                val destImg = File(imagesDir, photo.filename)
                sourceImg.copyTo(destImg, overwrite = true)
                copiedImageFiles.add(destImg)
            }

            // 8. Compute checksums and build manifest.json
            val manifestEntries = mutableListOf<ManifestFileEntry>()
            val filesToHash = listOf(
                sessionFile,
                wifiFile,
                imuFile,
                photoMetadataFile,
                annotationsFile
            ) + copiedImageFiles

            filesToHash.forEach { file ->
                val relativePath = file.relativeTo(stagingDir).path
                val sha256 = calculateSha256(file)
                manifestEntries.add(
                    ManifestFileEntry(
                        filename = relativePath,
                        sizeBytes = file.length(),
                        sha256 = sha256
                    )
                )
            }

            val totalReal = (if (!session.hasMockData) 1 else 0) +
                    wifiScans.count { !it.isMock } +
                    sensorReadings.count { !it.isMock } +
                    photos.count { !it.isMock } +
                    annotations.count { !it.isMock }

            val totalMock = (if (session.hasMockData) 1 else 0) +
                    wifiScans.count { it.isMock } +
                    sensorReadings.count { it.isMock } +
                    photos.count { it.isMock } +
                    annotations.count { it.isMock }

            val manifest = ExportManifest(
                exportVersion = "1.0.0",
                sessionId = session.sessionId,
                exportedAtMs = timestamp,
                exportedAtIso = TimestampRecord.formatIso(timestamp),
                recordCounts = ExportRecordCounts(
                    wifiCount = wifiScans.size,
                    imuCount = sensorReadings.size,
                    photoCount = photos.size,
                    annotationCount = annotations.size,
                    totalObservations = wifiScans.size + sensorReadings.size + photos.size + annotations.size
                ),
                sourceCounts = ExportSourceCounts(
                    realObservationCount = totalReal,
                    mockObservationCount = totalMock,
                    containsRealData = totalReal > 0,
                    containsMockData = totalMock > 0
                ),
                files = manifestEntries,
                validationStatus = "VALIDATED"
            )

            val manifestFile = File(stagingDir, "manifest.json")
            manifestFile.writeText(manifest.toJson(), Charsets.UTF_8)

            // 9. Compress staging directory to ZIP
            if (targetZipFile.exists()) targetZipFile.delete()
            zipDirectory(stagingDir, targetZipFile)

            return Result.success(targetZipFile)

        } catch (e: Throwable) {
            return Result.failure(e)
        } finally {
            stagingDir.deleteRecursively()
        }
    }

    private fun zipDirectory(sourceDir: File, zipFile: File) {
        ZipOutputStream(FileOutputStream(zipFile)).use { zos ->
            sourceDir.walkTopDown().forEach { file ->
                if (file.isFile) {
                    val relativePath = file.relativeTo(sourceDir).path
                    val entry = ZipEntry(relativePath)
                    zos.putNextEntry(entry)
                    FileInputStream(file).use { fis ->
                        fis.copyTo(zos)
                    }
                    zos.closeEntry()
                }
            }
        }
    }

    companion object {
        fun calculateSha256(file: File): String {
            val digest = MessageDigest.getInstance("SHA-256")
            FileInputStream(file).use { fis ->
                val buffer = ByteArray(8192)
                var bytesRead: Int
                while (fis.read(buffer).also { bytesRead = it } != -1) {
                    digest.update(buffer, 0, bytesRead)
                }
            }
            return digest.digest().joinToString("") { "%02x".format(it) }
        }
    }
}
