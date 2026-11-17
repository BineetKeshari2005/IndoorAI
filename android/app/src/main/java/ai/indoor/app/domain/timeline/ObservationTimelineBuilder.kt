package ai.indoor.app.domain.timeline

import ai.indoor.app.domain.models.SensorType
import ai.indoor.app.domain.models.SurveyAnnotation
import ai.indoor.app.domain.models.SurveyPhotoObservation
import ai.indoor.app.domain.models.SurveySensorObservation
import ai.indoor.app.domain.models.SurveyWifiObservation
import java.util.Locale

/**
 * Constructs a chronological multimodal timeline from disparate observation streams.
 */
object ObservationTimelineBuilder {

    fun buildTimeline(
        wifiScans: List<SurveyWifiObservation>,
        sensorReadings: List<SurveySensorObservation>,
        photos: List<SurveyPhotoObservation>,
        annotations: List<SurveyAnnotation>
    ): List<SessionTimelineItem> {
        val items = mutableListOf<SessionTimelineItem>()

        // 1. Process Wi-Fi sweeps
        wifiScans.forEach { wifi ->
            val apCount = wifi.accessPoints.size
            val strongestRssi = wifi.accessPoints.maxOfOrNull { it.rssi } ?: -105
            val ssids = wifi.accessPoints.map { it.ssid }.filter { it.isNotBlank() }.distinct().take(3).joinToString(", ")

            items.add(
                SessionTimelineItem(
                    id = wifi.observationId,
                    sessionId = wifi.sessionId,
                    type = TimelineObservationType.WIFI_SCAN,
                    normalizedTimestampMs = wifi.timestamp.normalizedEpochMs,
                    originalTimestamp = wifi.timestamp.originalTimestamp,
                    timeSourceDisplay = wifi.timestamp.timeSource.description,
                    formattedTime = SessionTimelineItem.formatTimestamp(wifi.timestamp.normalizedEpochMs),
                    title = "Wi-Fi Sweep ($apCount APs)",
                    shortDescription = "Strongest: $strongestRssi dBm ${if (ssids.isNotBlank()) "($ssids)" else ""}",
                    isMock = wifi.isMock,
                    isTimingCertain = wifi.timestamp.isReliable && (wifi.timestamp.uncertaintyMs ?: 999f) <= 50f,
                    uncertaintyDisplay = wifi.timestamp.uncertaintyDisplay,
                    detailKeyValues = mapOf(
                        "Access Points Count" to apCount.toString(),
                        "Radio Status" to if (wifi.isFresh) "Fresh Scan" else "Cached / Throttled",
                        "Strongest RSSI" to "$strongestRssi dBm",
                        "Sync Method" to wifi.timestamp.syncMethod.name,
                        "Original Timestamp" to wifi.timestamp.originalTimestamp.toString(),
                        "Ground Truth Label" to (wifi.manualLabel?.formattedDisplay ?: "Unlabeled")
                    )
                )
            )
        }

        // 2. Process Sensors (IMU / Magnetometer)
        sensorReadings.forEach { sensor ->
            val type = when (sensor.sensorType) {
                SensorType.ACCELEROMETER -> TimelineObservationType.ACCELEROMETER
                SensorType.GYROSCOPE -> TimelineObservationType.GYROSCOPE
                SensorType.MAGNETIC_FIELD -> TimelineObservationType.MAGNETOMETER
                SensorType.ROTATION_VECTOR -> TimelineObservationType.ORIENTATION
            }

            val summary = when (sensor.sensorType) {
                SensorType.ACCELEROMETER -> "Acc: %.2f m/s² (X: %.2f, Y: %.2f, Z: %.2f)".format(Locale.US, sensor.magnitude, sensor.x, sensor.y, sensor.z)
                SensorType.GYROSCOPE -> "Rot: %.2f rad/s (X: %.2f, Y: %.2f, Z: %.2f)".format(Locale.US, sensor.magnitude, sensor.x, sensor.y, sensor.z)
                SensorType.MAGNETIC_FIELD -> "Mag: %.1f µT (X: %.1f, Y: %.1f, Z: %.1f)".format(Locale.US, sensor.magnitude, sensor.x, sensor.y, sensor.z)
                SensorType.ROTATION_VECTOR -> "Heading: %.1f° (Pitch: %.1f°, Roll: %.1f°)".format(Locale.US, sensor.x, sensor.y, sensor.z)
            }

            items.add(
                SessionTimelineItem(
                    id = sensor.observationId,
                    sessionId = sensor.sessionId,
                    type = type,
                    normalizedTimestampMs = sensor.timestamp.normalizedEpochMs,
                    originalTimestamp = sensor.timestamp.originalTimestamp,
                    timeSourceDisplay = sensor.timestamp.timeSource.description,
                    formattedTime = SessionTimelineItem.formatTimestamp(sensor.timestamp.normalizedEpochMs),
                    title = "${sensor.sensorType.name} Sample",
                    shortDescription = summary,
                    isMock = sensor.isMock,
                    isTimingCertain = sensor.timestamp.isReliable,
                    uncertaintyDisplay = sensor.timestamp.uncertaintyDisplay,
                    detailKeyValues = mapOf(
                        "Sensor Type" to sensor.sensorType.name,
                        "Magnitude" to "%.3f".format(Locale.US, sensor.magnitude),
                        "Axes (X, Y, Z)" to "(%.3f, %.3f, %.3f)".format(Locale.US, sensor.x, sensor.y, sensor.z),
                        "Accuracy" to sensor.accuracy.name,
                        "Sampling Profile" to sensor.samplingRate,
                        "Sync Method" to sensor.timestamp.syncMethod.name,
                        "Hardware Nanos" to sensor.timestamp.originalTimestamp.toString()
                    )
                )
            )
        }

        // 3. Process Photos
        photos.forEach { photo ->
            items.add(
                SessionTimelineItem(
                    id = photo.photoId,
                    sessionId = photo.sessionId,
                    type = TimelineObservationType.PHOTO_CAPTURE,
                    normalizedTimestampMs = photo.timestamp.normalizedEpochMs,
                    originalTimestamp = photo.timestamp.originalTimestamp,
                    timeSourceDisplay = photo.timestamp.timeSource.description,
                    formattedTime = SessionTimelineItem.formatTimestamp(photo.timestamp.normalizedEpochMs),
                    title = "Photo: ${photo.landmarkType.displayName}",
                    shortDescription = "${photo.displayLocation} • ${if (photo.faceBlurApplied) "${photo.facesDetectedCount} faces blurred" else "0 faces"}",
                    isMock = photo.isMock,
                    isTimingCertain = photo.timestamp.isReliable,
                    uncertaintyDisplay = photo.timestamp.uncertaintyDisplay,
                    detailKeyValues = mapOf(
                        "Landmark Type" to photo.landmarkType.displayName,
                        "Filename" to photo.filename,
                        "Location" to photo.displayLocation,
                        "Dimensions" to "${photo.imageWidth}x${photo.imageHeight}",
                        "File Size" to "${photo.fileSizeBytes / 1024} KB",
                        "EXIF Sanitized" to photo.isExifSanitized.toString(),
                        "Face Detection" to photo.faceDetectionStatus.name,
                        "Description" to photo.description,
                        "Surveyor Notes" to photo.notes
                    )
                )
            )
        }

        // 4. Process Manual Annotations
        annotations.forEach { ann ->
            items.add(
                SessionTimelineItem(
                    id = ann.annotationId,
                    sessionId = ann.sessionId,
                    type = TimelineObservationType.ANNOTATION,
                    normalizedTimestampMs = ann.timestamp.normalizedEpochMs,
                    originalTimestamp = ann.timestamp.originalTimestamp,
                    timeSourceDisplay = ann.timestamp.timeSource.description,
                    formattedTime = SessionTimelineItem.formatTimestamp(ann.timestamp.normalizedEpochMs),
                    title = "Annotation: ${ann.type.displayName}",
                    shortDescription = "${ann.description} (${ann.displayLocation})",
                    isMock = ann.isMock,
                    isTimingCertain = ann.timestamp.isReliable,
                    uncertaintyDisplay = ann.timestamp.uncertaintyDisplay,
                    detailKeyValues = mapOf(
                        "Annotation Type" to ann.type.displayName,
                        "Description" to ann.description,
                        "Location" to ann.displayLocation,
                        "Related ID" to (ann.relatedObservationId ?: "None")
                    )
                )
            )
        }

        // Sort chronologically ascending
        return items.sortedWith { a, b ->
            val timeA = a.normalizedTimestampMs ?: Long.MAX_VALUE
            val timeB = b.normalizedTimestampMs ?: Long.MAX_VALUE
            timeA.compareTo(timeB)
        }
    }
}
