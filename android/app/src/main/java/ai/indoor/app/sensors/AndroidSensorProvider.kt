package ai.indoor.app.sensors

import ai.indoor.app.domain.models.ImuSample
import ai.indoor.app.domain.models.SensorAccuracy
import ai.indoor.app.domain.models.SensorSamplingRate
import ai.indoor.app.domain.models.SensorType
import ai.indoor.app.domain.models.SingleSensorReading
import ai.indoor.app.sensors.fusion.LowPassGravityFilter
import ai.indoor.app.sensors.fusion.SensorFusionManager
import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.Collections

/**
 * Production implementation of Android SensorEventListener managing real-time
 * Accelerometer, Gyroscope, Magnetometer, and Orientation streams.
 *
 * Adheres to strict Android energy and privacy requirements:
 * 1. Safe Lifecycle Management: Unregisters all listeners immediately on stop.
 * 2. Nanosecond Precision: Preserves raw hardware nanosecond timestamps (event.timestamp).
 * 3. Graceful Hardware Absence: Supports devices lacking gyroscopes or magnetometers without crashing.
 * 4. Gravity & Orientation Estimation: Applies LowPassGravityFilter and triad fusion.
 */
class AndroidSensorProvider(
    private val context: Context
) : SensorProvider, SensorEventListener {

    private val sensorManager: SensorManager? =
        context.applicationContext.getSystemService(Context.SENSOR_SERVICE) as? SensorManager

    private val accelerometer: Sensor? = sensorManager?.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val gyroscope: Sensor? = sensorManager?.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    private val magnetometer: Sensor? = sensorManager?.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)
    private val rotationVector: Sensor? = sensorManager?.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)

    override val availability: SensorAvailability = SensorAvailability(
        hasAccelerometer = accelerometer != null,
        hasGyroscope = gyroscope != null,
        hasMagnetometer = magnetometer != null,
        hasRotationVector = rotationVector != null,
        isMock = false
    )

    private val _sensorReadings = MutableStateFlow(ImuSample(isMock = false))
    override val sensorReadings: StateFlow<ImuSample> = _sensorReadings.asStateFlow()

    private val _isListening = MutableStateFlow(false)
    override val isListening: StateFlow<Boolean> = _isListening.asStateFlow()

    private val _samplingRate = MutableStateFlow(SensorSamplingRate.UI)
    override val samplingRate: StateFlow<SensorSamplingRate> = _samplingRate.asStateFlow()

    private val _collectedReadings = Collections.synchronizedList(mutableListOf<SingleSensorReading>())
    override val collectedReadings: List<SingleSensorReading>
        get() = synchronized(_collectedReadings) { _collectedReadings.toList() }

    private var activeSessionId: String? = null
    private val gravityFilter = LowPassGravityFilter(alpha = 0.8f)

    // Cached measurement vectors for orientation fusion
    private val lastGravity = FloatArray(3) { if (it == 2) 9.81f else 0f }
    private val lastGeomagnetic = FloatArray(3)
    private var hasGravitySample = false
    private var hasGeomagneticSample = false

    // Cached individual axis metrics
    private var lastAx = 0f
    private var lastAy = 0f
    private var lastAz = 9.81f
    private var lastAccelNs = 0L
    private var lastAccelAcc = SensorAccuracy.HIGH

    private var lastGx = 0f
    private var lastGy = 0f
    private var lastGz = 0f
    private var lastGyroNs = 0L
    private var lastGyroAcc = SensorAccuracy.HIGH

    private var lastMx = 0f
    private var lastMy = 0f
    private var lastMz = 0f
    private var lastMagNs = 0L
    private var lastMagAcc = SensorAccuracy.HIGH

    private var lastGravX = 0f
    private var lastGravY = 0f
    private var lastGravZ = 9.81f
    private var lastLinAx = 0f
    private var lastLinAy = 0f
    private var lastLinAz = 0f

    override fun setSamplingRate(rate: SensorSamplingRate) {
        _samplingRate.value = rate
        if (_isListening.value) {
            // Re-register with the updated sampling profile
            stopListening()
            startListening(activeSessionId)
        }
    }

    override fun startListening(sessionId: String?) {
        if (_isListening.value || sensorManager == null) return
        activeSessionId = sessionId

        val delayCode = _samplingRate.value.delayCode

        accelerometer?.let {
            sensorManager.registerListener(this, it, delayCode)
        }
        gyroscope?.let {
            sensorManager.registerListener(this, it, delayCode)
        }
        magnetometer?.let {
            sensorManager.registerListener(this, it, delayCode)
        }

        _isListening.value = true
    }

    override fun stopListening() {
        if (!_isListening.value || sensorManager == null) return

        sensorManager.unregisterListener(this)
        _isListening.value = false
        gravityFilter.reset()
    }

    override fun clearCollectedReadings() {
        synchronized(_collectedReadings) {
            _collectedReadings.clear()
        }
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (event == null) return

        val accuracy = SensorAccuracy.fromCode(event.accuracy)
        val rateLabel = _samplingRate.value.label

        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                lastAx = event.values[0]
                lastAy = event.values[1]
                lastAz = event.values[2]
                lastAccelNs = event.timestamp
                lastAccelAcc = accuracy

                // Low-pass filter to extract gravity vs linear dynamic motion
                val gravEstimate = gravityFilter.update(lastAx, lastAy, lastAz)
                lastGravX = gravEstimate.gx
                lastGravY = gravEstimate.gy
                lastGravZ = gravEstimate.gz
                lastLinAx = gravEstimate.linearAccelX
                lastLinAy = gravEstimate.linearAccelY
                lastLinAz = gravEstimate.linearAccelZ

                lastGravity[0] = lastGravX
                lastGravity[1] = lastGravY
                lastGravity[2] = lastGravZ
                hasGravitySample = true

                bufferReading(
                    SingleSensorReading(
                        sensorType = SensorType.ACCELEROMETER,
                        x = lastAx,
                        y = lastAy,
                        z = lastAz,
                        timestampNs = event.timestamp,
                        accuracy = accuracy,
                        sessionId = activeSessionId,
                        samplingRate = rateLabel,
                        isMock = false
                    )
                )
            }
            Sensor.TYPE_GYROSCOPE -> {
                lastGx = event.values[0]
                lastGy = event.values[1]
                lastGz = event.values[2]
                lastGyroNs = event.timestamp
                lastGyroAcc = accuracy

                bufferReading(
                    SingleSensorReading(
                        sensorType = SensorType.GYROSCOPE,
                        x = lastGx,
                        y = lastGy,
                        z = lastGz,
                        timestampNs = event.timestamp,
                        accuracy = accuracy,
                        sessionId = activeSessionId,
                        samplingRate = rateLabel,
                        isMock = false
                    )
                )
            }
            Sensor.TYPE_MAGNETIC_FIELD -> {
                lastMx = event.values[0]
                lastMy = event.values[1]
                lastMz = event.values[2]
                lastMagNs = event.timestamp
                lastMagAcc = accuracy

                lastGeomagnetic[0] = lastMx
                lastGeomagnetic[1] = lastMy
                lastGeomagnetic[2] = lastMz
                hasGeomagneticSample = true

                bufferReading(
                    SingleSensorReading(
                        sensorType = SensorType.MAGNETIC_FIELD,
                        x = lastMx,
                        y = lastMy,
                        z = lastMz,
                        timestampNs = event.timestamp,
                        accuracy = accuracy,
                        sessionId = activeSessionId,
                        samplingRate = rateLabel,
                        isMock = false
                    )
                )
            }
        }

        // Compute orientation if both gravity and magnetic flux vectors are available
        val orientation = if (hasGravitySample && hasGeomagneticSample) {
            SensorFusionManager.computeOrientation(lastGravity, lastGeomagnetic)
        } else {
            null
        }

        _sensorReadings.value = ImuSample(
            ax = lastAx,
            ay = lastAy,
            az = lastAz,
            accelTimestampNs = lastAccelNs,
            accelAccuracy = lastAccelAcc,

            gx = lastGx,
            gy = lastGy,
            gz = lastGz,
            gyroTimestampNs = lastGyroNs,
            gyroAccuracy = lastGyroAcc,

            mx = lastMx,
            my = lastMy,
            mz = lastMz,
            magTimestampNs = lastMagNs,
            magAccuracy = lastMagAcc,

            gravityX = lastGravX,
            gravityY = lastGravY,
            gravityZ = lastGravZ,
            linearAccelX = lastLinAx,
            linearAccelY = lastLinAy,
            linearAccelZ = lastLinAz,

            azimuthDeg = orientation?.azimuthDeg ?: 0f,
            pitchDeg = orientation?.pitchDeg ?: 0f,
            rollDeg = orientation?.rollDeg ?: 0f,
            hasOrientation = orientation != null,

            timestamp = System.currentTimeMillis(),
            isMock = false
        )
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // Updated accuracy will be propagated on subsequent onSensorChanged calls
    }

    private fun bufferReading(reading: SingleSensorReading) {
        synchronized(_collectedReadings) {
            if (_collectedReadings.size < MAX_BUFFER_CAPACITY) {
                _collectedReadings.add(reading)
            }
        }
    }

    companion object {
        private const val MAX_BUFFER_CAPACITY = 20_000
    }
}
