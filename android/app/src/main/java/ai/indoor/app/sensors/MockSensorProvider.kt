package ai.indoor.app.sensors

import ai.indoor.app.domain.models.ImuSample
import ai.indoor.app.domain.models.SensorAccuracy
import ai.indoor.app.domain.models.SensorSamplingRate
import ai.indoor.app.domain.models.SensorType
import ai.indoor.app.domain.models.SingleSensorReading
import ai.indoor.app.sensors.fusion.LowPassGravityFilter
import ai.indoor.app.sensors.fusion.SensorFusionManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.util.Collections
import kotlin.math.cos
import kotlin.math.sin
import kotlin.random.Random

/**
 * Synthetic sensor provider for Android emulators, testing, or devices
 * lacking hardware gyroscopes or magnetometers.
 *
 * All generated records are strictly tagged with `isMock = true`.
 */
class MockSensorProvider(
    private val scope: CoroutineScope = CoroutineScope(Dispatchers.Default)
) : SensorProvider {

    override val availability: SensorAvailability = SensorAvailability(
        hasAccelerometer = true,
        hasGyroscope = true,
        hasMagnetometer = true,
        hasRotationVector = true,
        isMock = true
    )

    private val _sensorReadings = MutableStateFlow(ImuSample(isMock = true))
    override val sensorReadings: StateFlow<ImuSample> = _sensorReadings.asStateFlow()

    private val _isListening = MutableStateFlow(false)
    override val isListening: StateFlow<Boolean> = _isListening.asStateFlow()

    private val _samplingRate = MutableStateFlow(SensorSamplingRate.UI)
    override val samplingRate: StateFlow<SensorSamplingRate> = _samplingRate.asStateFlow()

    private val _collectedReadings = Collections.synchronizedList(mutableListOf<SingleSensorReading>())
    override val collectedReadings: List<SingleSensorReading>
        get() = synchronized(_collectedReadings) { _collectedReadings.toList() }

    private var activeSessionId: String? = null
    private var simulationJob: Job? = null
    private var stepTime = 0.0
    private val gravityFilter = LowPassGravityFilter(alpha = 0.8f)

    override fun setSamplingRate(rate: SensorSamplingRate) {
        _samplingRate.value = rate
        if (_isListening.value) {
            stopListening()
            startListening(activeSessionId)
        }
    }

    override fun startListening(sessionId: String?) {
        if (_isListening.value) return
        activeSessionId = sessionId
        _isListening.value = true

        val delayIntervalMs = when (_samplingRate.value) {
            SensorSamplingRate.FASTEST -> 10L // ~100 Hz
            SensorSamplingRate.GAME -> 20L    // ~50 Hz
            SensorSamplingRate.UI -> 60L      // ~16 Hz
        }

        simulationJob = scope.launch {
            while (isActive) {
                delay(delayIntervalMs)
                stepTime += (delayIntervalMs / 1000.0)

                val nowNs = System.nanoTime()
                val rateLabel = _samplingRate.value.label

                // Realistic human handheld walking gait dynamics
                val tremor = (Random.nextFloat() - 0.5f) * 0.08f
                val walkingAccel = (sin(stepTime * 3.0) * 0.45f).toFloat()

                val newAx = (sin(stepTime * 0.5) * 0.15f).toFloat() + tremor
                val newAy = walkingAccel + tremor
                val newAz = 9.81f + (cos(stepTime * 3.0) * 0.25f).toFloat() + tremor

                val newGx = (cos(stepTime) * 0.03f).toFloat()
                val newGy = (sin(stepTime * 1.5) * 0.04f).toFloat()
                val newGz = (sin(stepTime * 0.8) * 0.02f).toFloat()

                // Earth magnetic field (~45 µT with gradual heading rotation)
                val headingAngle = stepTime * 0.1
                val newMx = (cos(headingAngle) * 22.0f).toFloat()
                val newMy = (sin(headingAngle) * 22.0f).toFloat()
                val newMz = 40.5f

                // Low-pass filter for gravity & dynamic motion
                val grav = gravityFilter.update(newAx, newAy, newAz)

                // Compute orientation
                val gravArr = floatArrayOf(grav.gx, grav.gy, grav.gz)
                val magArr = floatArrayOf(newMx, newMy, newMz)
                val orientation = SensorFusionManager.computeOrientation(gravArr, magArr)

                // Buffer individual readings
                synchronized(_collectedReadings) {
                    if (_collectedReadings.size < 20_000) {
                        _collectedReadings.add(
                            SingleSensorReading(
                                sensorType = SensorType.ACCELEROMETER,
                                x = newAx, y = newAy, z = newAz,
                                timestampNs = nowNs,
                                accuracy = SensorAccuracy.HIGH,
                                sessionId = activeSessionId,
                                samplingRate = rateLabel,
                                isMock = true
                            )
                        )
                        _collectedReadings.add(
                            SingleSensorReading(
                                sensorType = SensorType.GYROSCOPE,
                                x = newGx, y = newGy, z = newGz,
                                timestampNs = nowNs,
                                accuracy = SensorAccuracy.HIGH,
                                sessionId = activeSessionId,
                                samplingRate = rateLabel,
                                isMock = true
                            )
                        )
                        _collectedReadings.add(
                            SingleSensorReading(
                                sensorType = SensorType.MAGNETIC_FIELD,
                                x = newMx, y = newMy, z = newMz,
                                timestampNs = nowNs,
                                accuracy = SensorAccuracy.HIGH,
                                sessionId = activeSessionId,
                                samplingRate = rateLabel,
                                isMock = true
                            )
                        )
                    }
                }

                _sensorReadings.value = ImuSample(
                    ax = newAx,
                    ay = newAy,
                    az = newAz,
                    accelTimestampNs = nowNs,
                    accelAccuracy = SensorAccuracy.HIGH,

                    gx = newGx,
                    gy = newGy,
                    gz = newGz,
                    gyroTimestampNs = nowNs,
                    gyroAccuracy = SensorAccuracy.HIGH,

                    mx = newMx,
                    my = newMy,
                    mz = newMz,
                    magTimestampNs = nowNs,
                    magAccuracy = SensorAccuracy.HIGH,

                    gravityX = grav.gx,
                    gravityY = grav.gy,
                    gravityZ = grav.gz,
                    linearAccelX = grav.linearAccelX,
                    linearAccelY = grav.linearAccelY,
                    linearAccelZ = grav.linearAccelZ,

                    azimuthDeg = orientation?.azimuthDeg ?: 0f,
                    pitchDeg = orientation?.pitchDeg ?: 0f,
                    rollDeg = orientation?.rollDeg ?: 0f,
                    hasOrientation = orientation != null,

                    timestamp = System.currentTimeMillis(),
                    isMock = true
                )
            }
        }
    }

    override fun stopListening() {
        simulationJob?.cancel()
        simulationJob = null
        _isListening.value = false
        gravityFilter.reset()
    }

    override fun clearCollectedReadings() {
        synchronized(_collectedReadings) {
            _collectedReadings.clear()
        }
    }
}
