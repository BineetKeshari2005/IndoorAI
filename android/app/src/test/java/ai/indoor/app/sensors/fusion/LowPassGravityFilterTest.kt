package ai.indoor.app.sensors.fusion

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

class LowPassGravityFilterTest {

    private lateinit var filter: LowPassGravityFilter

    @Before
    fun setUp() {
        filter = LowPassGravityFilter(alpha = 0.8f)
    }

    @Test
    fun testStaticGravityInitialization() {
        // First reading initializes gravity filter directly
        val estimate = filter.update(0f, 0f, 9.81f)

        assertEquals(0f, estimate.gx, 0.001f)
        assertEquals(0f, estimate.gy, 0.001f)
        assertEquals(9.81f, estimate.gz, 0.001f)

        // Static device has zero dynamic linear acceleration
        assertEquals(0f, estimate.linearAccelX, 0.001f)
        assertEquals(0f, estimate.linearAccelY, 0.001f)
        assertEquals(0f, estimate.linearAccelZ, 0.001f)
        assertEquals(0f, estimate.linearMagnitude, 0.001f)
    }

    @Test
    fun testDynamicAccelerationSeparation() {
        // Initialize at rest
        filter.update(0f, 0f, 9.81f)

        // Sudden forward acceleration step (e.g. Ay spikes to 3.0 m/s^2)
        val step = filter.update(0f, 3.0f, 9.81f)

        // Low-pass filter smooths gravity: gy = 0.8 * 0 + 0.2 * 3.0 = 0.6 m/s^2
        assertEquals(0.6f, step.gy, 0.001f)

        // Linear dynamic acceleration captures the forward surge: 3.0 - 0.6 = 2.4 m/s^2
        assertEquals(2.4f, step.linearAccelY, 0.001f)
        assertTrue(step.linearMagnitude > 2.0f)
    }

    @Test
    fun testFilterReset() {
        filter.update(1.0f, 2.0f, 8.0f)
        filter.reset()

        // After reset, first reading re-initializes
        val estimate = filter.update(0f, 0f, 9.81f)
        assertEquals(9.81f, estimate.gz, 0.001f)
        assertEquals(0f, estimate.linearAccelZ, 0.001f)
    }
}
