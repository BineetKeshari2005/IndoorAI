package ai.indoor.app.sensors.fusion

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class SensorFusionManagerTest {

    @Test
    fun testOrientationFlatFacingNorth() {
        // Device flat on a table (screen facing up: Gz = 9.81 m/s^2)
        // Pointing North: Geomagnetic field vector has positive Y (horizontal North) and negative Z (inclination)
        val gravity = floatArrayOf(0f, 0f, 9.81f)
        val geomagnetic = floatArrayOf(0f, 22.0f, -40.0f)

        val orientation = SensorFusionManager.computeOrientation(gravity, geomagnetic)

        assertNotNull(orientation)
        assertTrue(orientation!!.isValid)

        // Flat device has pitch ~ 0 and roll ~ 0
        assertEquals(0f, orientation.pitchDeg, 1.0f)
        assertEquals(0f, orientation.rollDeg, 1.0f)

        // Pointing North has compass heading ~ 0° / 360°
        assertEquals(0f, orientation.compassHeadingDeg, 2.0f)
    }

    @Test
    fun testOrientationPitchTiltedForward() {
        // Device tilted 45 degrees forward: Gx = 0, Gy = 9.81 * sin(45°), Gz = 9.81 * cos(45°)
        val gY = 9.81f * 0.7071f
        val gZ = 9.81f * 0.7071f
        val gravity = floatArrayOf(0f, gY, gZ)
        val geomagnetic = floatArrayOf(0f, 22.0f, -40.0f)

        val orientation = SensorFusionManager.computeOrientation(gravity, geomagnetic)
        assertNotNull(orientation)

        // Pitch should indicate forward tilt (~ -45°)
        assertEquals(-45f, orientation!!.pitchDeg, 3.0f)
        // Roll should remain ~ 0°
        assertEquals(0f, orientation.rollDeg, 2.0f)
    }

    @Test
    fun testFreeFallDegenerateCaseReturnsNull() {
        // Zero gravity vector (free-fall state)
        val zeroGravity = floatArrayOf(0f, 0f, 0f)
        val geomagnetic = floatArrayOf(0f, 22.0f, -40.0f)

        val orientation = SensorFusionManager.computeOrientation(zeroGravity, geomagnetic)
        assertNull("Orientation cannot be resolved in free-fall without gravity anchor", orientation)
    }

    @Test
    fun testIncompleteArrayReturnsNull() {
        val shortGravity = floatArrayOf(0f, 9.81f)
        val geomagnetic = floatArrayOf(0f, 22.0f, -40.0f)

        val orientation = SensorFusionManager.computeOrientation(shortGravity, geomagnetic)
        assertNull(orientation)
    }
}
