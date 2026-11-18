package ai.indoor.app.realtime

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class PendingEventQueueTest {

    @Test
    fun testEnqueueAndPeekBatch() {
        val queue = PendingEventQueue(maxCapacity = 10)
        assertEquals(0, queue.size)
        assertTrue(queue.isEmpty)

        val ev1 = RealtimeEventDto(eventId = "EV-1", sessionId = "S1", eventType = RealtimeEventType.HEARTBEAT, deviceId = "D1")
        val ev2 = RealtimeEventDto(eventId = "EV-2", sessionId = "S1", eventType = RealtimeEventType.WIFI_SCAN, deviceId = "D1")

        queue.enqueue(ev1)
        queue.enqueue(ev2)

        assertEquals(2, queue.size)
        assertFalse(queue.isEmpty)
        assertEquals(2L, queue.totalEnqueued)

        val batch = queue.peekBatch(10)
        assertEquals(2, batch.size)
        assertEquals("EV-1", batch[0].eventId)
        assertEquals("EV-2", batch[1].eventId)
    }

    @Test
    fun testBoundedDropOldestPolicy() {
        val queue = PendingEventQueue(maxCapacity = 3)

        for (i in 1..5) {
            queue.enqueue(
                RealtimeEventDto(
                    eventId = "EV-$i",
                    sessionId = "S1",
                    eventType = RealtimeEventType.HEARTBEAT,
                    deviceId = "D1"
                )
            )
        }

        // Capacity is 3, so size should be 3
        assertEquals(3, queue.size)
        assertEquals(5L, queue.totalEnqueued)
        assertEquals(2L, queue.totalDropped)

        // Oldest (EV-1, EV-2) were evicted; remaining should be EV-3, EV-4, EV-5
        val batch = queue.peekBatch(10)
        assertEquals(3, batch.size)
        assertEquals("EV-3", batch[0].eventId)
        assertEquals("EV-4", batch[1].eventId)
        assertEquals("EV-5", batch[2].eventId)
    }

    @Test
    fun testRemoveAcknowledgedEvent() {
        val queue = PendingEventQueue(maxCapacity = 10)

        queue.enqueue(RealtimeEventDto(eventId = "EV-1", sessionId = "S1", eventType = RealtimeEventType.HEARTBEAT, deviceId = "D1"))
        queue.enqueue(RealtimeEventDto(eventId = "EV-2", sessionId = "S1", eventType = RealtimeEventType.HEARTBEAT, deviceId = "D1"))
        queue.enqueue(RealtimeEventDto(eventId = "EV-3", sessionId = "S1", eventType = RealtimeEventType.HEARTBEAT, deviceId = "D1"))

        assertEquals(3, queue.size)

        val removed = queue.removeAcknowledged("EV-2")
        assertTrue(removed)
        assertEquals(2, queue.size)
        assertEquals(1L, queue.totalAcknowledged)

        val remaining = queue.peekBatch(10)
        assertEquals(2, remaining.size)
        assertEquals("EV-1", remaining[0].eventId)
        assertEquals("EV-3", remaining[1].eventId)

        // Try removing non-existent event
        val notRemoved = queue.removeAcknowledged("EV-UNKNOWN")
        assertFalse(notRemoved)
        assertEquals(2, queue.size)
    }

    @Test
    fun testRemoveBatchAcknowledged() {
        val queue = PendingEventQueue(maxCapacity = 10)

        queue.enqueue(RealtimeEventDto(eventId = "EV-A", sessionId = "S1", eventType = RealtimeEventType.HEARTBEAT, deviceId = "D1"))
        queue.enqueue(RealtimeEventDto(eventId = "EV-B", sessionId = "S1", eventType = RealtimeEventType.HEARTBEAT, deviceId = "D1"))
        queue.enqueue(RealtimeEventDto(eventId = "EV-C", sessionId = "S1", eventType = RealtimeEventType.HEARTBEAT, deviceId = "D1"))

        val count = queue.removeBatchAcknowledged(listOf("EV-A", "EV-C", "EV-NONEXISTENT"))
        assertEquals(2, count)
        assertEquals(1, queue.size)
        assertEquals("EV-B", queue.peekBatch(10)[0].eventId)
    }

    @Test
    fun testClearQueue() {
        val queue = PendingEventQueue(maxCapacity = 10)
        queue.enqueue(RealtimeEventDto(eventId = "EV-1", sessionId = "S1", eventType = RealtimeEventType.HEARTBEAT, deviceId = "D1"))
        queue.enqueue(RealtimeEventDto(eventId = "EV-2", sessionId = "S1", eventType = RealtimeEventType.HEARTBEAT, deviceId = "D1"))

        assertEquals(2, queue.size)
        queue.clear()
        assertEquals(0, queue.size)
        assertTrue(queue.isEmpty)
    }
}
