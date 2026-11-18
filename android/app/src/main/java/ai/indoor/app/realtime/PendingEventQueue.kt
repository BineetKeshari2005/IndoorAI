package ai.indoor.app.realtime

import java.util.ArrayDeque

/**
 * Thread-safe, bounded in-memory event buffer for unacknowledged events during network outages.
 *
 * Enforces FIFO ordering, bounded drop-oldest backpressure, and idempotent event ID preservation.
 */
class PendingEventQueue(
    private val maxCapacity: Int = 1000
) {
    private val queue = ArrayDeque<RealtimeEventDto>()
    private val lock = Any()

    var totalEnqueued: Long = 0
        private set

    var totalDropped: Long = 0
        private set

    var totalAcknowledged: Long = 0
        private set

    val size: Int
        get() = synchronized(lock) { queue.size }

    val isEmpty: Boolean
        get() = synchronized(lock) { queue.isEmpty() }

    /**
     * Enqueues an event. If capacity is exceeded, evicts the oldest event to prevent memory exhaustion.
     */
    fun enqueue(event: RealtimeEventDto): Boolean {
        synchronized(lock) {
            if (queue.size >= maxCapacity) {
                queue.pollFirst()
                totalDropped++
            }
            queue.addLast(event)
            totalEnqueued++
            return true
        }
    }

    /**
     * Retrieves up to [maxCount] pending events without removing them.
     */
    fun peekBatch(maxCount: Int = 50): List<RealtimeEventDto> {
        synchronized(lock) {
            return queue.take(maxCount)
        }
    }

    /**
     * Removes an event from the pending queue when positive ACK is received from the server.
     */
    fun removeAcknowledged(eventId: String): Boolean {
        synchronized(lock) {
            val iterator = queue.iterator()
            while (iterator.hasNext()) {
                val item = iterator.next()
                if (item.eventId == eventId) {
                    iterator.remove()
                    totalAcknowledged++
                    return true
                }
            }
            return false
        }
    }

    /**
     * Removes multiple events by IDs upon batch ACK.
     */
    fun removeBatchAcknowledged(eventIds: Collection<String>): Int {
        var count = 0
        synchronized(lock) {
            val idSet = eventIds.toSet()
            val iterator = queue.iterator()
            while (iterator.hasNext()) {
                val item = iterator.next()
                if (idSet.contains(item.eventId)) {
                    iterator.remove()
                    totalAcknowledged++
                    count++
                }
            }
        }
        return count
    }

    /**
     * Clears all pending events.
     */
    fun clear() {
        synchronized(lock) {
            queue.clear()
        }
    }
}
