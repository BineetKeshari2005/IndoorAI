package ai.indoor.app.domain.timestamp

/**
 * Abstraction for system clocks, allowing monotonic and wall-clock time
 * acquisition across Android runtimes and JVM unit test suites.
 */
interface SystemClockProvider {
    fun currentTimeMillis(): Long
    fun elapsedRealtimeNanos(): Long

    companion object {
        val Default: SystemClockProvider = object : SystemClockProvider {
            override fun currentTimeMillis(): Long = System.currentTimeMillis()

            override fun elapsedRealtimeNanos(): Long {
                return try {
                    // Use Android's SystemClock if available on runtime classpath
                    val clazz = Class.forName("android.os.SystemClock")
                    val method = clazz.getMethod("elapsedRealtimeNanos")
                    method.invoke(null) as Long
                } catch (_: Throwable) {
                    // Fallback to high-resolution JVM nanoTime for unit testing
                    System.nanoTime()
                }
            }
        }
    }
}
