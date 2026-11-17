package ai.indoor.app.domain.models

/**
 * Data provenance classification distinguishing physical handset hardware observations
 * from synthetic benchmark / demonstration telemetry.
 */
enum class DataSourceType(val displayName: String) {
    /**
     * Exclusively collected from live physical hardware sensors and 802.11 radios.
     */
    REAL("Real Hardware"),

    /**
     * Exclusively generated from synthetic test / benchmark generators.
     */
    MOCK("Synthetic Mock"),

    /**
     * Mixed session containing both real and mock observations (distinguished per-observation).
     */
    HYBRID("Hybrid Real/Mock")
}
