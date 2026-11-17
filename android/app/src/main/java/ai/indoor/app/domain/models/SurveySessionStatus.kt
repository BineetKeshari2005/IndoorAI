package ai.indoor.app.domain.models

/**
 * Standard lifecycle states for a multimodal IndoorAI survey recording session.
 */
enum class SurveySessionStatus(val displayName: String) {
    /**
     * Session configuration prepared; awaiting start command.
     */
    CREATED("Created"),

    /**
     * Active survey recording; incoming observations are associated and ingested.
     */
    ACTIVE("Active"),

    /**
     * Survey temporarily paused; observation streams are suspended.
     */
    PAUSED("Paused"),

    /**
     * Survey concluded normally; dataset sealed for export and verification.
     */
    COMPLETED("Completed"),

    /**
     * Survey aborted by user or surveyor; data retained or flagged for deletion.
     */
    CANCELLED("Cancelled")
}
