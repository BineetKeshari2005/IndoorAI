package ai.indoor.app.data.api

import ai.indoor.app.data.models.HealthResponseDto
import ai.indoor.app.data.models.LocalizationRequestDto
import ai.indoor.app.data.models.LocalizationResponseDto
import ai.indoor.app.data.models.SessionTelemetryDto
import ai.indoor.app.data.models.SessionUploadResponseDto
import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST

/**
 * Retrofit contract for IndoorAI backend services.
 *
 * NOTE: The endpoints below describe the contract for the upcoming FastAPI
 * backend endpoints scheduled for Phase 8 (Tasks 27 & 28). If the backend is
 * offline or an endpoint is not yet activated, the repository layer provides
 * graceful offline mock handling.
 */
interface IndoorApiService {

    /**
     * Service health check endpoint.
     */
    @GET("api/v1/health")
    suspend fun checkHealth(): Response<HealthResponseDto>

    /**
     * Submit an ambient Wi-Fi / multimodal observation for live positioning prediction.
     */
    @POST("api/v1/predict")
    suspend fun requestLocalization(
        @Body request: LocalizationRequestDto
    ): Response<LocalizationResponseDto>

    /**
     * Upload an aggregated telemetry session batch.
     */
    @POST("api/v1/telemetry/session")
    suspend fun uploadSessionTelemetry(
        @Body session: SessionTelemetryDto
    ): Response<SessionUploadResponseDto>
}

