package ai.indoor.app.data.api

import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit

/**
 * Factory for creating configured Retrofit instances with custom base URLs.
 */
object ApiClient {

    /**
     * Standard local address for Android Studio Emulator pointing to host's 127.0.0.1.
     * For physical devices, this must be changed to the computer's LAN IP (e.g. 192.168.1.X).
     */
    const val EMULATOR_DEFAULT_BASE_URL = "http://10.0.2.2:8000/"

    fun create(baseUrl: String = EMULATOR_DEFAULT_BASE_URL): IndoorApiService {
        val normalizedUrl = if (baseUrl.endsWith("/")) baseUrl else "$baseUrl/"

        val loggingInterceptor = HttpLoggingInterceptor().apply {
            level = HttpLoggingInterceptor.Level.BASIC
        }

        val okHttpClient = OkHttpClient.Builder()
            .addInterceptor(loggingInterceptor)
            .connectTimeout(10, TimeUnit.SECONDS)
            .readTimeout(15, TimeUnit.SECONDS)
            .writeTimeout(15, TimeUnit.SECONDS)
            .build()

        return Retrofit.Builder()
            .baseUrl(normalizedUrl)
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(IndoorApiService::class.java)
    }
}

