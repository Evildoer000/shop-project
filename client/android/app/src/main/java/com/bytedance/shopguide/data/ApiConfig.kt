package com.bytedance.shopguide.data

import com.bytedance.shopguide.BuildConfig
import kotlinx.serialization.json.Json
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import java.util.concurrent.TimeUnit

/**
 * Centralised API configuration.
 *
 * The base URL is injected at build time via `BuildConfig.API_BASE_URL`, which
 * defaults to `http://10.0.2.2:8000/` (Android emulator → host machine).
 * Override on real devices by setting `API_BASE_URL=<value>` in
 * `local.properties` or as a Gradle property.
 */
object ApiConfig {
    val baseUrl: String = BuildConfig.API_BASE_URL.trimEnd('/') + "/"

    val json: Json = Json {
        ignoreUnknownKeys = true
        explicitNulls = false
        coerceInputValues = true
        isLenient = true
    }

    val sharedOkHttp: OkHttpClient by lazy {
        val logging = HttpLoggingInterceptor().apply {
            level = if (BuildConfig.DEBUG) {
                HttpLoggingInterceptor.Level.BASIC
            } else {
                HttpLoggingInterceptor.Level.NONE
            }
        }
        OkHttpClient.Builder()
            .connectTimeout(15, TimeUnit.SECONDS)
            .writeTimeout(30, TimeUnit.SECONDS)
            // readTimeout=0 by default so SSE consumers don't get killed; the
            // streaming client copies & overrides this on its own builder.
            .readTimeout(60, TimeUnit.SECONDS)
            .addInterceptor(logging)
            .build()
    }
}
