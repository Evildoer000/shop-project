package com.bytedance.shopguide.data

import com.bytedance.shopguide.model.EventReportRequestDto
import com.bytedance.shopguide.model.EventReportResponseDto
import com.bytedance.shopguide.model.RecommendationResponseDto
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import okhttp3.HttpUrl.Companion.toHttpUrl
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody

class BehaviorApi(
    baseUrl: String = ApiConfig.baseUrl,
    private val client: OkHttpClient = ApiConfig.sharedOkHttp,
    private val json: Json = ApiConfig.json,
) {
    private val eventsUrl = baseUrl.trimEnd('/') + "/api/events"
    private val recommendationsUrl = baseUrl.trimEnd('/') + "/api/recommendations"
    private val mediaType = "application/json; charset=utf-8".toMediaType()

    suspend fun fetchRecommendations(
        userId: String,
        size: Int = 24,
    ): Result<RecommendationResponseDto> = withContext(Dispatchers.IO) {
        runCatching {
            val url = recommendationsUrl.toHttpUrl().newBuilder()
                .addQueryParameter("user_id", userId)
                .addQueryParameter("size", size.toString())
                .build()
            val request = Request.Builder().url(url).get().build()
            client.newCall(request).execute().use { resp ->
                if (!resp.isSuccessful) error("HTTP ${resp.code}")
                json.decodeFromString(RecommendationResponseDto.serializer(), resp.body?.string().orEmpty())
            }
        }
    }

    suspend fun reportEvent(request: EventReportRequestDto): Result<EventReportResponseDto> =
        withContext(Dispatchers.IO) {
            runCatching {
                val body = json.encodeToString(request).toRequestBody(mediaType)
                val httpRequest = Request.Builder().url(eventsUrl).post(body).build()
                client.newCall(httpRequest).execute().use { resp ->
                    if (!resp.isSuccessful) error("HTTP ${resp.code}")
                    json.decodeFromString(EventReportResponseDto.serializer(), resp.body?.string().orEmpty())
                }
            }
        }
}
