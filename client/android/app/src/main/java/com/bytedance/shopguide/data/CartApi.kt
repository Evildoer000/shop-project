package com.bytedance.shopguide.data

import com.bytedance.shopguide.model.CartDto
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.HttpUrl.Companion.toHttpUrl
import okhttp3.OkHttpClient
import okhttp3.Request

class CartApi(
    baseUrl: String = ApiConfig.baseUrl,
    private val client: OkHttpClient = ApiConfig.sharedOkHttp,
    private val json: kotlinx.serialization.json.Json = ApiConfig.json,
) {
    private val cartUrl = baseUrl.trimEnd('/') + "/api/cart"

    suspend fun getCart(userId: String, sessionId: String): Result<CartDto> = withContext(Dispatchers.IO) {
        runCatching {
            val url = cartUrl.toHttpUrl().newBuilder()
                .addQueryParameter("user_id", userId)
                .addQueryParameter("session_id", sessionId)
                .build()
            val request = Request.Builder().url(url).get().build()
            client.newCall(request).execute().use { resp ->
                if (!resp.isSuccessful) error("HTTP ${resp.code}")
                json.decodeFromString(CartDto.serializer(), resp.body?.string().orEmpty())
            }
        }
    }
}
