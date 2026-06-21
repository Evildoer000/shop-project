package com.bytedance.shopguide.data

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody

@Serializable
data class EventReportPayload(
    @SerialName("user_id") val userId: String,
    @SerialName("session_id") val sessionId: String,
    @SerialName("event_type") val eventType: String,
    @SerialName("product_id") val productId: String,
    val context: Map<String, String> = emptyMap(),
)

@Serializable
data class EventReportResult(
    val ok: Boolean = true,
    @SerialName("event_id") val eventId: Long,
)

class EventApi(
    baseUrl: String = ApiConfig.baseUrl,
    private val client: OkHttpClient = ApiConfig.sharedOkHttp,
    private val json: kotlinx.serialization.json.Json = ApiConfig.json,
) {
    private val eventsUrl: String = baseUrl.trimEnd('/') + "/api/events"
    private val mediaType = "application/json; charset=utf-8".toMediaType()

    suspend fun addToCart(
        userId: String,
        sessionId: String,
        productId: String,
        source: String,
        sku: Map<String, String> = emptyMap(),
    ): Result<EventReportResult> {
        val ctx = mutableMapOf("from" to source)
        if (sku.isNotEmpty()) {
            // 把 sku 序列化成 JSON string 塞 context, 后端 cart_snapshot 会解析
            ctx["sku"] = json.encodeToString(sku)
        }
        return report(
            EventReportPayload(
                userId = userId,
                sessionId = sessionId,
                eventType = "cart_add",
                productId = productId,
                context = ctx,
            ),
        )
    }

    /** 用户点击商品卡片, 例如 chat 里跳商品详情时. */
    suspend fun reportClick(
        userId: String,
        sessionId: String,
        productId: String,
        source: String,
        query: String = "",
    ): Result<EventReportResult> {
        val ctx = mutableMapOf("from" to source)
        if (query.isNotEmpty()) ctx["query"] = query.take(120)
        return report(
            EventReportPayload(
                userId = userId,
                sessionId = sessionId,
                eventType = "click",
                productId = productId,
                context = ctx,
            ),
        )
    }

    suspend fun removeFromCart(
        userId: String,
        sessionId: String,
        productId: String,
        source: String,
        sku: Map<String, String> = emptyMap(),
    ): Result<EventReportResult> {
        val ctx = mutableMapOf("from" to source)
        if (sku.isNotEmpty()) {
            // 跟 addToCart 对齐: 把 sku 序列化成 JSON string 塞 context
            // 后端 cart_snapshot 按 (product_id, sku_signature) 聚合, 不传 sku 会减不到原来的 bucket
            ctx["sku"] = json.encodeToString(sku)
        }
        return report(
            EventReportPayload(
                userId = userId,
                sessionId = sessionId,
                eventType = "cart_remove",
                productId = productId,
                context = ctx,
            ),
        )
    }

    private suspend fun report(payload: EventReportPayload): Result<EventReportResult> = withContext(Dispatchers.IO) {
        runCatching {
            val request = Request.Builder()
                .url(eventsUrl)
                .post(json.encodeToString(payload).toRequestBody(mediaType))
                .build()
            client.newCall(request).execute().use { resp ->
                if (!resp.isSuccessful) error("HTTP ${resp.code}")
                json.decodeFromString(EventReportResult.serializer(), resp.body?.string().orEmpty())
            }
        }
    }
}
