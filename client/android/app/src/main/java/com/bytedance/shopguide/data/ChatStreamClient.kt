package com.bytedance.shopguide.data

import android.util.Log
import com.bytedance.shopguide.model.DecisionTraceDto
import com.bytedance.shopguide.model.ProductCardDto
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okio.BufferedSource
import java.io.IOException
import java.util.concurrent.TimeUnit

/**
 * Streams events from the backend `POST /api/chat/stream` SSE endpoint and
 * emits them as a typed Kotlin Flow.
 *
 * Why an in-house SSE parser? The backend uses `sse_starlette`, which writes
 * `event:` / `data:` line pairs separated by a blank line — i.e. a textbook
 * SSE stream. Adding a dedicated SSE library here would be overkill for the
 * handful of event types we care about, and OkHttp's streaming response body
 * already gives us everything we need.
 */
class ChatStreamClient(
    baseUrl: String,
    private val json: Json = ApiConfig.json,
    okHttp: OkHttpClient? = null,
) {
    private val endpoint = baseUrl.trimEnd('/') + "/api/chat/stream"

    // SSE responses are long-lived. We disable the read timeout so the client
    // doesn't kill the stream mid-answer when the LLM pauses between tokens.
    private val client: OkHttpClient = (okHttp ?: ApiConfig.sharedOkHttp).newBuilder()
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .build()

    /**
     * Connect to the SSE endpoint and emit each decoded event. Cancelling the
     * collecting coroutine cancels the underlying call.
     */
    fun stream(request: ChatStreamRequest): Flow<SseEvent> = flow {
        val body = json.encodeToString(ChatStreamRequest.serializer(), request)
            .toRequestBody(JSON_MEDIA_TYPE)

        val httpRequest = Request.Builder()
            .url(endpoint)
            .post(body)
            .header("Accept", "text/event-stream")
            .header("Cache-Control", "no-cache")
            .build()

        val call = client.newCall(httpRequest)
        val response = try {
            call.execute()
        } catch (e: IOException) {
            emit(SseEvent.Error("连接服务器失败：${e.message ?: "网络异常"}"))
            return@flow
        }

        response.use { resp ->
            if (!resp.isSuccessful) {
                emit(SseEvent.Error("服务器返回 ${resp.code}"))
                return@flow
            }
            val source = resp.body?.source()
                ?: run {
                    emit(SseEvent.Error("响应体为空"))
                    return@flow
                }
            try {
                parseStream(source) { event -> emit(event) }
            } catch (e: IOException) {
                // Most commonly: stream closed mid-flight or coroutine cancelled.
                if (!call.isCanceled()) {
                    emit(SseEvent.Error("流式连接中断：${e.message ?: "未知错误"}"))
                }
            } finally {
                if (!call.isCanceled()) call.cancel()
            }
        }
    }.flowOn(Dispatchers.IO)

    /**
     * Parse the raw SSE byte stream into [SseEvent]s. Each event ends with a
     * blank line per the SSE spec — until then we accumulate `event:` and
     * `data:` lines. The callback is `suspend` so it can forward straight to a
     * `FlowCollector.emit`.
     */
    private suspend fun parseStream(source: BufferedSource, emit: suspend (SseEvent) -> Unit) {
        var eventName: String? = null
        val dataBuffer = StringBuilder()

        suspend fun flushEvent() {
            if (dataBuffer.isEmpty() && eventName == null) return
            val ev = decode(eventName, dataBuffer.toString())
            if (ev != null) {
                logEvent(ev)
                emit(ev)
            }
            eventName = null
            dataBuffer.clear()
        }

        while (true) {
            val line = source.readUtf8Line()?.trimEnd('\r') ?: break

            if (line.isBlank()) {
                flushEvent()
                continue
            }
            if (line.startsWith(":")) continue // comment / keep-alive
            when {
                line.startsWith("event:") -> eventName = line.substring(6).trim()
                line.startsWith("data:") -> {
                    if (dataBuffer.isNotEmpty()) dataBuffer.append('\n')
                    dataBuffer.append(line.substring(5).trimStart())
                }
                else -> { /* ignore id:/retry:/unknown fields */ }
            }
        }

        // Flush any trailing event that wasn't terminated by a blank line.
        flushEvent()
    }

    private fun decode(eventName: String?, data: String): SseEvent? {
        val payload = data.trim()
        // The backend always puts a JSON object in `data:`. If it isn't valid
        // JSON we fall back to Unknown rather than crashing the stream.
        val obj = runCatching { json.parseToJsonElement(payload).jsonObject }
            .getOrElse { return SseEvent.Unknown(eventName ?: "message", payload) }

        val type = eventName?.takeIf { it.isNotBlank() }
            ?: obj["type"]?.jsonPrimitive?.contentOrNull
            ?: "message"

        return when (type) {
            "trace" -> SseEvent.Trace(
                stage = obj["stage"]?.jsonPrimitive?.contentOrNull.orEmpty(),
                content = obj["content"]?.jsonPrimitive?.contentOrNull.orEmpty(),
            )
            "decision_trace" -> {
                val traceObj = obj["trace"]?.jsonObject ?: return SseEvent.Unknown(type, payload)
                val trace = runCatching {
                    json.decodeFromJsonElement(DecisionTraceDto.serializer(), traceObj)
                }.getOrNull() ?: DecisionTraceDto()
                SseEvent.DecisionTrace(trace)
            }
            "token" -> SseEvent.Token(obj["content"]?.jsonPrimitive?.contentOrNull.orEmpty())
            "agent_update" -> SseEvent.AgentUpdate(
                stage = obj["stage"]?.jsonPrimitive?.contentOrNull.orEmpty(),
                title = obj["title"]?.jsonPrimitive?.contentOrNull.orEmpty(),
                contentDelta = obj["content_delta"]?.jsonPrimitive?.contentOrNull.orEmpty(),
                done = obj["done"]?.jsonPrimitive?.booleanOrNull ?: false,
            )
            "product_cards" -> {
                val products = obj["products"]?.jsonArray
                    ?.mapNotNull { el ->
                        runCatching {
                            json.decodeFromJsonElement(ProductCardDto.serializer(), el)
                        }.getOrNull()
                    } ?: emptyList()
                SseEvent.ProductCards(products)
            }
            "error" -> SseEvent.Error(
                obj["message"]?.jsonPrimitive?.contentOrNull
                    ?: obj["error"]?.jsonPrimitive?.contentOrNull
                    ?: "服务器返回错误事件"
            )
            "done" -> SseEvent.Done
            else -> SseEvent.Unknown(type, payload)
        }
    }

    private fun logEvent(event: SseEvent) {
        when (event) {
            is SseEvent.AgentUpdate -> Log.d(
                TAG,
                "event=agent_update stage=${event.stage} delta_len=${event.contentDelta.length} done=${event.done}",
            )
            is SseEvent.Token -> Log.d(TAG, "event=token len=${event.content.length}")
            is SseEvent.ProductCards -> Log.d(TAG, "event=product_cards count=${event.products.size}")
            is SseEvent.DecisionTrace -> Log.d(TAG, "event=decision_trace")
            is SseEvent.Trace -> Log.d(TAG, "event=trace stage=${event.stage}")
            is SseEvent.Done -> Log.d(TAG, "event=done")
            is SseEvent.Error -> Log.d(TAG, "event=error")
            is SseEvent.Unknown -> Log.d(TAG, "event=unknown type=${event.type}")
        }
    }

    companion object {
        private val JSON_MEDIA_TYPE = "application/json; charset=utf-8".toMediaType()
        private const val TAG = "ChatStreamClient"
    }
}
