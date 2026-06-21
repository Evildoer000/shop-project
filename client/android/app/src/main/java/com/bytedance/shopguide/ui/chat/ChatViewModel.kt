package com.bytedance.shopguide.ui.chat

import android.content.ContentResolver
import android.net.Uri
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.bytedance.shopguide.data.ApiConfig
import com.bytedance.shopguide.data.ChatStreamClient
import com.bytedance.shopguide.data.ChatStreamRequest
import com.bytedance.shopguide.data.EventApi
import com.bytedance.shopguide.data.ImageApiClient
import com.bytedance.shopguide.data.SseEvent
import com.bytedance.shopguide.model.AgentUpdateState
import com.bytedance.shopguide.model.ChatMessage
import com.bytedance.shopguide.model.Role
import com.bytedance.shopguide.model.TraceLog
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.util.UUID

/**
 * UI state for the chat screen.
 */
data class ChatUiState(
    val messages: List<ChatMessage> = emptyList(),
    val input: String = "",
    val isStreaming: Boolean = false,
    val errorMessage: String? = null,
    val sessionId: String = "android_session_" + UUID.randomUUID().toString().substring(0, 8),
    val userId: String = "android_user",
    val history: List<ChatSessionSummary> = emptyList(),
    val selectedImageUri: Uri? = null,
    val uploadingImage: Boolean = false,
    val uploadedImageId: String? = null,
    val uploadError: String? = null,
    val cartNotice: String? = null,
)

data class ChatSessionSummary(
    val sessionId: String,
    val title: String,
    val messages: List<ChatMessage>,
)

/**
 * Drives the chat screen. Owns the session id, accumulates streaming tokens,
 * appends product cards and decision traces to the in-flight assistant
 * message, and exposes a single [StateFlow] for the UI.
 */
class ChatViewModel(
    private val streamClient: ChatStreamClient = ChatStreamClient(ApiConfig.baseUrl),
    private val imageClient: ImageApiClient = ImageApiClient(ApiConfig.baseUrl),
    private val eventApi: EventApi = EventApi(ApiConfig.baseUrl),
) : ViewModel() {

    private val _state = MutableStateFlow(ChatUiState())
    val state: StateFlow<ChatUiState> = _state.asStateFlow()

    private var streamingJob: Job? = null
    private var uploadJob: Job? = null
    private val streamStartedAtByMessageId = mutableMapOf<String, Long>()
    private val firstVisibleTokenLogged = mutableSetOf<String>()
    private val lastVisibleDeltaRenderedAtByMessageId = mutableMapOf<String, Long>()

    fun onInputChanged(text: String) {
        _state.update { it.copy(input = text) }
    }

    fun newSession() {
        streamingJob?.cancel()
        uploadJob?.cancel()
        streamStartedAtByMessageId.clear()
        firstVisibleTokenLogged.clear()
        lastVisibleDeltaRenderedAtByMessageId.clear()
        _state.update {
            val archived = archiveCurrentSession(it)
            ChatUiState(
                sessionId = "android_session_" + UUID.randomUUID().toString().substring(0, 8),
                userId = it.userId,
                history = archived,
            )
        }
    }

    fun restoreSession(sessionId: String) {
        streamingJob?.cancel()
        uploadJob?.cancel()
        streamStartedAtByMessageId.clear()
        firstVisibleTokenLogged.clear()
        lastVisibleDeltaRenderedAtByMessageId.clear()
        _state.update { state ->
            val target = state.history.firstOrNull { it.sessionId == sessionId } ?: return@update state
            val archived = archiveCurrentSession(state).filterNot { it.sessionId == sessionId }
            state.copy(
                sessionId = target.sessionId,
                messages = target.messages,
                input = "",
                isStreaming = false,
                errorMessage = null,
                selectedImageUri = null,
                uploadingImage = false,
                uploadedImageId = null,
                uploadError = null,
                history = archived,
            )
        }
    }

    fun onImageSelected(uri: Uri, contentResolver: ContentResolver) {
        uploadJob?.cancel()
        _state.update {
            it.copy(
                selectedImageUri = uri,
                uploadingImage = true,
                uploadedImageId = null,
                uploadError = null,
            )
        }
        uploadJob = viewModelScope.launch {
            try {
                val response = imageClient.upload(contentResolver, uri)
                _state.update {
                    if (it.selectedImageUri != uri) it
                    else it.copy(uploadingImage = false, uploadedImageId = response.imageId, uploadError = null)
                }
            } catch (t: Throwable) {
                _state.update {
                    if (it.selectedImageUri != uri) it
                    else it.copy(uploadingImage = false, uploadedImageId = null, uploadError = t.message ?: "上传图片失败")
                }
            }
        }
    }

    fun clearSelectedImage() {
        uploadJob?.cancel()
        _state.update {
            it.copy(
                selectedImageUri = null,
                uploadingImage = false,
                uploadedImageId = null,
                uploadError = null,
            )
        }
    }

    fun addToCart(productId: String) {
        val current = _state.value
        viewModelScope.launch {
            val result = eventApi.addToCart(
                userId = current.userId,
                sessionId = current.sessionId,
                productId = productId,
                source = "chat_card",
            )
            _state.update {
                it.copy(
                    cartNotice = if (result.isSuccess) "已加入购物车" else "加入购物车失败",
                )
            }
        }
    }

    /**
     * 用户在 chat 里点商品卡片 (上报 click 事件, 影响商城推荐).
     * fire-and-forget: 失败不影响主链路.
     */
    fun reportProductClick(productId: String) {
        val current = _state.value
        // 找最近一条用户原话作为 chat 上下文 query
        val lastUserQuery = current.messages.lastOrNull { it.role.name == "USER" }?.content.orEmpty()
        viewModelScope.launch {
            eventApi.reportClick(
                userId = current.userId,
                sessionId = current.sessionId,
                productId = productId,
                source = "chat_card",
                query = lastUserQuery,
            )
        }
    }

    fun clearCartNotice() {
        _state.update { it.copy(cartNotice = null) }
    }

    fun sendMessage() {
        val current = _state.value
        val text = current.input.trim()
        val imageUri = current.selectedImageUri
        val imageId = current.uploadedImageId
        val hasImage = imageUri != null
        if ((!hasImage && text.isEmpty()) || current.isStreaming) return
        if (hasImage && current.uploadingImage) return
        if (hasImage && current.uploadError != null) {
            _state.update { it.copy(errorMessage = current.uploadError) }
            return
        }
        if (hasImage && imageId == null) {
            _state.update { it.copy(errorMessage = "图片还没有上传成功，请稍后重试") }
            return
        }

        val userMsg = ChatMessage(
            role = Role.USER,
            content = text,
            attachedImageUri = imageUri?.toString(),
        )
        val assistantMsg = ChatMessage(role = Role.ASSISTANT, isStreaming = true)

        _state.update {
            it.copy(
                messages = it.messages + userMsg + assistantMsg,
                input = "",
                isStreaming = true,
                errorMessage = null,
                selectedImageUri = null,
                uploadingImage = false,
                uploadedImageId = null,
                uploadError = null,
            )
        }

        streamingJob?.cancel()
        streamingJob = viewModelScope.launch {
            try {
                val request = ChatStreamRequest(
                    userId = current.userId,
                    sessionId = current.sessionId,
                    message = text,
                    imageId = imageId,
                )
                streamStartedAtByMessageId[assistantMsg.id] = System.currentTimeMillis()
                streamClient.stream(request).collect { event ->
                    handleEvent(assistantMsg.id, event)
                }
            } catch (t: Throwable) {
                applyError(assistantMsg.id, t.message ?: "未知错误")
            } finally {
                finalizeMessage(assistantMsg.id)
                _state.update {
                    it.copy(
                        selectedImageUri = null,
                        uploadingImage = false,
                        uploadedImageId = null,
                        uploadError = null,
                    )
                }
            }
        }
    }

    fun retryLast() {
        val last = _state.value.messages.lastOrNull { it.role == Role.USER } ?: return
        _state.update { it.copy(input = last.content) }
        sendMessage()
    }

    private suspend fun handleEvent(messageId: String, event: SseEvent) {
        when (event) {
            is SseEvent.Trace -> mutateMessage(messageId) { msg ->
                msg.copy(traceLogs = msg.traceLogs + TraceLog(event.stage, event.content))
            }
            is SseEvent.DecisionTrace -> mutateMessage(messageId) { msg ->
                msg.copy(decisionTrace = event.trace)
            }
            is SseEvent.AgentUpdate -> {
                recordFirstVisibleToken(messageId, event.contentDelta)
                paceVisibleDelta(messageId, event.contentDelta)
                mutateMessage(messageId) { msg ->
                    msg.copy(agentUpdates = msg.agentUpdates.withAgentUpdate(event))
                }
            }
            is SseEvent.Token -> {
                recordFirstVisibleToken(messageId, event.content)
                paceVisibleDelta(messageId, event.content)
                mutateMessage(messageId) { msg ->
                    msg.copy(content = msg.content + event.content)
                }
            }
            is SseEvent.ProductCards -> mutateMessage(messageId) { msg ->
                msg.copy(products = event.products)
            }
            is SseEvent.Done -> mutateMessage(messageId) { msg ->
                msg.copy(isStreaming = false)
            }
            is SseEvent.Error -> {
                applyError(messageId, event.message)
            }
            is SseEvent.Unknown -> { /* no-op: forward compat */ }
        }
    }

    private fun applyError(messageId: String, message: String) {
        mutateMessage(messageId) { msg ->
            msg.copy(
                isError = true,
                isStreaming = false,
                content = if (msg.content.isBlank()) "出错了：$message" else msg.content,
            )
        }
        _state.update { it.copy(errorMessage = message) }
    }

    private fun finalizeMessage(messageId: String) {
        mutateMessage(messageId) { msg -> msg.copy(isStreaming = false) }
        streamStartedAtByMessageId.remove(messageId)
        firstVisibleTokenLogged.remove(messageId)
        lastVisibleDeltaRenderedAtByMessageId.remove(messageId)
        _state.update { it.copy(isStreaming = false) }
    }

    private fun recordFirstVisibleToken(messageId: String, delta: String) {
        if (delta.isBlank() || messageId in firstVisibleTokenLogged) return
        val startedAt = streamStartedAtByMessageId[messageId] ?: return
        firstVisibleTokenLogged.add(messageId)
        val latencyMs = System.currentTimeMillis() - startedAt
        Log.i(TAG_FIRST_TOKEN, "message_id=$messageId first_visible_token_latency_ms=$latencyMs")
    }

    private suspend fun paceVisibleDelta(messageId: String, delta: String) {
        if (delta.isBlank()) return
        val now = System.currentTimeMillis()
        val lastRenderedAt = lastVisibleDeltaRenderedAtByMessageId[messageId]
        if (lastRenderedAt != null) {
            val waitMs = VISIBLE_DELTA_RENDER_INTERVAL_MS - (now - lastRenderedAt)
            if (waitMs > 0) delay(waitMs)
        }
        lastVisibleDeltaRenderedAtByMessageId[messageId] = System.currentTimeMillis()
    }

    private inline fun mutateMessage(id: String, transform: (ChatMessage) -> ChatMessage) {
        _state.update { state ->
            state.copy(
                messages = state.messages.map { if (it.id == id) transform(it) else it },
            )
        }
    }

    private fun archiveCurrentSession(state: ChatUiState): List<ChatSessionSummary> {
        if (state.messages.isEmpty()) return state.history
        val title = state.messages.firstOrNull { it.role == Role.USER }?.content
            ?.take(28)
            ?.ifBlank { "新会话" }
            ?: "新会话"
        val current = ChatSessionSummary(
            sessionId = state.sessionId,
            title = title,
            messages = state.messages,
        )
        return (listOf(current) + state.history.filterNot { it.sessionId == state.sessionId }).take(12)
    }

    companion object {
        private const val VISIBLE_DELTA_RENDER_INTERVAL_MS = 32L
        private const val TAG_FIRST_TOKEN = "FirstTokenTracker"
    }
}

private fun List<AgentUpdateState>.withAgentUpdate(event: SseEvent.AgentUpdate): List<AgentUpdateState> {
    val normalizedStage = event.stage.ifBlank { "agent" }
    val normalizedTitle = event.title.ifBlank { normalizedStage }
    val index = indexOfFirst { it.stage == normalizedStage }
    if (index < 0) {
        return this + AgentUpdateState(
            stage = normalizedStage,
            title = normalizedTitle,
            content = event.contentDelta,
            done = event.done,
        )
    }
    return mapIndexed { currentIndex, current ->
        if (currentIndex != index) current
        else current.copy(
            title = normalizedTitle,
            content = current.content + event.contentDelta,
            done = event.done || current.done,
        )
    }
}
