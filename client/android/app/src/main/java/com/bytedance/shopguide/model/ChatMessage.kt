package com.bytedance.shopguide.model

import java.util.UUID

/** Role of a chat message in the conversation. */
enum class Role { USER, ASSISTANT }

/**
 * A single message in the chat. Assistant messages are mutated in place as
 * streaming tokens arrive — see [ChatViewModel].
 */
data class ChatMessage(
    val id: String = UUID.randomUUID().toString(),
    val role: Role,
    val content: String = "",
    val isStreaming: Boolean = false,
    val isError: Boolean = false,
    val products: List<ProductCardDto> = emptyList(),
    val decisionTrace: DecisionTraceDto? = null,
    val agentUpdates: List<AgentUpdateState> = emptyList(),
    val traceLogs: List<TraceLog> = emptyList(),
    val attachedImageUri: String? = null,
)

/** User-visible progress text for one agent stage in the current answer. */
data class AgentUpdateState(
    val stage: String,
    val title: String,
    val content: String = "",
    val done: Boolean = false,
)

/** A small textual breadcrumb of the agent's intermediate stage. */
data class TraceLog(
    val stage: String,
    val content: String,
)
