package com.bytedance.shopguide.data

import com.bytedance.shopguide.model.DecisionTraceDto
import com.bytedance.shopguide.model.ProductCardDto

/**
 * A single decoded event from the `/api/chat/stream` SSE stream. The
 * orchestrator emits a fixed set of `type` values (see backend
 * `EcommerceOrchestrator.stream`), which we model here as a sealed class.
 *
 * Unknown event types are surfaced as [Unknown] so we never crash on an
 * event added on the server before the client knows about it.
 */
sealed class SseEvent {
    /** Intermediate breadcrumb like "已标准化用户输入". */
    data class Trace(val stage: String, val content: String) : SseEvent()

    /** Final structured trace shown in the decision panel. */
    data class DecisionTrace(val trace: DecisionTraceDto) : SseEvent()

    /** A chunk of the streaming natural-language answer. */
    data class Token(val content: String) : SseEvent()

    /** User-visible progress from an agent stage before the final answer. */
    data class AgentUpdate(
        val stage: String,
        val title: String,
        val contentDelta: String,
        val done: Boolean,
    ) : SseEvent()

    /** Final list of product cards attached to the answer. */
    data class ProductCards(val products: List<ProductCardDto>) : SseEvent()

    /** Sentinel emitted at the end of the stream. */
    data object Done : SseEvent()

    /** Backend-reported error. */
    data class Error(val message: String) : SseEvent()

    /** Forward-compat fallback. */
    data class Unknown(val type: String, val raw: String) : SseEvent()
}
