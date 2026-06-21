package com.bytedance.shopguide.data

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Mirrors the backend `ChatStreamRequest` body for `POST /api/chat/stream`.
 */
@Serializable
data class ChatStreamRequest(
    @SerialName("user_id") val userId: String,
    @SerialName("session_id") val sessionId: String,
    val message: String,
    @SerialName("image_id") val imageId: String? = null,
)
