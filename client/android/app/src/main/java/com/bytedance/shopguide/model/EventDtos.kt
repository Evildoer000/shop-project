package com.bytedance.shopguide.model

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject

@Serializable
data class EventReportRequestDto(
    @SerialName("user_id") val userId: String,
    @SerialName("session_id") val sessionId: String = "default",
    @SerialName("event_type") val eventType: String,
    @SerialName("product_id") val productId: String,
    @SerialName("turn_id") val turnId: Int? = null,
    val position: Int? = null,
    val context: JsonObject = buildJsonObject {},
)

@Serializable
data class EventReportResponseDto(
    val ok: Boolean = true,
    @SerialName("event_id") val eventId: Long = 0,
)
