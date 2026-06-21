package com.bytedance.shopguide.model

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class CartDto(
    @SerialName("user_id") val userId: String,
    @SerialName("session_id") val sessionId: String,
    val items: List<CartItemDto> = emptyList(),
    @SerialName("total_quantity") val totalQuantity: Int = 0,
    @SerialName("total_price") val totalPrice: Double = 0.0,
)

@Serializable
data class CartItemDto(
    @SerialName("product_id") val productId: String,
    val name: String,
    val category: String,
    @SerialName("sub_category") val subCategory: String? = null,
    val brand: String,
    val price: Double,
    @SerialName("image_url") val imageUrl: String = "",
    val quantity: Int = 1,
    val rating: Double = 0.0,
    val sku: Map<String, String> = emptyMap(),   // 选中规格 (例: {"尺码": "40 码"})
)
