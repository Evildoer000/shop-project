package com.bytedance.shopguide.model

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Mirror of the backend [app/schemas.py] `ProductCard` model. Returned in the SSE
 * `product_cards` event payload and rendered inline in chat messages.
 */
@Serializable
data class ProductCardDto(
    @SerialName("product_id") val productId: String,
    val name: String,
    val category: String,
    @SerialName("sub_category") val subCategory: String? = null,
    val brand: String,
    val price: Double,
    @SerialName("image_url") val imageUrl: String = "",
    val tags: List<String> = emptyList(),
    val rating: Double = 0.0,
    val reason: String = "",
)
