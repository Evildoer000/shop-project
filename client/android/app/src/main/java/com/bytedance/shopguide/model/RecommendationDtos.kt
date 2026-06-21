package com.bytedance.shopguide.model

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class RecommendationCardDto(
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
    val score: Double = 0.0,
)

@Serializable
data class RecommendationResponseDto(
    val products: List<RecommendationCardDto> = emptyList(),
    val stage: String = "cold",
    @SerialName("total_events") val totalEvents: Int = 0,
)

fun RecommendationCardDto.toProductCard(): ProductCardDto = ProductCardDto(
    productId = productId,
    name = name,
    category = category,
    subCategory = subCategory,
    brand = brand,
    price = price,
    imageUrl = imageUrl,
    tags = tags,
    rating = rating,
    reason = reason,
)
