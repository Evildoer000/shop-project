package com.bytedance.shopguide.model

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class ImageUploadResponse(
    @SerialName("image_id") val imageId: String,
    @SerialName("image_url") val imageUrl: String,
    val bytes: Int = 0,
)
