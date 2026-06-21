package com.bytedance.shopguide.data

fun resolveProductImageUrl(imageUrl: String, baseUrl: String = ApiConfig.baseUrl): String {
    val trimmed = imageUrl.trim()
    if (trimmed.isBlank()) return ""
    if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) return trimmed
    val cleanBase = baseUrl.trimEnd('/')
    val cleanPath = trimmed.trimStart('/')
    return "$cleanBase/$cleanPath"
}
