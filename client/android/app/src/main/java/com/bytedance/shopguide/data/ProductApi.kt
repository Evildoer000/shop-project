package com.bytedance.shopguide.data

import com.bytedance.shopguide.model.ProductDetailDto
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import okhttp3.OkHttpClient
import okhttp3.Request

/**
 * Small REST client for the non-streaming endpoints. Currently only
 * `GET /api/products/{product_id}` is needed (used by the detail screen).
 */
class ProductApi(
    baseUrl: String = ApiConfig.baseUrl,
    private val client: OkHttpClient = ApiConfig.sharedOkHttp,
    private val json: Json = ApiConfig.json,
) {
    private val productsPrefix: String = baseUrl.trimEnd('/') + "/api/products/"

    suspend fun getProduct(productId: String): Result<ProductDetailDto> = withContext(Dispatchers.IO) {
        runCatching {
            val request = Request.Builder()
                .url(productsPrefix + productId)
                .get()
                .build()
            client.newCall(request).execute().use { resp ->
                if (!resp.isSuccessful) error("HTTP ${resp.code}")
                val body = resp.body?.string().orEmpty()
                json.decodeFromString(ProductDetailDto.serializer(), body)
            }
        }
    }
}
