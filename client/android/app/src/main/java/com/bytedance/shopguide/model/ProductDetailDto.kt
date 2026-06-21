package com.bytedance.shopguide.model

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull

/**
 * Mirror of the backend [app/schemas.py] `ProductResponse` model, returned by
 * `GET /api/products/{product_id}`.
 *
 * `specs` and `structured_attributes` are kept as raw JSON because the backend
 * does not commit to a fixed shape across categories.
 */
@Serializable
data class ProductDetailDto(
    @SerialName("product_id") val productId: String,
    val name: String,
    val category: String,
    @SerialName("sub_category") val subCategory: String? = null,
    val brand: String,
    val price: Double,
    val stock: Int? = null,
    @SerialName("image_url") val imageUrl: String = "",
    val description: String = "",
    val specs: JsonObject = JsonObject(emptyMap()),
    @SerialName("ingredients_or_material") val ingredientsOrMaterial: String = "",
    @SerialName("suitable_for") val suitableFor: String = "",
    @SerialName("avoid_for") val avoidFor: String = "",
    val tags: List<String> = emptyList(),
    val rating: Double = 0.0,
    val sales: Int? = null,
    @SerialName("review_summary") val reviewSummary: String = "",
    @SerialName("image_caption") val imageCaption: String = "",
    @SerialName("structured_attributes") val structuredAttributes: JsonObject = JsonObject(emptyMap()),
) {
    fun skuOptionPairs(): List<Pair<String, String>> {
        val skus = specs["skus"] as? JsonArray ?: return emptyList()
        val options = linkedMapOf<String, LinkedHashSet<String>>()
        skus.forEach { sku ->
            val properties = (sku as? JsonObject)?.get("properties") as? JsonObject ?: return@forEach
            properties.forEach { (key, value) ->
                val display = jsonPrimitiveDisplay(value) ?: return@forEach
                options.getOrPut(key) { linkedSetOf() }.add(display)
            }
        }
        return options.mapNotNull { (key, values) ->
            val value = values.joinToString("、").takeIf { it.isNotBlank() } ?: return@mapNotNull null
            key to value
        }
    }

    /** 跟 skuOptionPairs 同源数据, 但每属性返回选项 list (供 UI 渲染 chip 选择). */
    fun skuOptionGroups(): List<Pair<String, List<String>>> {
        val skus = specs["skus"] as? JsonArray ?: return emptyList()
        val options = linkedMapOf<String, LinkedHashSet<String>>()
        skus.forEach { sku ->
            val properties = (sku as? JsonObject)?.get("properties") as? JsonObject ?: return@forEach
            properties.forEach { (key, value) ->
                val display = jsonPrimitiveDisplay(value) ?: return@forEach
                options.getOrPut(key) { linkedSetOf() }.add(display)
            }
        }
        return options.mapNotNull { (key, values) ->
            if (values.isEmpty()) return@mapNotNull null
            key to values.toList()
        }
    }

    fun displaySpecsPairs(): List<Pair<String, String>> =
        flattenJson(specs, hiddenKeys = setOf("skus", "source_json_path", "source_image_path"))

    fun displayStructuredAttributesPairs(): List<Pair<String, String>> =
        flattenJson(
            structuredAttributes,
            hiddenKeys = setOf("source", "category", "base_price", "image_path", "official_faq"),
        )

    private fun flattenJson(obj: JsonObject, hiddenKeys: Set<String>): List<Pair<String, String>> =
        obj.entries.mapNotNull { (key, value) ->
            if (key in hiddenKeys) return@mapNotNull null
            val display = jsonPrimitiveDisplay(value) ?: return@mapNotNull null
            if (display.length > 80) return@mapNotNull null
            key to display
        }

    private fun jsonPrimitiveDisplay(element: JsonElement): String? {
        val primitive = element as? JsonPrimitive ?: return null
        return primitive.contentOrNull?.trim()?.takeIf { it.isNotBlank() }
    }
}
