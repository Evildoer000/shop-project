package com.bytedance.shopguide.ui.mall

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.bytedance.shopguide.data.ApiConfig
import com.bytedance.shopguide.data.BehaviorApi
import com.bytedance.shopguide.model.EventReportRequestDto
import com.bytedance.shopguide.model.ProductCardDto
import com.bytedance.shopguide.model.toProductCard
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject

data class MallUiState(
    val products: List<ProductCardDto> = emptyList(),
    val stage: String = "",
    val totalEvents: Int = 0,
    val loading: Boolean = false,
    val error: String? = null,
    val userId: String = "android_user",
)

class MallViewModel(
    private val behaviorApi: BehaviorApi = BehaviorApi(ApiConfig.baseUrl),
) : ViewModel() {
    private val _state = MutableStateFlow(MallUiState())
    val state: StateFlow<MallUiState> = _state.asStateFlow()
    private val impressedProductIds = mutableSetOf<String>()

    fun load() {
        if (_state.value.loading) return
        _state.update { it.copy(loading = true, error = null) }
        viewModelScope.launch {
            behaviorApi.fetchRecommendations(_state.value.userId, size = 24).fold(
                onSuccess = { response ->
                    val cards = response.products.map { it.toProductCard() }
                    _state.update {
                        it.copy(
                            products = cards,
                            stage = response.stage,
                            totalEvents = response.totalEvents,
                            loading = false,
                            error = null,
                        )
                    }
                    reportImpressions(cards)
                },
                onFailure = { error ->
                    _state.update {
                        it.copy(loading = false, error = error.message ?: "加载推荐失败")
                    }
                },
            )
        }
    }

    fun reportProductOpen(productId: String, position: Int) {
        report(productId, position, "click")
        report(productId, position, "detail_view")
    }

    private fun reportImpressions(products: List<ProductCardDto>) {
        products.take(12).forEachIndexed { index, product ->
            if (impressedProductIds.add(product.productId)) {
                report(product.productId, index, "impression")
            }
        }
    }

    private fun report(productId: String, position: Int, eventType: String) {
        val state = _state.value
        val product = state.products.firstOrNull { it.productId == productId }
        val context = buildJsonObject {
            put("from", JsonPrimitive("mall"))
            put("page", JsonPrimitive("home"))
            put("stage", JsonPrimitive(state.stage))
            product?.let {
                put("brand", JsonPrimitive(it.brand))
                put("category", JsonPrimitive(it.category))
            }
        }
        viewModelScope.launch {
            behaviorApi.reportEvent(
                EventReportRequestDto(
                    userId = state.userId,
                    sessionId = "mall_session",
                    eventType = eventType,
                    productId = productId,
                    position = position,
                    context = context,
                ),
            )
        }
    }
}
