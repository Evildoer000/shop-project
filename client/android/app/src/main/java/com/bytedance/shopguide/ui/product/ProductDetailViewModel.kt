package com.bytedance.shopguide.ui.product

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.bytedance.shopguide.data.EventApi
import com.bytedance.shopguide.data.ProductApi
import com.bytedance.shopguide.model.ProductDetailDto
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

sealed class ProductDetailUiState {
    data object Loading : ProductDetailUiState()
    data class Success(val product: ProductDetailDto) : ProductDetailUiState()
    data class Error(val message: String) : ProductDetailUiState()
}

class ProductDetailViewModel(
    private val productId: String,
    private val api: ProductApi = ProductApi(),
    private val eventApi: EventApi = EventApi(),
) : ViewModel() {

    private val _state = MutableStateFlow<ProductDetailUiState>(ProductDetailUiState.Loading)
    val state: StateFlow<ProductDetailUiState> = _state.asStateFlow()
    private val _cartState = MutableStateFlow(CartActionState())
    val cartState: StateFlow<CartActionState> = _cartState.asStateFlow()

    /** 用户选中的规格. key=属性名 (尺码/款型), value=选中值 (40码/男款). */
    private val _selectedSku = MutableStateFlow<Map<String, String>>(emptyMap())
    val selectedSku: StateFlow<Map<String, String>> = _selectedSku.asStateFlow()

    fun selectSku(key: String, value: String) {
        _selectedSku.value = _selectedSku.value.toMutableMap().apply { put(key, value) }
    }

    init {
        load()
    }

    fun load() {
        _state.value = ProductDetailUiState.Loading
        viewModelScope.launch {
            api.getProduct(productId).fold(
                onSuccess = { _state.value = ProductDetailUiState.Success(it) },
                onFailure = { _state.value = ProductDetailUiState.Error(it.message ?: "未知错误") },
            )
        }
    }

    fun addToCart() {
        if (_cartState.value.loading || _cartState.value.added) return
        _cartState.value = _cartState.value.copy(loading = true)
        viewModelScope.launch {
            val result = eventApi.addToCart(
                userId = "android_user",
                sessionId = "product_detail",
                productId = productId,
                source = "product_detail",
                sku = _selectedSku.value,
            )
            _cartState.value = CartActionState(
                loading = false,
                added = result.isSuccess,
                notice = if (result.isSuccess) "已加入购物车" else "加入购物车失败",
            )
        }
    }

    fun clearCartNotice() {
        _cartState.value = _cartState.value.copy(notice = null)
    }
}

data class CartActionState(
    val loading: Boolean = false,
    val added: Boolean = false,
    val notice: String? = null,
)
