package com.bytedance.shopguide.ui.cart

import android.widget.Toast
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.ui.draw.clip
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Remove
import androidx.compose.material.icons.filled.ShoppingCart
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.ViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import com.bytedance.shopguide.data.CartApi
import com.bytedance.shopguide.data.EventApi
import com.bytedance.shopguide.model.CartDto
import com.bytedance.shopguide.model.CartItemDto
import com.bytedance.shopguide.ui.chat.ProductImage
import com.bytedance.shopguide.ui.chat.formatPrice
import com.bytedance.shopguide.ui.theme.MutedInk
import com.bytedance.shopguide.ui.theme.PriceRed
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CartScreen(
    onBack: () -> Unit,
    onProductClick: (String) -> Unit,
    viewModel: CartViewModel = viewModel(),
) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val notice by viewModel.notice.collectAsStateWithLifecycle()
    val context = androidx.compose.ui.platform.LocalContext.current

    LaunchedEffect(Unit) {
        viewModel.load()
    }
    LaunchedEffect(notice) {
        notice?.let {
            Toast.makeText(context, it, Toast.LENGTH_SHORT).show()
            viewModel.clearNotice()
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("购物车") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "返回")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.background,
                ),
            )
        },
        bottomBar = {
            val cart = (state as? CartUiState.Success)?.cart
            if (cart != null && cart.items.isNotEmpty()) {
                CartTotalBar(cart)
            }
        },
        containerColor = MaterialTheme.colorScheme.background,
    ) { padding ->
        Box(modifier = Modifier.fillMaxSize().padding(padding)) {
            when (val current = state) {
                CartUiState.Loading -> LoadingCart()
                is CartUiState.Error -> ErrorCart(current.message, onRetry = viewModel::load)
                is CartUiState.Success -> CartBody(
                    cart = current.cart,
                    removingProductId = current.removingProductId,
                    onProductClick = onProductClick,
                    onRemoveOne = { pid, sku -> viewModel.removeOne(pid, sku) },
                    onAddOne = { pid, sku -> viewModel.addOne(pid, sku) },
                )
            }
        }
    }
}

@Composable
private fun LoadingCart() {
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        CircularProgressIndicator()
    }
}

@Composable
private fun ErrorCart(message: String, onRetry: () -> Unit) {
    Column(
        modifier = Modifier.fillMaxSize().padding(24.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text("购物车加载失败", style = MaterialTheme.typography.titleMedium)
        Spacer(Modifier.height(6.dp))
        Text(message, style = MaterialTheme.typography.bodySmall, color = MutedInk)
        Spacer(Modifier.height(14.dp))
        Button(onClick = onRetry, shape = RoundedCornerShape(8.dp)) {
            Text("重试")
        }
    }
}

@Composable
private fun CartBody(
    cart: CartDto,
    removingProductId: String?,
    onProductClick: (String) -> Unit,
    onRemoveOne: (String, Map<String, String>) -> Unit,
    onAddOne: (String, Map<String, String>) -> Unit,
) {
    if (cart.items.isEmpty()) {
        EmptyCart()
        return
    }
    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(horizontal = 14.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        // 同一 productId 可能因为 SKU 不同而出现多次, key 必须组合 productId + sku
        items(cart.items, key = { "${it.productId}|${it.sku.entries.sortedBy { e -> e.key }.joinToString(",") { e -> "${e.key}=${e.value}" }}" }) { item ->
            CartItemCard(
                item = item,
                busy = item.productId == removingProductId,
                onClick = { onProductClick(item.productId) },
                onRemoveOne = { onRemoveOne(item.productId, item.sku) },
                onAddOne = { onAddOne(item.productId, item.sku) },
            )
        }
    }
}

@Composable
private fun EmptyCart() {
    Column(
        modifier = Modifier.fillMaxSize().padding(28.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Surface(
            color = MaterialTheme.colorScheme.surfaceVariant,
            shape = RoundedCornerShape(8.dp),
            modifier = Modifier.size(58.dp),
        ) {
            Box(contentAlignment = Alignment.Center) {
                Icon(Icons.Filled.ShoppingCart, contentDescription = null, tint = MutedInk)
            }
        }
        Spacer(Modifier.height(16.dp))
        Text("购物车还是空的", style = MaterialTheme.typography.titleMedium)
        Spacer(Modifier.height(6.dp))
        Text("在推荐商品卡或商品详情页点击加购后，会出现在这里。", style = MaterialTheme.typography.bodyMedium, color = MutedInk)
    }
}

@Composable
private fun CartItemCard(
    item: CartItemDto,
    busy: Boolean,
    onClick: () -> Unit,
    onRemoveOne: () -> Unit,
    onAddOne: () -> Unit,
) {
    Card(
        modifier = Modifier.fillMaxWidth().clickable(onClick = onClick),
        shape = RoundedCornerShape(8.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.55f)),
        elevation = CardDefaults.cardElevation(defaultElevation = 1.dp),
    ) {
        Row(modifier = Modifier.padding(12.dp), verticalAlignment = Alignment.Top) {
            ProductImage(
                imageUrl = item.imageUrl,
                modifier = Modifier.size(82.dp),
            )
            Spacer(Modifier.width(12.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = item.name,
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.SemiBold,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis,
                )
                Spacer(Modifier.height(4.dp))
                Text(
                    text = listOfNotNull(item.brand.takeIf { it.isNotBlank() }, item.category, item.subCategory)
                        .joinToString(" · "),
                    style = MaterialTheme.typography.bodySmall,
                    color = MutedInk,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
                if (item.sku.isNotEmpty()) {
                    Spacer(Modifier.height(6.dp))
                    Text(
                        text = item.sku.entries.joinToString(" · ") { "${it.key} ${it.value}" },
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.primary,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
                Spacer(Modifier.height(10.dp))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween,
                ) {
                    Text(
                        text = "¥${formatPrice(item.price)}",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold,
                        color = PriceRed,
                    )
                    QuantityStepper(
                        quantity = item.quantity,
                        busy = busy,
                        onMinus = onRemoveOne,
                        onPlus = onAddOne,
                    )
                }
            }
        }
    }
}

/**
 * 数量调整器: [-]  3  [+]
 * 加号 / 减号都是 32dp 圆形按钮, 中间显示当前数量.
 * busy 状态时全部 disable + 中间显示 loading.
 */
@Composable
private fun QuantityStepper(
    quantity: Int,
    busy: Boolean,
    onMinus: () -> Unit,
    onPlus: () -> Unit,
) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .clip(RoundedCornerShape(20.dp))
            .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f))
            .padding(horizontal = 4.dp, vertical = 4.dp),
    ) {
        StepperButton(
            icon = Icons.Filled.Remove,
            description = "减少一件",
            enabled = !busy,
            onClick = onMinus,
        )
        Box(
            modifier = Modifier.widthIn(min = 36.dp).padding(horizontal = 6.dp),
            contentAlignment = Alignment.Center,
        ) {
            if (busy) {
                CircularProgressIndicator(
                    modifier = Modifier.size(14.dp),
                    strokeWidth = 1.5.dp,
                )
            } else {
                Text(
                    text = quantity.toString(),
                    style = MaterialTheme.typography.bodyLarge,
                    fontWeight = FontWeight.SemiBold,
                )
            }
        }
        StepperButton(
            icon = Icons.Filled.Add,
            description = "增加一件",
            enabled = !busy,
            onClick = onPlus,
        )
    }
}

@Composable
private fun StepperButton(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    description: String,
    enabled: Boolean,
    onClick: () -> Unit,
) {
    androidx.compose.material3.Surface(
        onClick = onClick,
        enabled = enabled,
        shape = androidx.compose.foundation.shape.CircleShape,
        color = MaterialTheme.colorScheme.surface,
        modifier = Modifier.size(28.dp),
    ) {
        Box(contentAlignment = Alignment.Center) {
            Icon(
                imageVector = icon,
                contentDescription = description,
                modifier = Modifier.size(16.dp),
                tint = if (enabled) MaterialTheme.colorScheme.onSurface
                       else MaterialTheme.colorScheme.outline,
            )
        }
    }
}

@Composable
private fun CartTotalBar(cart: CartDto) {
    Surface(color = MaterialTheme.colorScheme.surface, tonalElevation = 3.dp) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 10.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text("共 ${cart.totalQuantity} 件", style = MaterialTheme.typography.bodySmall, color = MutedInk)
                Text(
                    text = "¥${formatPrice(cart.totalPrice)}",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                    color = PriceRed,
                )
            }
            Button(
                onClick = {},
                enabled = false,
                shape = RoundedCornerShape(8.dp),
                colors = ButtonDefaults.buttonColors(disabledContainerColor = MaterialTheme.colorScheme.surfaceVariant),
            ) {
                Text("下单演示")
            }
        }
    }
}

class CartViewModel(
    private val cartApi: CartApi = CartApi(),
    private val eventApi: EventApi = EventApi(),
) : ViewModel() {
    private val _state = MutableStateFlow<CartUiState>(CartUiState.Loading)
    val state: StateFlow<CartUiState> = _state.asStateFlow()
    private val _notice = MutableStateFlow<String?>(null)
    val notice: StateFlow<String?> = _notice.asStateFlow()

    fun load() {
        _state.value = CartUiState.Loading
        viewModelScope.launch {
            cartApi.getCart(USER_ID, "all").fold(
                onSuccess = { _state.value = CartUiState.Success(it) },
                onFailure = { _state.value = CartUiState.Error(it.message ?: "未知错误") },
            )
        }
    }

    fun removeOne(productId: String, sku: Map<String, String>) {
        val current = _state.value as? CartUiState.Success ?: return
        _state.value = current.copy(removingProductId = productId)
        viewModelScope.launch {
            val result = eventApi.removeFromCart(USER_ID, "cart", productId, "cart_screen", sku)
            if (result.isFailure) {
                _notice.value = "移除失败"
                _state.value = current.copy(removingProductId = null)
            } else {
                _notice.value = "已减少一件"
                load()
            }
        }
    }

    fun addOne(productId: String, sku: Map<String, String>) {
        val current = _state.value as? CartUiState.Success ?: return
        _state.value = current.copy(removingProductId = productId)
        viewModelScope.launch {
            val result = eventApi.addToCart(USER_ID, "cart", productId, "cart_screen", sku)
            if (result.isFailure) {
                _notice.value = "添加失败"
                _state.value = current.copy(removingProductId = null)
            } else {
                _notice.value = "已添加一件"
                load()
            }
        }
    }

    fun clearNotice() {
        _notice.value = null
    }

    companion object {
        private const val USER_ID = "android_user"
    }
}

sealed class CartUiState {
    data object Loading : CartUiState()
    data class Success(val cart: CartDto, val removingProductId: String? = null) : CartUiState()
    data class Error(val message: String) : CartUiState()
}
