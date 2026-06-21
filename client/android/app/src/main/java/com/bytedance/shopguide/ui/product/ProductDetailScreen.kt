package com.bytedance.shopguide.ui.product

import android.widget.Toast
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.ShoppingCart
import androidx.compose.material.icons.filled.Star
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
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
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.bytedance.shopguide.R
import com.bytedance.shopguide.model.ProductDetailDto
import com.bytedance.shopguide.ui.chat.ProductImage
import com.bytedance.shopguide.ui.chat.formatPrice

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ProductDetailScreen(
    productId: String,
    onBack: () -> Unit,
) {
    val viewModel: ProductDetailViewModel = viewModel(
        key = "product_$productId",
        factory = viewModelFactory {
            initializer { ProductDetailViewModel(productId) }
        },
    )
    val state by viewModel.state.collectAsStateWithLifecycle()
    val cartState by viewModel.cartState.collectAsStateWithLifecycle()
    val selectedSku by viewModel.selectedSku.collectAsStateWithLifecycle()
    val context = LocalContext.current

    LaunchedEffect(cartState.notice) {
        cartState.notice?.let {
            Toast.makeText(context, it, Toast.LENGTH_SHORT).show()
            viewModel.clearCartNotice()
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.product_detail_title)) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(
                            Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = stringResource(R.string.cd_back),
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.background,
                ),
            )
        },
        bottomBar = {
            if (state is ProductDetailUiState.Success) {
                AddToCartBar(
                    loading = cartState.loading,
                    added = cartState.added,
                    onAddToCart = viewModel::addToCart,
                )
            }
        },
        containerColor = MaterialTheme.colorScheme.background,
    ) { padding ->
        Box(modifier = Modifier.fillMaxSize().padding(padding)) {
            when (val s = state) {
                is ProductDetailUiState.Loading -> LoadingView()
                is ProductDetailUiState.Error -> ErrorView(s.message) { viewModel.load() }
                is ProductDetailUiState.Success -> DetailBody(
                    product = s.product,
                    selectedSku = selectedSku,
                    onSelectSku = viewModel::selectSku,
                )
            }
        }
    }
}

@Composable
private fun LoadingView() {
    Column(
        modifier = Modifier.fillMaxSize(),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        CircularProgressIndicator()
        Spacer(Modifier.height(8.dp))
        Text(stringResource(R.string.product_detail_loading))
    }
}

@Composable
private fun ErrorView(message: String, onRetry: () -> Unit) {
    Column(
        modifier = Modifier.fillMaxSize().padding(24.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(stringResource(R.string.product_detail_error))
        Spacer(Modifier.height(4.dp))
        Text(
            text = message,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f),
        )
        Spacer(Modifier.height(12.dp))
        Button(onClick = onRetry) {
            Text(stringResource(R.string.product_detail_retry))
        }
    }
}

@Composable
private fun DetailBody(
    product: ProductDetailDto,
    selectedSku: Map<String, String>,
    onSelectSku: (String, String) -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState()),
    ) {
        ProductImage(
            imageUrl = product.imageUrl,
            modifier = Modifier
                .fillMaxWidth()
                .aspectRatio(1.1f)
                .background(MaterialTheme.colorScheme.surfaceVariant),
        )
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = product.name,
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.SemiBold,
            )
            Spacer(Modifier.height(6.dp))
            Text(
                text = listOfNotNull(
                    product.brand.takeIf { it.isNotBlank() },
                    product.category,
                    product.subCategory,
                ).joinToString(" · "),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.65f),
            )
            Spacer(Modifier.height(12.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    text = "¥${formatPrice(product.price)}",
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.primary,
                )
                Spacer(modifier = Modifier.padding(end = 12.dp))
                if (product.rating > 0) {
                    Icon(
                        Icons.Filled.Star,
                        contentDescription = null,
                        tint = Color(0xFFFFB300),
                    )
                    Spacer(modifier = Modifier.padding(end = 4.dp))
                    Text(
                        text = String.format("%.1f", product.rating),
                        style = MaterialTheme.typography.bodyMedium,
                    )
                }
                if (product.sales != null && product.sales > 0) {
                    Spacer(modifier = Modifier.padding(end = 12.dp))
                    Text(
                        text = "销量 ${product.sales}",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.65f),
                    )
                }
            }
            if (product.tags.isNotEmpty()) {
                Spacer(Modifier.height(12.dp))
                TagRow(product.tags)
            }
            Spacer(Modifier.height(20.dp))

            DescriptionSection(stringResource(R.string.product_detail_description), product.description)
            ReviewSummarySection(stringResource(R.string.product_detail_review), product.reviewSummary)
            SectionRow(stringResource(R.string.product_detail_suitable), product.suitableFor)
            SectionRow(stringResource(R.string.product_detail_avoid), product.avoidFor)
            SectionRow(stringResource(R.string.product_detail_ingredients), product.ingredientsOrMaterial)

            val skuGroups = product.skuOptionGroups()
            if (skuGroups.isNotEmpty()) {
                Section(title = "可选规格") {
                    SkuSelector(
                        groups = skuGroups,
                        selected = selectedSku,
                        onSelect = onSelectSku,
                    )
                }
            }

            val specs = product.displaySpecsPairs().filter { it.second.isNotBlank() }
            if (specs.isNotEmpty()) {
                Section(title = "规格") {
                    SpecList(specs)
                }
            }

            val attrs = product.displayStructuredAttributesPairs().filter { it.second.isNotBlank() }
            if (attrs.isNotEmpty()) {
                Section(title = "更多属性") {
                    SpecList(attrs)
                }
            }
        }
    }
}

@Composable
private fun AddToCartBar(
    loading: Boolean,
    added: Boolean,
    onAddToCart: () -> Unit,
) {
    Surface(
        color = MaterialTheme.colorScheme.surface,
        tonalElevation = 3.dp,
    ) {
        Button(
            onClick = onAddToCart,
            enabled = !loading && !added,
            shape = RoundedCornerShape(8.dp),
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 10.dp),
            contentPadding = PaddingValues(vertical = 13.dp),
            colors = if (added) {
                ButtonDefaults.buttonColors(
                    containerColor = MaterialTheme.colorScheme.tertiaryContainer,
                    contentColor = MaterialTheme.colorScheme.onTertiaryContainer,
                    disabledContainerColor = MaterialTheme.colorScheme.tertiaryContainer,
                    disabledContentColor = MaterialTheme.colorScheme.onTertiaryContainer,
                )
            } else {
                ButtonDefaults.buttonColors()
            },
        ) {
            when {
                loading -> CircularProgressIndicator(
                    modifier = Modifier.size(18.dp),
                    strokeWidth = 2.dp,
                    color = MaterialTheme.colorScheme.onPrimary,
                )
                added -> Icon(Icons.Filled.Check, contentDescription = null, modifier = Modifier.size(18.dp))
                else -> Icon(Icons.Filled.ShoppingCart, contentDescription = null, modifier = Modifier.size(18.dp))
            }
            Spacer(Modifier.width(8.dp))
            Text(
                when {
                    loading -> "正在加入"
                    added -> "已加入购物车"
                    else -> "加入购物车"
                }
            )
        }
    }
}

@Composable
private fun SectionRow(title: String, value: String) {
    if (value.isBlank()) return
    Section(title = title) {
        Text(value, style = MaterialTheme.typography.bodyMedium)
    }
}

/** 商品描述: 浅灰圆角卡片包裹, 行高放松, 字体偏阅读体验. */
@Composable
private fun DescriptionSection(title: String, value: String) {
    if (value.isBlank()) return
    Section(title = title) {
        androidx.compose.material3.Surface(
            shape = RoundedCornerShape(12.dp),
            color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.4f),
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text(
                text = value,
                modifier = Modifier.padding(14.dp),
                style = MaterialTheme.typography.bodyLarge.copy(
                    lineHeight = 26.sp,
                ),
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.85f),
            )
        }
    }
}

/** 用户评价: 解析 "昵称(N/5)：xxx" 格式, 每条评价独立卡片 + 星星评分. */
@Composable
private fun ReviewSummarySection(title: String, value: String) {
    if (value.isBlank()) return
    val reviews = remember(value) { parseReviews(value) }
    Section(title = title) {
        if (reviews.isEmpty()) {
            // 解析失败兜底, 退化成普通 text
            Text(value, style = MaterialTheme.typography.bodyMedium)
            return@Section
        }
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            reviews.forEach { ReviewCard(it) }
        }
    }
}

@Composable
private fun ReviewCard(review: Review) {
    val ratingColor = when {
        review.rating >= 4 -> Color(0xFF137333)   // 绿: 好评
        review.rating >= 3 -> Color(0xFFB58105)   // 橙: 中评
        else -> Color(0xFFC53929)                  // 红: 差评
    }
    androidx.compose.material3.Surface(
        shape = RoundedCornerShape(12.dp),
        color = MaterialTheme.colorScheme.surface,
        tonalElevation = 1.dp,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            // 头部: 头像 + 昵称 + 评分
            Row(verticalAlignment = Alignment.CenterVertically) {
                // 圆形头像 (用昵称首字)
                Box(
                    modifier = Modifier
                        .size(32.dp)
                        .clip(CircleShape)
                        .background(MaterialTheme.colorScheme.primaryContainer),
                    contentAlignment = Alignment.Center,
                ) {
                    Text(
                        text = review.author.take(1),
                        style = MaterialTheme.typography.labelLarge,
                        color = MaterialTheme.colorScheme.onPrimaryContainer,
                        fontWeight = FontWeight.SemiBold,
                    )
                }
                Spacer(Modifier.width(10.dp))
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = review.author,
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.SemiBold,
                    )
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        repeat(5) { idx ->
                            Icon(
                                imageVector = Icons.Filled.Star,
                                contentDescription = null,
                                tint = if (idx < review.rating) ratingColor else MaterialTheme.colorScheme.outlineVariant,
                                modifier = Modifier.size(14.dp),
                            )
                        }
                        Spacer(Modifier.width(6.dp))
                        Text(
                            text = "${review.rating}/5",
                            style = MaterialTheme.typography.labelMedium,
                            color = ratingColor,
                            fontWeight = FontWeight.SemiBold,
                        )
                    }
                }
            }
            Spacer(Modifier.height(8.dp))
            HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.3f))
            Spacer(Modifier.height(8.dp))
            // 评价正文
            Text(
                text = review.content,
                style = MaterialTheme.typography.bodyMedium.copy(lineHeight = 22.sp),
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.85f),
            )
        }
    }
}

private data class Review(
    val author: String,
    val rating: Int,
    val content: String,
)

/**
 * 解析后端的评价摘要文本. 典型格式:
 *   阿凯(5/5)：内容...
 *   阿远(3/5)：内容...
 *
 * 也兼容半角冒号 + 全角斜杠等变体.
 */
private fun parseReviews(text: String): List<Review> {
    if (text.isBlank()) return emptyList()
    // 匹配 "昵称(N/5)：" 或 "昵称(N/5):"
    val pattern = Regex("([^()\\n]{1,20})\\((\\d)\\s*[/／]\\s*5\\)\\s*[：:]\\s*")
    val matches = pattern.findAll(text).toList()
    if (matches.isEmpty()) return emptyList()

    val results = mutableListOf<Review>()
    for ((i, m) in matches.withIndex()) {
        val author = m.groupValues[1].trim()
        val rating = m.groupValues[2].toIntOrNull()?.coerceIn(0, 5) ?: 0
        val contentStart = m.range.last + 1
        val contentEnd = if (i + 1 < matches.size) matches[i + 1].range.first else text.length
        val content = text.substring(contentStart, contentEnd).trim().trimEnd('\n')
        if (content.isNotEmpty()) {
            results.add(Review(author, rating, content))
        }
    }
    return results
}

@Composable
private fun Section(title: String, content: @Composable () -> Unit) {
    Column(modifier = Modifier.padding(vertical = 6.dp)) {
        Text(
            text = title,
            style = MaterialTheme.typography.titleMedium,
            color = MaterialTheme.colorScheme.primary,
        )
        Spacer(Modifier.height(4.dp))
        content()
    }
}

/**
 * SKU 规格选择: 每个属性 (尺码/款型/鞋楦) 显示一组 chip, 单选.
 * 状态在 ViewModel, 加购时一起带出去.
 *
 * @param selected 当前每属性选中的值; 没选时按"第 1 个"展示, 但 sku map 仍为空, 加购前需用户主动点
 */
@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun SkuSelector(
    groups: List<Pair<String, List<String>>>,
    selected: Map<String, String>,
    onSelect: (String, String) -> Unit,
) {
    // 默认每属性选第 1 个 (首次进页面就有合理的初始值)
    LaunchedEffect(groups) {
        groups.forEach { (key, values) ->
            if (key !in selected && values.isNotEmpty()) {
                onSelect(key, values.first())
            }
        }
    }
    Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
        groups.forEach { (key, values) ->
            val current = selected[key] ?: values.firstOrNull() ?: ""
            Column {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = key,
                        style = MaterialTheme.typography.labelLarge,
                        fontWeight = FontWeight.SemiBold,
                        color = MaterialTheme.colorScheme.onSurface,
                    )
                    Spacer(Modifier.width(8.dp))
                    Text(
                        text = current,
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.primary,
                    )
                }
                Spacer(Modifier.height(8.dp))
                androidx.compose.foundation.layout.FlowRow(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    values.forEach { value ->
                        SkuChip(
                            text = value,
                            selected = current == value,
                            onClick = { onSelect(key, value) },
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun SkuChip(
    text: String,
    selected: Boolean,
    onClick: () -> Unit,
) {
    androidx.compose.material3.Surface(
        onClick = onClick,
        shape = RoundedCornerShape(8.dp),
        color = if (selected) {
            MaterialTheme.colorScheme.primary
        } else {
            MaterialTheme.colorScheme.surface
        },
        contentColor = if (selected) {
            MaterialTheme.colorScheme.onPrimary
        } else {
            MaterialTheme.colorScheme.onSurface
        },
        border = androidx.compose.foundation.BorderStroke(
            width = if (selected) 0.dp else 1.dp,
            color = if (selected) MaterialTheme.colorScheme.primary
                    else MaterialTheme.colorScheme.outlineVariant,
        ),
    ) {
        Text(
            text = text,
            modifier = Modifier.padding(horizontal = 14.dp, vertical = 8.dp),
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Normal,
        )
    }
}

@Composable
private fun SpecList(items: List<Pair<String, String>>) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        items.forEach { (key, value) ->
            Text(
                text = "$key：$value",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.82f),
            )
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun TagRow(tags: List<String>) {
    androidx.compose.foundation.layout.FlowRow(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(6.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        tags.forEach { tag ->
            Surface(
                color = MaterialTheme.colorScheme.surfaceVariant,
                shape = RoundedCornerShape(6.dp),
            ) {
                Text(
                    text = tag,
                    style = MaterialTheme.typography.labelSmall,
                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                )
            }
        }
    }
}
