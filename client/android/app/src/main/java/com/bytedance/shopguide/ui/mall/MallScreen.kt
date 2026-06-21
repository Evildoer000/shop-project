package com.bytedance.shopguide.ui.mall

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.Button
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
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.bytedance.shopguide.ui.chat.ProductCardItem
import com.bytedance.shopguide.ui.theme.MutedInk

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MallScreen(
    onBack: () -> Unit,
    onProductClick: (String) -> Unit,
    viewModel: MallViewModel = viewModel(),
) {
    val state by viewModel.state.collectAsStateWithLifecycle()

    LaunchedEffect(Unit) {
        if (state.products.isEmpty()) viewModel.load()
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text("商城推荐", style = MaterialTheme.typography.titleLarge)
                        Text(
                            text = "${stageText(state.stage)} · ${state.totalEvents} 条行为",
                            style = MaterialTheme.typography.labelMedium,
                            color = MutedInk,
                        )
                    }
                },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "返回")
                    }
                },
                actions = {
                    IconButton(onClick = viewModel::load) {
                        Icon(Icons.Filled.Refresh, contentDescription = "刷新")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.background,
                ),
            )
        },
        containerColor = MaterialTheme.colorScheme.background,
    ) { padding ->
        when {
            state.loading && state.products.isEmpty() -> LoadingMall(Modifier.padding(padding))
            state.error != null && state.products.isEmpty() -> ErrorMall(
                message = state.error.orEmpty(),
                onRetry = viewModel::load,
                modifier = Modifier.padding(padding),
            )
            state.products.isEmpty() -> EmptyMall(Modifier.padding(padding))
            else -> RecommendationList(
                state = state,
                onProductClick = onProductClick,
                onOpen = viewModel::reportProductOpen,
                modifier = Modifier.padding(padding),
            )
        }
    }
}

@Composable
private fun RecommendationList(
    state: MallUiState,
    onProductClick: (String) -> Unit,
    onOpen: (String, Int) -> Unit,
    modifier: Modifier,
) {
    LazyColumn(
        modifier = modifier.fillMaxSize(),
        contentPadding = PaddingValues(horizontal = 14.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        item {
            Surface(
                color = MaterialTheme.colorScheme.primaryContainer,
                shape = RoundedCornerShape(8.dp),
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(
                    text = "根据热门度与浏览行为生成推荐，不进入 Agent 决策链。",
                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 9.dp),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onPrimaryContainer,
                )
            }
        }
        itemsIndexed(state.products, key = { _, item -> item.productId }) { index, product ->
            ProductCardItem(
                product = product,
                onClick = {
                    onOpen(product.productId, index)
                    onProductClick(product.productId)
                },
            )
        }
    }
}

@Composable
private fun LoadingMall(modifier: Modifier) {
    Box(modifier = modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        CircularProgressIndicator()
    }
}

@Composable
private fun ErrorMall(message: String, onRetry: () -> Unit, modifier: Modifier) {
    Column(
        modifier = modifier.fillMaxSize().padding(24.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text("推荐加载失败", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
        Spacer(Modifier.height(6.dp))
        Text(message, style = MaterialTheme.typography.bodySmall, color = MutedInk)
        Spacer(Modifier.height(14.dp))
        Button(onClick = onRetry, shape = RoundedCornerShape(8.dp)) {
            Text("重试")
        }
    }
}

@Composable
private fun EmptyMall(modifier: Modifier) {
    Box(modifier = modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Text("暂无推荐商品", color = MutedInk)
    }
}

private fun stageText(stage: String): String = when (stage) {
    "cold" -> "冷启动热门"
    "warmup" -> "行为预热"
    "warm" -> "个性化"
    else -> "推荐"
}
