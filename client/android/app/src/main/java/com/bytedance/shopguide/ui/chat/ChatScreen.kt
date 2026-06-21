package com.bytedance.shopguide.ui.chat

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
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
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.ShoppingCart
import androidx.compose.material.icons.filled.Storefront
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import android.widget.Toast
import com.bytedance.shopguide.ui.theme.BrandAccent
import com.bytedance.shopguide.ui.theme.MutedInk

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(
    onProductClick: (String) -> Unit,
    onMallClick: () -> Unit,
    onCartClick: () -> Unit,
    viewModel: ChatViewModel = viewModel(),
) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val listState = rememberLazyListState()
    var showHistory by remember { mutableStateOf(false) }
    val context = LocalContext.current
    val imagePicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia(),
    ) { uri ->
        if (uri != null) {
            viewModel.onImageSelected(uri, context.contentResolver)
        }
    }
    val lastMessage = state.messages.lastOrNull()
    val lastAgentProgressLength = lastMessage?.agentUpdates?.sumOf { it.content.length } ?: 0
    val cartNotice = state.cartNotice

    LaunchedEffect(cartNotice) {
        if (cartNotice != null) {
            Toast.makeText(context, cartNotice, Toast.LENGTH_SHORT).show()
            viewModel.clearCartNotice()
        }
    }

    LaunchedEffect(
        state.messages.size,
        lastMessage?.content?.length,
        lastAgentProgressLength,
        lastMessage?.products?.size,
        lastMessage?.isStreaming,
    ) {
        val lastIndex = state.messages.lastIndex
        if (lastIndex >= 0) listState.animateScrollToItem(lastIndex)
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text(
                            text = "智购助手",
                            style = MaterialTheme.typography.titleLarge,
                        )
                        Text(
                            text = "把需求说清楚，剩下的交给我筛",
                            style = MaterialTheme.typography.labelMedium,
                            color = MutedInk,
                        )
                    }
                },
                actions = {
                    IconButton(onClick = onMallClick) {
                        Icon(Icons.Filled.Storefront, contentDescription = "商城推荐")
                    }
                    IconButton(onClick = onCartClick) {
                        Icon(Icons.Filled.ShoppingCart, contentDescription = "购物车")
                    }
                    IconButton(onClick = { showHistory = true }) {
                        Icon(Icons.Filled.History, contentDescription = "历史会话")
                    }
                    IconButton(onClick = { viewModel.newSession() }) {
                        Icon(Icons.Filled.Add, contentDescription = "新会话")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.background,
                ),
            )
        },
        containerColor = MaterialTheme.colorScheme.background,
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding),
        ) {
            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth(),
            ) {
                if (state.messages.isEmpty()) {
                    EmptyState(onSuggestionClick = viewModel::onInputChanged)
                } else {
                    LazyColumn(
                        state = listState,
                        modifier = Modifier.fillMaxSize(),
                        contentPadding = PaddingValues(horizontal = 14.dp, vertical = 12.dp),
                        verticalArrangement = Arrangement.spacedBy(10.dp),
                    ) {
                        items(state.messages, key = { it.id }) { msg ->
                            MessageBubble(
                                message = msg,
                                onProductClick = { pid ->
                                    viewModel.reportProductClick(pid)   // ★ 上报 click 影响商城推荐
                                    onProductClick(pid)                  // 原导航
                                },
                                onProductAddToCart = viewModel::addToCart,
                            )
                        }
                    }
                }
            }

            HorizontalDivider(color = MaterialTheme.colorScheme.surfaceVariant)

            Surface(color = MaterialTheme.colorScheme.surface) {
                ChatInputBar(
                    input = state.input,
                    isSending = state.isStreaming,
                    selectedImageUri = state.selectedImageUri,
                    isUploadingImage = state.uploadingImage,
                    uploadError = state.uploadError,
                    onInputChanged = viewModel::onInputChanged,
                    onPickImage = {
                        imagePicker.launch(
                            PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly),
                        )
                    },
                    onClearImage = viewModel::clearSelectedImage,
                    onSend = viewModel::sendMessage,
                )
            }
        }
    }

    if (showHistory) {
        HistorySheet(
            history = state.history,
            onDismiss = { showHistory = false },
            onOpen = {
                showHistory = false
                viewModel.restoreSession(it)
            },
        )
    }

}

@Composable
private fun EmptyState(onSuggestionClick: (String) -> Unit) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 28.dp),
        contentAlignment = Alignment.Center,
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Surface(
                color = MaterialTheme.colorScheme.surface,
                shape = RoundedCornerShape(18.dp),
                tonalElevation = 2.dp,
            ) {
                Text(
                    text = "购物前先问一句",
                    modifier = Modifier.padding(horizontal = 18.dp, vertical = 10.dp),
                    style = MaterialTheme.typography.labelLarge,
                    color = BrandAccent,
                )
            }
            Spacer(Modifier.height(18.dp))
            Text(
                text = "智购助手",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.SemiBold,
            )
            Spacer(Modifier.height(8.dp))
            Text(
                text = "告诉我预算、场景、偏好或排除项，我会帮你筛商品、解释理由，也能继续追问调整。",
                style = MaterialTheme.typography.bodyMedium,
                color = MutedInk,
                textAlign = TextAlign.Center,
            )
            Spacer(Modifier.height(28.dp))
            SuggestionChip(
                text = "我是油皮，预算 150 以内，推荐一款夏天用不闷的防晒",
                onClick = onSuggestionClick,
            )
            Spacer(Modifier.height(10.dp))
            SuggestionChip(
                text = "下周去三亚玩，帮我配一套防晒和晒后修复，预算 300",
                onClick = onSuggestionClick,
            )
            Spacer(Modifier.height(10.dp))
            SuggestionChip(
                text = "预算 2000，推荐一款降噪蓝牙耳机，长途飞行用",
                onClick = onSuggestionClick,
            )
        }
    }
}

@Composable
private fun SuggestionChip(
    text: String,
    onClick: (String) -> Unit,
) {
    Surface(
        color = MaterialTheme.colorScheme.surface,
        shape = RoundedCornerShape(16.dp),
        tonalElevation = 1.dp,
        modifier = Modifier.clickable { onClick(text) },
    ) {
        Text(
            text = text,
            style = MaterialTheme.typography.bodyMedium,
            maxLines = 2,
            overflow = TextOverflow.Ellipsis,
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 11.dp),
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun HistorySheet(
    history: List<ChatSessionSummary>,
    onDismiss: () -> Unit,
    onOpen: (String) -> Unit,
) {
    ModalBottomSheet(onDismissRequest = onDismiss) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 20.dp, vertical = 8.dp),
        ) {
            Text("历史会话", style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.height(12.dp))
            if (history.isEmpty()) {
                Text(
                    text = "还没有历史会话。点右上角 + 开始新会话后，旧会话会出现在这里。",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MutedInk,
                    modifier = Modifier.padding(bottom = 24.dp),
                )
            } else {
                history.forEach { item ->
                    Surface(
                        color = MaterialTheme.colorScheme.surfaceVariant,
                        shape = RoundedCornerShape(14.dp),
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(bottom = 10.dp)
                            .clickable { onOpen(item.sessionId) },
                    ) {
                        Column(modifier = Modifier.padding(14.dp)) {
                            Text(
                                text = item.title,
                                style = MaterialTheme.typography.labelLarge,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis,
                            )
                            Spacer(Modifier.height(4.dp))
                            Text(
                                text = "${item.messages.size} 条消息",
                                style = MaterialTheme.typography.bodySmall,
                                color = MutedInk,
                            )
                        }
                    }
                }
            }
            Spacer(Modifier.height(20.dp))
        }
    }
}
