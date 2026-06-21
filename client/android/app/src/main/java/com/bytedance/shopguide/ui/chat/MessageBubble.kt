package com.bytedance.shopguide.ui.chat

import android.net.Uri
import android.widget.Toast
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import com.bytedance.shopguide.model.AgentUpdateState
import com.bytedance.shopguide.model.ChatMessage
import com.bytedance.shopguide.model.Role
import com.bytedance.shopguide.ui.theme.AssistantBubble
import com.bytedance.shopguide.ui.theme.BrandPrimary
import com.bytedance.shopguide.ui.theme.MutedInk
import com.bytedance.shopguide.ui.theme.TraceTint

@Composable
fun MessageBubble(
    message: ChatMessage,
    onProductClick: (String) -> Unit,
    onProductAddToCart: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    when (message.role) {
        Role.USER -> UserBubble(message, modifier)
        Role.ASSISTANT -> AssistantBubble(message, onProductClick, onProductAddToCart, modifier)
    }
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun UserBubble(message: ChatMessage, modifier: Modifier) {
    val clipboard = LocalClipboardManager.current
    val context = LocalContext.current
    val displayText = message.content.toDisplayText()
    Row(
        modifier = modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.End,
    ) {
        Column(
            modifier = Modifier.widthIn(max = 300.dp),
            horizontalAlignment = Alignment.End,
        ) {
            message.attachedImageUri?.let { uriString ->
                Surface(
                    shape = RoundedCornerShape(14.dp),
                    color = MaterialTheme.colorScheme.surfaceVariant,
                    modifier = Modifier.padding(bottom = 6.dp),
                ) {
                    AsyncImage(
                        model = Uri.parse(uriString),
                        contentDescription = "用户上传的图片",
                        contentScale = ContentScale.Crop,
                        modifier = Modifier
                            .size(width = 180.dp, height = 180.dp)
                            .clip(RoundedCornerShape(14.dp)),
                    )
                }
            }

            if (displayText.isNotBlank()) {
                Surface(
                    color = MaterialTheme.colorScheme.primaryContainer,
                    shape = RoundedCornerShape(18.dp, 6.dp, 18.dp, 18.dp),
                    modifier = Modifier.combinedClickable(
                        onClick = {},
                        onLongClick = {
                            clipboard.setText(AnnotatedString(displayText))
                            Toast.makeText(context, "已复制消息", Toast.LENGTH_SHORT).show()
                        },
                    ),
                ) {
                    SelectionContainer {
                        Text(
                            text = displayText,
                            modifier = Modifier.padding(horizontal = 15.dp, vertical = 10.dp),
                            color = MaterialTheme.colorScheme.onPrimaryContainer,
                            style = MaterialTheme.typography.bodyLarge,
                        )
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun AssistantBubble(
    message: ChatMessage,
    onProductClick: (String) -> Unit,
    onProductAddToCart: (String) -> Unit,
    modifier: Modifier,
) {
    val clipboard = LocalClipboardManager.current
    val context = LocalContext.current
    val displayText = message.content.toDisplayText()
    val progress = message.currentProgress()

    Column(modifier = modifier.fillMaxWidth()) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Surface(
                color = MaterialTheme.colorScheme.primary,
                shape = RoundedCornerShape(50),
                modifier = Modifier.size(30.dp),
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth().padding(2.dp),
                    horizontalArrangement = Arrangement.Center,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        "AI",
                        color = MaterialTheme.colorScheme.onPrimary,
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = FontWeight.SemiBold,
                    )
                }
            }
            Spacer(Modifier.width(8.dp))
            Text(
                text = "导购助手",
                style = MaterialTheme.typography.labelLarge,
                color = MutedInk,
            )
        }
        Spacer(Modifier.height(7.dp))

        if (progress != null) {
            AgentProgressStrip(progress, message.agentUpdates, message.isStreaming)
            Spacer(Modifier.height(8.dp))
        }

        if (displayText.isNotBlank() || (message.isStreaming && progress == null)) {
            Surface(
                color = AssistantBubble,
                shape = RoundedCornerShape(8.dp, 18.dp, 18.dp, 18.dp),
                modifier = Modifier
                    .fillMaxWidth()
                    .combinedClickable(
                        onClick = {},
                        onLongClick = {
                            if (displayText.isNotBlank()) {
                                clipboard.setText(AnnotatedString(displayText))
                                Toast.makeText(context, "已复制回复", Toast.LENGTH_SHORT).show()
                            }
                        },
                    ),
                tonalElevation = 1.dp,
            ) {
                Column(modifier = Modifier.padding(15.dp)) {
                    if (displayText.isNotBlank()) {
                        if (message.isError) {
                            SelectionContainer {
                                Text(
                                    text = displayText,
                                    style = MaterialTheme.typography.bodyLarge,
                                    color = MaterialTheme.colorScheme.error,
                                )
                            }
                        } else {
                            // ★ markdown 渲染: 商品标题 + 价格 chip + 副标 + 段落正文
                            MarkdownText(content = displayText)
                        }
                    } else {
                        Text(
                            text = "正在准备回复…",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MutedInk,
                        )
                    }
                }
            }
        }

        if (message.products.isNotEmpty()) {
            Spacer(Modifier.height(10.dp))
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                message.products.forEach { product ->
                    ProductCardItem(
                        product = product,
                        onClick = { onProductClick(product.productId) },
                        onAddToCart = { onProductAddToCart(product.productId) },
                    )
                }
            }
        }

        if (!message.isStreaming && (message.decisionTrace != null || message.traceLogs.isNotEmpty())) {
            Spacer(Modifier.height(10.dp))
            DecisionTracePanel(
                trace = message.decisionTrace,
                traceLogs = message.traceLogs,
            )
        }
    }
}

@Composable
private fun AgentProgressStrip(
    progress: AgentProgressUi,
    updates: List<AgentUpdateState>,
    isStreaming: Boolean,
) {
    Surface(
        color = TraceTint,
        shape = RoundedCornerShape(8.dp),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(modifier = Modifier.padding(horizontal = 12.dp, vertical = 10.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                if (isStreaming) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(14.dp),
                        strokeWidth = 2.dp,
                        color = BrandPrimary,
                        trackColor = BrandPrimary.copy(alpha = 0.14f),
                    )
                } else {
                    Surface(
                        color = BrandPrimary,
                        shape = RoundedCornerShape(50),
                        modifier = Modifier.size(8.dp),
                    ) {}
                }
                Spacer(Modifier.width(8.dp))
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = progress.title,
                        style = MaterialTheme.typography.labelLarge,
                        color = MaterialTheme.colorScheme.onSurface,
                        fontWeight = FontWeight.SemiBold,
                    )
                    if (progress.content.isNotBlank()) {
                        Text(
                            text = progress.content.toDisplayText(),
                            style = MaterialTheme.typography.bodySmall,
                            color = MutedInk,
                            maxLines = 2,
                            overflow = TextOverflow.Ellipsis,
                        )
                    }
                }
            }
            Spacer(Modifier.height(9.dp))
            ProgressSteps(updates)
        }
    }
}

@Composable
private fun ProgressSteps(updates: List<AgentUpdateState>) {
    val steps = listOf(
        "planner" to "理解",
        "retrieval" to "检索",
        "corrective" to "校验",
        "answer" to "回答",
    )
    val seenStages = updates.map { it.stage }.toSet()
    val activeStage = updates.lastOrNull { !it.done }?.stage ?: updates.lastOrNull()?.stage
    Row(horizontalArrangement = Arrangement.spacedBy(7.dp)) {
        steps.forEach { (stage, label) ->
            val active = stage == activeStage
            val reached = stage in seenStages
            Surface(
                color = when {
                    active -> BrandPrimary
                    reached -> BrandPrimary.copy(alpha = 0.16f)
                    else -> MaterialTheme.colorScheme.surface
                },
                shape = RoundedCornerShape(50),
            ) {
                Text(
                    text = label,
                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp),
                    style = MaterialTheme.typography.labelSmall,
                    color = if (active) MaterialTheme.colorScheme.onPrimary else MutedInk,
                    maxLines = 1,
                )
            }
        }
    }
}

private data class AgentProgressUi(
    val title: String,
    val content: String,
)

private fun ChatMessage.currentProgress(): AgentProgressUi? {
    if (!isStreaming) return null
    val activeUpdate = agentUpdates.lastOrNull { !it.done && it.title.isNotBlank() }
    val latestUpdate = activeUpdate ?: agentUpdates.lastOrNull { it.title.isNotBlank() }
    if (latestUpdate != null) {
        return AgentProgressUi(
            title = latestUpdate.title,
            content = latestUpdate.content,
        )
    }
    if (content.isNotBlank()) {
        return AgentProgressUi("生成回答", "正在组织最终回复。")
    }
    return when (traceLogs.lastOrNull()?.stage) {
        "single_retrieval_worker_execution",
        "multi_need_retrieval",
        "image_retrieval_worker_execution" -> AgentProgressUi("检索商品", "正在整理候选商品。")
        "corrective_reflection" -> AgentProgressUi("校验证据", "正在核对商品依据。")
        "intent_planning" -> AgentProgressUi("规划完成", "准备进入商品检索。")
        "input" -> AgentProgressUi("理解需求", "正在理解你的需求。")
        else -> AgentProgressUi("理解需求", "正在启动导购链路。")
    }
}

private fun String.toDisplayText(): String {
    if (isBlank()) return this
    return this
        .replace(Regex("\\*\\*(.*?)\\*\\*"), "$1")
        .replace(Regex("\\s*\\(\\s*product_id\\s*:\\s*[^)]*\\)", RegexOption.IGNORE_CASE), "")
        .replace(Regex("product_id\\s*:\\s*\\S+", RegexOption.IGNORE_CASE), "")
        .replace(" - 理由：", "\n理由：")
        .replace(Regex("\n{3,}"), "\n\n")
        .trim()
}
