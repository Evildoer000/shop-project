package com.bytedance.shopguide.ui.chat

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.bytedance.shopguide.ui.theme.MutedInk
import com.bytedance.shopguide.ui.theme.PriceRed

/**
 * 轻量 markdown 渲染, 专为 LLM 推荐回答的格式优化:
 *   ## 1. 商品名     →  圆形 badge + 商品名大字粗体
 *   **¥899** ...     →  红色大字加粗
 *   *副标:*          →  灰色斜体小字
 *   普通段落          →  正文
 */
@Composable
fun MarkdownText(
    content: String,
    textColor: Color = MaterialTheme.colorScheme.onSurface,
) {
    val blocks = parseBlocks(content)
    SelectionContainer {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            blocks.forEach { block ->
                when (block) {
                    is MdBlock.H2Product -> ProductHeading(block.number, block.title)
                    is MdBlock.PriceLine -> PriceMetaLine(block.line)
                    is MdBlock.Italic -> Text(
                        text = block.text,
                        style = MaterialTheme.typography.labelMedium.copy(
                            fontStyle = FontStyle.Italic,
                            color = MutedInk,
                        ),
                    )
                    is MdBlock.Paragraph -> Text(
                        text = renderInline(block.text, textColor),
                        style = MaterialTheme.typography.bodyLarge.copy(
                            lineHeight = 24.sp,
                            color = textColor,
                        ),
                    )
                    is MdBlock.Spacer -> Spacer(modifier = Modifier.height(4.dp))
                }
            }
        }
    }
}

@Composable
private fun ProductHeading(number: String, title: String) {
    Spacer(modifier = Modifier.height(6.dp))
    Row(verticalAlignment = Alignment.CenterVertically) {
        Box(
            modifier = Modifier
                .size(26.dp)
                .clip(CircleShape)
                .background(MaterialTheme.colorScheme.primary),
            contentAlignment = Alignment.Center,
        ) {
            Text(
                text = number,
                color = MaterialTheme.colorScheme.onPrimary,
                style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold),
            )
        }
        Spacer(modifier = Modifier.width(10.dp))
        Text(
            text = title,
            style = MaterialTheme.typography.titleMedium.copy(
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onSurface,
            ),
        )
    }
}

@Composable
private fun PriceMetaLine(line: String) {
    // 解析 "**¥899**　耐克 · 跑步鞋"
    val priceRegex = Regex("\\*\\*([^*]+)\\*\\*")
    val match = priceRegex.find(line)
    val priceText = match?.groupValues?.get(1) ?: ""
    val rest = if (match != null) line.substring(match.range.last + 1).trim() else line
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(8.dp))
            .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.4f))
            .padding(horizontal = 10.dp, vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        if (priceText.isNotBlank()) {
            Text(
                text = priceText,
                style = MaterialTheme.typography.titleLarge.copy(
                    fontWeight = FontWeight.Bold,
                    color = PriceRed,
                ),
            )
            if (rest.isNotBlank()) {
                Spacer(modifier = Modifier.width(12.dp))
            }
        }
        if (rest.isNotBlank()) {
            Text(
                text = rest.trimStart('　', ' ', '·').trim(),
                style = MaterialTheme.typography.bodyMedium.copy(color = MutedInk),
            )
        }
    }
}

private fun renderInline(text: String, baseColor: Color): AnnotatedString = buildAnnotatedString {
    var i = 0
    while (i < text.length) {
        when {
            text.startsWith("**", i) -> {
                val end = text.indexOf("**", i + 2)
                if (end == -1) {
                    append(text.substring(i)); i = text.length
                } else {
                    withStyle(SpanStyle(fontWeight = FontWeight.Bold)) {
                        append(text.substring(i + 2, end))
                    }
                    i = end + 2
                }
            }
            text[i] == '*' -> {
                val end = text.indexOf('*', i + 1)
                if (end == -1) {
                    append(text.substring(i)); i = text.length
                } else {
                    withStyle(SpanStyle(fontStyle = FontStyle.Italic, color = baseColor.copy(alpha = 0.7f))) {
                        append(text.substring(i + 1, end))
                    }
                    i = end + 1
                }
            }
            else -> {
                append(text[i])
                i += 1
            }
        }
    }
}

private sealed class MdBlock {
    data class H2Product(val number: String, val title: String) : MdBlock()
    data class PriceLine(val line: String) : MdBlock()
    data class Italic(val text: String) : MdBlock()
    data class Paragraph(val text: String) : MdBlock()
    data object Spacer : MdBlock()
}

private fun parseBlocks(text: String): List<MdBlock> {
    val blocks = mutableListOf<MdBlock>()
    val lines = text.split("\n")
    var paragraphBuf = StringBuilder()

    fun flushParagraph() {
        val s = paragraphBuf.toString().trim()
        if (s.isNotEmpty()) {
            blocks += MdBlock.Paragraph(s)
        }
        paragraphBuf = StringBuilder()
    }

    val h2NumPattern = Regex("^##\\s*(\\d+)\\.?\\s*(.*)")
    val h2GenericPattern = Regex("^##\\s+(.*)")
    val italicWholePattern = Regex("^\\*([^*]+)[\\*]?[:：]?\\s*$")
    val priceLinePattern = Regex("^\\s*\\*\\*[¥￥]")

    for (raw in lines) {
        val line = raw.trim()
        if (line.isEmpty()) {
            flushParagraph()
            blocks += MdBlock.Spacer
            continue
        }
        // ## 1. xxx
        h2NumPattern.matchEntire(line)?.let { m ->
            flushParagraph()
            blocks += MdBlock.H2Product(m.groupValues[1], m.groupValues[2])
            return@let
        } ?: run {
            // ## xxx (无数字)
            h2GenericPattern.matchEntire(line)?.let { m ->
                flushParagraph()
                blocks += MdBlock.H2Product("•", m.groupValues[1])
                return@run
            }

            // **¥xxx** ... (整行价格元数据行)
            if (priceLinePattern.containsMatchIn(line)) {
                flushParagraph()
                blocks += MdBlock.PriceLine(line)
                return@run
            }

            // *xxx：* (整行斜体副标)
            italicWholePattern.matchEntire(line)?.let { m ->
                flushParagraph()
                blocks += MdBlock.Italic(m.groupValues[1].trimEnd('：', ':').trim())
                return@run
            } ?: run {
                if (paragraphBuf.isNotEmpty()) paragraphBuf.append('\n')
                paragraphBuf.append(line)
            }
        }
    }
    flushParagraph()
    return blocks
}
