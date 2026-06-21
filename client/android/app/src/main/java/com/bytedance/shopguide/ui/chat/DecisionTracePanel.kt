package com.bytedance.shopguide.ui.chat

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ExpandLess
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material.icons.filled.TipsAndUpdates
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.bytedance.shopguide.model.DecisionTraceDto
import com.bytedance.shopguide.model.TraceLog
import com.bytedance.shopguide.model.primitiveContent
import com.bytedance.shopguide.ui.theme.MutedInk
import com.bytedance.shopguide.ui.theme.TraceTint
import kotlinx.serialization.json.jsonArray

@Composable
fun DecisionTracePanel(
    trace: DecisionTraceDto?,
    traceLogs: List<TraceLog>,
    modifier: Modifier = Modifier,
) {
    if (trace == null && traceLogs.isEmpty()) return

    var expanded by remember { mutableStateOf(false) }
    val shortReason = trace?.finalReason
        ?.takeIf { it.isNotBlank() }
        ?: trace?.retrievalSummaryPairs()?.firstOrNull { it.first == "reason" }?.second
        ?: "已根据你的需求完成筛选。"

    Surface(
        modifier = modifier.fillMaxWidth(),
        color = TraceTint,
        shape = RoundedCornerShape(16.dp),
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Icon(
                    Icons.Filled.TipsAndUpdates,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary,
                )
                Spacer(Modifier.width(8.dp))
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "为什么这样推荐",
                        style = MaterialTheme.typography.labelLarge,
                    )
                    Text(
                        text = shortReason.toHumanTraceText(),
                        style = MaterialTheme.typography.bodySmall,
                        color = MutedInk,
                        maxLines = if (expanded) Int.MAX_VALUE else 2,
                    )
                }
                TextButton(onClick = { expanded = !expanded }) {
                    Text(if (expanded) "收起" else "详情")
                    Icon(
                        if (expanded) Icons.Filled.ExpandLess else Icons.Filled.ExpandMore,
                        contentDescription = null,
                    )
                }
            }
            AnimatedVisibility(
                visible = expanded,
                enter = expandVertically() + fadeIn(),
                exit = shrinkVertically() + fadeOut(),
            ) {
                Column(
                    modifier = Modifier.padding(top = 10.dp),
                    verticalArrangement = Arrangement.spacedBy(9.dp),
                ) {
                    val items = timelineItems(trace, traceLogs)
                    items.forEachIndexed { index, item ->
                        DecisionTimelineRow(
                            item = item,
                            isLast = index == items.lastIndex,
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun DecisionTimelineRow(item: TimelineItem, isLast: Boolean) {
    Row(modifier = Modifier.fillMaxWidth()) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Box(
                modifier = Modifier
                    .size(9.dp)
                    .clip(RoundedCornerShape(50))
                    .background(MaterialTheme.colorScheme.primary),
            )
            if (!isLast) {
                Box(
                    modifier = Modifier
                        .width(1.dp)
                        .height(34.dp)
                        .background(MaterialTheme.colorScheme.primary.copy(alpha = 0.22f)),
                )
            }
        }
        Spacer(Modifier.width(10.dp))
        Column(modifier = Modifier.weight(1f).padding(bottom = if (isLast) 0.dp else 5.dp)) {
            Text(
                text = item.title,
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.primary,
                fontWeight = FontWeight.SemiBold,
            )
            Text(
                text = item.component,
                style = MaterialTheme.typography.labelSmall,
                color = MutedInk,
            )
            if (item.description.isNotBlank()) {
                Spacer(Modifier.height(2.dp))
                Text(
                    text = item.description.toHumanTraceText(),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.78f),
                )
            }
        }
    }
}

private data class TimelineItem(
    val title: String,
    val component: String,
    val description: String,
)

private fun timelineItems(trace: DecisionTraceDto?, traceLogs: List<TraceLog>): List<TimelineItem> {
    val safeTrace = trace ?: return traceLogs.map {
        TimelineItem(stageTitle(it.stage), componentForStage(it.stage), it.content)
    }
    return listOf(
        TimelineItem(
            "输入理解",
            "InputProcessor",
            stageReason(safeTrace, "input") ?: traceLogs.firstOrNull { it.stage == "input" }?.content.orEmpty(),
        ),
        TimelineItem(
            "图片理解",
            "ImageAttributeExtractor",
            imageAttributeDescription(safeTrace),
        ),
        TimelineItem(
            "计划意图",
            "IntentPlanner",
            stageReason(safeTrace, "intent_planning")
                ?: safeTrace.plannerProposal["summary"]?.primitiveContent()?.takeIf { it.isNotBlank() }
                ?: "已形成本轮执行计划。",
        ),
        TimelineItem(
            "流程裁决",
            "Orchestrator",
            safeTrace.route.takeIf { it.isNotBlank() }?.let { "最终路线：${it.toHumanTraceText()}" }.orEmpty(),
        ),
        TimelineItem(
            "检索召回",
            retrievalComponent(safeTrace),
            retrievalDescription(safeTrace),
        ),
        TimelineItem(
            "证据反思",
            "CorrectiveAgent",
            safeTrace.finalReason.ifBlank { stageReason(safeTrace, "corrective_reflection").orEmpty() },
        ),
        TimelineItem(
            "回答生成",
            "AnswerGenerator",
            "根据已裁决路线组织自然语言回复。",
        ),
    ).filter { it.description.isNotBlank() || it.title in setOf("流程裁决", "回答生成") }
}

private fun imageAttributeDescription(trace: DecisionTraceDto): String {
    val attributes = trace.imageAttributes
    if (attributes.isEmpty()) return ""
    val available = attributes["available"]?.primitiveContent()?.toBooleanStrictOrNull() ?: false
    val note = attributes["uncertainty_note"]?.primitiveContent().orEmpty()
    if (!available) {
        return note.takeIf { it.isNotBlank() }?.let { "图片理解暂不可用：$it" }.orEmpty()
    }
    val productType = attributes["product_type_guess"]?.primitiveContent().orEmpty()
    val category = attributes["category_guess"]?.primitiveContent().orEmpty()
    val colors = attributes.stringList("colors")
    val styles = attributes.stringList("style_tags")
    val occasions = attributes.stringList("occasion_tags")
    val confidence = attributes["confidence"]?.primitiveContent().orEmpty()

    val brief = listOfNotNull(
        productType.takeIf { it.isNotBlank() },
        colors.takeIf { it.isNotEmpty() }?.joinToString("、"),
        styles.takeIf { it.isNotEmpty() }?.joinToString("、"),
        occasions.takeIf { it.isNotEmpty() }?.joinToString("、"),
        category.takeIf { it.isNotBlank() && it != productType },
    ).joinToString(" / ")
    return listOfNotNull(
        brief.takeIf { it.isNotBlank() }?.let { "图片推测：$it" },
        confidence.takeIf { it.isNotBlank() && it != "0.0" }?.let { "置信度 $it" },
        note.takeIf { it.isNotBlank() },
    ).joinToString("；")
}

private fun kotlinx.serialization.json.JsonObject.stringList(key: String): List<String> =
    try {
        get(key)?.jsonArray?.mapNotNull { it.primitiveContent()?.takeIf { text -> text.isNotBlank() } }.orEmpty()
    } catch (_: Throwable) {
        emptyList()
    }

private fun stageReason(trace: DecisionTraceDto, name: String): String? =
    trace.stages.firstOrNull { it["name"]?.primitiveContent() == name }
        ?.get("reason")
        ?.primitiveContent()
        ?.takeIf { it.isNotBlank() }

private fun retrievalComponent(trace: DecisionTraceDto): String =
    when {
        trace.task["execution_path"]?.primitiveContent() == "image_retrieval" -> "ImageRetrievalWorker"
        trace.agentPath.any { it["node"]?.primitiveContent() == "MultiNeedRetrievalCoordinator" } -> "RetrievalWorker"
        else -> "RetrievalWorker"
    }

private fun retrievalDescription(trace: DecisionTraceDto): String {
    val afterCorrective = trace.candidateCounts["after_corrective"]?.primitiveContent().orEmpty()
    val total = trace.candidateCounts["hybrid_candidates"]
        ?.primitiveContent()
        .orEmpty()
        .ifBlank { trace.candidateCounts["multi_need_slot_candidates"]?.primitiveContent().orEmpty() }
    return when {
        afterCorrective.isNotBlank() && total.isNotBlank() -> "召回 $total 个候选，证据校验后保留 $afterCorrective 个。"
        afterCorrective.isNotBlank() -> "证据校验后保留 $afterCorrective 个候选。"
        else -> stageReason(trace, "single_retrieval_worker_execution")
            ?: stageReason(trace, "multi_need_retrieval")
            ?: stageReason(trace, "image_retrieval_worker_execution")
            ?: trace.retrievalSummary["reason"]?.primitiveContent().orEmpty()
    }
}

private fun stageTitle(stage: String): String =
    when (stage) {
        "input" -> "输入理解"
        "intent_planning" -> "计划意图"
        "corrective_reflection" -> "证据反思"
        else -> "处理进度"
    }

private fun componentForStage(stage: String): String =
    when (stage) {
        "input" -> "InputProcessor"
        "intent_planning" -> "IntentPlanner"
        "corrective_reflection" -> "CorrectiveAgent"
        else -> "Orchestrator"
    }

private fun String.toHumanTraceText(): String =
    replace("ReAct 决策判断", "系统判断")
        .replace("direct_answer", "直接回答")
        .replace("rewrite_used_llm", "智能改写")
        .replace("query_rewrite", "需求理解")
        .replace("input", "输入处理")
        .replace(Regex("\\b(route|answer_mode|reason)\\b\\s*[:：]"), "")
        .replace(Regex("\\s+"), " ")
        .trim()
