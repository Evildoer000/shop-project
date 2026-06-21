package com.bytedance.shopguide.model

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/**
 * Mirror of the backend [app/schemas.py] `DecisionTrace`. Pushed via the SSE
 * `decision_trace` event and rendered inside the Decision Trace panel.
 *
 * Free-form sub-objects (`query_understanding`, `retrieval_summary`) are kept
 * as JsonObject so we don't have to keep schemas perfectly in sync.
 */
@Serializable
data class DecisionTraceDto(
    @SerialName("query_understanding")
    val queryUnderstanding: JsonObject = JsonObject(emptyMap()),
    @SerialName("image_attributes")
    val imageAttributes: JsonObject = JsonObject(emptyMap()),
    @SerialName("memory_used") val memoryUsed: List<String> = emptyList(),
    val filters: List<String> = emptyList(),
    @SerialName("retrieval_summary")
    val retrievalSummary: JsonObject = JsonObject(emptyMap()),
    @SerialName("agent_path") val agentPath: List<JsonObject> = emptyList(),
    @SerialName("planner_proposal")
    val plannerProposal: JsonObject = JsonObject(emptyMap()),
    @SerialName("orchestrator_decisions")
    val orchestratorDecisions: List<JsonObject> = emptyList(),
    val task: JsonObject = JsonObject(emptyMap()),
    val route: String = "",
    @SerialName("candidate_counts")
    val candidateCounts: JsonObject = JsonObject(emptyMap()),
    val stages: List<JsonObject> = emptyList(),
    @SerialName("rerank_factors") val rerankFactors: List<String> = emptyList(),
    @SerialName("final_reason") val finalReason: String = "",
) {
    /** Flatten arbitrary JsonObject into a list of "key: value" rows. */
    fun queryUnderstandingPairs(): List<Pair<String, String>> = flatten(queryUnderstanding)

    fun retrievalSummaryPairs(): List<Pair<String, String>> = flatten(retrievalSummary)

    private fun flatten(obj: JsonObject): List<Pair<String, String>> =
        obj.entries.map { (k, v) -> k to v.toDisplay() }

    private fun JsonElement.toDisplay(): String = when (this) {
        is JsonPrimitive -> this.contentOrEmpty()
        is JsonObject -> jsonObject.entries.joinToString(", ") {
            "${it.key}: ${it.value.toDisplay()}"
        }
        else -> try {
            jsonArray.joinToString(", ") { it.toDisplay() }
        } catch (_: Throwable) {
            toString()
        }
    }

    private fun JsonPrimitive.contentOrEmpty(): String =
        try { content } catch (_: Throwable) { toString() }
}

// Convenience extension to read primitive-as-string in a forgiving way.
internal fun JsonElement.primitiveContent(): String? = try {
    (this as? JsonPrimitive)?.jsonPrimitive?.content
} catch (_: Throwable) {
    null
}
