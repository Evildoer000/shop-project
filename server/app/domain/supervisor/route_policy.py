from __future__ import annotations

from dataclasses import dataclass, field
import re

from app.schemas import AgentTaskProposal, IntentItem, IntentPlanV3


_PLATFORM_ALIASES = {
    "taobao": ("淘宝", "taobao"),
    "douyin_ec": ("抖音电商", "抖音商城", "抖音", "douyin"),
    "xiaohongshu": ("小红书", "xiaohongshu", "xhs"),
}


@dataclass(frozen=True)
class RouteDecision:
    approved: bool
    reason: str
    details: dict = field(default_factory=dict)


class RoutePolicy:
    """Capability checks scoped to one intent task; no global execution mode."""

    def evaluate_task(
        self,
        plan: IntentPlanV3,
        task: AgentTaskProposal,
    ) -> RouteDecision:
        if len(task.intent_ids) != 1:
            return RouteDecision(False, "Each task must belong to exactly one intent.")
        intent = self._intent(plan, task.intent_ids[0])
        if intent is None:
            return RouteDecision(False, "Task references an unknown intent.")

        capability = task.capability
        if capability == "profile_preference":
            return RouteDecision(
                bool(task.parameters.query.strip()),
                "The task has an intent-scoped profile lookup goal."
                if task.parameters.query.strip()
                else "Profile lookup requires a task-specific query.",
            )
        if capability == "clarification":
            blocking = any(item.blocking for item in intent.uncertainties)
            vague = (
                intent.route_basis.target_clarity == "vague_effect_or_use"
                and not intent.route_basis.product_family.strip()
                and not intent.product_needs
            )
            approved = blocking or vague or bool(task.parameters.missing_fields)
            return RouteDecision(
                approved,
                "The intent has a blocking ambiguity."
                if approved
                else "Clarification requires a blocking uncertainty or missing field.",
            )
        if capability == "single_product_recommendation":
            needs = self._selected_needs(intent, task)
            if intent.recommendation_policy.candidate_source == "context_only":
                return RouteDecision(
                    False,
                    "Context-only recommendation must reuse referenced products without a new retrieval task.",
                )
            explicit_target = bool(
                intent.route_basis.product_family.strip()
                or needs
                or intent.route_basis.target_clarity in {"explicit_product", "context_product"}
            )
            approved = intent.intent_type == "product_recommendation" and explicit_target and len(needs) <= 1
            return RouteDecision(
                approved,
                "The task owns one executable product target."
                if approved
                else "Single-product recommendation requires one concrete recommendation target.",
                {"product_need_count": len(needs)},
            )
        if capability == "multi_product_bundle":
            needs = self._selected_needs(intent, task)
            approved = intent.intent_type == "product_recommendation" and len(needs) >= 2
            return RouteDecision(
                approved,
                "The task owns an executable multi-product bundle."
                if approved
                else "Multi-product bundle requires at least two product needs in one intent.",
                {"product_need_count": len(needs)},
            )
        if capability == "comparison":
            approved = intent.intent_type == "product_comparison"
            return RouteDecision(
                approved,
                "The intent explicitly requests product comparison."
                if approved
                else "Comparison requires a product_comparison intent.",
            )
        if capability == "knowledge_research":
            return self._knowledge_decision(plan, intent, task)
        if capability == "commerce_research":
            return self._commerce_decision(plan, intent, task)
        return RouteDecision(True, "No additional intent routing restriction applies.")

    def _knowledge_decision(
        self,
        plan: IntentPlanV3,
        intent: IntentItem,
        task: AgentTaskProposal,
    ) -> RouteDecision:
        knowledge_mode = str(
            getattr(task.parameters, "knowledge_mode", "knowledge_answer")
        )
        external_need = intent.route_basis.external_information_need
        supported_intent = intent.intent_type in {
            "product_qa",
            "shopping_knowledge",
            "product_comparison",
            "product_recommendation",
        }
        knowledge_bridge = knowledge_mode == "concept_bridge"
        explicit_research = external_need in {"explicit_web", "freshness_required"}
        approved = supported_intent or knowledge_bridge or explicit_research
        if not approved:
            return RouteDecision(
                False,
                "Product knowledge enrichment is not grounded in this intent's knowledge or bridge requirement.",
            )
        expected_trigger = {
            "explicit_web": "explicit_web_request",
            "freshness_required": "freshness_required",
            "knowledge_bridge": "knowledge_bridge",
        }.get(external_need)
        if knowledge_bridge:
            expected_trigger = "knowledge_bridge"
        if expected_trigger and task.parameters.trigger_type != expected_trigger:
            return RouteDecision(
                False,
                "Knowledge task trigger_type does not match the intent route basis.",
                {"expected_trigger_type": expected_trigger},
            )
        if not task.parameters.query.strip():
            return RouteDecision(False, "Product knowledge enrichment requires a task-specific query.")
        if task.parameters.trigger_text and task.parameters.trigger_type != "knowledge_bridge":
            if not self._grounded(plan.original_query, task.parameters.trigger_text):
                return RouteDecision(False, "Knowledge trigger_text is not grounded in the current query.")
        return RouteDecision(
            True,
            "The intent has an explicit knowledge requirement or a grounded knowledge bridge.",
            {"trigger_type": task.parameters.trigger_type},
        )

    def _commerce_decision(
        self,
        plan: IntentPlanV3,
        intent: IntentItem,
        task: AgentTaskProposal,
    ) -> RouteDecision:
        parameters = task.parameters
        if intent.route_basis.external_information_need != "explicit_platform":
            return RouteDecision(False, "Commerce research requires an explicit platform request.")
        if parameters.trigger_type != "explicit_platform_request" or not parameters.platforms:
            return RouteDecision(False, "Commerce research requires explicit platforms and trigger_type.")
        if not parameters.query.strip():
            return RouteDecision(False, "Commerce research requires a task-specific query.")
        normalized_query = self._normalized(plan.original_query)
        ungrounded = [
            platform
            for platform in parameters.platforms
            if not any(
                self._normalized(alias) in normalized_query
                for alias in _PLATFORM_ALIASES.get(platform, ())
            )
        ]
        if ungrounded:
            return RouteDecision(
                False,
                "One or more commerce platforms were not named by the user.",
                {"ungrounded_platforms": ungrounded},
            )
        return RouteDecision(
            True,
            "The user explicitly requested the selected commerce platforms.",
            {"platforms": list(parameters.platforms)},
        )

    def _selected_needs(
        self,
        intent: IntentItem,
        task: AgentTaskProposal,
    ) -> list:
        selected = set(task.parameters.product_need_ids)
        return [
            need
            for need in intent.product_needs
            if not selected or need.need_id in selected
        ]

    def _intent(self, plan: IntentPlanV3, intent_id: str) -> IntentItem | None:
        return next((item for item in plan.intents if item.intent_id == intent_id), None)

    def _grounded(self, query: str, fragment: str) -> bool:
        normalized = self._normalized(fragment)
        return bool(normalized) and normalized in self._normalized(query)

    def _normalized(self, value: str) -> str:
        return re.sub(r"[\W_]+", "", str(value or "").lower(), flags=re.UNICODE)
