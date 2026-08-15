from __future__ import annotations

from dataclasses import dataclass, field
import re

from app.domain.retrieval_plan_builder import RetrievalPlanBuilder
from app.schemas import AgentTaskProposal, IntentItem, IntentPlan, ResearchRequest


EXTERNAL_TRIGGER_TYPES = frozenset(
    {
        "explicit_web_request",
        "explicit_platform_request",
        "freshness_required",
        "knowledge_bridge",
    }
)

_WEB_SIGNALS = (
    "联网",
    "上网",
    "网上",
    "网络搜索",
    "搜网页",
    "查资料",
    "查一下资料",
    "搜索资料",
    "找资料",
)
_FRESHNESS_SIGNALS = (
    "最新",
    "近期",
    "最近",
    "实时",
    "今天",
    "今日",
    "本周",
    "本月",
    "今年",
    "刚发布",
    "新发布",
)
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
    """Deterministic business routing rules applied after the LLM proposal."""

    def evaluate_proposal(
        self,
        plan: IntentPlan,
        proposal: AgentTaskProposal,
    ) -> RouteDecision:
        intents = self._proposal_intents(plan, proposal)
        intent_types = {intent.intent_type for intent in intents}
        capability = proposal.capability

        if not intents and capability != "profile_preference":
            return RouteDecision(False, "Proposal does not reference an approved business intent.")
        if capability == "profile_preference":
            approved = any(request.context_type == "long_term_profile" for request in plan.context_requests)
            return RouteDecision(
                approved,
                "Long-term profile context was explicitly requested."
                if approved
                else "No long-term profile context request exists.",
            )
        if capability == "clarification":
            approved = plan.execution_mode == "clarify" and plan.clarification.blocking
            return RouteDecision(
                approved,
                "A blocking clarification is required."
                if approved
                else "Clarification is only allowed for a blocking clarify route.",
            )
        if capability == "single_product_recommendation":
            approved = plan.execution_mode == "single_product" and "product_recommendation" in intent_types
            return RouteDecision(
                approved,
                "The plan has one executable product target."
                if approved
                else "Single-product retrieval requires a single_product recommendation intent.",
            )
        if capability == "multi_product_bundle":
            approved = (
                plan.execution_mode == "multi_product"
                and "product_recommendation" in intent_types
                and bool(plan.need_slots)
            )
            return RouteDecision(
                approved,
                "The plan has an executable multi-slot product target."
                if approved
                else "Multi-product routing requires product slots and a recommendation intent.",
            )
        if capability == "comparison":
            approved = "product_comparison" in intent_types
            return RouteDecision(
                approved,
                "The referenced intent explicitly requests product comparison."
                if approved
                else "Comparison cannot be inferred without a product_comparison intent.",
            )
        if capability == "knowledge_research":
            approved = bool(intent_types.intersection({"product_qa", "shopping_knowledge"})) or (
                self.is_knowledge_bridge_plan(plan)
                and any(
                    intent.route_basis.external_information_need == "knowledge_bridge"
                    for intent in intents
                )
            )
            return RouteDecision(
                approved,
                "The referenced intent requests product facts, shopping knowledge, or an approved knowledge bridge."
                if approved
                else "Knowledge research requires a product_qa or shopping_knowledge intent.",
            )
        if capability == "commerce_research":
            approved = any(
                request.intent_id in proposal.intent_ids
                and request.mode in {"marketplace", "social_content"}
                for request in plan.research_requests
            )
            return RouteDecision(
                approved,
                "A matching marketplace or social-content request exists."
                if approved
                else "Commerce research requires an explicit platform research request.",
            )
        return RouteDecision(True, "No additional business routing restriction applies.")

    def evaluate_research_request(
        self,
        plan: IntentPlan,
        request: ResearchRequest,
    ) -> RouteDecision:
        intent = next((item for item in plan.intents if item.intent_id == request.intent_id), None)
        if intent is None:
            return RouteDecision(False, "Research request references an unknown intent.")
        if request.trigger_type not in EXTERNAL_TRIGGER_TYPES:
            return RouteDecision(False, "External research trigger_type is not allowlisted.")
        if not self._trigger_is_grounded(plan, request.trigger_text):
            return RouteDecision(
                False,
                "External research trigger_text is not a verbatim fragment of the current query.",
            )

        route_basis = intent.route_basis
        expected_need = {
            "explicit_web_request": "explicit_web",
            "explicit_platform_request": "explicit_platform",
            "freshness_required": "freshness_required",
            "knowledge_bridge": "knowledge_bridge",
        }[request.trigger_type]
        if route_basis.external_information_need != expected_need:
            return RouteDecision(
                False,
                "Intent route_basis does not support the proposed external research trigger.",
                {
                    "expected_external_information_need": expected_need,
                    "actual_external_information_need": route_basis.external_information_need,
                },
            )

        if request.trigger_type == "explicit_web_request":
            if request.mode != "web_general" or not self._contains_any(plan.original_query, _WEB_SIGNALS):
                return RouteDecision(False, "The current query does not explicitly request web research.")
        elif request.trigger_type == "explicit_platform_request":
            decision = self._evaluate_platform_trigger(plan, request)
            if not decision.approved:
                return decision
        elif request.trigger_type == "freshness_required":
            if request.mode != "web_general" or not self._contains_any(plan.original_query, _FRESHNESS_SIGNALS):
                return RouteDecision(False, "The current query has no explicit freshness requirement.")
        elif request.trigger_type == "knowledge_bridge":
            decision = self._evaluate_knowledge_bridge(plan, intent, request)
            if not decision.approved:
                return decision

        return RouteDecision(
            True,
            "External research has a structured, query-grounded routing trigger.",
            {
                "trigger_type": request.trigger_type,
                "trigger_text": request.trigger_text,
                "mode": request.mode,
                "platforms": list(request.platforms),
            },
        )

    def is_core_proposal(self, plan: IntentPlan, proposal: AgentTaskProposal) -> bool:
        capability = proposal.capability
        if plan.execution_mode == "clarify":
            return capability == "clarification"
        if plan.execution_mode == "single_product":
            return capability == "single_product_recommendation"
        if plan.execution_mode == "multi_product":
            return capability == "multi_product_bundle"
        if plan.execution_mode != "context_evidence":
            return False

        if capability == "comparison" and plan.primary_intent == "product_comparison":
            return True
        if capability == "knowledge_research" and plan.primary_intent in {"product_qa", "shopping_knowledge"}:
            return True
        if capability == "knowledge_research" and self.is_knowledge_bridge_plan(plan):
            return True
        return False

    def core_proposal_ids(self, plan: IntentPlan) -> list[str]:
        return [
            proposal.proposal_id
            for proposal in plan.agent_proposals
            if self.is_core_proposal(plan, proposal)
        ]

    def is_core_research_request(self, plan: IntentPlan, request: ResearchRequest) -> bool:
        return request.trigger_type == "knowledge_bridge" and self.is_knowledge_bridge_plan(plan)

    def is_knowledge_bridge_plan(self, plan: IntentPlan) -> bool:
        if (
            plan.execution_mode != "context_evidence"
            or self._referenced_product_ids(plan)
            or self._explicit_product_family(plan.original_query)
        ):
            return False
        return any(
            intent.route_basis.target_clarity == "vague_effect_or_use"
            and intent.route_basis.local_catalog_status == "insufficient"
            and intent.route_basis.external_information_need == "knowledge_bridge"
            and not intent.route_basis.product_family.strip()
            for intent in plan.intents
            if intent.intent_type in {"product_recommendation", "shopping_knowledge"}
        )

    def _evaluate_platform_trigger(
        self,
        plan: IntentPlan,
        request: ResearchRequest,
    ) -> RouteDecision:
        if request.mode not in {"marketplace", "social_content"} or not request.platforms:
            return RouteDecision(False, "Platform research requires marketplace/social mode and platforms.")
        query = self._normalized(plan.original_query)
        ungrounded = [
            platform
            for platform in request.platforms
            if not any(self._normalized(alias) in query for alias in _PLATFORM_ALIASES.get(platform, ()))
        ]
        if ungrounded:
            return RouteDecision(
                False,
                "One or more proposed platforms were not named in the current query.",
                {"ungrounded_platforms": ungrounded},
            )
        return RouteDecision(True, "All requested platforms are grounded in the current query.")

    def _evaluate_knowledge_bridge(
        self,
        plan: IntentPlan,
        intent: IntentItem,
        request: ResearchRequest,
    ) -> RouteDecision:
        basis = intent.route_basis
        errors: list[str] = []
        if request.mode != "web_general":
            errors.append("knowledge bridge only supports web_general")
        if plan.execution_mode != "context_evidence":
            errors.append("execution_mode is not context_evidence")
        if basis.target_clarity != "vague_effect_or_use":
            errors.append("target is not marked vague_effect_or_use")
        if basis.local_catalog_status != "insufficient":
            errors.append("local catalog is not marked insufficient")
        if basis.product_family.strip():
            errors.append("a concrete product family is already available")
        explicit_family = self._explicit_product_family(plan.original_query)
        if explicit_family:
            errors.append(f"current query already contains concrete product family: {explicit_family}")
        if not request.local_catalog_gap.strip():
            errors.append("local_catalog_gap is missing")
        if errors:
            return RouteDecision(
                False,
                "Knowledge bridge prerequisites are not satisfied.",
                {"errors": errors},
            )
        return RouteDecision(True, "A vague goal requires an evidence-backed product-family bridge.")

    def _proposal_intents(
        self,
        plan: IntentPlan,
        proposal: AgentTaskProposal,
    ) -> list[IntentItem]:
        selected = set(proposal.intent_ids)
        return [intent for intent in plan.intents if intent.intent_id in selected]

    def _referenced_product_ids(self, plan: IntentPlan) -> set[str]:
        result = set(plan.referenced_product_ids)
        for intent in plan.intents:
            result.update(intent.referenced_product_ids)
        return result

    def _trigger_is_grounded(self, plan: IntentPlan, trigger_text: str) -> bool:
        trigger = self._normalized(trigger_text)
        if not trigger:
            return False
        return trigger in self._normalized(plan.original_query)

    def _contains_any(self, value: str, signals: tuple[str, ...]) -> bool:
        normalized = self._normalized(value)
        return any(self._normalized(signal) in normalized for signal in signals)

    def _explicit_product_family(self, value: str) -> str:
        normalized = self._normalized(value)
        matches: list[tuple[int, str]] = []
        for category, keywords in RetrievalPlanBuilder.CATEGORY_KEYWORDS.items():
            for term in (category, *keywords):
                candidate = self._normalized(term)
                if candidate and candidate in normalized:
                    matches.append((len(candidate), category))
        for category in RetrievalPlanBuilder.TOP_LEVEL_CATEGORIES:
            candidate = self._normalized(category)
            if candidate and candidate in normalized:
                matches.append((len(candidate), category))
        return max(matches, default=(0, ""))[1]

    def _normalized(self, value: str) -> str:
        return re.sub(r"[\W_]+", "", str(value or "").lower(), flags=re.UNICODE)
