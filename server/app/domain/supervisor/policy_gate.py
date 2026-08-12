from __future__ import annotations

from pydantic import BaseModel, Field

from app.domain.supervisor.agent_registry import AgentRegistry
from app.domain.supervisor.capability_catalog import CapabilityCatalog, build_default_capability_catalog
from app.domain.supervisor.validators import IntentPlanContractError, validate_intent_plan_contract
from app.schemas import AgentTaskProposal, IntentPlan, ResearchRequest


class PolicyDecision(BaseModel):
    subject_id: str
    decision: str
    approved: bool
    reason: str
    capability: str = ""
    selected_agent_id: str = ""
    details: dict = Field(default_factory=dict)


class PolicyEvaluation(BaseModel):
    approved: bool
    decisions: list[PolicyDecision] = Field(default_factory=list)
    selections: dict[str, str] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class KnowledgeFollowUpDecision(BaseModel):
    """Supervisor-owned approval for turning research into product retrieval."""

    decision: str
    approved: bool
    reason: str
    candidate_concepts: list[str] = Field(default_factory=list)
    approved_concepts: list[str] = Field(default_factory=list)
    rejected_concepts: list[dict] = Field(default_factory=list)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    approved_query: str = ""


class SupervisorPolicyGate:
    """Deterministically approves planner proposals against local policy."""

    def __init__(
        self,
        registry: AgentRegistry,
        catalog: CapabilityCatalog | None = None,
    ) -> None:
        self.registry = registry
        self.catalog = catalog or build_default_capability_catalog()

    def evaluate(self, plan: IntentPlan) -> PolicyEvaluation:
        try:
            validate_intent_plan_contract(plan, self.catalog)
        except IntentPlanContractError as exc:
            return PolicyEvaluation(approved=False, errors=exc.errors)

        decisions: list[PolicyDecision] = []
        selections: dict[str, str] = {}
        required_rejections: list[str] = []
        for proposal in plan.agent_proposals:
            decision = self._evaluate_proposal(plan, proposal)
            decisions.append(decision)
            if decision.approved:
                selections[proposal.proposal_id] = decision.selected_agent_id
            elif proposal.required:
                required_rejections.append(proposal.proposal_id)

        errors = []
        if required_rejections:
            errors.append(f"required proposals rejected: {sorted(required_rejections)}")
        for request in plan.research_requests:
            decision = self._evaluate_research_request(plan, request, selections)
            decisions.append(decision)
            if request.required and not decision.approved:
                errors.append(f"required research request rejected: {request.request_id}")
        return PolicyEvaluation(
            approved=not errors,
            decisions=decisions,
            selections=selections,
            errors=errors,
        )

    def approve_knowledge_follow_up(
        self,
        plan: IntentPlan,
        *,
        knowledge_artifact: dict,
        evidence_refs: list[object],
        existing_product_path: bool = False,
    ) -> KnowledgeFollowUpDecision:
        """Approve grounded concept proposals; KnowledgeResearchAgent cannot approve itself."""
        candidate_concepts = self._strings(knowledge_artifact.get("candidate_concepts", []), limit=12)
        risks = self._strings(knowledge_artifact.get("risks", []), limit=12)
        if not any(intent.intent_type == "product_recommendation" for intent in plan.intents):
            return KnowledgeFollowUpDecision(
                decision="not_applicable",
                approved=False,
                reason="当前计划没有商品推荐意图，知识结果只供解释，不追加商品检索。",
                candidate_concepts=candidate_concepts,
                risk_flags=risks,
            )
        if existing_product_path:
            return KnowledgeFollowUpDecision(
                decision="already_planned",
                approved=False,
                reason="原计划已经包含商品检索节点，不重复追加检索。",
                candidate_concepts=candidate_concepts,
                risk_flags=risks,
            )

        evidence_by_id = {
            str(getattr(ref, "evidence_id", "")): ref
            for ref in evidence_refs
            if str(getattr(ref, "evidence_id", ""))
        }
        approved: list[str] = []
        rejected: list[dict] = []
        supporting_ids: list[str] = []
        proposals = knowledge_artifact.get("concept_proposals") or []
        for proposal in proposals[:12]:
            if not isinstance(proposal, dict):
                continue
            concept = str(proposal.get("concept") or "").strip()
            references = self._strings(proposal.get("supporting_evidence_ids", []), limit=12)
            missing = [item for item in references if item not in evidence_by_id]
            invalid_sources = [
                item
                for item in references
                if item in evidence_by_id
                and str(getattr(evidence_by_id[item], "source_type", ""))
                not in {"web_search", "commerce_mcp"}
            ]
            rejection_reason = ""
            if len(concept) < 2 or len(concept) > 48:
                rejection_reason = "商品概念长度不在受控范围内。"
            elif not references:
                rejection_reason = "没有提供逐项支撑证据。"
            elif missing:
                rejection_reason = f"引用了不存在的证据：{missing}。"
            elif invalid_sources:
                rejection_reason = f"支撑证据来源不能驱动商品检索：{invalid_sources}。"
            if rejection_reason:
                rejected.append(
                    {
                        "concept": concept,
                        "reason": rejection_reason,
                        "supporting_evidence_ids": references,
                    }
                )
                continue
            approved.append(concept)
            supporting_ids.extend(references)
            if len(approved) >= 4:
                break

        approved = list(dict.fromkeys(approved))
        supporting_ids = list(dict.fromkeys(supporting_ids))
        if not approved:
            return KnowledgeFollowUpDecision(
                decision="reject_knowledge_to_retrieval",
                approved=False,
                reason="没有逐项由外部证据支撑的商品概念，停止追加检索。",
                candidate_concepts=candidate_concepts,
                rejected_concepts=rejected,
                risk_flags=risks,
            )

        query_parts = [plan.normalized_query or plan.original_query, *approved]
        approved_query = " ".join(item.strip() for item in query_parts if item.strip())[:300]
        return KnowledgeFollowUpDecision(
            decision="approve_knowledge_to_retrieval",
            approved=True,
            reason="商品概念具有逐项外部证据引用，允许追加一次受控的本地商品检索。",
            candidate_concepts=candidate_concepts,
            approved_concepts=approved,
            rejected_concepts=rejected,
            supporting_evidence_ids=supporting_ids,
            risk_flags=risks,
            approved_query=approved_query,
        )

    def _strings(self, values: object, *, limit: int) -> list[str]:
        if not isinstance(values, list):
            return []
        return list(
            dict.fromkeys(str(item).strip() for item in values if str(item).strip())
        )[:limit]

    def _evaluate_proposal(self, plan: IntentPlan, proposal: AgentTaskProposal) -> PolicyDecision:
        definition = self.catalog.get(proposal.capability)
        if definition is None:
            return self._reject(proposal, "Capability is not registered.")
        if not definition.planner_proposable or definition.supervisor_managed:
            return self._reject(proposal, "Capability can only be inserted by the Supervisor.")

        if proposal.capability == "profile_preference":
            profile_requests = [
                request for request in plan.context_requests if request.context_type == "long_term_profile"
            ]
            if not profile_requests:
                return self._reject(proposal, "No approved long-term profile context request exists.")

        registration = self.registry.select_for_capability(
            proposal.capability,
            execution_mode=plan.execution_mode,
        )
        if registration is None:
            return self._reject(
                proposal,
                f"No enabled Agent supports capability={proposal.capability} in mode={plan.execution_mode}.",
            )

        return PolicyDecision(
            subject_id=proposal.proposal_id,
            decision="approve_agent_proposal",
            approved=True,
            capability=proposal.capability,
            selected_agent_id=registration.manifest.agent_id,
            reason="Proposal is allowlisted and an enabled Agent satisfies the execution mode.",
            details={
                "allowed_tools": list(registration.manifest.allowed_tools),
                "required": proposal.required,
            },
        )

    def _evaluate_research_request(
        self,
        plan: IntentPlan,
        request: ResearchRequest,
        selections: dict[str, str],
    ) -> PolicyDecision:
        required_tool = "web_search" if request.mode == "web_general" else "commerce_search"
        selected_agent_ids: list[str] = []
        for proposal in plan.agent_proposals:
            selected_agent_id = selections.get(proposal.proposal_id)
            if not selected_agent_id or request.intent_id not in proposal.intent_ids:
                continue
            registration = self.registry.require(selected_agent_id)
            if required_tool in registration.manifest.allowed_tools:
                selected_agent_ids.append(selected_agent_id)

        approved = bool(selected_agent_ids)
        return PolicyDecision(
            subject_id=request.request_id,
            decision="approve_research_request" if approved else "reject_research_request",
            approved=approved,
            capability="external_research",
            selected_agent_id=selected_agent_ids[0] if selected_agent_ids else "",
            reason=(
                f"Research request is covered by allowlisted tool {required_tool}."
                if approved
                else f"No approved Agent for intent={request.intent_id} is allowlisted for {required_tool}."
            ),
            details={
                "mode": request.mode,
                "platforms": list(request.platforms),
                "required": request.required,
                "required_tool": required_tool,
                "eligible_agent_ids": selected_agent_ids,
            },
        )

    def _reject(self, proposal: AgentTaskProposal, reason: str) -> PolicyDecision:
        return PolicyDecision(
            subject_id=proposal.proposal_id,
            decision="reject_agent_proposal",
            approved=False,
            capability=proposal.capability,
            reason=reason,
            details={"required": proposal.required},
        )
