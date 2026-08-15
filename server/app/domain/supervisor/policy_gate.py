from __future__ import annotations

from pydantic import BaseModel, Field

from app.domain.supervisor.agent_registry import AgentRegistry
from app.domain.supervisor.capability_catalog import CapabilityCatalog, build_default_capability_catalog
from app.domain.supervisor.route_policy import RoutePolicy
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
    approved_research_request_ids: list[str] = Field(default_factory=list)
    research_selections: dict[str, str] = Field(default_factory=dict)
    core_proposal_ids: list[str] = Field(default_factory=list)
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
    verification_goal: str = ""
    approved_query: str = ""


class SupervisorPolicyGate:
    """Deterministically approves planner proposals against local policy."""

    def __init__(
        self,
        registry: AgentRegistry,
        catalog: CapabilityCatalog | None = None,
        route_policy: RoutePolicy | None = None,
    ) -> None:
        self.registry = registry
        self.catalog = catalog or build_default_capability_catalog()
        self.route_policy = route_policy or RoutePolicy()

    def evaluate(self, plan: IntentPlan) -> PolicyEvaluation:
        try:
            validate_intent_plan_contract(plan, self.catalog)
        except IntentPlanContractError as exc:
            return PolicyEvaluation(approved=False, errors=exc.errors)

        proposal_decisions: dict[str, PolicyDecision] = {}
        selections: dict[str, str] = {}
        for proposal in plan.agent_proposals:
            decision = self._evaluate_proposal(plan, proposal)
            proposal_decisions[proposal.proposal_id] = decision
            if decision.approved:
                selections[proposal.proposal_id] = decision.selected_agent_id

        approved_research_request_ids: list[str] = []
        research_selections: dict[str, str] = {}
        research_decisions: dict[str, PolicyDecision] = {}
        for request in plan.research_requests:
            decision = self._evaluate_research_request(plan, request, selections)
            research_decisions[request.request_id] = decision
            if decision.approved:
                approved_research_request_ids.append(request.request_id)
                research_selections[request.request_id] = decision.selected_agent_id

        # CommerceResearchAgent exists only to execute approved external requests.
        # A hallucinated or rejected platform request must not create an empty
        # Agent node merely because the Agent itself is registered.
        approved_request_set = set(approved_research_request_ids)
        for proposal in plan.agent_proposals:
            if proposal.capability != "commerce_research" or proposal.proposal_id not in selections:
                continue
            covered = any(
                request.request_id in approved_request_set
                and request.consumer_capability == proposal.capability
                and request.intent_id in proposal.intent_ids
                for request in plan.research_requests
            )
            if covered:
                continue
            selections.pop(proposal.proposal_id, None)
            proposal_decisions[proposal.proposal_id] = self._reject(
                proposal,
                "No approved external research request is assigned to this Agent.",
                decision="reject_no_approved_research",
            )

        # A hard dependency can never disappear from the compiled graph. If its
        # proposal is rejected, reject only the dependent branch and propagate
        # that result; unrelated branches remain independently executable.
        changed = True
        while changed:
            changed = False
            for proposal in plan.agent_proposals:
                if proposal.proposal_id not in selections:
                    continue
                unavailable = [
                    dependency_id
                    for dependency_id in proposal.depends_on
                    if dependency_id not in selections
                ]
                if not unavailable:
                    continue
                selections.pop(proposal.proposal_id, None)
                proposal_decisions[proposal.proposal_id] = self._reject(
                    proposal,
                    "A hard dependency was rejected by policy.",
                    decision="reject_hard_dependency",
                    details={"unavailable_dependency_ids": unavailable},
                )
                changed = True

        # Research approval cannot outlive the Agent branch that owns it.
        for request in plan.research_requests:
            if request.request_id not in approved_request_set:
                continue
            consumers = [
                proposal
                for proposal in plan.agent_proposals
                if proposal.proposal_id in selections
                and proposal.capability == request.consumer_capability
                and request.intent_id in proposal.intent_ids
            ]
            if consumers:
                continue
            approved_request_set.remove(request.request_id)
            approved_research_request_ids.remove(request.request_id)
            research_selections.pop(request.request_id, None)
            research_decisions[request.request_id] = PolicyDecision(
                subject_id=request.request_id,
                decision="reject_consumer_branch",
                approved=False,
                capability="external_research",
                reason="The Agent branch assigned to this research request was rejected.",
                details={
                    "consumer_capability": request.consumer_capability,
                    "trigger_type": request.trigger_type,
                },
            )

        core_proposal_ids = self.route_policy.core_proposal_ids(plan)
        core_rejections = [
            proposal_id for proposal_id in core_proposal_ids if proposal_id not in selections
        ]
        errors: list[str] = []
        if plan.execution_mode != "direct" and not core_proposal_ids:
            errors.append("no core proposal satisfies the execution route")
        if core_rejections:
            errors.append(f"core proposals rejected: {sorted(core_rejections)}")
        core_research_ids = [
            request.request_id
            for request in plan.research_requests
            if self.route_policy.is_core_research_request(plan, request)
        ]
        if self.route_policy.is_knowledge_bridge_plan(plan) and not core_research_ids:
            errors.append("knowledge bridge requires an approved web research request")
        for request_id in core_research_ids:
            if request_id not in approved_request_set:
                errors.append(f"core research request rejected: {request_id}")

        decisions = [
            proposal_decisions[proposal.proposal_id]
            for proposal in plan.agent_proposals
        ] + [
            research_decisions[request.request_id]
            for request in plan.research_requests
        ]
        return PolicyEvaluation(
            approved=not errors,
            decisions=decisions,
            selections=selections,
            approved_research_request_ids=approved_research_request_ids,
            research_selections=research_selections,
            core_proposal_ids=core_proposal_ids,
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
            concept_type = str(proposal.get("concept_type") or "").strip()
            references = self._strings(proposal.get("supporting_evidence_ids", []), limit=12)
            missing = [item for item in references if item not in evidence_by_id]
            invalid_sources = [
                item
                for item in references
                if item in evidence_by_id
                and str(getattr(evidence_by_id[item], "source_type", ""))
                not in {"web_search", "commerce_mcp"}
            ]
            unsupported = [
                item
                for item in references
                if item in evidence_by_id
                and not self._evidence_mentions(evidence_by_id[item], concept)
            ]
            rejection_reason = ""
            if len(concept) < 2 or len(concept) > 48:
                rejection_reason = "商品概念长度不在受控范围内。"
            elif concept_type != "product_type":
                rejection_reason = (
                    f"概念类型 {concept_type or 'unknown'} 不是可直接检索的商品品类。"
                )
            elif not references:
                rejection_reason = "没有提供逐项支撑证据。"
            elif missing:
                rejection_reason = f"引用了不存在的证据：{missing}。"
            elif invalid_sources:
                rejection_reason = f"支撑证据来源不能驱动商品检索：{invalid_sources}。"
            elif unsupported:
                rejection_reason = f"引用证据正文没有明确出现该商品概念：{unsupported}。"
            if rejection_reason:
                rejected.append(
                    {
                        "concept": concept,
                        "concept_type": concept_type or "unknown",
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

        # Retrieval uses only Supervisor-approved product types. The original
        # vague efficacy/symptom goal remains available to the Verifier through
        # original_query and must not pollute BM25/vector queries.
        approved_query = " ".join(approved)[:300]
        return KnowledgeFollowUpDecision(
            decision="approve_knowledge_to_retrieval",
            approved=True,
            reason="商品概念具有逐项外部证据引用，允许追加一次受控的本地商品检索。",
            candidate_concepts=candidate_concepts,
            approved_concepts=approved,
            rejected_concepts=rejected,
            supporting_evidence_ids=supporting_ids,
            risk_flags=risks,
            verification_goal=plan.original_query or plan.normalized_query,
            approved_query=approved_query,
        )

    def _evidence_mentions(self, ref: object, concept: str) -> bool:
        normalized_concept = "".join(concept.lower().split())
        if not normalized_concept:
            return False
        evidence_text = " ".join(
            str(getattr(ref, field, "") or "")
            for field in ("claim", "summary")
        )
        normalized_evidence = "".join(evidence_text.lower().split())
        return normalized_concept in normalized_evidence

    def _strings(self, values: object, *, limit: int) -> list[str]:
        if not isinstance(values, list):
            return []
        return list(
            dict.fromkeys(str(item).strip() for item in values if str(item).strip())
        )[:limit]

    def _evaluate_proposal(self, plan: IntentPlan, proposal: AgentTaskProposal) -> PolicyDecision:
        route_decision = self.route_policy.evaluate_proposal(plan, proposal)
        if not route_decision.approved:
            return self._reject(
                proposal,
                route_decision.reason,
                decision="reject_route_policy",
                details=route_decision.details,
            )
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
                "planner_required": proposal.required,
                "effective_core": self.route_policy.is_core_proposal(plan, proposal),
                "route_policy": route_decision.details,
            },
        )

    def _evaluate_research_request(
        self,
        plan: IntentPlan,
        request: ResearchRequest,
        selections: dict[str, str],
    ) -> PolicyDecision:
        route_decision = self.route_policy.evaluate_research_request(plan, request)
        if not route_decision.approved:
            return PolicyDecision(
                subject_id=request.request_id,
                decision="reject_research_route_policy",
                approved=False,
                capability="external_research",
                reason=route_decision.reason,
                details={
                    **route_decision.details,
                    "mode": request.mode,
                    "platforms": list(request.platforms),
                    "trigger_type": request.trigger_type,
                    "trigger_text": request.trigger_text,
                    "planner_required": request.required,
                    "effective_core": self.route_policy.is_core_research_request(plan, request),
                },
            )
        required_tool = "web_search" if request.mode == "web_general" else "commerce_search"
        selected_agent_ids: list[str] = []
        for proposal in plan.agent_proposals:
            if proposal.capability != request.consumer_capability:
                continue
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
                "trigger_type": request.trigger_type,
                "trigger_text": request.trigger_text,
                "consumer_capability": request.consumer_capability,
                "planner_required": request.required,
                "effective_core": self.route_policy.is_core_research_request(plan, request),
                "required_tool": required_tool,
                "eligible_agent_ids": selected_agent_ids,
                "route_policy": route_decision.details,
            },
        )

    def _reject(
        self,
        proposal: AgentTaskProposal,
        reason: str,
        *,
        decision: str = "reject_agent_proposal",
        details: dict | None = None,
    ) -> PolicyDecision:
        return PolicyDecision(
            subject_id=proposal.proposal_id,
            decision=decision,
            approved=False,
            capability=proposal.capability,
            reason=reason,
            details={"planner_required": proposal.required, **(details or {})},
        )
