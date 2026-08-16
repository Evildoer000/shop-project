from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.domain.supervisor.agent_registry import AgentRegistry
from app.domain.supervisor.capability_catalog import (
    CapabilityCatalog,
    build_default_capability_catalog,
)
from app.domain.supervisor.route_policy import RoutePolicy
from app.domain.supervisor.validators import (
    IntentPlanContractError,
    validate_intent_plan_contract,
)
from app.schemas import AgentTaskProposal, IntentPlanV3


class PolicyDecision(BaseModel):
    subject_id: str
    decision: str
    approved: bool
    reason: str
    capability: str = ""
    selected_agent_id: str = ""
    details: dict = Field(default_factory=dict)


class IntentPolicyStatus(BaseModel):
    intent_id: str
    status: Literal[
        "ready",
        "awaiting_evidence",
        "needs_clarification",
        "unsupported",
    ]
    approved_task_ids: list[str] = Field(default_factory=list)
    rejected_task_ids: list[str] = Field(default_factory=list)
    reason: str = ""


class PolicyEvaluation(BaseModel):
    approved: bool
    decisions: list[PolicyDecision] = Field(default_factory=list)
    selections: dict[str, str] = Field(default_factory=dict)
    required_task_ids: list[str] = Field(default_factory=list)
    rejected_task_ids: list[str] = Field(default_factory=list)
    intent_statuses: list[IntentPolicyStatus] = Field(default_factory=list)
    uncovered_required_intent_ids: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class SupervisorPolicyGate:
    """Validate and approve each intent task independently."""

    def __init__(
        self,
        registry: AgentRegistry,
        catalog: CapabilityCatalog | None = None,
        route_policy: RoutePolicy | None = None,
    ) -> None:
        self.registry = registry
        self.catalog = catalog or build_default_capability_catalog()
        self.route_policy = route_policy or RoutePolicy()

    def evaluate(self, plan: IntentPlanV3) -> PolicyEvaluation:
        try:
            validate_intent_plan_contract(plan, self.catalog)
        except IntentPlanContractError as exc:
            return PolicyEvaluation(approved=False, errors=exc.errors)

        decisions: dict[str, PolicyDecision] = {}
        selections: dict[str, str] = {}
        for task in plan.task_proposals:
            decision = self._evaluate_task(plan, task)
            decisions[task.task_id] = decision
            if decision.approved:
                selections[task.task_id] = decision.selected_agent_id

        # Reject only hard descendants of rejected tasks. Independent roots and
        # optional consumers remain executable.
        changed = True
        while changed:
            changed = False
            for task in plan.task_proposals:
                if task.task_id not in selections:
                    continue
                unavailable = [
                    dependency_id
                    for dependency_id in task.depends_on
                    if dependency_id not in selections
                ]
                if not unavailable:
                    continue
                selections.pop(task.task_id, None)
                decisions[task.task_id] = self._reject(
                    task,
                    "A hard dependency was rejected; only this descendant branch is rejected.",
                    decision="reject_hard_dependency",
                    details={"unavailable_dependency_ids": unavailable},
                )
                changed = True

        rejected_task_ids = [
            task.task_id for task in plan.task_proposals if task.task_id not in selections
        ]
        intent_statuses = self._intent_statuses(plan, selections)
        uncovered_required = [
            status.intent_id
            for status in intent_statuses
            if status.status == "unsupported"
            and next(
                intent.priority for intent in plan.intents if intent.intent_id == status.intent_id
            )
            == "required"
        ]
        required_task_ids = [
            task.task_id
            for task in plan.task_proposals
            if task.task_id in selections
            and task.capability != "profile_preference"
            and any(
                intent.intent_id in task.intent_ids and intent.priority == "required"
                for intent in plan.intents
            )
        ]
        executable_required = [
            status
            for status in intent_statuses
            if next(
                intent.priority for intent in plan.intents if intent.intent_id == status.intent_id
            )
            == "required"
            and status.status != "unsupported"
        ]
        errors: list[str] = []
        if not executable_required:
            errors.append("no required intent has an executable task branch")

        return PolicyEvaluation(
            approved=not errors,
            decisions=[decisions[task.task_id] for task in plan.task_proposals],
            selections=selections,
            required_task_ids=required_task_ids,
            rejected_task_ids=rejected_task_ids,
            intent_statuses=intent_statuses,
            uncovered_required_intent_ids=uncovered_required,
            errors=errors,
        )

    def _evaluate_task(
        self,
        plan: IntentPlanV3,
        task: AgentTaskProposal,
    ) -> PolicyDecision:
        route = self.route_policy.evaluate_task(plan, task)
        if not route.approved:
            return self._reject(
                task,
                route.reason,
                decision="reject_route_policy",
                details=route.details,
            )
        definition = self.catalog.get(task.capability)
        if definition is None:
            return self._reject(task, "Capability is not registered.")
        if not definition.planner_proposable or definition.supervisor_managed:
            return self._reject(task, "Capability can only be inserted by the Supervisor.")
        registration = self.registry.select_for_capability(task.capability)
        if registration is None:
            return self._reject(
                task,
                f"No enabled Agent provides capability={task.capability}.",
            )
        return PolicyDecision(
            subject_id=task.task_id,
            decision="approve_agent_task",
            approved=True,
            capability=task.capability,
            selected_agent_id=registration.manifest.agent_id,
            reason="The task is intent-compatible and an enabled Agent provides the capability.",
            details={
                "intent_ids": list(task.intent_ids),
                "allowed_tools": list(registration.manifest.allowed_tools),
                "route_policy": route.details,
            },
        )

    def _intent_statuses(
        self,
        plan: IntentPlanV3,
        selections: dict[str, str],
    ) -> list[IntentPolicyStatus]:
        result: list[IntentPolicyStatus] = []
        for intent in plan.intents:
            approved = [
                task
                for task in plan.task_proposals
                if task.task_id in selections and intent.intent_id in task.intent_ids
            ]
            rejected = [
                task.task_id
                for task in plan.task_proposals
                if task.task_id not in selections and intent.intent_id in task.intent_ids
            ]
            capabilities = {task.capability for task in approved}
            if "clarification" in capabilities:
                status = "needs_clarification"
                reason = "This intent has a blocking clarification task."
            elif (
                capabilities == {"knowledge_research"}
                and intent.route_basis.external_information_need == "knowledge_bridge"
            ):
                status = "awaiting_evidence"
                reason = "Knowledge evidence must return before this intent is replanned."
            elif approved or intent.intent_type == "social_chat":
                status = "ready"
                reason = "This intent has an independently executable branch."
            elif self._is_context_only_recommendation(intent):
                status = "ready"
                reason = (
                    "This intent can be executed from referenced conversation "
                    "products without a new retrieval task."
                )
            else:
                status = "unsupported"
                reason = "No task for this intent passed policy."
            result.append(
                IntentPolicyStatus(
                    intent_id=intent.intent_id,
                    status=status,  # type: ignore[arg-type]
                    approved_task_ids=[task.task_id for task in approved],
                    rejected_task_ids=rejected,
                    reason=reason,
                )
            )
        return result

    def _is_context_only_recommendation(self, intent) -> bool:
        return bool(
            intent.intent_type == "product_recommendation"
            and intent.recommendation_policy.candidate_source == "context_only"
            and intent.referenced_product_ids
        )

    def _reject(
        self,
        task: AgentTaskProposal,
        reason: str,
        *,
        decision: str = "reject_agent_task",
        details: dict | None = None,
    ) -> PolicyDecision:
        return PolicyDecision(
            subject_id=task.task_id,
            decision=decision,
            approved=False,
            capability=task.capability,
            reason=reason,
            details=details or {},
        )
