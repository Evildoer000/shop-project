from __future__ import annotations

from collections.abc import Iterable

from app.domain.supervisor.capability_catalog import CapabilityCatalog, build_default_capability_catalog
from app.schemas import IntentPlan


COMMERCE_RESEARCH_PLATFORMS = frozenset({"taobao", "douyin_ec", "xiaohongshu"})
RESEARCH_TRIGGER_NEEDS = {
    "explicit_web_request": "explicit_web",
    "explicit_platform_request": "explicit_platform",
    "freshness_required": "freshness_required",
    "knowledge_bridge": "knowledge_bridge",
}


class IntentPlanContractError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("Invalid IntentPlan contract: " + "; ".join(errors))


def validate_intent_plan_contract(
    plan: IntentPlan,
    catalog: CapabilityCatalog | None = None,
) -> None:
    resolved_catalog = catalog or build_default_capability_catalog()
    errors: list[str] = []

    if plan.schema_version != "2.0":
        errors.append("schema_version must be 2.0")
    if not plan.normalized_query.strip() and not plan.original_query.strip():
        errors.append("normalized_query or original_query is required")
    if not plan.input_modalities:
        errors.append("input_modalities cannot be empty")
    if len(plan.input_modalities) != len(set(plan.input_modalities)):
        errors.append("input_modalities cannot contain duplicates")

    intent_ids = [intent.intent_id.strip() for intent in plan.intents]
    if not intent_ids:
        errors.append("intents cannot be empty")
    _validate_unique_non_empty("intent_id", intent_ids, errors)
    intent_id_set = set(intent_ids)
    intent_types = {intent.intent_type for intent in plan.intents}
    if plan.intents and plan.primary_intent not in intent_types:
        errors.append("primary_intent must match one item in intents")
    intent_dependencies = {intent.intent_id: intent.depends_on for intent in plan.intents}
    _validate_dependencies("intent", intent_dependencies, intent_id_set, errors)

    if plan.constraints.budget_min is not None and plan.constraints.budget_min < 0:
        errors.append("constraints.budget_min must be non-negative")
    if plan.constraints.budget_max is not None and plan.constraints.budget_max < 0:
        errors.append("constraints.budget_max must be non-negative")
    if (
        plan.constraints.budget_min is not None
        and plan.constraints.budget_max is not None
        and plan.constraints.budget_min > plan.constraints.budget_max
    ):
        errors.append("constraints.budget_min cannot exceed constraints.budget_max")
    if any(item.source == "long_term_profile" and item.strength == "hard" for item in plan.constraints.items):
        errors.append("long_term_profile constraints must be soft")

    slot_ids = [slot.slot_id.strip() for slot in plan.need_slots]
    _validate_unique_non_empty("slot_id", slot_ids, errors)
    for slot in plan.need_slots:
        if slot.intent_id and slot.intent_id not in intent_id_set:
            errors.append(f"slot {slot.slot_id} references unknown intent_id {slot.intent_id}")
        if not any([slot.query.strip(), slot.semantic_query.strip(), slot.keyword_query.strip()]):
            errors.append(f"slot {slot.slot_id} requires a retrieval query")
    if plan.execution_mode == "multi_product" and not plan.need_slots:
        errors.append("multi_product execution requires need_slots")

    referenced_ids = set(plan.referenced_product_ids)
    for intent in plan.intents:
        referenced_ids.update(intent.referenced_product_ids)
    if (
        plan.execution_mode == "context_evidence"
        and not referenced_ids
        and not any(proposal.capability == "knowledge_research" for proposal in plan.agent_proposals)
    ):
        errors.append("context_evidence execution requires referenced products or knowledge_research")
    if plan.execution_mode == "clarify":
        if not plan.clarification.required or not plan.clarification.blocking:
            errors.append("clarify execution requires a blocking clarification proposal")
    if plan.clarification.blocking and plan.execution_mode != "clarify":
        errors.append("blocking clarification requires execution_mode=clarify")

    context_request_ids = [request.request_id.strip() for request in plan.context_requests]
    _validate_unique_non_empty("context request_id", context_request_ids, errors)
    if any(request.context_type == "long_term_profile" and not request.reason.strip() for request in plan.context_requests):
        errors.append("long_term profile requests require a reason")

    research_request_ids = [request.request_id.strip() for request in plan.research_requests]
    _validate_unique_non_empty("research request_id", research_request_ids, errors)
    intents_by_id = {intent.intent_id: intent for intent in plan.intents}
    for request in plan.research_requests:
        if request.intent_id not in intent_id_set:
            errors.append(f"research request {request.request_id} references unknown intent_id {request.intent_id}")
            continue
        intent = intents_by_id[request.intent_id]
        if not request.trigger_text.strip():
            errors.append(f"research request {request.request_id} requires trigger_text")
        expected_need = RESEARCH_TRIGGER_NEEDS[request.trigger_type]
        if intent.route_basis.external_information_need != expected_need:
            errors.append(
                f"research request {request.request_id} trigger_type does not match intent route_basis"
            )
        if request.mode == "web_general" and request.consumer_capability == "commerce_research":
            errors.append(
                f"research request {request.request_id} web_general cannot be consumed by commerce_research"
            )
        if request.mode in {"marketplace", "social_content"} and request.consumer_capability != "commerce_research":
            errors.append(
                f"research request {request.request_id} commerce mode requires consumer_capability=commerce_research"
            )
        if request.trigger_type == "explicit_platform_request" and request.mode not in {
            "marketplace",
            "social_content",
        }:
            errors.append(
                f"research request {request.request_id} explicit platform trigger requires commerce mode"
            )
        if request.trigger_type != "explicit_platform_request" and request.mode in {
            "marketplace",
            "social_content",
        }:
            errors.append(
                f"research request {request.request_id} commerce mode requires explicit_platform_request"
            )
        if request.trigger_type == "knowledge_bridge" and not request.local_catalog_gap.strip():
            errors.append(f"research request {request.request_id} knowledge_bridge requires local_catalog_gap")
        if request.mode in {"marketplace", "social_content"} and not request.platforms:
            errors.append(f"research request {request.request_id} requires at least one platform")
        if request.mode in {"marketplace", "social_content"}:
            unsupported = sorted(set(request.platforms) - COMMERCE_RESEARCH_PLATFORMS)
            if unsupported:
                errors.append(
                    f"research request {request.request_id} uses unsupported commerce platforms {unsupported}"
                )

    proposal_ids = [proposal.proposal_id.strip() for proposal in plan.agent_proposals]
    _validate_unique_non_empty("proposal_id", proposal_ids, errors)
    proposal_capabilities = [proposal.capability for proposal in plan.agent_proposals]
    duplicate_capabilities = sorted(
        {
            capability
            for capability in proposal_capabilities
            if proposal_capabilities.count(capability) > 1
        }
    )
    if duplicate_capabilities:
        errors.append(f"duplicate proposal capabilities: {duplicate_capabilities}")
    proposal_id_set = set(proposal_ids)
    proposal_dependencies = {proposal.proposal_id: proposal.depends_on for proposal in plan.agent_proposals}
    _validate_dependencies("proposal", proposal_dependencies, proposal_id_set, errors)
    optional_dependencies = {
        proposal.proposal_id: proposal.optional_context_from
        for proposal in plan.agent_proposals
    }
    _validate_dependency_references(
        "proposal optional_context_from",
        optional_dependencies,
        proposal_id_set,
        errors,
    )
    for proposal in plan.agent_proposals:
        definition = resolved_catalog.get(proposal.capability)
        if definition is None:
            errors.append(f"proposal {proposal.proposal_id} uses unknown capability {proposal.capability}")
        elif not definition.planner_proposable:
            errors.append(
                f"proposal {proposal.proposal_id} cannot request Supervisor-managed capability {proposal.capability}"
            )
        unknown_intents = set(proposal.intent_ids) - intent_id_set
        if unknown_intents:
            errors.append(
                f"proposal {proposal.proposal_id} references unknown intent_ids {sorted(unknown_intents)}"
            )
        overlap = set(proposal.depends_on).intersection(proposal.optional_context_from)
        if overlap:
            errors.append(
                f"proposal {proposal.proposal_id} cannot use the same dependency as hard and optional: {sorted(overlap)}"
            )

    proposals_by_capability = {
        proposal.capability: proposal for proposal in plan.agent_proposals
    }
    for request in plan.research_requests:
        consumer = proposals_by_capability.get(request.consumer_capability)
        if consumer is None or request.intent_id not in consumer.intent_ids:
            errors.append(
                f"research request {request.request_id} has no matching consumer proposal"
            )

    proposed_capabilities = {proposal.capability for proposal in plan.agent_proposals}
    expected_capability = {
        "clarify": "clarification",
        "single_product": "single_product_recommendation",
        "multi_product": "multi_product_bundle",
    }.get(plan.execution_mode)
    if expected_capability and expected_capability not in proposed_capabilities:
        errors.append(f"execution_mode={plan.execution_mode} requires proposal capability {expected_capability}")
    if "profile_preference" in proposed_capabilities and not any(
        request.context_type == "long_term_profile" for request in plan.context_requests
    ):
        errors.append("profile_preference proposal requires a long_term_profile context request")

    if errors:
        raise IntentPlanContractError(errors)


def _validate_unique_non_empty(label: str, values: list[str], errors: list[str]) -> None:
    if any(not value for value in values):
        errors.append(f"{label} cannot be empty")
    duplicates = sorted({value for value in values if value and values.count(value) > 1})
    if duplicates:
        errors.append(f"duplicate {label}: {duplicates}")


def _validate_dependencies(
    label: str,
    dependencies: dict[str, list[str]],
    known_ids: set[str],
    errors: list[str],
) -> None:
    for node_id, dependency_ids in dependencies.items():
        unknown = set(dependency_ids) - known_ids
        if unknown:
            errors.append(f"{label} {node_id} has unknown dependencies {sorted(unknown)}")
        if node_id in dependency_ids:
            errors.append(f"{label} {node_id} cannot depend on itself")
    cycle = _find_cycle(dependencies)
    if cycle:
        errors.append(f"{label} dependency cycle detected: {' -> '.join(cycle)}")


def _validate_dependency_references(
    label: str,
    dependencies: dict[str, list[str]],
    known_ids: set[str],
    errors: list[str],
) -> None:
    for node_id, dependency_ids in dependencies.items():
        unknown = set(dependency_ids) - known_ids
        if unknown:
            errors.append(f"{label} {node_id} has unknown references {sorted(unknown)}")
        if node_id in dependency_ids:
            errors.append(f"{label} {node_id} cannot reference itself")


def _find_cycle(dependencies: dict[str, Iterable[str]]) -> list[str]:
    visiting: list[str] = []
    visited: set[str] = set()

    def visit(node_id: str) -> list[str]:
        if node_id in visiting:
            start = visiting.index(node_id)
            return visiting[start:] + [node_id]
        if node_id in visited:
            return []
        visiting.append(node_id)
        for dependency_id in dependencies.get(node_id, []):
            if dependency_id not in dependencies:
                continue
            cycle = visit(dependency_id)
            if cycle:
                return cycle
        visiting.pop()
        visited.add(node_id)
        return []

    for candidate in dependencies:
        cycle = visit(candidate)
        if cycle:
            return cycle
    return []
