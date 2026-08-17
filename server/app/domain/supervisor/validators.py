from __future__ import annotations

from app.domain.supervisor.capability_catalog import (
    CapabilityCatalog,
    build_default_capability_catalog,
)
from app.schemas import IntentPlanV3


class IntentPlanContractError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("Invalid IntentPlan V3 contract: " + "; ".join(errors))


def validate_intent_plan_contract(
    plan: IntentPlanV3,
    catalog: CapabilityCatalog | None = None,
) -> None:
    resolved_catalog = catalog or build_default_capability_catalog()
    errors: list[str] = []

    if plan.schema_version != "3.0":
        errors.append("schema_version must be 3.0")
    if not plan.original_query.strip() and not plan.normalized_query.strip():
        errors.append("original_query or normalized_query is required")
    if not plan.intents:
        errors.append("intents cannot be empty")

    intent_ids = [item.intent_id.strip() for item in plan.intents]
    _validate_unique_non_empty("intent_id", intent_ids, errors)
    intent_by_id = {item.intent_id: item for item in plan.intents}

    need_owner: dict[str, str] = {}
    for intent in plan.intents:
        if not intent.goal.strip() or not intent.resolved_query.strip():
            errors.append(f"intent {intent.intent_id} requires goal and resolved_query")
        for constraint in intent.constraints:
            if constraint.source == "long_term_profile" and constraint.strength == "hard":
                errors.append(
                    f"intent {intent.intent_id} long_term_profile constraints must be soft"
                )
        for need in intent.product_needs:
            if not need.need_id.strip():
                errors.append(f"intent {intent.intent_id} has an empty product need_id")
                continue
            if need.need_id in need_owner:
                errors.append(
                    f"duplicate product need_id {need.need_id} in intents "
                    f"{need_owner[need.need_id]} and {intent.intent_id}"
                )
            need_owner[need.need_id] = intent.intent_id
            if not need.goal.strip():
                errors.append(f"product need {need.need_id} requires a goal")
            for constraint in need.constraints:
                if constraint.source == "long_term_profile" and constraint.strength == "hard":
                    errors.append(
                        f"product need {need.need_id} long_term_profile constraints must be soft"
                    )
        policy = intent.recommendation_policy
        if (
            intent.budget is not None
            and intent.budget.minimum is not None
            and intent.budget.maximum is not None
            and intent.budget.minimum > intent.budget.maximum
        ):
            errors.append(
                f"intent {intent.intent_id} budget minimum cannot exceed maximum"
            )
        if plan.references_resolved:
            unknown_references = sorted(
                set(intent.referenced_product_ids)
                - set(plan.trusted_context_product_ids)
            )
            if unknown_references:
                errors.append(
                    f"intent {intent.intent_id} references products outside the trusted "
                    f"user/session context: {unknown_references}"
                )
        if intent.intent_type == "product_recommendation":
            if (
                policy.candidate_source in {"context_only", "context_plus_new"}
                and not intent.referenced_product_ids
            ):
                errors.append(
                    f"recommendation intent {intent.intent_id} uses "
                    f"{policy.candidate_source} without referenced_product_ids"
                )
            if (
                policy.reference_policy != "none"
                and not intent.referenced_product_ids
            ):
                errors.append(
                    f"recommendation intent {intent.intent_id} uses "
                    f"{policy.reference_policy} without referenced_product_ids"
                )
            if (
                policy.requested_count is None
                and policy.count_mode != "unknown"
            ):
                errors.append(
                    f"recommendation intent {intent.intent_id} has count_mode "
                    "without requested_count"
                )

    task_ids = [item.task_id.strip() for item in plan.task_proposals]
    _validate_unique_non_empty("task_id", task_ids, errors)
    known_tasks = set(task_ids)
    known_intents = set(intent_ids)
    dependencies = {item.task_id: item.depends_on for item in plan.task_proposals}
    optional_dependencies = {
        item.task_id: item.optional_upstream_task_ids
        for item in plan.task_proposals
    }
    _validate_dependency_references(
        "task depends_on",
        dependencies,
        known_tasks,
        errors,
    )
    _validate_dependency_references(
        "task optional_upstream_task_ids",
        optional_dependencies,
        known_tasks,
        errors,
    )
    combined_dependencies = {
        task_id: list(
            dict.fromkeys(
                [
                    *dependencies.get(task_id, []),
                    *optional_dependencies.get(task_id, []),
                ]
            )
        )
        for task_id in task_ids
    }
    _validate_dependency_graph("task", combined_dependencies, errors)

    for task in plan.task_proposals:
        definition = resolved_catalog.get(task.capability)
        if definition is None:
            errors.append(f"task {task.task_id} uses unknown capability {task.capability}")
        elif not definition.planner_proposable:
            errors.append(
                f"task {task.task_id} cannot request Supervisor-managed capability {task.capability}"
            )
        if len(task.intent_ids) != 1 or task.intent_ids[0] != task.intent_id:
            errors.append(
                f"task {task.task_id} must belong to exactly one intent; split independent work into separate tasks"
            )
            continue
        intent_id = task.intent_id
        if intent_id not in known_intents:
            errors.append(f"task {task.task_id} references unknown intent_id {intent_id}")
            continue
        overlap = set(task.depends_on).intersection(task.optional_upstream_task_ids)
        if overlap:
            errors.append(
                f"task {task.task_id} cannot use the same dependency as hard and optional: {sorted(overlap)}"
            )
        intent = intent_by_id[intent_id]
        selected_need_ids = list(
            getattr(task.parameters, "product_need_ids", []) or []
        )
        unknown_needs = [
            need_id
            for need_id in selected_need_ids
            if need_owner.get(need_id) != intent_id
        ]
        if unknown_needs:
            errors.append(
                f"task {task.task_id} references product needs outside intent {intent_id}: {unknown_needs}"
            )
        effective_needs = [
            need
            for need in intent.product_needs
            if not selected_need_ids or need.need_id in selected_need_ids
        ]
        if task.capability == "multi_product_bundle" and len(effective_needs) < 2:
            errors.append(f"task {task.task_id} multi_product_bundle requires at least two product needs")
        if task.capability == "single_product_recommendation" and len(effective_needs) > 1:
            errors.append(
                f"task {task.task_id} single_product_recommendation cannot own multiple product needs"
            )
        if (
            intent.recommendation_policy.candidate_source == "context_only"
            and task.capability
            in {"single_product_recommendation", "multi_product_bundle"}
        ):
            errors.append(
                f"context-only recommendation intent {intent_id} must not create "
                f"product retrieval task {task.task_id}"
            )
        parameters = task.parameters
        if task.capability in {"knowledge_research", "commerce_research"} and not str(
            parameters.query or ""
        ).strip():
            errors.append(f"task {task.task_id} requires a task-specific query")
        if task.capability == "profile_preference" and not str(
            parameters.query or ""
        ).strip():
            errors.append(f"task {task.task_id} profile lookup requires a query")
        if task.capability == "commerce_research" and not parameters.platforms:
            errors.append(f"task {task.task_id} commerce research requires platforms")

    if errors:
        raise IntentPlanContractError(errors)


def _is_context_only_recommendation(intent) -> bool:
    return bool(
        intent.intent_type == "product_recommendation"
        and intent.recommendation_policy.candidate_source == "context_only"
        and intent.referenced_product_ids
    )


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
    _validate_dependency_references(label, dependencies, known_ids, errors)
    _validate_dependency_graph(label, dependencies, errors)


def _validate_dependency_graph(
    label: str,
    dependencies: dict[str, list[str]],
    errors: list[str],
) -> None:
    for node_id, dependency_ids in dependencies.items():
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
            errors.append(f"{label} {node_id} has unknown dependencies {sorted(unknown)}")
        if len(dependency_ids) != len(set(dependency_ids)):
            errors.append(f"{label} {node_id} has duplicate dependencies")


def _find_cycle(dependencies: dict[str, list[str]]) -> list[str]:
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
            cycle = visit(dependency_id)
            if cycle:
                return cycle
        visiting.pop()
        visited.add(node_id)
        return []

    for node_id in dependencies:
        cycle = visit(node_id)
        if cycle:
            return cycle
    return []
