from __future__ import annotations

import uuid
from typing import Any

from app.domain.supervisor.agent_registry import (
    AgentRegistry,
    build_foundation_agent_registry,
)
from app.domain.supervisor.capability_catalog import (
    CapabilityCatalog,
    build_default_capability_catalog,
)
from app.domain.supervisor.policy_gate import PolicyEvaluation, SupervisorPolicyGate
from app.domain.supervisor.task_graph import TaskGraph, TaskGraphNode
from app.schemas import AgentTaskProposal, IntentItem, IntentPlanV3, IntentProductNeed


class SupervisorCompilationError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("Supervisor could not compile task graph: " + "; ".join(errors))


class SupervisorPolicyRejectedError(SupervisorCompilationError):
    """The deterministic PolicyGate rejected the complete planner proposal."""

    def __init__(self, evaluation: PolicyEvaluation) -> None:
        self.evaluation = evaluation
        self.policy_span_payload: dict | None = None
        super().__init__(list(evaluation.errors))


class SupervisorPlanCompiler:
    """Compile intent-scoped V3 tasks into independently executable branches."""

    def __init__(
        self,
        registry: AgentRegistry | None = None,
        catalog: CapabilityCatalog | None = None,
        policy_gate: SupervisorPolicyGate | None = None,
    ) -> None:
        self.catalog = catalog or build_default_capability_catalog()
        self.registry = registry or build_foundation_agent_registry(self.catalog)
        self.policy_gate = policy_gate or SupervisorPolicyGate(self.registry, self.catalog)

    def compile(self, plan: IntentPlanV3, *, turn_id: str | None = None) -> TaskGraph:
        return self.compile_evaluated(plan, evaluation=self.evaluate(plan), turn_id=turn_id)

    def evaluate(self, plan: IntentPlanV3) -> PolicyEvaluation:
        return self.policy_gate.evaluate(plan)

    def compile_evaluated(
        self,
        plan: IntentPlanV3,
        *,
        evaluation: PolicyEvaluation,
        turn_id: str | None = None,
    ) -> TaskGraph:
        if not evaluation.approved:
            raise SupervisorPolicyRejectedError(evaluation)

        resolved_turn_id = turn_id or uuid.uuid4().hex
        graph = TaskGraph(
            schema_version="3.0",
            graph_id=uuid.uuid4().hex,
            turn_id=resolved_turn_id,
            metadata={
                "intent_plan_schema_version": plan.schema_version,
                "intent_order": [intent.intent_id for intent in plan.intents],
                "intent_goals": {
                    intent.intent_id: intent.goal for intent in plan.intents
                },
                "intent_statuses": {
                    item.intent_id: item.model_dump() for item in evaluation.intent_statuses
                },
                "policy_evaluation": evaluation.model_dump(),
                "repair_history": [],
                "branch_terminal_nodes": {},
            },
        )
        intent_node = self._fixed_node(
            graph,
            capability="intent_understanding",
            node_id="system:intent_understanding",
            depends_on=[],
            intent_ids=[intent.intent_id for intent in plan.intents],
            status="succeeded",
            metadata={"plan_schema_version": plan.schema_version},
        )
        graph.entry_node_ids = [intent_node.node_id]
        policy_node = TaskGraphNode(
            node_id="system:policy_gate",
            task_id=f"{resolved_turn_id}:policy_gate",
            agent_id="supervisor_policy_gate",
            capability="policy_gate",
            intent_ids=[intent.intent_id for intent in plan.intents],
            depends_on=[intent_node.node_id],
            status="succeeded",
            metadata={
                "phase": "initial_policy_approval",
                "decision_count": len(evaluation.decisions),
                "approved_task_ids": sorted(evaluation.selections),
                "rejected_task_ids": list(evaluation.rejected_task_ids),
                "implementation": "deterministic_code",
            },
        )
        graph.add_node(policy_node)

        task_outputs = self._append_task_nodes(
            graph,
            plan,
            evaluation,
            root_node_id=policy_node.node_id,
            prefix="proposal",
        )
        self._append_branch_boundaries(graph, plan, evaluation, task_outputs)
        graph.refresh_terminal_nodes()
        graph.terminal_node_ids = ["system:memory_distillation"]
        graph.validate_graph()
        return graph

    def append_revised_branch(
        self,
        graph: TaskGraph,
        plan: IntentPlanV3,
        evaluation: PolicyEvaluation,
        *,
        intent_id: str,
        root_node_id: str,
        prefix: str,
    ) -> list[str]:
        """Append one evidence-informed intent revision without touching siblings."""
        if not evaluation.approved:
            return []
        task_outputs = self._append_task_nodes(
            graph,
            plan,
            evaluation,
            root_node_id=root_node_id,
            prefix=prefix,
        )
        tasks = [
            task
            for task in plan.task_proposals
            if task.task_id in task_outputs and intent_id in task.intent_ids
        ]
        answer = graph.require_node("system:answer_generation")
        old_terminal = str(
            graph.metadata.get("branch_terminal_nodes", {}).get(intent_id) or ""
        )
        clarification = next(
            (task_outputs[task.task_id] for task in tasks if task.capability == "clarification"),
            "",
        )
        if clarification:
            verifier = graph.get_node(f"system:verify:{intent_id}")
            if verifier is not None and verifier.status == "pending":
                verifier.status = "skipped"
                verifier.metadata["skipped_reason"] = "intent_replanned_to_clarification"
            new_terminal = clarification
        else:
            evidence_outputs = self._leaf_evidence_outputs(tasks, task_outputs)
            verifier = graph.get_node(f"system:verify:{intent_id}")
            if verifier is not None and evidence_outputs:
                verifier.depends_on = _unique([*verifier.depends_on, *evidence_outputs])
                verifier.input_refs = [
                    value for value in verifier.input_refs if value not in evidence_outputs
                ]
                verifier.metadata["revision_root_node_id"] = root_node_id
                verifier.metadata["evidence_node_ids"] = list(verifier.depends_on)
                new_terminal = verifier.node_id
                if any(task.capability == "multi_product_bundle" for task in tasks):
                    optimizer_id = f"system:optimize:{intent_id}"
                    optimizer = graph.get_node(optimizer_id)
                    if optimizer is None:
                        optimizer = self._fixed_node(
                            graph,
                            capability="bundle_optimization",
                            node_id=optimizer_id,
                            depends_on=[verifier.node_id],
                            intent_ids=[intent_id],
                        )
                    new_terminal = optimizer.node_id
            else:
                new_terminal = old_terminal or root_node_id

        if old_terminal:
            answer.depends_on = [
                new_terminal if value == old_terminal else value
                for value in answer.depends_on
            ]
        elif new_terminal not in answer.depends_on:
            answer.depends_on.append(new_terminal)
        answer.depends_on = _unique(answer.depends_on)
        graph.metadata.setdefault("branch_terminal_nodes", {})[intent_id] = new_terminal
        graph.metadata.setdefault("intent_revisions", []).append(
            {
                "intent_id": intent_id,
                "root_node_id": root_node_id,
                "task_node_ids": list(task_outputs.values()),
                "terminal_node_id": new_terminal,
            }
        )
        graph.refresh_terminal_nodes()
        graph.terminal_node_ids = ["system:memory_distillation"]
        graph.validate_graph()
        return list(task_outputs.values())

    def _append_task_nodes(
        self,
        graph: TaskGraph,
        plan: IntentPlanV3,
        evaluation: PolicyEvaluation,
        *,
        root_node_id: str,
        prefix: str,
    ) -> dict[str, str]:
        approved_tasks = [
            task for task in plan.task_proposals if task.task_id in evaluation.selections
        ]
        node_ids = {
            task.task_id: f"{prefix}:{task.task_id}" for task in approved_tasks
        }
        for task in approved_tasks:
            registration = self.registry.require(evaluation.selections[task.task_id])
            hard_dependencies = [
                node_ids[dependency_id]
                for dependency_id in task.depends_on
                if dependency_id in node_ids
            ]
            hard_dependencies = _unique([*hard_dependencies, root_node_id])
            optional_inputs = [
                node_ids[dependency_id]
                for dependency_id in task.optional_context_from
                if dependency_id in node_ids and node_ids[dependency_id] not in hard_dependencies
            ]
            graph.add_node(
                TaskGraphNode(
                    node_id=node_ids[task.task_id],
                    task_id=f"{graph.turn_id}:{prefix}:{task.task_id}",
                    agent_id=registration.manifest.agent_id,
                    capability=task.capability,
                    intent_ids=list(task.intent_ids),
                    depends_on=hard_dependencies,
                    input_refs=optional_inputs,
                    required=task.task_id in evaluation.required_task_ids,
                    max_attempts=registration.manifest.max_attempts,
                    metadata={
                        "planner_task_id": task.task_id,
                        "objective": task.objective,
                        "reason": task.reason,
                        "parameters": task.parameters.model_dump(),
                        "allowed_tools": list(registration.manifest.allowed_tools),
                        "supervisor_approved": True,
                    },
                )
            )

        task_outputs = dict(node_ids)
        for task in approved_tasks:
            if task.capability != "multi_product_bundle":
                continue
            dispatch = graph.require_node(node_ids[task.task_id])
            intent = self._intent_for_task(plan, task)
            needs = self._selected_needs(intent, task) if intent is not None else []
            slot_node_ids: list[str] = []
            required_slot_ids: list[str] = []
            optional_slot_ids: list[str] = []
            for need in needs:
                slot_node_id = f"{dispatch.node_id}:slot:{need.need_id}"
                slot = self._need_payload(intent, need)
                slot_node = self._fixed_node(
                    graph,
                    capability="slot_product_retrieval",
                    node_id=slot_node_id,
                    depends_on=[dispatch.node_id],
                    intent_ids=list(task.intent_ids),
                    required=need.priority == "required",
                    metadata={
                        "phase": "slot_retrieval",
                        "parent_bundle_node_id": dispatch.node_id,
                        "planner_task_id": task.task_id,
                        "slot": slot,
                    },
                )
                slot_node_ids.append(slot_node.node_id)
                (required_slot_ids if need.priority == "required" else optional_slot_ids).append(
                    slot_node.node_id
                )
            merge_node_id = f"{dispatch.node_id}:merge"
            merge = self._fixed_node(
                graph,
                capability="multi_product_bundle",
                node_id=merge_node_id,
                depends_on=required_slot_ids or [dispatch.node_id],
                input_refs=optional_slot_ids,
                intent_ids=list(task.intent_ids),
                required=dispatch.required,
                metadata={
                    "phase": "merge_slot_evidence",
                    "dispatch_node_id": dispatch.node_id,
                    "slot_node_ids": slot_node_ids,
                    "planner_task_id": task.task_id,
                },
            )
            task_outputs[task.task_id] = merge.node_id
            for node in graph.nodes:
                if node.node_id in {dispatch.node_id, merge.node_id, *slot_node_ids}:
                    continue
                node.depends_on = [
                    merge.node_id if value == dispatch.node_id else value
                    for value in node.depends_on
                ]
                node.input_refs = [
                    merge.node_id if value == dispatch.node_id else value
                    for value in node.input_refs
                ]
        return task_outputs

    def _append_branch_boundaries(
        self,
        graph: TaskGraph,
        plan: IntentPlanV3,
        evaluation: PolicyEvaluation,
        task_outputs: dict[str, str],
    ) -> None:
        branch_terminals: dict[str, str] = {}
        profile_nodes: list[str] = []
        for task in plan.task_proposals:
            if task.task_id in task_outputs and task.capability == "profile_preference":
                profile_nodes.append(task_outputs[task.task_id])

        for intent in plan.intents:
            tasks = [
                task
                for task in plan.task_proposals
                if task.task_id in task_outputs and intent.intent_id in task.intent_ids
            ]
            clarification = next(
                (task_outputs[task.task_id] for task in tasks if task.capability == "clarification"),
                "",
            )
            if clarification:
                branch_terminals[intent.intent_id] = clarification
                continue
            evidence_outputs = self._leaf_evidence_outputs(tasks, task_outputs)
            context_only = self._is_context_only_recommendation(intent)
            if evidence_outputs or context_only:
                verifier = self._fixed_node(
                    graph,
                    capability="evidence_verification",
                    node_id=f"system:verify:{intent.intent_id}",
                    depends_on=evidence_outputs or ["system:policy_gate"],
                    input_refs=[
                        node_id
                        for node_id in profile_nodes
                        if node_id not in evidence_outputs
                    ],
                    intent_ids=[intent.intent_id],
                    metadata={
                        "evidence_node_ids": evidence_outputs,
                        "intent_goal": intent.goal,
                        "candidate_source": intent.recommendation_policy.candidate_source,
                        "reference_policy": intent.recommendation_policy.reference_policy,
                        "requested_count": intent.recommendation_policy.requested_count,
                        "count_mode": intent.recommendation_policy.count_mode,
                    },
                )
                terminal = verifier.node_id
                if any(task.capability == "multi_product_bundle" for task in tasks):
                    optimizer = self._fixed_node(
                        graph,
                        capability="bundle_optimization",
                        node_id=f"system:optimize:{intent.intent_id}",
                        depends_on=[verifier.node_id],
                        intent_ids=[intent.intent_id],
                    )
                    terminal = optimizer.node_id
                branch_terminals[intent.intent_id] = terminal
                continue
            non_profile = [
                task_outputs[task.task_id]
                for task in tasks
                if task.capability != "profile_preference"
            ]
            branch_terminals[intent.intent_id] = (
                non_profile[-1] if non_profile else "system:policy_gate"
            )

        answer = self._fixed_node(
            graph,
            capability="answer_generation",
            node_id="system:answer_generation",
            depends_on=_unique(list(branch_terminals.values())),
            input_refs=[
                node_id
                for node_id in profile_nodes
                if node_id not in branch_terminals.values()
            ],
            intent_ids=[intent.intent_id for intent in plan.intents],
            dependency_policy="all_terminal",
            metadata={
                "intent_order": [intent.intent_id for intent in plan.intents],
                "branch_terminal_nodes": dict(branch_terminals),
                "uncovered_required_intent_ids": list(
                    evaluation.uncovered_required_intent_ids
                ),
            },
        )
        self._fixed_node(
            graph,
            capability="memory_distillation",
            node_id="system:memory_distillation",
            depends_on=[answer.node_id],
            intent_ids=[intent.intent_id for intent in plan.intents],
            asynchronous=True,
            required=False,
            metadata={"trigger": "answer_emitted"},
        )
        graph.metadata["branch_terminal_nodes"] = branch_terminals

    def _leaf_evidence_outputs(
        self,
        tasks: list[AgentTaskProposal],
        task_outputs: dict[str, str],
    ) -> list[str]:
        task_ids = {task.task_id for task in tasks}
        depended_on = {
            dependency_id
            for task in tasks
            for dependency_id in task.depends_on
            if dependency_id in task_ids
        }
        leaves = [
            task
            for task in tasks
            if task.task_id not in depended_on
            and (definition := self.catalog.get(task.capability)) is not None
            and definition.evidence_producing
        ]
        return _unique([task_outputs[task.task_id] for task in leaves])

    def _is_context_only_recommendation(self, intent: IntentItem) -> bool:
        return bool(
            intent.intent_type == "product_recommendation"
            and intent.recommendation_policy.candidate_source == "context_only"
            and intent.referenced_product_ids
        )

    def schedule_repair(
        self,
        graph: TaskGraph,
        failed_node_ids: list[str],
        *,
        reason: str,
    ) -> list[str]:
        graph.validate_graph()
        failed_nodes = [graph.require_node(node_id) for node_id in _unique(failed_node_ids)]
        if not failed_nodes:
            raise SupervisorCompilationError(["repair requires at least one failed node"])
        invalid = [node.node_id for node in failed_nodes if node.status not in {"failed", "timeout"}]
        if invalid:
            raise SupervisorCompilationError(
                [f"repair targets must be failed or timeout nodes: {invalid}"]
            )
        exhausted = [node.node_id for node in failed_nodes if node.attempt >= node.max_attempts]
        if exhausted:
            raise SupervisorCompilationError([f"repair attempts exhausted for nodes: {exhausted}"])

        history = graph.metadata.setdefault("repair_history", [])
        repair_cycle = len(history) + 1
        repair_registration = self._require_fixed_registration("repair")
        repair_node_id = f"runtime:repair:{repair_cycle}"
        replacements = {
            node.node_id: f"runtime:retry:{node.node_id}:{node.attempt + 1}"
            for node in failed_nodes
        }
        repair_node = TaskGraphNode(
            node_id=repair_node_id,
            task_id=f"{graph.turn_id}:repair:{repair_cycle}",
            agent_id=repair_registration.manifest.agent_id,
            capability="repair",
            intent_ids=_unique(
                [intent_id for node in failed_nodes for intent_id in node.intent_ids]
            ),
            depends_on=[node.node_id for node in failed_nodes],
            dependency_policy="all_terminal",
            max_attempts=repair_registration.manifest.max_attempts,
            metadata={
                "reason": reason,
                "repair_cycle": repair_cycle,
                "failed_node_ids": [node.node_id for node in failed_nodes],
            },
        )
        graph.add_node(repair_node)

        retries: list[TaskGraphNode] = []
        for failed in failed_nodes:
            retry_id = replacements[failed.node_id]
            failed.metadata = {
                **failed.metadata,
                "superseded_by": retry_id,
                "repair_cycle": repair_cycle,
            }
            retries.append(
                failed.model_copy(
                    update={
                        "node_id": retry_id,
                        "task_id": f"{graph.turn_id}:{retry_id}",
                        "depends_on": _unique(
                            [
                                repair_node_id,
                                *[
                                    replacements.get(value, value)
                                    for value in failed.depends_on
                                    if replacements.get(value, value) != retry_id
                                ],
                            ]
                        ),
                        "input_refs": _unique(
                            [replacements.get(value, value) for value in failed.input_refs]
                        ),
                        "dependency_policy": "all_succeeded",
                        "status": "pending",
                        "attempt": failed.attempt + 1,
                        "metadata": {
                            **failed.metadata,
                            "retry_of": failed.node_id,
                            "repair_reason": reason,
                            "repair_cycle": repair_cycle,
                        },
                    }
                )
            )

        for node in graph.nodes:
            if (
                node.status != "pending"
                or node.node_id in replacements
                or node.node_id == repair_node_id
            ):
                continue
            node.depends_on = _unique(
                [replacements.get(value, value) for value in node.depends_on]
            )
            node.input_refs = _unique(
                [replacements.get(value, value) for value in node.input_refs]
            )
        for retry in retries:
            graph.add_node(retry)
        history.append(
            {
                "cycle": repair_cycle,
                "reason": reason,
                "intent_ids": repair_node.intent_ids,
                "failed_node_ids": [node.node_id for node in failed_nodes],
                "retry_node_ids": [node.node_id for node in retries],
            }
        )
        graph.refresh_terminal_nodes()
        graph.validate_graph()
        return [node.node_id for node in retries]

    def _need_payload(self, intent: IntentItem, need: IntentProductNeed) -> dict[str, Any]:
        constraints = [*intent.constraints, *need.constraints]
        hard = [
            f"{item.name}={item.value}" for item in constraints if item.strength == "hard"
        ]
        soft = [
            f"{item.name}={item.value}" for item in constraints if item.strength == "soft"
        ]
        query = " ".join(
            dict.fromkeys(
                value
                for value in [need.product_type, need.goal, *hard, *soft]
                if str(value).strip()
            )
        )
        return {
            "slot_id": need.need_id,
            "intent_id": intent.intent_id,
            "need_type": need.priority,
            "goal": need.goal,
            "product_type": need.product_type,
            "query": query or intent.resolved_query,
            "hard_constraints": hard,
            "soft_constraints": soft,
            "exclude_terms": list(need.exclusions),
            "min_candidates": 1,
        }

    def _selected_needs(
        self,
        intent: IntentItem,
        task: AgentTaskProposal,
    ) -> list[IntentProductNeed]:
        selected = set(task.parameters.product_need_ids)
        return [
            need
            for need in intent.product_needs
            if not selected or need.need_id in selected
        ]

    def _intent_for_task(
        self,
        plan: IntentPlanV3,
        task: AgentTaskProposal,
    ) -> IntentItem | None:
        selected = set(task.intent_ids)
        return next((intent for intent in plan.intents if intent.intent_id in selected), None)

    def _fixed_node(
        self,
        graph: TaskGraph,
        *,
        capability: str,
        node_id: str,
        depends_on: list[str],
        input_refs: list[str] | None = None,
        intent_ids: list[str] | None = None,
        status: str = "pending",
        attempt: int = 1,
        asynchronous: bool = False,
        required: bool = True,
        dependency_policy: str = "all_succeeded",
        metadata: dict | None = None,
    ) -> TaskGraphNode:
        registration = self._require_fixed_registration(capability)
        node = TaskGraphNode(
            node_id=node_id,
            task_id=f"{graph.turn_id}:{node_id}",
            agent_id=registration.manifest.agent_id,
            capability=capability,
            intent_ids=intent_ids or [],
            depends_on=depends_on,
            input_refs=input_refs or [],
            status=status,  # type: ignore[arg-type]
            attempt=attempt,
            max_attempts=max(attempt, registration.manifest.max_attempts),
            asynchronous=asynchronous,
            required=required,
            dependency_policy=dependency_policy,  # type: ignore[arg-type]
            metadata=metadata or {},
        )
        graph.add_node(node)
        return node

    def _require_fixed_registration(self, capability: str):
        registration = self.registry.select_for_capability(capability)
        if registration is None:
            raise SupervisorCompilationError(
                [f"no enabled Supervisor Agent provides capability={capability}"]
            )
        return registration


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
