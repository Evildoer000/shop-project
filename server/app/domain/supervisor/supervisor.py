from __future__ import annotations

import uuid

from app.domain.supervisor.agent_registry import AgentRegistry, build_foundation_agent_registry
from app.domain.supervisor.capability_catalog import CapabilityCatalog, build_default_capability_catalog
from app.domain.supervisor.policy_gate import PolicyEvaluation, SupervisorPolicyGate
from app.domain.supervisor.task_graph import TaskGraph, TaskGraphNode
from app.schemas import AgentTaskProposal, IntentPlan


class SupervisorCompilationError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("Supervisor could not compile task graph: " + "; ".join(errors))


class SupervisorPlanCompiler:
    """Compiles an IntentPlan proposal into a deterministic execution DAG."""

    def __init__(
        self,
        registry: AgentRegistry | None = None,
        catalog: CapabilityCatalog | None = None,
        policy_gate: SupervisorPolicyGate | None = None,
    ) -> None:
        self.catalog = catalog or build_default_capability_catalog()
        self.registry = registry or build_foundation_agent_registry(self.catalog)
        self.policy_gate = policy_gate or SupervisorPolicyGate(self.registry, self.catalog)

    def compile(self, plan: IntentPlan, *, turn_id: str | None = None) -> TaskGraph:
        evaluation = self.policy_gate.evaluate(plan)
        if not evaluation.approved:
            raise SupervisorCompilationError(evaluation.errors)

        resolved_turn_id = turn_id or uuid.uuid4().hex
        graph = TaskGraph(
            graph_id=uuid.uuid4().hex,
            turn_id=resolved_turn_id,
            metadata={
                "intent_plan_schema_version": plan.schema_version,
                "execution_mode": plan.execution_mode,
                "primary_intent": plan.primary_intent,
                "input_modalities": list(plan.input_modalities),
                "policy_evaluation": evaluation.model_dump(),
                "repair_history": [],
            },
        )
        intent_node = self._fixed_node(
            graph,
            capability="intent_understanding",
            node_id="system:intent_understanding",
            depends_on=[],
            status="succeeded",
            metadata={"plan_schema_version": plan.schema_version},
        )
        graph.entry_node_ids = [intent_node.node_id]

        proposal_node_ids = {
            proposal.proposal_id: f"proposal:{proposal.proposal_id}"
            for proposal in plan.agent_proposals
            if proposal.proposal_id in evaluation.selections
        }
        profile_proposals = [
            proposal
            for proposal in plan.agent_proposals
            if proposal.proposal_id in evaluation.selections and proposal.capability == "profile_preference"
        ]
        needs_intent_refinement = any(
            request.context_type == "long_term_profile" and request.usage == "intent_refinement"
            for request in plan.context_requests
        )
        refine_node_id = ""
        if needs_intent_refinement and profile_proposals:
            refine_node_id = "system:intent_refinement"

        for proposal in plan.agent_proposals:
            selected_agent_id = evaluation.selections.get(proposal.proposal_id)
            if not selected_agent_id:
                continue
            dependency_ids = [
                proposal_node_ids[dependency_id]
                for dependency_id in proposal.depends_on
                if dependency_id in proposal_node_ids
            ]
            if not dependency_ids:
                if refine_node_id and proposal.capability != "profile_preference":
                    dependency_ids = [refine_node_id]
                else:
                    dependency_ids = [intent_node.node_id]
            registration = self.registry.require(selected_agent_id)
            context_request = next(
                (
                    request
                    for request in plan.context_requests
                    if proposal.capability == "profile_preference"
                    and request.context_type == "long_term_profile"
                ),
                None,
            )
            graph.add_node(
                TaskGraphNode(
                    node_id=proposal_node_ids[proposal.proposal_id],
                    task_id=f"{resolved_turn_id}:{proposal.proposal_id}",
                    agent_id=selected_agent_id,
                    capability=proposal.capability,
                    intent_ids=list(proposal.intent_ids),
                    depends_on=dependency_ids,
                    input_refs=list(proposal.optional_context_from),
                    required=proposal.required,
                    max_attempts=registration.manifest.max_attempts,
                    metadata={
                        "proposal_id": proposal.proposal_id,
                        "reason": proposal.reason,
                        "allowed_tools": list(registration.manifest.allowed_tools),
                        "research_requests": [
                            request.model_dump()
                            for request in plan.research_requests
                            if request.intent_id in proposal.intent_ids
                        ],
                        **(
                            {
                                "query": context_request.query,
                                "usage": context_request.usage,
                                "context_request_id": context_request.request_id,
                            }
                            if context_request is not None
                            else {}
                        ),
                    },
                )
            )

        if refine_node_id:
            profile_node_ids = [proposal_node_ids[proposal.proposal_id] for proposal in profile_proposals]
            self._fixed_node(
                graph,
                capability="intent_understanding",
                node_id=refine_node_id,
                depends_on=profile_node_ids,
                attempt=2,
                metadata={"reason": "Long-term profile was explicitly requested for intent refinement."},
            )

        proposal_nodes = [
            graph.require_node(proposal_node_ids[proposal_id])
            for proposal_id in proposal_node_ids
        ]
        profile_node_ids = [
            node.node_id for node in proposal_nodes if node.capability == "profile_preference"
        ]
        ranking_profile_requested = any(
            request.context_type == "long_term_profile" and request.usage == "ranking_only"
            for request in plan.context_requests
        )
        if ranking_profile_requested and profile_node_ids:
            for node in proposal_nodes:
                if node.capability not in {"single_product_recommendation", "multi_product_bundle"}:
                    continue
                if refine_node_id and refine_node_id in node.depends_on:
                    continue
                node.depends_on = _unique([*node.depends_on, *profile_node_ids])

        # A bundle has an explicit dispatch node, one isolated retrieval node per
        # slot, and a merge node. This keeps parallel slot work visible in the
        # execution graph without allowing a slot to see its siblings' inputs.
        bundle_dispatch_ids: set[str] = set()
        bundle_merge_ids: set[str] = set()
        for bundle_node in [node for node in proposal_nodes if node.capability == "multi_product_bundle"]:
            bundle_dispatch_ids.add(bundle_node.node_id)
            slot_node_ids: list[str] = []
            for slot in plan.need_slots:
                slot_node_id = f"{bundle_node.node_id}:slot:{slot.slot_id}"
                self._fixed_node(
                    graph,
                    capability="slot_product_retrieval",
                    node_id=slot_node_id,
                    depends_on=[bundle_node.node_id],
                    required=slot.need_type == "required",
                    metadata={
                        "phase": "slot_retrieval",
                        "parent_bundle_node_id": bundle_node.node_id,
                        "slot": slot.model_dump(),
                    },
                )
                slot_node_ids.append(slot_node_id)
            merge_node_id = f"{bundle_node.node_id}:merge"
            merge_dependencies = slot_node_ids or [bundle_node.node_id]
            merge_node = self._fixed_node(
                graph,
                capability="multi_product_bundle",
                node_id=merge_node_id,
                depends_on=merge_dependencies,
                required=bundle_node.required,
                metadata={
                    "phase": "merge_slot_evidence",
                    "dispatch_node_id": bundle_node.node_id,
                    "slot_node_ids": slot_node_ids,
                },
            )
            bundle_merge_ids.add(merge_node.node_id)

            # Downstream proposals consume the merged bundle evidence, never the
            # coordinator's pre-retrieval dispatch output.
            for node in graph.nodes:
                if node.node_id in bundle_dispatch_ids or node.node_id in bundle_merge_ids or node.node_id in slot_node_ids:
                    continue
                if bundle_node.node_id not in node.depends_on:
                    continue
                node.depends_on = [
                    merge_node_id if dependency_id == bundle_node.node_id else dependency_id
                    for dependency_id in node.depends_on
                ]

        if plan.execution_mode == "clarify":
            clarification_nodes = [node for node in proposal_nodes if node.capability == "clarification"]
            clarification_dependency = [node.node_id for node in clarification_nodes]
            memory_node = self._fixed_node(
                graph,
                capability="memory_distillation",
                node_id="system:memory_distillation",
                depends_on=clarification_dependency,
                asynchronous=True,
                required=False,
                metadata={"trigger": "clarification_emitted"},
            )
            graph.terminal_node_ids = [memory_node.node_id]
            graph.validate_graph()
            return graph

        evidence_node_ids = []
        for node in graph.nodes:
            if node.node_id in bundle_dispatch_ids:
                continue
            if node.node_id in bundle_merge_ids:
                evidence_node_ids.append(node.node_id)
                continue
            if node.capability == "slot_product_retrieval":
                # The merge node is the single evidence boundary for a bundle.
                continue
            definition = self.catalog.get(node.capability)
            if definition and definition.evidence_producing:
                evidence_node_ids.append(node.node_id)
        final_dependency_ids: list[str]
        if evidence_node_ids:
            verifier_node = self._fixed_node(
                graph,
                capability="evidence_verification",
                node_id="system:evidence_verification",
                depends_on=_unique(evidence_node_ids + profile_node_ids),
                metadata={"evidence_node_ids": evidence_node_ids},
            )
            final_dependency_ids = [verifier_node.node_id]
            if plan.execution_mode == "multi_product":
                optimizer_node = self._fixed_node(
                    graph,
                    capability="bundle_optimization",
                    node_id="system:bundle_optimization",
                    depends_on=[verifier_node.node_id],
                )
                final_dependency_ids = [optimizer_node.node_id]
        else:
            final_dependency_ids = [node.node_id for node in proposal_nodes] or [intent_node.node_id]

        final_dependency_ids = _unique(final_dependency_ids + profile_node_ids)
        answer_node = self._fixed_node(
            graph,
            capability="answer_generation",
            node_id="system:answer_generation",
            depends_on=final_dependency_ids,
        )
        memory_node = self._fixed_node(
            graph,
            capability="memory_distillation",
            node_id="system:memory_distillation",
            depends_on=[answer_node.node_id],
            asynchronous=True,
            required=False,
            metadata={"trigger": "answer_emitted"},
        )
        graph.terminal_node_ids = [memory_node.node_id]
        graph.validate_graph()
        return graph

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
        invalid_statuses = [node.node_id for node in failed_nodes if node.status not in {"failed", "timeout"}]
        if invalid_statuses:
            raise SupervisorCompilationError(
                [f"repair targets must be failed or timeout nodes: {invalid_statuses}"]
            )
        exhausted = [node.node_id for node in failed_nodes if node.attempt >= node.max_attempts]
        if exhausted:
            raise SupervisorCompilationError([f"repair attempts exhausted for nodes: {exhausted}"])

        history = graph.metadata.setdefault("repair_history", [])
        repair_attempt = len(history) + 1
        repair_registration = self._require_fixed_registration("repair", graph.metadata.get("execution_mode", ""))
        repair_node_id = f"runtime:repair:{repair_attempt}"

        retry_nodes: list[TaskGraphNode] = []
        replacements: dict[str, str] = {}
        for failed_node in failed_nodes:
            retry_node_id = f"runtime:retry:{failed_node.node_id}:{failed_node.attempt + 1}"
            replacements[failed_node.node_id] = retry_node_id
        for failed_node in failed_nodes:
            retry_node_id = replacements[failed_node.node_id]
            failed_node.metadata = {
                **failed_node.metadata,
                "superseded_by": retry_node_id,
                "repair_cycle": repair_attempt,
            }
            retry_dependencies = [repair_node_id]
            for dependency_id in failed_node.depends_on:
                replacement = replacements.get(dependency_id, dependency_id)
                if replacement != retry_node_id and replacement not in retry_dependencies:
                    retry_dependencies.append(replacement)
            retry_nodes.append(
                failed_node.model_copy(
                    update={
                        "node_id": retry_node_id,
                        "depends_on": retry_dependencies,
                        "dependency_policy": "all_succeeded",
                        "status": "pending",
                        "attempt": failed_node.attempt + 1,
                        "metadata": {
                            **failed_node.metadata,
                            "retry_of": failed_node.node_id,
                            "repair_reason": reason,
                            "repair_cycle": repair_attempt,
                        },
                    }
                )
            )

        existing_nodes = list(graph.nodes)
        for node in existing_nodes:
            if node.status != "pending" or node.node_id in replacements:
                continue
            updated_dependencies: list[str] = []
            for dependency_id in node.depends_on:
                replacement = replacements.get(dependency_id, dependency_id)
                if replacement not in updated_dependencies:
                    updated_dependencies.append(replacement)
            node.depends_on = updated_dependencies

        graph.add_node(
            TaskGraphNode(
                node_id=repair_node_id,
                task_id=f"{graph.turn_id}:repair:{repair_attempt}",
                agent_id=repair_registration.manifest.agent_id,
                capability="repair",
                depends_on=[node.node_id for node in failed_nodes],
                dependency_policy="all_terminal",
                attempt=1,
                max_attempts=repair_registration.manifest.max_attempts,
                metadata={
                    "reason": reason,
                    "repair_cycle": repair_attempt,
                    "failed_node_ids": [node.node_id for node in failed_nodes],
                },
            )
        )
        for retry_node in retry_nodes:
            graph.add_node(retry_node)
        history.append(
            {
                "attempt": repair_attempt,
                "reason": reason,
                "failed_node_ids": [node.node_id for node in failed_nodes],
                "retry_node_ids": [node.node_id for node in retry_nodes],
            }
        )
        graph.refresh_terminal_nodes()
        graph.validate_graph()
        return [node.node_id for node in retry_nodes]

    def _fixed_node(
        self,
        graph: TaskGraph,
        *,
        capability: str,
        node_id: str,
        depends_on: list[str],
        status: str = "pending",
        attempt: int = 1,
        asynchronous: bool = False,
        required: bool = True,
        metadata: dict | None = None,
    ) -> TaskGraphNode:
        registration = self._require_fixed_registration(capability, graph.metadata.get("execution_mode", ""))
        node = TaskGraphNode(
            node_id=node_id,
            task_id=f"{graph.turn_id}:{node_id}",
            agent_id=registration.manifest.agent_id,
            capability=capability,
            depends_on=depends_on,
            status=status,  # type: ignore[arg-type]
            attempt=attempt,
            max_attempts=max(attempt, registration.manifest.max_attempts),
            asynchronous=asynchronous,
            required=required,
            metadata=metadata or {},
        )
        graph.add_node(node)
        return node

    def _require_fixed_registration(self, capability: str, execution_mode: str):
        registration = self.registry.select_for_capability(capability, execution_mode=execution_mode)
        if registration is None:
            raise SupervisorCompilationError(
                [f"no enabled Supervisor Agent provides capability={capability} for mode={execution_mode}"]
            )
        return registration


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
