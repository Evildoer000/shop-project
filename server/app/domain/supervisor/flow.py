from __future__ import annotations

from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

from app.domain.agents import (
    AgentExecutionContext,
    AgentExecutor,
    AgentResult,
    BusinessAgentHandlers,
    BusinessAgentServices,
    ExecutionReport,
)
from app.domain.supervisor.agent_registry import AgentRegistry
from app.domain.supervisor.prompts import PromptRegistry, build_default_prompt_registry
from app.domain.supervisor.supervisor import (
    SupervisorPlanCompiler,
    SupervisorPolicyRejectedError,
)
from app.domain.supervisor.task_graph import TaskGraph, TaskGraphNode
from app.schemas import IntentPlanV3, ReflectionResult


@dataclass
class SupervisorFlowResult:
    intent_plan: IntentPlanV3
    graph: TaskGraph
    report: ExecutionReport
    route: str = "no_product"
    answer_text: str = ""
    answer_tokens: list[str] = field(default_factory=list)
    cards: list[dict[str, Any]] = field(default_factory=list)
    product_ids: list[str] = field(default_factory=list)
    reflection: Any = None
    trace: dict[str, Any] = field(default_factory=dict)


class SupervisorFlow:
    """Execute independent intent branches and aggregate them in source order."""

    def __init__(
        self,
        *,
        services: BusinessAgentServices,
        tool_registry: Any,
        span_recorder: Any,
        budget_manager: Any,
        registry: AgentRegistry | None = None,
        prompt_registry: PromptRegistry | None = None,
    ) -> None:
        self.services = services
        self.prompt_registry = prompt_registry or build_default_prompt_registry()
        self.compiler = SupervisorPlanCompiler(registry=registry)
        handlers = BusinessAgentHandlers(services).as_mapping()
        handlers["policy_gate"] = self._execute_policy_gate
        self.executor = AgentExecutor(
            agent_registry=self.compiler.registry,
            tool_registry=tool_registry,
            span_recorder=span_recorder,
            budget_manager=budget_manager,
            handlers=handlers,
            prompt_registry=self.prompt_registry,
        )
        self.last_result: SupervisorFlowResult | None = None

    async def _execute_policy_gate(self, context: AgentExecutionContext) -> AgentResult:
        plan = context.artifact("intent_plan", context.intent_plan)
        if not isinstance(plan, IntentPlanV3):
            revision_id = str(context.node.metadata.get("revision_intent_id") or "")
            plan = context.artifact(f"intent_plan_revision:{revision_id}")
        if not isinstance(plan, IntentPlanV3):
            return AgentResult.failure_result(
                "missing_intent_plan",
                "PolicyGate requires an IntentPlan V3 artifact.",
                retryable=False,
                termination_reason="policy_input_missing",
            )
        evaluation = self.compiler.evaluate(plan)
        if not evaluation.approved:
            return AgentResult.failure_result(
                "policy_rejected",
                "; ".join(evaluation.errors) or "PolicyGate rejected the revised branch.",
                retryable=False,
                termination_reason="policy_rejected",
            )
        return AgentResult.success(
            {"policy_evaluation": evaluation.model_dump()},
            artifacts={"policy_evaluation": evaluation},
            termination_reason="policy_approved",
        )

    async def run(
        self,
        *,
        query: str,
        user_id: str,
        session_id: str,
        turn_id: str,
        intent_plan: IntentPlanV3,
        conversation_context: Any = None,
        query_plan: Any = None,
        image_attributes: Any = None,
        image_path: str | None = None,
        intent_span_key: str | None = None,
        policy_attempt: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        evaluation, policy_span_payload = self._evaluate_initial_policy(
            intent_plan,
            turn_id=turn_id,
            intent_span_key=intent_span_key,
            attempt=policy_attempt,
        )
        try:
            graph = self.compiler.compile_evaluated(
                intent_plan,
                evaluation=evaluation,
                turn_id=turn_id,
            )
        except SupervisorPolicyRejectedError as exc:
            exc.policy_span_payload = policy_span_payload
            raise

        graph.metadata["policy_attempt"] = max(1, int(policy_attempt or 1))
        artifacts: dict[str, Any] = {
            "intent_plan": intent_plan,
            "policy_evaluation": evaluation,
        }
        base_context = AgentExecutionContext(
            node=graph.require_node(graph.entry_node_ids[0]),
            graph=graph,
            query=query,
            user_id=user_id,
            session_id=session_id,
            turn_id=turn_id,
            intent_plan=intent_plan,
            conversation_context=conversation_context,
            query_plan=query_plan,
            image_attributes=image_attributes,
            artifacts=artifacts,
            services=self.services,
            span_recorder=self.executor.span_recorder,
            budget_manager=self.executor.budget_manager,
            metadata={
                **(metadata or {}),
                "image_path": image_path,
            },
        )
        completed_span_keys: dict[str, str] = {}
        if intent_span_key:
            completed_span_keys["system:intent_understanding"] = intent_span_key
        policy_span_key = str((policy_span_payload or {}).get("span_key") or "")
        if policy_span_key:
            completed_span_keys["system:policy_gate"] = policy_span_key
        self.executor.begin_graph(completed_span_keys=completed_span_keys)
        report = ExecutionReport(graph=graph, artifacts=artifacts)
        emitted_answer_output = False

        if policy_span_payload is not None:
            yield self.executor.span_recorder.timing_event(policy_span_payload)
        yield self._graph_snapshot_event(graph, intent_plan, "compiled")

        pre_verifier_stop = {
            "evidence_verification",
            "bundle_optimization",
            "answer_generation",
            "memory_distillation",
        }
        self._reset_phase_report(report)
        async for event in self._execute_phase(
            graph,
            base_context,
            report,
            intent_plan,
            stop_before=pre_verifier_stop,
        ):
            if event.get("type") in {"token", "product_cards"}:
                emitted_answer_output = True
            yield event

        # Runtime repair stays branch-local: each failed node gets its own Repair
        # lineage, while unrelated ready roots continue to run.
        runtime_cycle = 0
        while report.failed_node_ids and runtime_cycle < 2:
            repairable = self._repairable_nodes(graph, report.failed_node_ids)
            if not repairable:
                break
            retry_ids: list[str] = []
            for node in repairable:
                retry_ids.extend(
                    self.compiler.schedule_repair(
                        graph,
                        [node.node_id],
                        reason="Agent 执行失败，只修复当前意图分支。",
                    )
                )
            runtime_cycle += 1
            yield self._graph_snapshot_event(graph, intent_plan, "runtime_repair_scheduled")
            self._reset_phase_report(report)
            async for event in self._execute_phase(
                graph,
                base_context,
                report,
                intent_plan,
                stop_before=pre_verifier_stop,
            ):
                yield event

        # A knowledge bridge is not converted into a product route by fixed
        # Supervisor rules. The same IntentUnderstandingAgent revises only the
        # intent whose evidence just returned.
        revision_nodes = self._append_knowledge_revision_nodes(
            graph,
            intent_plan,
            report,
        )
        if revision_nodes:
            yield self._graph_snapshot_event(graph, intent_plan, "intent_revision_scheduled")
            self._reset_phase_report(report)
            async for event in self._execute_phase(
                graph,
                base_context,
                report,
                intent_plan,
                stop_before=pre_verifier_stop,
            ):
                yield event

            appended_business_nodes: list[str] = []
            for revision_node in revision_nodes:
                intent_id = revision_node.intent_ids[0]
                scoped = report.artifacts.get(f"artifacts:{revision_node.node_id}")
                revised_plan = (
                    scoped.get(f"intent_plan_revision:{intent_id}")
                    if isinstance(scoped, dict)
                    else None
                )
                if not isinstance(revised_plan, IntentPlanV3):
                    graph.metadata.setdefault("intent_revision_errors", []).append(
                        {"intent_id": intent_id, "reason": "revision_plan_missing"}
                    )
                    continue
                revision_evaluation = self.compiler.evaluate(revised_plan)
                policy_node = self._append_revision_policy_node(
                    graph,
                    revision_node,
                    revision_evaluation,
                )
                if not revision_evaluation.approved:
                    continue
                appended_business_nodes.extend(
                    self.compiler.append_revised_branch(
                        graph,
                        revised_plan,
                        revision_evaluation,
                        intent_id=intent_id,
                        root_node_id=policy_node.node_id,
                        prefix=f"revision:{intent_id}",
                    )
                )
            if appended_business_nodes:
                yield self._graph_snapshot_event(
                    graph,
                    intent_plan,
                    "intent_revision_compiled",
                )
                self._reset_phase_report(report)
                async for event in self._execute_phase(
                    graph,
                    base_context,
                    report,
                    intent_plan,
                    stop_before=pre_verifier_stop,
                ):
                    yield event

        # All intent verifiers can run concurrently. A failed verifier blocks
        # only its branch terminal; AnswerGenerator waits for terminal status,
        # not universal success.
        self._reset_phase_report(report)
        async for event in self._execute_phase(
            graph,
            base_context,
            report,
            intent_plan,
            stop_before={"bundle_optimization", "answer_generation", "memory_distillation"},
        ):
            yield event

        # A verifier can finish successfully while still reporting that its
        # candidate evidence is repairable. Repair only the retrieval lineage
        # for that intent, then attach a fresh verifier to the retried evidence.
        quality_cycle = 0
        while quality_cycle < 2:
            repair_requests = self._quality_repair_requests(graph, report)
            if not repair_requests:
                break
            scheduled = 0
            for verifier, target_ids in repair_requests:
                verifier.metadata["quality_repair_checked"] = True
                for node_id in target_ids:
                    target = graph.require_node(node_id)
                    if target.status == "succeeded":
                        target.status = "failed"
                repairable = self._repairable_nodes(graph, target_ids)
                if not repairable:
                    continue
                retry_ids = self.compiler.schedule_repair(
                    graph,
                    [node.node_id for node in repairable],
                    reason=(
                        "EvidenceVerifier 标记当前意图的候选证据不足或不匹配，"
                        "只重试该意图的检索节点。"
                    ),
                )
                repair_node_ids = {
                    dependency_id
                    for retry_id in retry_ids
                    for dependency_id in graph.require_node(retry_id).depends_on
                    if graph.require_node(dependency_id).capability == "repair"
                }
                for repair_node_id in repair_node_ids:
                    repair_node = graph.require_node(repair_node_id)
                    if verifier.node_id not in repair_node.input_refs:
                        repair_node.input_refs.append(verifier.node_id)
                next_verifier = self._append_verifier_retry(
                    graph,
                    previous_verifier=verifier,
                    retry_node_ids=retry_ids,
                    cycle=quality_cycle + 1,
                )
                history = graph.metadata.get("repair_history", [])
                if history:
                    history[-1].update(
                        {
                            "kind": "evidence_quality",
                            "intent_id": verifier.intent_ids[0]
                            if verifier.intent_ids
                            else "",
                            "verifier_node_id": next_verifier.node_id,
                        }
                    )
                scheduled += 1
            if not scheduled:
                break

            quality_cycle += 1
            yield self._graph_snapshot_event(
                graph,
                intent_plan,
                "branch_quality_repair_scheduled",
            )
            self._reset_phase_report(report)
            async for event in self._execute_phase(
                graph,
                base_context,
                report,
                intent_plan,
                stop_before=pre_verifier_stop,
            ):
                yield event

            self._reset_phase_report(report)
            async for event in self._execute_phase(
                graph,
                base_context,
                report,
                intent_plan,
                stop_before={
                    "bundle_optimization",
                    "answer_generation",
                    "memory_distillation",
                },
            ):
                yield event

        self._reset_phase_report(report)
        async for event in self._execute_phase(
            graph,
            base_context,
            report,
            intent_plan,
        ):
            if event.get("type") in {"token", "product_cards"}:
                emitted_answer_output = True
            yield event

        report.failed_node_ids = self._unresolved_failure_ids(graph)
        report.blocked_node_ids = [
            node.node_id
            for node in graph.nodes
            if node.status == "skipped"
            and node.metadata.get("skipped_reason")
            in {"dependency_failed", "optional_branch_failed"}
            and not node.metadata.get("superseded_by")
        ]
        report.completed = graph.all_terminal()

        answer_node = graph.get_node("system:answer_generation")
        answer_result = report.results.get(answer_node.node_id) if answer_node is not None else None
        answer_artifacts = answer_result.artifacts if answer_result is not None else {}
        output = answer_result.output if answer_result is not None else {}
        answer_text = str(answer_artifacts.get("answer_text") or "")
        answer_tokens = list(answer_artifacts.get("answer_tokens") or [])
        cards = list(answer_artifacts.get("answer_cards") or [])
        product_ids = list(answer_artifacts.get("answer_product_ids") or [])
        route = str(answer_artifacts.get("answer_route") or output.get("route") or "no_product")
        reflections = self._branch_reflections(graph, report)
        trace = self._trace(graph, intent_plan, report, route, reflections)
        self.last_result = SupervisorFlowResult(
            intent_plan=intent_plan,
            graph=graph,
            report=report,
            route=route,
            answer_text=answer_text,
            answer_tokens=answer_tokens,
            cards=cards,
            product_ids=product_ids,
            reflection=reflections,
            trace=trace,
        )
        yield {"type": "decision_trace", "trace": trace}
        if not emitted_answer_output:
            if cards:
                yield {"type": "product_cards", "products": cards}
            for token in answer_tokens:
                yield {"type": "token", "content": token}
        yield {"type": "done"}

    async def _execute_phase(
        self,
        graph: TaskGraph,
        base_context: AgentExecutionContext,
        report: ExecutionReport,
        plan: IntentPlanV3,
        *,
        stop_before: set[str] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        stream = self.executor.execute_stream(
            graph,
            base_context=base_context,
            report=report,
            stop_before_capabilities=stop_before,
        )
        async with aclosing(stream) as phase_stream:
            async for event in phase_stream:
                yield self._executor_event(event, graph, plan)

    def _append_knowledge_revision_nodes(
        self,
        graph: TaskGraph,
        plan: IntentPlanV3,
        report: ExecutionReport,
    ) -> list[TaskGraphNode]:
        result: list[TaskGraphNode] = []
        registration = self.compiler.registry.select_for_capability("intent_understanding")
        if registration is None:
            return result
        for intent in plan.intents:
            knowledge_nodes = [
                node
                for node in graph.nodes
                if node.capability == "knowledge_research"
                and intent.intent_id in node.intent_ids
                and node.status == "succeeded"
                and str(
                    (node.metadata.get("parameters") or {}).get("knowledge_mode")
                    or ""
                )
                == "concept_bridge"
            ]
            if not knowledge_nodes:
                continue
            knowledge_node = knowledge_nodes[-1]
            knowledge_artifacts = report.artifacts.get(
                f"artifacts:{knowledge_node.node_id}",
                {},
            )
            knowledge_result = (
                knowledge_artifacts.get("knowledge_research")
                if isinstance(knowledge_artifacts, dict)
                else None
            )
            node_id = f"runtime:intent_revision:{intent.intent_id}:1"
            if graph.get_node(node_id) is not None:
                continue
            node = TaskGraphNode(
                node_id=node_id,
                task_id=f"{graph.turn_id}:intent_revision:{intent.intent_id}:1",
                agent_id=registration.manifest.agent_id,
                capability="intent_understanding",
                intent_ids=[intent.intent_id],
                depends_on=[knowledge_node.node_id],
                max_attempts=registration.manifest.max_attempts,
                metadata={
                    "phase": "intent_revision_after_knowledge",
                    "intent_revision": {
                        "intent_id": intent.intent_id,
                        "target_intent": intent.model_dump(),
                        "completed_capability": "knowledge_research",
                        "knowledge_result": knowledge_result or {},
                        "instruction": "只修订当前意图的后续任务，不得重建其它意图。",
                    },
                },
            )
            graph.add_node(node)
            result.append(node)
        if result:
            graph.validate_graph()
        return result

    def _append_revision_policy_node(
        self,
        graph: TaskGraph,
        revision_node: TaskGraphNode,
        evaluation: Any,
    ) -> TaskGraphNode:
        intent_id = revision_node.intent_ids[0]
        node = TaskGraphNode(
            node_id=f"runtime:policy_gate:intent_revision:{intent_id}",
            task_id=f"{graph.turn_id}:policy_gate:intent_revision:{intent_id}",
            agent_id="supervisor_policy_gate",
            capability="policy_gate",
            intent_ids=[intent_id],
            depends_on=[revision_node.node_id],
            status="succeeded",
            metadata={
                "phase": "intent_revision_policy_approval",
                "implementation": "deterministic_code",
                "approved": evaluation.approved,
                "evaluation": evaluation.model_dump(),
            },
        )
        graph.add_node(node)
        graph.metadata.setdefault("supervisor_decisions", []).append(
            {
                "subject": node.node_id,
                "intent_id": intent_id,
                **evaluation.model_dump(),
            }
        )
        recorder = self.executor.span_recorder
        if recorder is not None:
            payload = recorder.record_completed_span(
                "supervisor_policy_gate:intent_revision",
                duration_ms=0,
                label="单意图修订策略审批",
                parent_span_key=self.executor.span_key_for_node(revision_node.node_id),
                task_id=node.task_id,
                agent_id=node.agent_id,
                span_type="policy",
                status="succeeded",
                input_summary={"intent_id": intent_id},
                output_summary=evaluation.model_dump(),
                termination_reason=(
                    "policy_approved" if evaluation.approved else "policy_rejected"
                ),
            )
            self.executor.register_completed_node_span(
                node.node_id,
                str(payload.get("span_key") or ""),
            )
        graph.validate_graph()
        return node

    def _evaluate_initial_policy(
        self,
        plan: IntentPlanV3,
        *,
        turn_id: str,
        intent_span_key: str | None,
        attempt: int = 1,
    ) -> tuple[Any, dict[str, Any] | None]:
        recorder = self.executor.span_recorder
        resolved_attempt = max(1, int(attempt or 1))
        span = None
        if recorder is not None:
            span = recorder.start_span(
                "supervisor_policy_gate",
                label=(
                    "Supervisor 初始策略审批"
                    if resolved_attempt == 1
                    else "Supervisor 重规划策略审批"
                ),
                agent="SupervisorPolicyGate",
                parent_span_key=intent_span_key,
                task_id=f"{turn_id}:policy_gate:attempt:{resolved_attempt}",
                agent_id="supervisor_policy_gate",
                span_type="policy",
                attempt=resolved_attempt,
                input_summary={
                    "intent_count": len(plan.intents),
                    "task_count": len(plan.task_proposals),
                    "plan_schema_version": plan.schema_version,
                    "implementation": "deterministic_code",
                },
            )
        evaluation = self.compiler.evaluate(plan)
        payload = None
        if span is not None:
            payload = recorder.finish_span(
                span,
                status="succeeded",
                output_summary=evaluation.model_dump(),
                metrics={
                    "approved": evaluation.approved,
                    "decision_count": len(evaluation.decisions),
                    "selected_agent_count": len(evaluation.selections),
                    "uncovered_intent_count": len(
                        evaluation.uncovered_required_intent_ids
                    ),
                    "hard_error_count": len(evaluation.hard_errors),
                    "warning_count": len(evaluation.warnings),
                },
                termination_reason=(
                    "policy_approved" if evaluation.approved else "policy_rejected"
                ),
            )
        return evaluation, payload

    def _repairable_nodes(
        self,
        graph: TaskGraph,
        node_ids: list[str],
    ) -> list[TaskGraphNode]:
        return [
            node
            for node_id in dict.fromkeys(node_ids)
            if (node := graph.get_node(node_id)) is not None
            and node.status in {"failed", "timeout"}
            and node.attempt < node.max_attempts
            and node.capability
            not in {
                "intent_understanding",
                "policy_gate",
                "answer_generation",
                "memory_distillation",
            }
        ]

    def _quality_repair_requests(
        self,
        graph: TaskGraph,
        report: ExecutionReport,
    ) -> list[tuple[TaskGraphNode, list[str]]]:
        requests: list[tuple[TaskGraphNode, list[str]]] = []
        for verifier in graph.nodes:
            if (
                verifier.capability != "evidence_verification"
                or verifier.status != "succeeded"
                or verifier.metadata.get("superseded_by")
                or verifier.metadata.get("quality_repair_checked")
            ):
                continue
            scoped = report.artifacts.get(f"artifacts:{verifier.node_id}")
            reflection = scoped.get("reflection") if isinstance(scoped, dict) else None
            if (
                not isinstance(reflection, ReflectionResult)
                or not reflection.repair_hint.repairable
            ):
                verifier.metadata["quality_repair_checked"] = True
                continue
            targets = self._quality_failure_targets(
                graph,
                verifier,
                reflection,
            )
            if targets:
                requests.append((verifier, targets))
            else:
                verifier.metadata["quality_repair_checked"] = True
        return requests

    def _quality_failure_targets(
        self,
        graph: TaskGraph,
        verifier: TaskGraphNode,
        reflection: ReflectionResult,
    ) -> list[str]:
        intent_ids = set(verifier.intent_ids)
        ancestors = set(graph.ancestor_node_ids(verifier.node_id))
        hinted_slots = set(reflection.repair_hint.target_slot_ids)

        single_nodes = [
            node
            for node in graph.nodes
            if node.node_id in ancestors
            and node.capability == "single_product_recommendation"
            and node.status == "succeeded"
            and not node.metadata.get("superseded_by")
            and (not intent_ids or intent_ids.intersection(node.intent_ids))
        ]
        if single_nodes and (not hinted_slots or "single" in hinted_slots):
            return [max(single_nodes, key=lambda item: item.attempt).node_id]

        slots_by_id: dict[str, list[TaskGraphNode]] = {}
        for node in graph.nodes:
            if (
                node.node_id not in ancestors
                or node.capability != "slot_product_retrieval"
                or node.status != "succeeded"
                or node.metadata.get("superseded_by")
                or intent_ids
                and not intent_ids.intersection(node.intent_ids)
            ):
                continue
            slot_id = str((node.metadata.get("slot") or {}).get("slot_id") or "")
            if slot_id and (not hinted_slots or slot_id in hinted_slots):
                slots_by_id.setdefault(slot_id, []).append(node)
        selected_slots = [
            max(nodes, key=lambda item: item.attempt)
            for nodes in slots_by_id.values()
        ]
        if not selected_slots:
            return []

        targets = [node.node_id for node in selected_slots]
        merge_nodes = [
            node
            for node in graph.nodes
            if node.node_id in ancestors
            and node.capability == "multi_product_bundle"
            and node.metadata.get("phase") == "merge_slot_evidence"
            and node.status == "succeeded"
            and not node.metadata.get("superseded_by")
            and (not intent_ids or intent_ids.intersection(node.intent_ids))
        ]
        if merge_nodes:
            targets.append(max(merge_nodes, key=lambda item: item.attempt).node_id)
        return list(dict.fromkeys(targets))

    def _append_verifier_retry(
        self,
        graph: TaskGraph,
        *,
        previous_verifier: TaskGraphNode,
        retry_node_ids: list[str],
        cycle: int,
    ) -> TaskGraphNode:
        registration = self.compiler.registry.select_for_capability(
            "evidence_verification"
        )
        if registration is None:
            raise RuntimeError("EvidenceVerifierAgent is not registered")
        replacements = {
            str(graph.require_node(node_id).metadata.get("retry_of") or ""): node_id
            for node_id in retry_node_ids
        }
        dependencies = list(
            dict.fromkeys(
                replacements.get(node_id, node_id)
                for node_id in previous_verifier.depends_on
            )
        )
        optional_inputs = [
            replacements.get(node_id, node_id)
            for node_id in previous_verifier.input_refs
        ]
        intent_id = previous_verifier.intent_ids[0] if previous_verifier.intent_ids else "unknown"
        attempt = previous_verifier.attempt + 1
        node = TaskGraphNode(
            node_id=f"runtime:verify:{intent_id}:{attempt}",
            task_id=f"{graph.turn_id}:verify:{intent_id}:{attempt}",
            agent_id=registration.manifest.agent_id,
            capability="evidence_verification",
            intent_ids=list(previous_verifier.intent_ids),
            depends_on=dependencies,
            input_refs=list(
                dict.fromkeys(
                    node_id
                    for node_id in optional_inputs
                    if node_id not in dependencies
                )
            ),
            attempt=attempt,
            max_attempts=max(attempt, registration.manifest.max_attempts),
            metadata={
                **previous_verifier.metadata,
                "phase": "evidence_reverification",
                "previous_verifier_node_id": previous_verifier.node_id,
                "retry_node_ids": list(retry_node_ids),
                "quality_repair_cycle": cycle,
                "quality_repair_checked": False,
            },
        )
        previous_verifier.metadata["superseded_by"] = node.node_id
        for downstream in graph.nodes:
            if downstream.status != "pending" or downstream.capability == "repair":
                continue
            downstream.depends_on = [
                node.node_id if value == previous_verifier.node_id else value
                for value in downstream.depends_on
            ]
            downstream.input_refs = [
                node.node_id if value == previous_verifier.node_id else value
                for value in downstream.input_refs
            ]
        graph.add_node(node)
        terminal_nodes = graph.metadata.setdefault("branch_terminal_nodes", {})
        if terminal_nodes.get(intent_id) == previous_verifier.node_id:
            terminal_nodes[intent_id] = node.node_id
        graph.refresh_terminal_nodes()
        graph.terminal_node_ids = ["system:memory_distillation"]
        graph.validate_graph()
        return node

    def _reset_phase_report(self, report: ExecutionReport) -> None:
        report.failed_node_ids = []
        report.blocked_node_ids = []
        report.completed = False

    def _unresolved_failure_ids(self, graph: TaskGraph) -> list[str]:
        unresolved: list[str] = []
        for node in graph.nodes:
            if node.status not in {"failed", "timeout", "cancelled"}:
                continue
            replacement_id = str(node.metadata.get("superseded_by") or "")
            replacement = graph.get_node(replacement_id) if replacement_id else None
            seen: set[str] = set()
            while replacement is not None and replacement.node_id not in seen:
                seen.add(replacement.node_id)
                next_id = str(replacement.metadata.get("superseded_by") or "")
                if not next_id:
                    break
                replacement = graph.get_node(next_id)
            if replacement is None or replacement.status != "succeeded":
                unresolved.append(node.node_id)
        return unresolved

    def _branch_reflections(
        self,
        graph: TaskGraph,
        report: ExecutionReport,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for intent_id, terminal_id in graph.metadata.get("branch_terminal_nodes", {}).items():
            node_ids = [*graph.ancestor_node_ids(terminal_id), terminal_id]
            for node_id in reversed(node_ids):
                scoped = report.artifacts.get(f"artifacts:{node_id}")
                reflection = scoped.get("reflection") if isinstance(scoped, dict) else None
                if isinstance(reflection, ReflectionResult):
                    result[intent_id] = reflection.model_dump()
                    break
        return result

    def _executor_event(
        self,
        event: Any,
        graph: TaskGraph,
        plan: IntentPlanV3,
    ) -> dict[str, Any]:
        payload = event.as_dict()
        if event.kind == "agent_output":
            client_event = payload.get("event")
            if isinstance(client_event, dict):
                return client_event
            return {
                "type": "trace",
                "stage": "agent_output",
                "content": "Agent emitted an invalid client event.",
            }
        if event.kind == "node_started":
            node = graph.get_node(event.node_id)
            goals = graph.metadata.get("intent_goals", {})
            goal = "；".join(goals.get(item, item) for item in (node.intent_ids if node else []))
            return {
                "type": "agent_update",
                "stage": self._stage_for_capability(payload.get("capability", "")),
                "title": f"{payload.get('agent_id', 'Agent')} 执行中",
                "content_delta": (
                    f"正在处理：{goal}。" if goal else f"正在执行 {payload.get('capability', '')}。"
                ),
                "done": False,
                "run_id": getattr(self.executor.span_recorder, "run_id", ""),
                "task_id": payload.get("task_id", ""),
                "agent_id": payload.get("agent_id", ""),
                "intent_ids": list(node.intent_ids) if node else [],
            }
        if event.kind == "node_finished":
            span = self._latest_span(payload.get("task_id", ""))
            timing = (
                self.executor.span_recorder.timing_event(span)
                if span is not None
                else {
                    "type": "timing_update",
                    "run_id": getattr(self.executor.span_recorder, "run_id", ""),
                }
            )
            node = graph.get_node(event.node_id)
            return {
                "type": "timing_update",
                **timing,
                "node": {
                    "node_id": event.node_id,
                    "task_id": payload.get("task_id", ""),
                    "agent_id": payload.get("agent_id", ""),
                    "capability": payload.get("capability", ""),
                    "status": payload.get("status", ""),
                    "attempt": payload.get("attempt", 1),
                    "intent_ids": list(node.intent_ids) if node else [],
                },
            }
        if event.kind == "node_blocked":
            return {
                "type": "agent_update",
                "stage": "supervisor",
                "title": "分支依赖未完成",
                "content_delta": f"节点 {event.node_id} 的硬依赖失败，仅跳过当前分支后续。",
                "done": True,
            }
        if event.kind == "handoff_updated":
            handoff = payload.get("handoff") if isinstance(payload.get("handoff"), dict) else {}
            return {
                "type": "handoff_update",
                "run_id": getattr(self.executor.span_recorder, "run_id", ""),
                "handoff": handoff,
                "span_key": payload.get("span_key", ""),
            }
        return {
            "type": "decision_trace",
            "trace": self._trace(graph, plan, None, "", {}),
            "executor_event": payload,
        }

    def _latest_span(self, task_id: str) -> dict[str, Any] | None:
        spans = getattr(self.executor.span_recorder, "spans", [])
        for span in reversed(spans):
            if not task_id or span.get("task_id") == task_id:
                return span
        return spans[-1] if spans else None

    def _trace(
        self,
        graph: TaskGraph,
        plan: IntentPlanV3,
        report: ExecutionReport | None,
        route: str,
        reflections: Any,
    ) -> dict[str, Any]:
        nodes = [
            {
                "node_id": node.node_id,
                "task_id": node.task_id,
                "agent_id": node.agent_id,
                "capability": node.capability,
                "intent_ids": list(node.intent_ids),
                "depends_on": list(node.depends_on),
                "input_refs": list(node.input_refs),
                "required": node.required,
                "status": node.status,
                "attempt": node.attempt,
                "phase": node.metadata.get("phase", ""),
                "objective": node.metadata.get("objective", ""),
                "skipped_reason": node.metadata.get("skipped_reason", ""),
            }
            for node in graph.nodes
        ]
        tool_calls = self._trace_tool_calls(graph, report)
        graph.sync_handoffs()
        handoffs = [item.model_dump() for item in sorted(graph.handoffs, key=lambda item: item.sequence)]
        unresolved = list(report.failed_node_ids) if report else []
        blocked = list(report.blocked_node_ids) if report else []
        pending = [node.node_id for node in graph.nodes if node.status in {"pending", "running"}]
        task_status = (
            "running"
            if pending
            else "degraded"
            if unresolved or blocked
            else "succeeded"
        )
        return {
            "trace_schema_version": "v3",
            "run_id": str(getattr(self.executor.span_recorder, "run_id", "") or ""),
            "route": route,
            "task_status": task_status,
            "task": {
                "graph_id": graph.graph_id,
                "turn_id": graph.turn_id,
                "schema_version": graph.schema_version,
                "intent_order": graph.metadata.get("intent_order", []),
                "intent_statuses": graph.metadata.get("intent_statuses", {}),
                "branch_terminal_nodes": graph.metadata.get("branch_terminal_nodes", {}),
                "repair_history": graph.metadata.get("repair_history", []),
                "intent_revisions": graph.metadata.get("intent_revisions", []),
                "supervisor_decisions": graph.metadata.get("supervisor_decisions", []),
                "handoff_count": len(handoffs),
                "tool_call_count": len(tool_calls),
            },
            "planner_proposal": plan.model_dump(),
            "agent_path": nodes,
            "tool_calls": tool_calls,
            "handoffs": handoffs,
            "reflections_by_intent": reflections or {},
            "failed_node_ids": unresolved,
            "blocked_node_ids": blocked,
        }

    def _trace_tool_calls(
        self,
        graph: TaskGraph,
        report: ExecutionReport | None,
    ) -> list[dict[str, Any]]:
        if report is None:
            return []
        calls: list[dict[str, Any]] = []
        for node in graph.nodes:
            result = report.results.get(node.node_id)
            if result is None:
                continue
            for index, call in enumerate(result.tool_calls, start=1):
                if not isinstance(call, dict):
                    continue
                calls.append(
                    {
                        **call,
                        "call_id": f"{node.task_id}:tool:{index}",
                        "node_id": node.node_id,
                        "capability": node.capability,
                        "intent_ids": list(node.intent_ids),
                    }
                )
        return calls

    def _stage_for_capability(self, capability: str) -> str:
        if capability in {"intent_understanding", "profile_preference", "clarification"}:
            return "planner"
        if capability in {
            "single_product_recommendation",
            "multi_product_bundle",
            "slot_product_retrieval",
            "commerce_research",
            "knowledge_research",
            "comparison",
        }:
            return "retrieval"
        if capability in {"evidence_verification", "repair", "bundle_optimization"}:
            return "corrective"
        if capability == "answer_generation":
            return "answer"
        return "supervisor"

    def _graph_snapshot_event(
        self,
        graph: TaskGraph,
        plan: IntentPlanV3,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "type": "decision_trace",
            "trace": self._trace(graph, plan, None, "", {}),
            "supervisor_decision": reason,
        }
