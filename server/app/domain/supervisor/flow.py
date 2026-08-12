from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from contextlib import aclosing
from typing import Any, AsyncGenerator

from app.domain.agents import (
    AgentExecutionContext,
    AgentExecutor,
    BusinessAgentHandlers,
    BusinessAgentServices,
    ExecutionReport,
)
from app.domain.supervisor.agent_registry import AgentRegistry
from app.domain.supervisor.prompts import PromptRegistry, build_default_prompt_registry
from app.domain.supervisor.supervisor import SupervisorPlanCompiler
from app.domain.supervisor.task_graph import TaskGraph, TaskGraphNode
from app.schemas import AgentTaskProposal, IntentPlan, ReflectionResult


@dataclass
class SupervisorFlowResult:
    intent_plan: IntentPlan
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
    """Request-scoped Supervisor execution for the decoupled Agent graph."""

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
        self.executor = AgentExecutor(
            agent_registry=self.compiler.registry,
            tool_registry=tool_registry,
            span_recorder=span_recorder,
            budget_manager=budget_manager,
            handlers=BusinessAgentHandlers(services).as_mapping(),
            prompt_registry=self.prompt_registry,
        )
        self.last_result: SupervisorFlowResult | None = None

    async def run(
        self,
        *,
        query: str,
        user_id: str,
        session_id: str,
        turn_id: str,
        intent_plan: IntentPlan,
        conversation_context: Any = None,
        query_plan: Any = None,
        image_attributes: Any = None,
        image_path: str | None = None,
        intent_span_key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        graph = self.compiler.compile(intent_plan, turn_id=turn_id)
        artifacts: dict[str, Any] = {"intent_plan": intent_plan}
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
        self.executor.begin_graph(
            completed_span_keys=(
                {"system:intent_understanding": intent_span_key}
                if intent_span_key
                else None
            )
        )
        emitted_answer_output = False

        yield self._graph_snapshot_event(graph, intent_plan, "compiled")

        pre_verifier_stop = {
            "evidence_verification",
            "bundle_optimization",
            "answer_generation",
            "memory_distillation",
        }
        report = ExecutionReport(graph=graph, artifacts=artifacts)

        # Execute all approved business work first. Runtime failures are repaired
        # before a pending Verifier can be marked blocked by its dependency.
        self._reset_phase_report(report)
        executor_stream = self.executor.execute_stream(
            graph,
            base_context=base_context,
            report=report,
            stop_before_capabilities=pre_verifier_stop,
        )
        async with aclosing(executor_stream) as phase_stream:
            async for event in phase_stream:
                translated = self._executor_event(event, graph, intent_plan)
                if event.kind == "agent_output":
                    emitted_answer_output = True
                yield translated

        runtime_repair_cycle = 0
        while report.failed_node_ids and runtime_repair_cycle < 2:
            repairable = self._repairable_nodes(graph, report.failed_node_ids)
            if not repairable:
                break
            retry_ids = self.compiler.schedule_repair(
                graph,
                [node.node_id for node in repairable],
                reason="Agent 执行失败，Supervisor 只重试失败节点。",
            )
            runtime_repair_cycle += 1
            self._annotate_last_repair(
                graph,
                kind="runtime_failure",
                cycle=runtime_repair_cycle,
                retry_ids=retry_ids,
            )
            yield self._graph_snapshot_event(graph, intent_plan, "runtime_repair_scheduled")
            self._reset_phase_report(report)
            executor_stream = self.executor.execute_stream(
                graph,
                base_context=base_context,
                report=report,
                stop_before_capabilities=pre_verifier_stop,
            )
            async with aclosing(executor_stream) as phase_stream:
                async for event in phase_stream:
                    translated = self._executor_event(event, graph, intent_plan)
                    if event.kind == "agent_output":
                        emitted_answer_output = True
                    yield translated

        pre_verifier_failed = bool(report.failed_node_ids)
        knowledge_decision = None
        if not pre_verifier_failed:
            knowledge_decision = self._knowledge_follow_up_decision(graph, base_context)
        if knowledge_decision is not None:
            decision_payload = knowledge_decision.model_dump()
            artifacts["knowledge_follow_up_decision"] = decision_payload
            graph.metadata.setdefault("supervisor_decisions", []).append(
                {
                    "subject": "knowledge_follow_up",
                    **decision_payload,
                }
            )
            if knowledge_decision.approved:
                knowledge_plan = self._knowledge_retrieval_plan(intent_plan, knowledge_decision)
                artifacts["knowledge_retrieval_plan"] = knowledge_plan
                artifacts["intent_plan"] = knowledge_plan
                self._append_knowledge_follow_up(graph, knowledge_decision)
                yield self._graph_snapshot_event(graph, intent_plan, "knowledge_follow_up_approved")
            elif knowledge_decision.decision not in {"not_applicable", "already_planned"}:
                yield self._graph_snapshot_event(graph, intent_plan, "knowledge_follow_up_rejected")

        verifier_node = next(
            (node for node in graph.nodes if node.capability == "evidence_verification" and node.status == "pending"),
            None,
        )
        if verifier_node is not None and not pre_verifier_failed:
            # Run Verifier alone so its structured repair decision is handled
            # before BundleOptimizer or AnswerGenerator sees the evidence.
            self._reset_phase_report(report)
            executor_stream = self.executor.execute_stream(
                graph,
                base_context=base_context,
                report=report,
                stop_before_capabilities={"bundle_optimization", "answer_generation", "memory_distillation"},
            )
            async with aclosing(executor_stream) as phase_stream:
                async for event in phase_stream:
                    translated = self._executor_event(event, graph, intent_plan)
                    if event.kind == "agent_output":
                        emitted_answer_output = True
                    yield translated

            quality_repair_cycle = 0
            current_verifier = verifier_node
            while not report.failed_node_ids and quality_repair_cycle < 2:
                targets = self._quality_failure_targets(graph, artifacts)
                if not targets:
                    break
                target_nodes = [graph.require_node(node_id) for node_id in targets]
                for node in target_nodes:
                    if node.status == "succeeded":
                        node.status = "failed"
                repairable = self._repairable_nodes(graph, targets)
                if not repairable:
                    break
                retry_ids = self.compiler.schedule_repair(
                    graph,
                    [node.node_id for node in repairable],
                    reason="EvidenceVerifier 标记候选证据不足或不匹配。",
                )
                quality_repair_cycle += 1
                next_verifier = self._append_verifier_retry(
                    graph,
                    previous_verifier=current_verifier,
                    retry_node_ids=retry_ids,
                    cycle=quality_repair_cycle,
                )
                self._annotate_last_repair(
                    graph,
                    kind="evidence_quality",
                    cycle=quality_repair_cycle,
                    retry_ids=retry_ids,
                    verifier_node_id=next_verifier.node_id,
                )
                yield self._graph_snapshot_event(graph, intent_plan, "repair_scheduled")
                self._reset_phase_report(report)
                executor_stream = self.executor.execute_stream(
                    graph,
                    base_context=base_context,
                    report=report,
                    stop_before_capabilities={"evidence_verification", "bundle_optimization", "answer_generation", "memory_distillation"},
                )
                async with aclosing(executor_stream) as phase_stream:
                    async for event in phase_stream:
                        translated = self._executor_event(event, graph, intent_plan)
                        if event.kind == "agent_output":
                            emitted_answer_output = True
                        yield translated
                if report.failed_node_ids:
                    break

                # The retry evidence is complete. The new verifier has explicit
                # dependencies on retry nodes, so this is a real second review.
                self._reset_phase_report(report)
                executor_stream = self.executor.execute_stream(
                    graph,
                    base_context=base_context,
                    report=report,
                    stop_before_capabilities={"bundle_optimization", "answer_generation", "memory_distillation"},
                )
                async with aclosing(executor_stream) as phase_stream:
                    async for event in phase_stream:
                        translated = self._executor_event(event, graph, intent_plan)
                        if event.kind == "agent_output":
                            emitted_answer_output = True
                        yield translated
                current_verifier = next_verifier

        # Finish optimizer, answer and the lightweight memory scheduling node.
        self._reset_phase_report(report)
        executor_stream = self.executor.execute_stream(
            graph,
            base_context=base_context,
            report=report,
        )
        async with aclosing(executor_stream) as phase_stream:
            async for event in phase_stream:
                translated = self._executor_event(event, graph, intent_plan)
                if event.kind == "agent_output":
                    emitted_answer_output = True
                yield translated

        # Required upstream failures must still terminate with an honest answer.
        # A dedicated fallback node preserves the failed lineage instead of
        # mutating and pretending the original answer path succeeded.
        if self._needs_fallback_answer(graph):
            artifacts["reflection"] = self._fallback_reflection(artifacts.get("reflection"))
            fallback_answer = self._append_fallback_answer(graph)
            yield self._graph_snapshot_event(graph, intent_plan, "fallback_answer_scheduled")
            self._reset_phase_report(report)
            executor_stream = self.executor.execute_stream(
                graph,
                base_context=base_context,
                report=report,
            )
            async with aclosing(executor_stream) as phase_stream:
                async for event in phase_stream:
                    translated = self._executor_event(event, graph, intent_plan)
                    if event.kind == "agent_output":
                        emitted_answer_output = True
                    yield translated
            if fallback_answer.status != "succeeded":
                graph.metadata["termination_reason"] = "fallback_answer_failed"

        report.failed_node_ids = self._unresolved_failure_ids(graph)
        report.blocked_node_ids = [
            node.node_id for node in graph.nodes
            if node.required and node.status == "skipped" and not node.metadata.get("superseded_by")
        ]
        report.completed = graph.all_terminal()

        answer_result = next(
            (result for node_id, result in reversed(list(report.results.items())) if graph.get_node(node_id) and graph.get_node(node_id).capability == "answer_generation"),
            None,
        )
        if answer_result is None:
            # A clarification graph has no AnswerGenerator node by design.
            answer_result = next(
                (result for node_id, result in reversed(list(report.results.items())) if graph.get_node(node_id) and graph.get_node(node_id).capability == "clarification"),
                None,
            )
        output = answer_result.output if answer_result is not None else {}
        answer_text = str(artifacts.get("answer_text") or output.get("question") or "")
        answer_tokens = list(artifacts.get("answer_tokens") or ([answer_text] if answer_text else []))
        cards = list(artifacts.get("answer_cards") or [])
        product_ids = list(artifacts.get("answer_product_ids") or [])
        route = str(artifacts.get("answer_route") or self._route_from_plan(intent_plan, output))
        reflection = artifacts.get("reflection")
        trace = self._trace(graph, intent_plan, report, route, reflection)
        self.last_result = SupervisorFlowResult(
            intent_plan=intent_plan,
            graph=graph,
            report=report,
            route=route,
            answer_text=answer_text,
            answer_tokens=answer_tokens,
            cards=cards,
            product_ids=product_ids,
            reflection=reflection,
            trace=trace,
        )
        yield {"type": "decision_trace", "trace": trace}
        if not emitted_answer_output:
            if cards:
                yield {"type": "product_cards", "products": cards}
            for token in answer_tokens:
                yield {"type": "token", "content": token}
        yield {"type": "done"}

    def _reset_phase_report(self, report: ExecutionReport) -> None:
        report.failed_node_ids = []
        report.blocked_node_ids = []
        report.completed = False

    def _repairable_nodes(self, graph: TaskGraph, node_ids: list[str]) -> list[TaskGraphNode]:
        result: list[TaskGraphNode] = []
        for node_id in self._unique(node_ids):
            node = graph.get_node(node_id)
            if node is None:
                continue
            if node.status not in {"failed", "timeout"}:
                continue
            if node.attempt >= node.max_attempts:
                continue
            result.append(node)
        return result

    def _annotate_last_repair(
        self,
        graph: TaskGraph,
        *,
        kind: str,
        cycle: int,
        retry_ids: list[str],
        verifier_node_id: str = "",
    ) -> None:
        history = graph.metadata.setdefault("repair_history", [])
        if not history:
            return
        history[-1].update(
            {
                "kind": kind,
                "flow_cycle": cycle,
                "retry_node_ids": list(retry_ids),
                **({"verifier_node_id": verifier_node_id} if verifier_node_id else {}),
            }
        )

    def _append_verifier_retry(
        self,
        graph: TaskGraph,
        *,
        previous_verifier: TaskGraphNode,
        retry_node_ids: list[str],
        cycle: int,
    ) -> TaskGraphNode:
        registration = self.compiler.registry.select_for_capability(
            "evidence_verification",
            execution_mode=str(graph.metadata.get("execution_mode") or ""),
        )
        if registration is None:
            raise RuntimeError("EvidenceVerifierAgent is not registered")
        replacements = {
            str(graph.require_node(node_id).metadata.get("retry_of") or ""): node_id
            for node_id in retry_node_ids
        }
        dependencies = self._unique(
            [replacements.get(dependency_id, dependency_id) for dependency_id in previous_verifier.depends_on]
        )
        attempt = previous_verifier.attempt + 1
        node = TaskGraphNode(
            node_id=f"runtime:evidence_verification:{cycle + 1}",
            task_id=f"{graph.turn_id}:evidence_verification:{attempt}",
            agent_id=registration.manifest.agent_id,
            capability="evidence_verification",
            depends_on=dependencies,
            attempt=attempt,
            max_attempts=max(attempt, registration.manifest.max_attempts),
            metadata={
                "phase": "evidence_reverification",
                "previous_verifier_node_id": previous_verifier.node_id,
                "retry_node_ids": list(retry_node_ids),
                "allowed_tools": list(registration.manifest.allowed_tools),
            },
        )
        for downstream in graph.nodes:
            if downstream.status != "pending" or previous_verifier.node_id not in downstream.depends_on:
                continue
            downstream.depends_on = [
                node.node_id if dependency_id == previous_verifier.node_id else dependency_id
                for dependency_id in downstream.depends_on
            ]
        graph.add_node(node)
        graph.refresh_terminal_nodes()
        graph.validate_graph()
        return node

    def _needs_fallback_answer(self, graph: TaskGraph) -> bool:
        answer_nodes = [node for node in graph.nodes if node.capability == "answer_generation"]
        return bool(answer_nodes) and not any(node.status == "succeeded" for node in answer_nodes)

    def _append_fallback_answer(self, graph: TaskGraph) -> TaskGraphNode:
        existing = graph.get_node("runtime:fallback_answer")
        if existing is not None:
            return existing
        registration = self.compiler.registry.select_for_capability(
            "answer_generation",
            execution_mode=str(graph.metadata.get("execution_mode") or ""),
        )
        if registration is None:
            raise RuntimeError("AnswerGenerator is not registered")
        node = TaskGraphNode(
            node_id="runtime:fallback_answer",
            task_id=f"{graph.turn_id}:fallback_answer",
            agent_id=registration.manifest.agent_id,
            capability="answer_generation",
            depends_on=[],
            metadata={
                "phase": "fallback_answer",
                "reason": "Required upstream evidence path did not complete.",
                "allowed_tools": list(registration.manifest.allowed_tools),
            },
        )
        for answer_node in [item for item in graph.nodes if item.capability == "answer_generation"]:
            answer_node.metadata = {**answer_node.metadata, "superseded_by": node.node_id}
        for pending_node in graph.nodes:
            if pending_node.status != "pending":
                continue
            pending_node.status = "skipped"
            pending_node.metadata = {
                **pending_node.metadata,
                "skipped_reason": "fallback_path_selected",
                "superseded_by": node.node_id,
            }
        graph.add_node(node)
        for memory_node in [item for item in graph.nodes if item.capability == "memory_distillation"]:
            if memory_node.status != "succeeded":
                memory_node.status = "pending"
                memory_node.depends_on = [node.node_id]
                memory_node.metadata = {
                    **memory_node.metadata,
                    "fallback_answer_node_id": node.node_id,
                }
        graph.refresh_terminal_nodes()
        graph.validate_graph()
        return node

    def _fallback_reflection(self, value: Any) -> ReflectionResult:
        if isinstance(value, ReflectionResult):
            return value.model_copy(
                update={
                    "has_passed_products": False,
                    "passed_product_ids": [],
                    "fallback_plan": "no_product",
                    "reason": value.reason or "上游 Agent 执行失败，无法形成可靠商品证据。",
                }
            )
        return ReflectionResult(
            has_passed_products=False,
            reason="上游 Agent 执行失败，无法形成可靠商品证据。",
            fallback_plan="no_product",
        )

    def _unresolved_failure_ids(self, graph: TaskGraph) -> list[str]:
        unresolved: list[str] = []
        for node in graph.nodes:
            if node.status not in {"failed", "timeout", "cancelled"}:
                continue
            replacement_id = str(node.metadata.get("superseded_by") or "")
            seen: set[str] = set()
            replacement = graph.get_node(replacement_id) if replacement_id else None
            while replacement is not None and replacement.node_id not in seen:
                seen.add(replacement.node_id)
                next_id = str(replacement.metadata.get("superseded_by") or "")
                if not next_id:
                    break
                replacement = graph.get_node(next_id)
            if replacement is None or replacement.status != "succeeded":
                unresolved.append(node.node_id)
        return unresolved

    def _executor_event(self, event: Any, graph: TaskGraph, plan: IntentPlan) -> dict[str, Any]:
        payload = event.as_dict()
        if event.kind == "agent_output":
            client_event = payload.get("event")
            if isinstance(client_event, dict):
                return client_event
            return {"type": "trace", "stage": "agent_output", "content": "Agent emitted an invalid client event."}
        if event.kind == "node_started":
            return {
                "type": "agent_update",
                "stage": self._stage_for_capability(payload.get("capability", "")),
                "title": f"{payload.get('agent_id', 'Agent')} 执行中",
                "content_delta": f"正在执行 {payload.get('capability', '')}。",
                "done": False,
                "run_id": getattr(self.executor.span_recorder, "run_id", ""),
                "task_id": payload.get("task_id", ""),
                "agent_id": payload.get("agent_id", ""),
            }
        if event.kind == "node_finished":
            span = self._latest_span(payload.get("task_id", ""))
            timing = self.executor.span_recorder.timing_event(span) if span is not None else {
                "type": "timing_update",
                "run_id": getattr(self.executor.span_recorder, "run_id", ""),
            }
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
                },
            }
        if event.kind == "node_blocked":
            return {
                "type": "agent_update",
                "stage": "supervisor",
                "title": "Supervisor 路由",
                "content_delta": f"节点 {event.node_id} 因依赖失败被跳过。",
                "done": True,
            }
        return {
            "type": "decision_trace",
            "trace": self._trace(graph, plan, None, "", None),
            "executor_event": payload,
        }

    def _latest_span(self, task_id: str) -> dict[str, Any] | None:
        spans = getattr(self.executor.span_recorder, "spans", [])
        for span in reversed(spans):
            if not task_id or span.get("task_id") == task_id:
                return span
        return spans[-1] if spans else None

    def _knowledge_follow_up_decision(self, graph: TaskGraph, context: AgentExecutionContext) -> Any:
        knowledge = context.artifact("knowledge_research")
        if not isinstance(knowledge, dict):
            return None
        plan = context.artifact("intent_plan", context.intent_plan)
        if not isinstance(plan, IntentPlan):
            return None
        knowledge_node = next((node for node in graph.nodes if node.capability == "knowledge_research"), None)
        evidence = (
            context.artifact(f"evidence:{knowledge_node.node_id}", [])
            if knowledge_node is not None
            else []
        )
        existing_product_path = any(
            node.capability in {
                "single_product_recommendation",
                "multi_product_bundle",
                "slot_product_retrieval",
            }
            for node in graph.nodes
        )
        decision = self.compiler.policy_gate.approve_knowledge_follow_up(
            plan,
            knowledge_artifact=knowledge,
            evidence_refs=evidence if isinstance(evidence, list) else [],
            existing_product_path=existing_product_path,
        )
        if not decision.approved:
            return decision
        registration = self.compiler.registry.select_for_capability("single_product_recommendation", execution_mode="single_product")
        verifier = next((node for node in graph.nodes if node.capability == "evidence_verification"), None)
        if registration is None or verifier is None or knowledge_node is None:
            return decision.model_copy(
                update={
                    "decision": "reject_knowledge_to_retrieval",
                    "approved": False,
                    "reason": "Supervisor 无法创建完整的推荐与校验节点，停止追加检索。",
                    "approved_query": "",
                }
            )
        node_id = "runtime:knowledge_recommendation"
        if graph.get_node(node_id) is not None:
            return decision.model_copy(
                update={
                    "decision": "already_scheduled",
                    "approved": False,
                    "reason": "知识衍生商品检索节点已经存在，不重复调度。",
                    "approved_query": "",
                }
            )
        return decision

    def _append_knowledge_follow_up(self, graph: TaskGraph, decision: Any) -> None:
        registration = self.compiler.registry.select_for_capability(
            "single_product_recommendation",
            execution_mode="single_product",
        )
        verifier = next((node for node in graph.nodes if node.capability == "evidence_verification"), None)
        knowledge_node = next((node for node in graph.nodes if node.capability == "knowledge_research"), None)
        if registration is None or verifier is None or knowledge_node is None:
            raise RuntimeError("Approved knowledge follow-up cannot be materialized")
        node_id = "runtime:knowledge_recommendation"
        graph.add_node(
            TaskGraphNode(
                node_id=node_id,
                task_id=f"{graph.turn_id}:knowledge_recommendation",
                agent_id=registration.manifest.agent_id,
                capability="single_product_recommendation",
                depends_on=[knowledge_node.node_id],
                max_attempts=registration.manifest.max_attempts,
                metadata={
                    "reason": decision.reason,
                    "query_override": decision.approved_query,
                    "supervisor_approved": True,
                    "approved_concepts": list(decision.approved_concepts),
                    "supporting_evidence_ids": list(decision.supporting_evidence_ids),
                    "risk_flags": list(decision.risk_flags),
                    "intent_plan_artifact": "knowledge_retrieval_plan",
                    "allowed_tools": list(registration.manifest.allowed_tools),
                },
            )
        )
        verifier.depends_on = [
            node_id if dependency_id == knowledge_node.node_id else dependency_id
            for dependency_id in verifier.depends_on
        ]
        # Preserve the source knowledge node as evidence lineage as well as the
        # derived recommendation node.
        if knowledge_node.node_id not in verifier.depends_on:
            verifier.depends_on.append(knowledge_node.node_id)
        if node_id not in verifier.depends_on:
            verifier.depends_on.append(node_id)
        graph.validate_graph()

    def _knowledge_retrieval_plan(
        self,
        original: IntentPlan,
        decision: Any,
    ) -> IntentPlan:
        query = str(decision.approved_query or original.normalized_query or original.original_query)
        return original.model_copy(
            update={
                "normalized_query": query,
                "execution_mode": "single_product",
                "plan_type": "single_retrieval",
                "vector_query": query,
                "keyword_query": query,
                "plan_reason": "KnowledgeResearchAgent 证据经 Supervisor 审批后形成商品检索概念。",
            }
        )

    def _quality_failure_targets(self, graph: TaskGraph, artifacts: dict[str, Any]) -> list[str]:
        reflection = artifacts.get("reflection")
        if reflection is None or not getattr(reflection, "repair_hint", None) or not reflection.repair_hint.repairable:
            return []
        targets = set(reflection.repair_hint.target_slot_ids or [])
        nodes: list[str] = []

        single_nodes = [
            node for node in graph.nodes
            if node.capability == "single_product_recommendation" and node.status == "succeeded"
        ]
        if single_nodes and (not targets or "single" in targets):
            nodes.append(max(single_nodes, key=lambda item: item.attempt).node_id)

        selected_slots: list[TaskGraphNode] = []
        by_slot: dict[str, list[TaskGraphNode]] = {}
        for node in graph.nodes:
            if node.capability != "slot_product_retrieval" or node.status != "succeeded":
                continue
            slot_id = str(node.metadata.get("slot", {}).get("slot_id") or "")
            if slot_id and (not targets or slot_id in targets):
                by_slot.setdefault(slot_id, []).append(node)
        for candidates in by_slot.values():
            selected_slots.append(max(candidates, key=lambda item: item.attempt))
        nodes.extend(node.node_id for node in selected_slots)

        # A quality retry for one or more slots must also rebuild the merged
        # MultiNeedState; otherwise the next Verifier would review stale evidence.
        if selected_slots:
            merge_nodes = [
                node for node in graph.nodes
                if node.capability == "multi_product_bundle"
                and node.metadata.get("phase") == "merge_slot_evidence"
                and node.status == "succeeded"
            ]
            if merge_nodes:
                nodes.append(max(merge_nodes, key=lambda item: item.attempt).node_id)
        return self._unique(nodes)

    def _route_from_plan(self, plan: IntentPlan, output: dict[str, Any]) -> str:
        if plan.plan_type == "clarify":
            return "clarify"
        if plan.plan_type == "direct_answer":
            return "direct_answer"
        return str(output.get("route") or "no_product")

    def _trace(
        self,
        graph: TaskGraph,
        plan: IntentPlan,
        report: ExecutionReport | None,
        route: str,
        reflection: Any,
    ) -> dict[str, Any]:
        nodes = [
            {
                "node_id": node.node_id,
                "task_id": node.task_id,
                "agent_id": node.agent_id,
                "capability": node.capability,
                "depends_on": list(node.depends_on),
                "status": node.status,
                "attempt": node.attempt,
                "phase": node.metadata.get("phase", ""),
            }
            for node in graph.nodes
        ]
        unresolved_failures = list(report.failed_node_ids) if report else []
        pending = [node.node_id for node in graph.nodes if node.status in {"pending", "running"}]
        task_status = "running" if pending else ("failed" if unresolved_failures else "succeeded")
        return {
            "route": route,
            "task_status": task_status,
            "task": {
                "graph_id": graph.graph_id,
                "turn_id": graph.turn_id,
                "schema_version": graph.schema_version,
                "execution_mode": graph.metadata.get("execution_mode"),
                "repair_history": graph.metadata.get("repair_history", []),
                "supervisor_decisions": graph.metadata.get("supervisor_decisions", []),
            },
            "planner_proposal": plan.model_dump(),
            "agent_path": nodes,
            "reflection": reflection.model_dump() if hasattr(reflection, "model_dump") else reflection or {},
            "failed_node_ids": list(report.failed_node_ids) if report else [],
            "blocked_node_ids": list(report.blocked_node_ids) if report else [],
        }

    def _stage_for_capability(self, capability: str) -> str:
        if capability in {"intent_understanding", "profile_preference", "clarification"}:
            return "planner"
        if capability in {"single_product_recommendation", "multi_product_bundle", "slot_product_retrieval", "commerce_research", "knowledge_research", "comparison"}:
            return "retrieval"
        if capability in {"evidence_verification", "repair", "bundle_optimization"}:
            return "corrective"
        if capability == "answer_generation":
            return "answer"
        return "supervisor"

    def _unique(self, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value for value in values if value))

    def _graph_snapshot_event(self, graph: TaskGraph, plan: IntentPlan, reason: str) -> dict[str, Any]:
        return {
            "type": "decision_trace",
            "trace": self._trace(graph, plan, None, "", None),
            "supervisor_decision": reason,
        }
