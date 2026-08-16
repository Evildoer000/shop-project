from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from typing import Any

from app.domain.agents.contracts import (
    AgentExecutionContext,
    AgentHandler,
    AgentResult,
    ExecutionReport,
)
from app.domain.supervisor.agent_registry import AgentRegistry
from app.domain.supervisor.task_graph import (
    FAILURE_STATUSES,
    AgentHandoff,
    TaskGraph,
    TaskGraphNode,
)
from app.domain.supervisor.tool_access import AgentToolAccess
from app.harness.tool_registry import ToolRegistry
from app.services.llm_client import bind_llm_trace_context


class GraphExecutionError(RuntimeError):
    pass


@dataclass
class ExecutorEvent:
    kind: str
    node_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "node_id": self.node_id, **self.payload}


class AgentExecutor:
    """Runs approved graph nodes; it never chooses a business route."""

    def __init__(
        self,
        *,
        agent_registry: AgentRegistry,
        tool_registry: ToolRegistry,
        span_recorder: Any = None,
        budget_manager: Any = None,
        handlers: dict[str, AgentHandler] | None = None,
        prompt_registry: Any = None,
    ) -> None:
        self.agent_registry = agent_registry
        self.tool_registry = tool_registry
        self.tool_access = AgentToolAccess(tool_registry)
        self.span_recorder = span_recorder
        self.budget_manager = budget_manager
        self.handlers = handlers or {}
        self.prompt_registry = prompt_registry
        self._span_keys: dict[str, str] = {}
        self._handoff_span_keys: dict[str, str] = {}
        self._observed_handoff_statuses: dict[str, str] = {}

    def register(self, capability: str, handler: AgentHandler) -> None:
        if capability in self.handlers:
            raise ValueError(f"Agent handler already registered: {capability}")
        self.handlers[capability] = handler

    def begin_graph(self, *, completed_span_keys: dict[str, str] | None = None) -> None:
        """Reset request-scoped lineage before executing a newly compiled graph."""
        self._span_keys = dict(completed_span_keys or {})
        self._handoff_span_keys = {}
        self._observed_handoff_statuses = {}

    def register_completed_node_span(self, node_id: str, span_key: str | None) -> None:
        """Attach a code-owned system node to the same request execution tree."""
        if node_id and span_key:
            self._span_keys[node_id] = span_key

    def span_key_for_node(self, node_id: str) -> str | None:
        return self._span_keys.get(node_id)

    async def execute(
        self,
        graph: TaskGraph,
        *,
        base_context: AgentExecutionContext,
        stop_before_capabilities: set[str] | None = None,
    ) -> ExecutionReport:
        report = ExecutionReport(graph=graph)
        async for _ in self.execute_stream(
            graph,
            base_context=base_context,
            report=report,
            stop_before_capabilities=stop_before_capabilities,
        ):
            pass
        return report

    async def execute_stream(
        self,
        graph: TaskGraph,
        *,
        base_context: AgentExecutionContext,
        report: ExecutionReport | None = None,
        stop_before_capabilities: set[str] | None = None,
    ) -> AsyncGenerator[ExecutorEvent, None]:
        graph.validate_graph()
        execution_report = report or ExecutionReport(graph=graph)
        # All node outputs are visible to later dependency nodes, but remain
        # scoped to this one request's execution context.
        if execution_report.artifacts is not base_context.artifacts:
            execution_report.artifacts.update(base_context.artifacts)
            base_context.artifacts = execution_report.artifacts
        event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        base_context.metadata["executor_event_queue"] = event_queue
        # Keep prior span keys when a Supervisor appends a Repair/retry cycle;
        # a retry must remain linked to the original failed node's lineage.
        if not self._span_keys:
            self._span_keys = {}
        active_nodes: list[TaskGraphNode] = []
        node_tasks: dict[str, asyncio.Task[AgentResult]] = {}
        queue_task: asyncio.Task[dict[str, Any]] | None = None
        interruption_reason = "executor_stream_closed"

        try:
            for handoff_event in self._handoff_events(graph):
                yield handoff_event
            while True:
                while True:
                    blocked = graph.mark_blocked_nodes(
                        exclude_capabilities=stop_before_capabilities,
                    )
                    if not blocked:
                        break
                    execution_report.blocked_node_ids.extend(blocked)
                    for node_id in blocked:
                        yield ExecutorEvent(
                            kind="node_blocked",
                            node_id=node_id,
                            payload={"reason": "dependency_failed"},
                        )
                    for handoff_event in self._handoff_events(graph):
                        yield handoff_event

                ready = graph.ready_nodes()
                if not ready:
                    pending = [node for node in graph.nodes if node.status == "pending"]
                    if pending:
                        deferred = [
                            node
                            for node in pending
                            if stop_before_capabilities
                            and node.capability in stop_before_capabilities
                        ]
                        if len(deferred) == len(pending):
                            break
                        raise GraphExecutionError(
                            "Task graph has pending nodes but none are ready: "
                            + ", ".join(node.node_id for node in pending)
                        )
                    break

                if stop_before_capabilities:
                    deferred = [
                        node for node in ready
                        if node.capability in stop_before_capabilities
                    ]
                    ready = [node for node in ready if node not in deferred]
                    if not ready:
                        # Supervisor intentionally pauses before a terminal phase
                        # so it can approve a follow-up graph node.
                        break

                active_nodes = list(ready)
                for node in active_nodes:
                    graph.accept_handoffs(node.node_id)
                for node in active_nodes:
                    node.status = "running"
                node_tasks = {
                    node.node_id: asyncio.create_task(
                        self._run_node(node, base_context, graph),
                        name=f"agent:{node.agent_id}:{node.task_id}",
                    )
                    for node in active_nodes
                }
                for handoff_event in self._handoff_events(graph):
                    yield handoff_event

                # Tasks exist before a start event is exposed. If the SSE
                # consumer closes on that event, the finally block can cancel
                # and account for every node in this batch.
                for node in active_nodes:
                    yield ExecutorEvent(
                        kind="node_started",
                        node_id=node.node_id,
                        payload={
                            "task_id": node.task_id,
                            "agent_id": node.agent_id,
                            "capability": node.capability,
                            "attempt": node.attempt,
                        },
                    )

                pending_tasks = set(node_tasks.values())
                while pending_tasks:
                    queue_task = asyncio.create_task(event_queue.get())
                    done, _ = await asyncio.wait(
                        {*pending_tasks, queue_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    pending_tasks.difference_update(done)
                    if queue_task in done:
                        queued = queue_task.result()
                        queue_task = None
                        yield ExecutorEvent(
                            kind=str(queued.get("kind") or "agent_output"),
                            node_id=str(queued.get("node_id") or ""),
                            payload=dict(queued.get("payload") or {}),
                        )
                    else:
                        await self._cancel_task(queue_task)
                        queue_task = None

                while not event_queue.empty():
                    queued = event_queue.get_nowait()
                    yield ExecutorEvent(
                        kind=str(queued.get("kind") or "agent_output"),
                        node_id=str(queued.get("node_id") or ""),
                        payload=dict(queued.get("payload") or {}),
                    )

                for node in active_nodes:
                    result = node_tasks[node.node_id].result()
                    self._store_result(execution_report, node, result)
                    graph.complete_source_handoffs(
                        node.node_id,
                        evidence_refs=[item.evidence_id for item in result.evidence if item.evidence_id],
                    )
                    graph.consume_handoffs(node.node_id)
                    yield ExecutorEvent(
                        kind="node_finished",
                        node_id=node.node_id,
                        payload={
                            "task_id": node.task_id,
                            "agent_id": node.agent_id,
                            "capability": node.capability,
                            "status": node.status,
                            "attempt": node.attempt,
                            "result": result.compact(),
                        },
                    )
                    for handoff_event in self._handoff_events(graph):
                        yield handoff_event

                active_nodes = []
                node_tasks = {}

            execution_report.completed = not [node for node in graph.nodes if node.status == "pending"]
            for handoff_event in self._handoff_events(graph):
                yield handoff_event
            yield ExecutorEvent(
                kind="graph_finished",
                payload={
                    "completed": execution_report.completed,
                    "failed_node_ids": list(dict.fromkeys(execution_report.failed_node_ids)),
                    "blocked_node_ids": list(dict.fromkeys(execution_report.blocked_node_ids)),
                },
            )
        except asyncio.CancelledError:
            interruption_reason = "executor_task_cancelled"
            raise
        except GeneratorExit:
            interruption_reason = "executor_stream_closed"
            raise
        except BaseException:
            interruption_reason = "executor_aborted_by_exception"
            raise
        finally:
            await self._cancel_task(queue_task)
            if node_tasks:
                await self._finalize_interrupted_batch(
                    active_nodes,
                    node_tasks,
                    execution_report,
                    interruption_reason,
                )
                self._handoff_events(graph)
            if base_context.metadata.get("executor_event_queue") is event_queue:
                base_context.metadata.pop("executor_event_queue", None)

    async def _run_node(
        self,
        node: TaskGraphNode,
        base_context: AgentExecutionContext,
        graph: TaskGraph,
    ) -> AgentResult:
        started = time.perf_counter()
        try:
            registration = self.agent_registry.require(node.agent_id)
        except Exception as exc:
            result = AgentResult.failure_result(
                "agent_not_registered",
                str(exc),
                retryable=False,
                termination_reason="configuration_error",
            )
            self._record_synthetic_failure_span(node, result, started, graph)
            return result
        if registration.manifest.capability != node.capability:
            result = AgentResult.failure_result(
                "agent_capability_mismatch",
                (
                    f"Agent {node.agent_id} provides {registration.manifest.capability}, "
                    f"not {node.capability}"
                ),
                retryable=False,
                termination_reason="configuration_error",
            )
            self._record_synthetic_failure_span(node, result, started, graph)
            return result
        handler = self.handlers.get(node.capability)
        parent_span_key = self._parent_span_key(node, graph)
        prompt_spec = self.prompt_registry.get(node.agent_id) if self.prompt_registry is not None else None
        span = None
        if self.span_recorder is not None:
            span = self.span_recorder.start_span(
                (
                    f"policy:{node.agent_id}"
                    if node.capability == "policy_gate"
                    else f"agent:{node.agent_id}"
                ),
                label=node.metadata.get("phase") or node.capability,
                agent=node.agent_id,
                agent_id=node.agent_id,
                task_id=node.task_id,
                span_type="policy" if node.capability == "policy_gate" else "agent",
                attempt=node.attempt,
                parent_span_key=parent_span_key,
                input_summary={
                    "node_id": node.node_id,
                    "capability": node.capability,
                    "depends_on": list(node.depends_on),
                    "input_refs": list(node.input_refs),
                    "handoff_ids": [item.handoff_id for item in graph.handoffs_to(node.node_id)],
                    "dependency_span_keys": [
                        self._span_keys.get(item)
                        for item in self._dependency_ids(node)
                    ],
                    "prompt_id": prompt_spec.agent_id if prompt_spec is not None else "",
                    "prompt_version": prompt_spec.version if prompt_spec is not None else "",
                    "allowed_tools": list(registration.manifest.allowed_tools),
                },
            )

        if handler is None:
            result = AgentResult.failure_result(
                "handler_not_registered",
                f"No handler registered for capability={node.capability}",
                retryable=False,
            )
        else:
            dependency_outputs = {
                dependency_id: self._output_for_dependency(dependency_id, base_context)
                for dependency_id in self._dependency_ids(node)
            }
            dependency_artifacts = {
                dependency_id: self._artifacts_for_dependency(
                    dependency_id,
                    base_context,
                )
                for dependency_id in graph.ancestor_node_ids(node.node_id)
            }
            context = AgentExecutionContext(
                node=node,
                graph=graph,
                query=base_context.query,
                user_id=base_context.user_id,
                session_id=base_context.session_id,
                turn_id=base_context.turn_id,
                intent_plan=base_context.artifacts.get("intent_plan", base_context.intent_plan),
                conversation_context=base_context.conversation_context,
                query_plan=base_context.query_plan,
                image_attributes=base_context.image_attributes,
                artifacts=base_context.artifacts,
                dependency_outputs=dependency_outputs,
                dependency_artifacts=dependency_artifacts,
                tool_access=self.tool_access,
                prompt_registry=self.prompt_registry,
                span_recorder=self.span_recorder,
                budget_manager=self.budget_manager,
                services=base_context.services,
                current_span=span,
                metadata=dict(base_context.metadata),
            )
            context.metadata.update(node.metadata)
            manifest_tools = tuple(registration.manifest.allowed_tools)
            requested_tools = context.metadata.get("allowed_tools")
            if requested_tools is None:
                effective_tools = list(manifest_tools)
            else:
                effective_tools = [
                    str(tool_name)
                    for tool_name in requested_tools
                    if str(tool_name) in manifest_tools
                ]
            context.metadata["allowed_tools"] = effective_tools
            context.node.metadata["allowed_tools"] = effective_tools
            if prompt_spec is not None:
                allowed_tool_descriptions = self.tool_access.describe_for_agent(
                    node.agent_id,
                    tuple(effective_tools),
                )
                context.metadata["prompt_id"] = prompt_spec.agent_id
                context.metadata["prompt_version"] = prompt_spec.version
                context.metadata["agent_system_prompt"] = prompt_spec.render_system(
                    allowed_tools=allowed_tool_descriptions
                )
            try:
                with bind_llm_trace_context(
                    run_id=getattr(self.span_recorder, "run_id", ""),
                    parent_span_key=getattr(span, "span_key", ""),
                    task_id=node.task_id,
                    agent_id=node.agent_id,
                    attempt=node.attempt,
                    span_recorder=self.span_recorder,
                ):
                    value = handler(context)
                    if inspect.isawaitable(value):
                        value = await asyncio.wait_for(value, timeout=registration.manifest.timeout_ms / 1000)
                result = value if isinstance(value, AgentResult) else AgentResult.success(dict(value or {}))
                if context.tool_call_records:
                    result.tool_calls.extend(context.tool_call_records)
            except asyncio.TimeoutError:
                result = AgentResult.failure_result(
                    "agent_timeout",
                    f"Agent exceeded timeout_ms={registration.manifest.timeout_ms}",
                    retryable=True,
                    timeout=True,
                    termination_reason="timeout",
                )
            except asyncio.CancelledError:
                result = AgentResult.cancelled_result()
            except Exception as exc:  # Agent failures are data for Repair, not process crashes.
                result = AgentResult.failure_result(
                    type(exc).__name__,
                    str(exc),
                    retryable=True,
                    termination_reason="unexpected_exception",
                )

        if span is not None:
            self._span_keys[node.node_id] = span.span_key
            self.span_recorder.finish_span(
                span,
                status=result.status,
                output_summary={
                    "keys": sorted(result.output),
                    "evidence_count": len(result.evidence),
                    "tool_call_count": len(result.tool_calls),
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                },
                metrics={
                    "evidence_count": len(result.evidence),
                    "tool_call_count": len(result.tool_calls),
                    "attempt": node.attempt,
                },
                error_type=result.failure.error_type if result.failure else "",
                error_message=result.failure.message if result.failure else "",
                termination_reason=result.termination_reason,
            )
        return result

    def _record_synthetic_failure_span(
        self,
        node: TaskGraphNode,
        result: AgentResult,
        started: float,
        graph: TaskGraph,
    ) -> None:
        """Keep graph configuration failures attributable to their task node."""
        if self.span_recorder is None or node.node_id in self._span_keys:
            return
        record = getattr(self.span_recorder, "record_completed_span", None)
        if not callable(record):
            return
        failure = result.failure
        payload = record(
            (
                f"policy:{node.agent_id}"
                if node.capability == "policy_gate"
                else f"agent:{node.agent_id}"
            ),
            duration_ms=max(0.0, (time.perf_counter() - started) * 1000),
            label=node.metadata.get("phase") or node.capability,
            parent_span_key=self._parent_span_key(node, graph),
            task_id=node.task_id,
            agent_id=node.agent_id,
            span_type="policy" if node.capability == "policy_gate" else "agent",
            attempt=node.attempt,
            status=result.status,
            input_summary={
                "node_id": node.node_id,
                "capability": node.capability,
                "depends_on": list(node.depends_on),
                "input_refs": list(node.input_refs),
                "dependency_span_keys": [
                    self._span_keys.get(item)
                    for item in self._dependency_ids(node)
                ],
            },
            output_summary={"configuration_valid": False},
            error_type=failure.error_type if failure else "",
            error_message=failure.message if failure else "",
            termination_reason=result.termination_reason or "configuration_error",
        )
        span_key = str(payload.get("span_key") or "")
        if span_key:
            self._span_keys[node.node_id] = span_key

    async def _finalize_interrupted_batch(
        self,
        nodes: list[TaskGraphNode],
        tasks: dict[str, asyncio.Task[AgentResult]],
        report: ExecutionReport,
        reason: str,
    ) -> None:
        for task in tasks.values():
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks.values(), return_exceptions=True)

        cancelled_ids: list[str] = []
        for node in nodes:
            if node.status != "running":
                continue
            task = tasks.get(node.node_id)
            result: AgentResult | None = None
            if task is not None and task.done() and not task.cancelled():
                try:
                    value = task.result()
                except BaseException:
                    value = None
                if isinstance(value, AgentResult):
                    result = value
            if result is None:
                result = AgentResult.cancelled_result(
                    "Agent batch ended before the node produced a result",
                    termination_reason=reason,
                )
                self._record_synthetic_cancelled_span(node, reason, report.graph)
            self._store_result(report, node, result, interrupted=True)
            report.graph.complete_source_handoffs(node.node_id)
            report.graph.consume_handoffs(node.node_id)
            if node.status == "cancelled":
                cancelled_ids.append(node.node_id)

        if cancelled_ids:
            graph_metadata = report.graph.metadata
            graph_metadata["termination_reason"] = reason
            graph_metadata["cancelled_node_ids"] = list(
                dict.fromkeys([*graph_metadata.get("cancelled_node_ids", []), *cancelled_ids])
            )
        report.completed = report.graph.all_terminal()

    def _store_result(
        self,
        report: ExecutionReport,
        node: TaskGraphNode,
        result: AgentResult,
        *,
        interrupted: bool = False,
    ) -> None:
        report.results[node.node_id] = result
        report.artifacts[node.node_id] = result.output
        report.artifacts[f"evidence:{node.node_id}"] = list(result.evidence)
        report.artifacts[f"tool_calls:{node.node_id}"] = list(result.tool_calls)
        report.artifacts[f"artifacts:{node.node_id}"] = dict(result.artifacts)
        if result.status in {"failed", "timeout", "cancelled"}:
            if node.required or interrupted:
                node.status = result.status  # type: ignore[assignment]
                report.failed_node_ids.append(node.node_id)
            else:
                node.status = "skipped"
                node.metadata = {
                    **node.metadata,
                    "skipped_reason": "optional_branch_failed",
                    "optional_failure": result.failure.compact() if result.failure else {},
                }
                report.blocked_node_ids.append(node.node_id)
        else:
            node.status = "succeeded"
        report.failed_node_ids = list(dict.fromkeys(report.failed_node_ids))
        report.blocked_node_ids = list(dict.fromkeys(report.blocked_node_ids))

    def _record_synthetic_cancelled_span(
        self,
        node: TaskGraphNode,
        reason: str,
        graph: TaskGraph,
    ) -> None:
        if self.span_recorder is None or node.node_id in self._span_keys:
            return
        record = getattr(self.span_recorder, "record_completed_span", None)
        if not callable(record):
            return
        payload = record(
            (
                f"policy:{node.agent_id}"
                if node.capability == "policy_gate"
                else f"agent:{node.agent_id}"
            ),
            duration_ms=0,
            label=node.metadata.get("phase") or node.capability,
            parent_span_key=self._parent_span_key(node, graph),
            task_id=node.task_id,
            agent_id=node.agent_id,
            span_type="policy" if node.capability == "policy_gate" else "agent",
            attempt=node.attempt,
            status="cancelled",
            input_summary={
                "node_id": node.node_id,
                "capability": node.capability,
                "depends_on": list(node.depends_on),
                "input_refs": list(node.input_refs),
            },
            error_type="CancelledError",
            error_message="Agent task was cancelled before its span started",
            termination_reason=reason,
        )
        span_key = str(payload.get("span_key") or "")
        if span_key:
            self._span_keys[node.node_id] = span_key

    async def _cancel_task(self, task: asyncio.Task[Any] | None) -> None:
        if task is None:
            return
        if not task.done():
            task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    def _parent_span_key(self, node: TaskGraphNode, graph: TaskGraph) -> str | None:
        incoming = graph.handoffs_to(node.node_id)
        ordered = [
            *[item for item in incoming if item.required],
            *[item for item in incoming if not item.required],
        ]
        for handoff in ordered:
            span_key = self._handoff_span_keys.get(handoff.handoff_id)
            if span_key:
                return span_key
        for dependency_id in self._dependency_ids(node):
            span_key = self._span_keys.get(dependency_id)
            if span_key:
                return span_key
        return None

    def _dependency_ids(self, node: TaskGraphNode) -> list[str]:
        return list(dict.fromkeys([*node.depends_on, *node.input_refs]))

    def _handoff_events(self, graph: TaskGraph) -> list[ExecutorEvent]:
        graph.sync_handoffs()
        events: list[ExecutorEvent] = []
        for handoff in sorted(graph.handoffs, key=lambda item: item.sequence):
            previous = self._observed_handoff_statuses.get(handoff.handoff_id)
            if previous == handoff.status:
                continue
            if handoff.status in {
                "accepted",
                "consumed",
                "unavailable",
                "blocked",
                "superseded",
            }:
                self._record_handoff_span(handoff, graph)
            self._observed_handoff_statuses[handoff.handoff_id] = handoff.status
            events.append(
                ExecutorEvent(
                    kind="handoff_updated",
                    node_id=handoff.to_node_id,
                    payload={
                        "handoff": handoff.model_dump(),
                        "span_key": self._handoff_span_keys.get(handoff.handoff_id, ""),
                    },
                )
            )
        return events

    def _record_handoff_span(self, handoff: AgentHandoff, graph: TaskGraph) -> None:
        if handoff.handoff_id in self._handoff_span_keys or self.span_recorder is None:
            return
        record = getattr(self.span_recorder, "record_completed_span", None)
        if not callable(record):
            return
        target = graph.get_node(handoff.to_node_id)
        source = graph.get_node(handoff.from_node_id)
        status = "succeeded"
        if handoff.status == "blocked":
            status = "failed"
        elif handoff.status in {"unavailable", "superseded"}:
            status = "degraded"
        payload = record(
            f"handoff:{handoff.handoff_type}",
            duration_ms=0,
            label=f"Agent 交接：{handoff.from_agent_id} -> {handoff.to_agent_id}",
            parent_span_key=self._span_keys.get(handoff.from_node_id),
            task_id=f"{graph.turn_id}:handoff:{handoff.sequence}",
            agent_id="supervisor_handoff",
            span_type="handoff",
            attempt=target.attempt if target is not None else 1,
            status=status,
            input_summary={
                "handoff_id": handoff.handoff_id,
                "handoff_type": handoff.handoff_type,
                "from_node_id": handoff.from_node_id,
                "from_agent_id": handoff.from_agent_id,
                "from_task_id": source.task_id if source is not None else "",
                "to_node_id": handoff.to_node_id,
                "to_agent_id": handoff.to_agent_id,
                "to_task_id": target.task_id if target is not None else "",
                "required": handoff.required,
                "artifact_refs": list(handoff.artifact_refs),
                "evidence_refs": list(handoff.evidence_refs),
            },
            output_summary={"handoff_status": handoff.status, "reason": handoff.reason},
            metrics={
                "required": handoff.required,
                "artifact_ref_count": len(handoff.artifact_refs),
                "evidence_ref_count": len(handoff.evidence_refs),
            },
            termination_reason=f"handoff_{handoff.status}",
        )
        span_key = str(payload.get("span_key") or "")
        if span_key:
            self._handoff_span_keys[handoff.handoff_id] = span_key

    def _output_for_dependency(self, node_id: str, base_context: AgentExecutionContext) -> dict[str, Any]:
        output = base_context.artifacts.get(node_id)
        return output if isinstance(output, dict) else {}

    def _artifacts_for_dependency(
        self,
        node_id: str,
        base_context: AgentExecutionContext,
    ) -> dict[str, Any]:
        value = base_context.artifacts.get(f"artifacts:{node_id}")
        return value if isinstance(value, dict) else {}
