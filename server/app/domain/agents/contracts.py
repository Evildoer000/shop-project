from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol

from app.domain.supervisor.task_graph import TaskGraph, TaskGraphNode


AgentResultStatus = Literal["succeeded", "failed", "timeout", "cancelled", "degraded"]


@dataclass(frozen=True)
class EvidenceRef:
    """A traceable fact reference, never an implicit model thought."""

    evidence_id: str
    source_type: str
    source_id: str
    claim: str = ""
    summary: str = ""
    product_id: str = ""
    slot_id: str = ""
    score: float | None = None
    verified: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)

    def compact(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "claim": self.claim[:240],
            "summary": self.summary[:500],
            "product_id": self.product_id,
            "slot_id": self.slot_id,
            "score": self.score,
            "verified": self.verified,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class AgentFailure:
    error_type: str
    message: str
    retryable: bool = False
    termination_reason: str = ""

    def compact(self) -> dict[str, Any]:
        return {
            "error_type": self.error_type,
            "message": self.message[:500],
            "retryable": self.retryable,
            "termination_reason": self.termination_reason,
        }


@dataclass
class AgentResult:
    status: AgentResultStatus = "succeeded"
    output: dict[str, Any] = field(default_factory=dict)
    evidence: list[EvidenceRef] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    failure: AgentFailure | None = None
    termination_reason: str = ""

    @classmethod
    def success(
        cls,
        output: dict[str, Any] | None = None,
        *,
        evidence: list[EvidenceRef] | None = None,
        artifacts: dict[str, Any] | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        termination_reason: str = "completed",
    ) -> "AgentResult":
        return cls(
            status="succeeded",
            output=output or {},
            evidence=evidence or [],
            artifacts=artifacts or {},
            tool_calls=tool_calls or [],
            termination_reason=termination_reason,
        )

    @classmethod
    def failure_result(
        cls,
        error_type: str,
        message: str,
        *,
        retryable: bool = False,
        timeout: bool = False,
        termination_reason: str = "agent_failed",
    ) -> "AgentResult":
        return cls(
            status="timeout" if timeout else "failed",
            failure=AgentFailure(
                error_type=error_type,
                message=message,
                retryable=retryable,
                termination_reason=termination_reason,
            ),
            termination_reason=termination_reason,
        )

    @classmethod
    def cancelled_result(
        cls,
        message: str = "Agent task was cancelled",
        *,
        termination_reason: str = "cancelled",
    ) -> "AgentResult":
        return cls(
            status="cancelled",
            failure=AgentFailure(
                error_type="CancelledError",
                message=message,
                retryable=False,
                termination_reason=termination_reason,
            ),
            termination_reason=termination_reason,
        )

    def compact(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "output": _compact_value(self.output),
            "evidence": [item.compact() for item in self.evidence],
            "artifacts": {key: _artifact_summary(value) for key, value in self.artifacts.items()},
            "tool_calls": _compact_value(self.tool_calls),
            "failure": self.failure.compact() if self.failure else None,
            "termination_reason": self.termination_reason,
        }


@dataclass
class AgentExecutionContext:
    """The smallest context an Agent receives for one task graph node."""

    node: TaskGraphNode
    graph: TaskGraph
    query: str
    user_id: str
    session_id: str
    turn_id: str
    intent_plan: Any
    conversation_context: Any = None
    query_plan: Any = None
    image_attributes: Any = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    dependency_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    dependency_artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    tool_access: Any = None
    prompt_registry: Any = None
    span_recorder: Any = None
    budget_manager: Any = None
    services: Any = None
    current_span: Any = None
    tool_call_records: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def output_from(self, node_id: str) -> dict[str, Any]:
        return self.dependency_outputs.get(node_id, {})

    def artifact(self, key: str, default: Any = None) -> Any:
        for node_artifacts in reversed(list(self.dependency_artifacts.values())):
            if key in node_artifacts:
                return node_artifacts[key]
        return self.artifacts.get(key, default)

    def artifact_from(self, node_id: str, key: str, default: Any = None) -> Any:
        return self.dependency_artifacts.get(node_id, {}).get(key, default)

    def artifacts_from(self, node_id: str) -> dict[str, Any]:
        return dict(self.dependency_artifacts.get(node_id, {}))

    def tool(self, name: str) -> Any:
        if self.tool_access is None:
            raise RuntimeError("Agent tool access is not configured")
        # AgentExecutor has already intersected node requests with the immutable
        # Agent Manifest. Nodes may narrow access but can never grant themselves
        # another Agent's Tool.
        allowed_values = self.metadata.get("allowed_tools", [])
        allowed = tuple(str(item) for item in allowed_values)
        return self.tool_access.get(self.node.agent_id, name, allowed)

    async def execute_tool(
        self,
        name: str,
        operation: str,
        callback: Callable[[Any], Any],
        *,
        input_summary: dict[str, Any] | None = None,
    ) -> Any:
        """Authorize, execute and trace one high-level Tool operation."""
        if self.tool_access is None:
            raise RuntimeError("Agent tool access is not configured")
        allowed_values = self.metadata.get("allowed_tools", [])
        allowed = tuple(str(item) for item in allowed_values)
        registration = self.tool_access.registration_for(self.node.agent_id, name, allowed)
        span = None
        if self.span_recorder is not None:
            span = self.span_recorder.start_span(
                f"tool:{name}:{operation}",
                label=f"{name}.{operation}",
                agent=name,
                agent_id=f"tool:{name}",
                task_id=self.node.task_id,
                span_type="tool",
                attempt=self.node.attempt,
                parent_span_key=getattr(self.current_span, "span_key", None),
                input_summary={
                    "agent_id": self.node.agent_id,
                    "tool": name,
                    "operation": operation,
                    **(input_summary or {}),
                },
            )
        record = {
            "tool": name,
            "operation": operation,
            "agent_id": self.node.agent_id,
            "task_id": self.node.task_id,
            "attempt": self.node.attempt,
            "status": "running",
        }
        try:
            value = callback(registration.instance)
            if inspect.isawaitable(value):
                value = await asyncio.wait_for(value, timeout=registration.timeout_ms / 1000)
            record["status"] = "succeeded"
            record["result"] = _tool_result_summary(value)
            if span is not None:
                self.span_recorder.finish_span(
                    span,
                    output_summary=record["result"],
                    metrics={"networked": registration.networked},
                    termination_reason="completed",
                )
            return value
        except asyncio.CancelledError:
            record["status"] = "cancelled"
            if span is not None:
                self.span_recorder.finish_span(
                    span,
                    status="cancelled",
                    error_type="CancelledError",
                    termination_reason="cancelled",
                )
            raise
        except Exception as exc:
            record["status"] = "failed"
            record["error_type"] = type(exc).__name__
            record["error_message"] = str(exc)[:300]
            if span is not None:
                self.span_recorder.finish_span(
                    span,
                    status="failed",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    termination_reason="tool_failed",
                )
            raise
        finally:
            self.tool_call_records.append(record)

    def allowed_tool_descriptions(self) -> list[dict[str, Any]]:
        if self.tool_access is None:
            return []
        allowed_values = self.metadata.get("allowed_tools", [])
        allowed = tuple(str(item) for item in allowed_values)
        return self.tool_access.describe_for_agent(self.node.agent_id, allowed)

    async def emit_client_event(self, event: dict[str, Any]) -> None:
        queue = self.metadata.get("executor_event_queue")
        if queue is None:
            return
        await queue.put(
            {
                "kind": "agent_output",
                "node_id": self.node.node_id,
                "payload": {
                    "task_id": self.node.task_id,
                    "agent_id": self.node.agent_id,
                    "capability": self.node.capability,
                    "event": _compact_value(event),
                },
            }
        )


class AgentHandler(Protocol):
    async def __call__(self, context: AgentExecutionContext) -> AgentResult | dict[str, Any]: ...


@dataclass
class ExecutionReport:
    graph: TaskGraph
    results: dict[str, AgentResult] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    failed_node_ids: list[str] = field(default_factory=list)
    blocked_node_ids: list[str] = field(default_factory=list)
    completed: bool = False

    @property
    def succeeded(self) -> bool:
        return not self.failed_node_ids and self.completed


def _compact_value(value: Any, *, depth: int = 0) -> Any:
    """Keep executor events JSON-safe without changing in-process artifacts."""
    if depth > 4:
        return "<truncated>"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _compact_value(item, depth=depth + 1) for key, item in list(value.items())[:80]}
    if isinstance(value, (list, tuple, set)):
        return [_compact_value(item, depth=depth + 1) for item in list(value)[:80]]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _compact_value(model_dump(), depth=depth + 1)
        except Exception:
            pass
    return str(value)


def _artifact_summary(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {"kind": "dict", "keys": sorted(str(key) for key in value)[:40]}
    if isinstance(value, (list, tuple, set)):
        return {"kind": "collection", "count": len(value)}
    return {"kind": type(value).__name__}


def _tool_result_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        items = value.get("items")
        error = value.get("error") if isinstance(value.get("error"), dict) else {}
        return {
            "kind": "dict",
            "available": value.get("available"),
            "item_count": len(items) if isinstance(items, list) else None,
            "error_code": error.get("code"),
            "latency_ms": value.get("latency_ms"),
            "keys": sorted(str(key) for key in value)[:30],
        }
    if isinstance(value, (list, tuple, set)):
        return {"kind": "collection", "count": len(value)}
    ranked = getattr(value, "ranked", None)
    if isinstance(ranked, list):
        return {"kind": type(value).__name__, "candidate_count": len(ranked)}
    candidates = getattr(value, "candidates", None)
    if isinstance(candidates, list):
        return {"kind": type(value).__name__, "candidate_count": len(candidates)}
    return {"kind": type(value).__name__}
