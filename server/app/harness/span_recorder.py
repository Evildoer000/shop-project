from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from app.db.models import AgentRun, AgentRunSpan
from app.db.session import get_sessionmaker

logger = logging.getLogger(__name__)


@dataclass
class ActiveSpan:
    span_key: str
    parent_span_key: str | None
    task_id: str
    agent_id: str
    span_type: str
    attempt: int
    sequence: int
    trace_schema_version: str
    name: str
    label: str
    agent: str
    started_at: datetime
    started_perf: float
    input_summary: dict[str, Any] = field(default_factory=dict)


class SpanRecorder:
    """Local first observability recorder for one Agent request.

    The recorder is intentionally best-effort: monitoring failures must not
    change the recommendation, retrieval, repair, or answer generation logic.
    """

    def __init__(self) -> None:
        self.run_id = f"run_{uuid.uuid4().hex}"
        self.trace_schema_version = "v2"
        self.user_id = ""
        self.session_id = ""
        self.turn_id = ""
        self.query_summary = ""
        self.started_at = datetime.now(timezone.utc)
        self.started_perf = time.perf_counter()
        self.finished_at: datetime | None = None
        self.status = "running"
        self.termination_reason = ""
        self.route = ""
        self.plan_type = ""
        self.first_token_latency_ms: float | None = None
        self.product_ids: list[str] = []
        self.evaluation_summary: dict[str, Any] = {}
        self.spans: list[dict[str, Any]] = []
        self._sequence = 0
        self._root_span: ActiveSpan | None = None
        self._last_span_by_name: dict[str, ActiveSpan] = {}
        self._active_spans: dict[str, ActiveSpan] = {}
        self._finished_span_payloads: dict[str, dict[str, Any]] = {}

    def start_run(self, *, user_id: str, session_id: str, turn_id: str, query_summary: str) -> dict[str, Any]:
        self.user_id = user_id
        self.session_id = session_id
        self.turn_id = turn_id
        self.query_summary = self._compact(query_summary, 500)
        self.started_at = datetime.now(timezone.utc)
        self.started_perf = time.perf_counter()
        self.finished_at = None
        self.status = "running"
        self.termination_reason = ""
        self.route = ""
        self.plan_type = ""
        self.first_token_latency_ms = None
        self.product_ids = []
        self.evaluation_summary = {}
        self._sequence = 0
        self.spans = []
        self._last_span_by_name = {}
        self._active_spans = {}
        self._finished_span_payloads = {}
        self._root_span = ActiveSpan(
            span_key=f"span_{uuid.uuid4().hex}",
            parent_span_key=None,
            task_id=turn_id or self.run_id,
            agent_id="EcommerceOrchestrator",
            span_type="run",
            attempt=1,
            sequence=0,
            trace_schema_version=self.trace_schema_version,
            name="orchestrator_run",
            label="本次请求执行根节点",
            agent="EcommerceOrchestrator",
            started_at=self.started_at,
            started_perf=self.started_perf,
        )
        self._persist_run()
        return self.run_payload()

    def start_span(
        self,
        name: str,
        *,
        label: str = "",
        agent: str = "",
        input_summary: dict[str, Any] | None = None,
        parent_span_key: str | None = None,
        task_id: str = "",
        agent_id: str = "",
        span_type: str = "",
        attempt: int = 1,
    ) -> ActiveSpan:
        safe_input = self._json_safe(input_summary or {})
        inferred_agent_id = agent_id or agent or self._infer_agent_id(name)
        inferred_span_type = span_type or self._infer_span_type(agent=inferred_agent_id, name=name)
        inferred_attempt = self._infer_attempt(name, safe_input, attempt)
        self._sequence += 1
        span = ActiveSpan(
            span_key=f"span_{uuid.uuid4().hex}",
            parent_span_key=parent_span_key if parent_span_key is not None else self._infer_parent_span_key(name),
            task_id=task_id or self.turn_id or self.run_id,
            agent_id=inferred_agent_id,
            span_type=inferred_span_type,
            attempt=inferred_attempt,
            sequence=self._sequence,
            trace_schema_version=self.trace_schema_version,
            name=name,
            label=label or name,
            agent=agent,
            started_at=datetime.now(timezone.utc),
            started_perf=time.perf_counter(),
            input_summary=safe_input,
        )
        self._last_span_by_name[name] = span
        self._active_spans[span.span_key] = span
        return span

    def child_span(
        self,
        parent: ActiveSpan,
        name: str,
        *,
        label: str = "",
        agent: str = "",
        input_summary: dict[str, Any] | None = None,
        task_id: str = "",
        agent_id: str = "",
        span_type: str = "",
        attempt: int = 1,
    ) -> ActiveSpan:
        """Start a span with an explicit parent for nested or concurrent work."""
        return self.start_span(
            name,
            label=label,
            agent=agent,
            input_summary=input_summary,
            parent_span_key=parent.span_key,
            task_id=task_id,
            agent_id=agent_id,
            span_type=span_type,
            attempt=attempt,
        )

    def _infer_parent_span_key(self, name: str) -> str | None:
        parent_names: dict[str, tuple[str, ...]] = {
            "input_normalize": (),
            "memory_context_load": (),
            "image_attribute_extraction": (),
            "intent_planning": (),
            "profile_lookup": ("intent_planning",),
            "intent_planning_profile_refine": ("profile_lookup", "intent_planning"),
            "retrieval_plan_builder": ("intent_planning_profile_refine", "intent_planning"),
            "single_retrieval_worker_execution": ("retrieval_plan_builder", "intent_planning"),
            "image_retrieval_worker_execution": ("retrieval_plan_builder", "intent_planning"),
            "multi_need_retrieval": ("retrieval_plan_builder", "intent_planning"),
            "corrective_reflection": (
                "single_retrieval_worker_execution",
                "image_retrieval_worker_execution",
                "multi_need_retrieval",
            ),
            "repair_plan_generated": (
                "corrective_reflection_after_repair",
                "corrective_reflection",
            ),
            "repair_search_executed": ("repair_plan_generated",),
            "corrective_reflection_after_repair": ("repair_search_executed",),
            "answer_generation": (
                "corrective_reflection_after_repair",
                "corrective_reflection",
                "multi_need_retrieval",
                "image_retrieval_worker_execution",
                "single_retrieval_worker_execution",
                "intent_planning_profile_refine",
                "intent_planning",
            ),
        }
        for parent_name in parent_names.get(name, ()):
            parent = self._last_span_by_name.get(parent_name)
            if parent is not None:
                return parent.span_key
        return self._root_span.span_key if self._root_span is not None else None

    def _infer_agent_id(self, name: str) -> str:
        return {
            "input_normalize": "InputProcessor",
            "memory_context_load": "MemoryManager",
            "image_attribute_extraction": "ImageAttributeExtractor",
            "intent_planning": "IntentPlanner",
            "intent_planning_profile_refine": "IntentPlanner",
            "profile_lookup": "ProfileLookupTool",
            "retrieval_plan_builder": "RetrievalPlanBuilder",
            "single_retrieval_worker_execution": "RetrievalWorker",
            "image_retrieval_worker_execution": "ImageRetrievalWorker",
            "multi_need_retrieval": "RetrievalWorker",
            "corrective_reflection": "CorrectiveAgent",
            "repair_plan_generated": "RepairAgent",
            "repair_search_executed": "RetrievalWorker",
            "corrective_reflection_after_repair": "CorrectiveAgent",
            "answer_generation": "AnswerGenerator",
        }.get(name, "EcommerceOrchestrator")

    def _infer_span_type(self, *, agent: str, name: str) -> str:
        if name == "orchestrator_run":
            return "run"
        if agent.endswith("Tool"):
            return "tool"
        if agent in {
            "InputProcessor",
            "MemoryManager",
            "RetrievalPlanBuilder",
        }:
            return "stage"
        if agent:
            return "agent"
        return "stage"

    def _infer_attempt(self, name: str, input_summary: dict[str, Any], attempt: int) -> int:
        if attempt > 1:
            return int(attempt)
        raw_attempt = input_summary.get("repair_attempt")
        if raw_attempt is not None:
            try:
                return max(1, int(raw_attempt))
            except (TypeError, ValueError):
                pass
        if name == "corrective_reflection_after_repair":
            repair_span = self._last_span_by_name.get("repair_plan_generated")
            if repair_span is not None:
                return repair_span.attempt
        return 1

    def finish_span(
        self,
        span: ActiveSpan,
        *,
        status: str = "succeeded",
        output_summary: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        error_type: str = "",
        error_message: str = "",
        termination_reason: str = "",
    ) -> dict[str, Any]:
        existing = self._finished_span_payloads.get(span.span_key)
        if existing is not None:
            return existing
        finished_at = datetime.now(timezone.utc)
        duration_ms = max(0.0, (time.perf_counter() - span.started_perf) * 1000)
        resolved_termination_reason = termination_reason
        if not resolved_termination_reason and status not in {"succeeded", "running"}:
            resolved_termination_reason = status
        payload = {
            "run_id": self.run_id,
            "span_key": span.span_key,
            "parent_span_key": span.parent_span_key,
            "task_id": span.task_id,
            "agent_id": span.agent_id,
            "span_type": span.span_type,
            "attempt": span.attempt,
            "sequence": span.sequence,
            "trace_schema_version": span.trace_schema_version,
            "name": span.name,
            "label": span.label,
            "agent": span.agent,
            "status": status,
            "started_at": span.started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_ms": round(duration_ms, 2),
            "input_summary": self._json_safe(span.input_summary),
            "output_summary": self._json_safe(output_summary or {}),
            "metrics": self._json_safe(metrics or {}),
            "error_type": error_type,
            "error_message": self._compact(error_message, 500),
            "termination_reason": self._compact(resolved_termination_reason, 128),
        }
        self._active_spans.pop(span.span_key, None)
        self._finished_span_payloads[span.span_key] = payload
        self.spans.append(payload)
        self._persist_span(payload, span.started_at, finished_at)
        logger.info(
            "agent_span run_id=%s name=%s status=%s duration_ms=%.2f metrics=%s",
            self.run_id,
            span.name,
            status,
            duration_ms,
            payload["metrics"],
        )
        return payload

    def record_completed_span(
        self,
        name: str,
        *,
        duration_ms: float,
        label: str = "",
        parent_span_key: str | None = None,
        task_id: str = "",
        agent_id: str = "",
        span_type: str = "stage",
        attempt: int = 1,
        status: str = "succeeded",
        input_summary: dict[str, Any] | None = None,
        output_summary: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        error_type: str = "",
        error_message: str = "",
        termination_reason: str = "",
    ) -> dict[str, Any]:
        """Persist an already completed external call with its measured duration."""
        finished_at = datetime.now(timezone.utc)
        resolved_duration = max(0.0, float(duration_ms or 0.0))
        started_at = finished_at - timedelta(milliseconds=resolved_duration)
        self._sequence += 1
        payload = {
            "run_id": self.run_id,
            "span_key": f"span_{uuid.uuid4().hex}",
            "parent_span_key": parent_span_key or (self._root_span.span_key if self._root_span else None),
            "task_id": task_id or self.turn_id or self.run_id,
            "agent_id": agent_id,
            "span_type": span_type,
            "attempt": max(1, int(attempt or 1)),
            "sequence": self._sequence,
            "trace_schema_version": self.trace_schema_version,
            "name": name,
            "label": label or name,
            "agent": agent_id,
            "status": status,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_ms": round(resolved_duration, 2),
            "input_summary": self._json_safe(input_summary or {}),
            "output_summary": self._json_safe(output_summary or {}),
            "metrics": self._json_safe(metrics or {}),
            "error_type": error_type,
            "error_message": self._compact(error_message, 500),
            "termination_reason": self._compact(termination_reason or status, 128),
        }
        self._finished_span_payloads[str(payload["span_key"])] = payload
        self.spans.append(payload)
        self._persist_span(payload, started_at, finished_at)
        logger.info(
            "agent_span run_id=%s name=%s status=%s duration_ms=%.2f metrics=%s",
            self.run_id,
            name,
            status,
            resolved_duration,
            payload["metrics"],
        )
        return payload

    def finish_open_spans(
        self,
        *,
        status: str,
        termination_reason: str,
        error_type: str = "",
        error_message: str = "",
    ) -> list[dict[str, Any]]:
        """Force every non-root in-flight span to a terminal state."""
        root_key = self._root_span.span_key if self._root_span is not None else ""
        finished: list[dict[str, Any]] = []
        for span in list(self._active_spans.values()):
            if span.span_key == root_key:
                continue
            finished.append(
                self.finish_span(
                    span,
                    status=status,
                    error_type=error_type,
                    error_message=error_message,
                    termination_reason=termination_reason,
                )
            )
        return finished

    def mark_first_token(self) -> None:
        if self.first_token_latency_ms is None:
            self.first_token_latency_ms = round((time.perf_counter() - self.started_perf) * 1000, 2)

    def finish_run(
        self,
        *,
        route: str,
        plan_type: str = "",
        product_ids: list[str] | None = None,
        evaluation_summary: dict[str, Any] | None = None,
        status: str = "succeeded",
        termination_reason: str = "",
    ) -> dict[str, Any]:
        if self.finished_at is not None:
            return self.run_payload()
        self.finish_open_spans(
            status="cancelled" if status == "cancelled" else "failed",
            termination_reason=termination_reason or "run_ended_with_open_span",
            error_type="CancelledError" if status == "cancelled" else "RunTerminated",
            error_message="Request ended before this span completed.",
        )
        self.status = status
        self.termination_reason = self._compact(termination_reason or status, 128)
        self.route = route
        self.plan_type = plan_type
        self.product_ids = list(dict.fromkeys(product_ids or []))
        self.evaluation_summary = self._json_safe(evaluation_summary or {})
        self.finished_at = datetime.now(timezone.utc)
        if self._root_span is not None and not any(
            item.get("span_key") == self._root_span.span_key for item in self.spans
        ):
            self.finish_span(
                self._root_span,
                status=status,
                output_summary={
                    "route": route,
                    "plan_type": plan_type,
                    "product_count": len(self.product_ids),
                },
                metrics={"completed_child_spans": len(self.spans)},
                termination_reason=termination_reason,
            )
        self._persist_run()
        payload = self.run_payload()
        logger.info(
            "agent_run run_id=%s status=%s route=%s plan_type=%s total_latency_ms=%s evaluation=%s",
            self.run_id,
            status,
            route,
            plan_type,
            payload["summary"]["total_latency_ms"],
            self.evaluation_summary,
        )
        return payload

    def run_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "summary": {
                "status": self.status,
                "termination_reason": self.termination_reason,
                "route": self.route,
                "plan_type": self.plan_type,
                "total_latency_ms": self.total_latency_ms(),
                "first_token_latency_ms": self.first_token_latency_ms,
                "completed_spans": len(self.spans),
                "root_span_key": self._root_span.span_key if self._root_span is not None else None,
                "trace_schema_version": self.trace_schema_version,
            },
            "evaluation": self.evaluation_summary,
        }

    def timing_event(self, span_payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {"type": "timing_update", **self.run_payload()}
        if span_payload is not None:
            payload["span"] = span_payload
        return payload

    def total_latency_ms(self) -> float:
        end_perf = self.started_perf + (
            ((self.finished_at or datetime.now(timezone.utc)) - self.started_at).total_seconds()
            if self.finished_at
            else (time.perf_counter() - self.started_perf)
        )
        if self.finished_at is not None:
            return round(max(0.0, (self.finished_at - self.started_at).total_seconds() * 1000), 2)
        return round(max(0.0, (end_perf - self.started_perf) * 1000), 2)

    def _persist_run(self) -> None:
        try:
            SessionLocal = get_sessionmaker()
            with SessionLocal() as db:
                row = db.get(AgentRun, self.run_id)
                if row is None:
                    row = AgentRun(
                        run_id=self.run_id,
                        user_id=self.user_id,
                        session_id=self.session_id,
                        turn_id=self.turn_id,
                        query_summary=self.query_summary,
                        status=self.status,
                    )
                    db.add(row)
                row.user_id = self.user_id
                row.session_id = self.session_id
                row.turn_id = self.turn_id
                row.query_summary = self.query_summary
                row.route = self.route
                row.plan_type = self.plan_type
                row.status = self.status
                row.termination_reason = self.termination_reason
                row.total_latency_ms = self.total_latency_ms()
                row.first_token_latency_ms = self.first_token_latency_ms
                row.product_ids = self.product_ids
                row.evaluation_summary = self.evaluation_summary
                db.commit()
        except Exception as exc:
            logger.warning("persist agent run failed for %s: %s", self.run_id, exc)

    def _persist_span(self, payload: dict[str, Any], started_at: datetime, finished_at: datetime) -> None:
        try:
            SessionLocal = get_sessionmaker()
            with SessionLocal() as db:
                db.add(
                    AgentRunSpan(
                        run_id=self.run_id,
                        span_key=str(payload.get("span_key") or ""),
                        parent_span_key=payload.get("parent_span_key"),
                        task_id=str(payload.get("task_id") or ""),
                        agent_id=str(payload.get("agent_id") or ""),
                        span_type=str(payload.get("span_type") or "stage"),
                        attempt=int(payload.get("attempt") or 1),
                        sequence=int(payload.get("sequence") or 0),
                        trace_schema_version=str(payload.get("trace_schema_version") or self.trace_schema_version),
                        name=str(payload["name"]),
                        label=str(payload["label"]),
                        agent=str(payload["agent"]),
                        status=str(payload["status"]),
                        started_at=started_at,
                        finished_at=finished_at,
                        duration_ms=float(payload["duration_ms"]),
                        input_summary=payload["input_summary"],
                        output_summary=payload["output_summary"],
                        metrics=payload["metrics"],
                        error_type=str(payload.get("error_type") or ""),
                        error_message=str(payload.get("error_message") or ""),
                        termination_reason=str(payload.get("termination_reason") or ""),
                    )
                )
                db.commit()
        except Exception as exc:
            logger.warning("persist agent span failed for %s/%s: %s", self.run_id, payload.get("name"), exc)

    def _json_safe(self, value: Any) -> Any:
        return json.loads(json.dumps(value, ensure_ascii=False, default=self._json_default))

    def _json_default(self, value: Any) -> Any:
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

    def _compact(self, value: str, limit: int) -> str:
        text = " ".join(str(value or "").split())
        if len(text) <= limit:
            return text
        return f"{text[:limit]}..."
