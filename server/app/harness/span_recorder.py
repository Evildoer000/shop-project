from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from app.db.models import AgentRun, AgentRunSpan
from app.db.session import get_sessionmaker

logger = logging.getLogger(__name__)


@dataclass
class ActiveSpan:
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
        self.user_id = ""
        self.session_id = ""
        self.turn_id = ""
        self.query_summary = ""
        self.started_at = datetime.now(timezone.utc)
        self.started_perf = time.perf_counter()
        self.finished_at: datetime | None = None
        self.status = "running"
        self.route = ""
        self.plan_type = ""
        self.first_token_latency_ms: float | None = None
        self.product_ids: list[str] = []
        self.evaluation_summary: dict[str, Any] = {}
        self.spans: list[dict[str, Any]] = []

    def start_run(self, *, user_id: str, session_id: str, turn_id: str, query_summary: str) -> dict[str, Any]:
        self.user_id = user_id
        self.session_id = session_id
        self.turn_id = turn_id
        self.query_summary = self._compact(query_summary, 500)
        self.started_at = datetime.now(timezone.utc)
        self.started_perf = time.perf_counter()
        self._persist_run()
        return self.run_payload()

    def start_span(
        self,
        name: str,
        *,
        label: str = "",
        agent: str = "",
        input_summary: dict[str, Any] | None = None,
    ) -> ActiveSpan:
        return ActiveSpan(
            name=name,
            label=label or name,
            agent=agent,
            started_at=datetime.now(timezone.utc),
            started_perf=time.perf_counter(),
            input_summary=self._json_safe(input_summary or {}),
        )

    def finish_span(
        self,
        span: ActiveSpan,
        *,
        status: str = "succeeded",
        output_summary: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        error_type: str = "",
        error_message: str = "",
    ) -> dict[str, Any]:
        finished_at = datetime.now(timezone.utc)
        duration_ms = max(0.0, (time.perf_counter() - span.started_perf) * 1000)
        payload = {
            "run_id": self.run_id,
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
        }
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
    ) -> dict[str, Any]:
        self.status = status
        self.route = route
        self.plan_type = plan_type
        self.product_ids = list(dict.fromkeys(product_ids or []))
        self.evaluation_summary = self._json_safe(evaluation_summary or {})
        self.finished_at = datetime.now(timezone.utc)
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
                "route": self.route,
                "plan_type": self.plan_type,
                "total_latency_ms": self.total_latency_ms(),
                "first_token_latency_ms": self.first_token_latency_ms,
                "completed_spans": len(self.spans),
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
