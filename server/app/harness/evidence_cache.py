from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol


@dataclass
class EvidenceCandidate:
    product_id: str
    score: float | None = None
    slot_id: str = ""
    stage: str = ""
    reason: str = ""
    display_order: int | None = None
    compact_product: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceSlot:
    slot_id: str
    goal: str = ""
    selected_product_ids: list[str] = field(default_factory=list)
    candidate_product_ids: list[str] = field(default_factory=list)
    rejected_product_ids: list[str] = field(default_factory=list)
    coverage_status: str = ""
    reason: str = ""


@dataclass
class EvidenceBundle:
    user_id: str
    session_id: str
    turn_id: str
    query: str
    execution_path: str
    final_route: str
    displayed_product_ids: list[str] = field(default_factory=list)
    selected_product_ids: list[str] = field(default_factory=list)
    candidate_product_ids: list[str] = field(default_factory=list)
    rejected_product_ids: list[str] = field(default_factory=list)
    candidates: list[EvidenceCandidate] = field(default_factory=list)
    slots: list[EvidenceSlot] = field(default_factory=list)
    reflection_summary: dict[str, Any] = field(default_factory=dict)
    trace_summary: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 1

    def compact(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "query": self.query,
            "execution_path": self.execution_path,
            "final_route": self.final_route,
            "displayed_product_ids": self.displayed_product_ids,
            "selected_product_ids": self.selected_product_ids,
            "candidate_product_ids": self.candidate_product_ids,
            "rejected_product_ids": self.rejected_product_ids,
            "candidates": [
                {
                    "product_id": candidate.product_id,
                    "score": candidate.score,
                    "slot_id": candidate.slot_id,
                    "stage": candidate.stage,
                    "reason": candidate.reason,
                    "display_order": candidate.display_order,
                    "product": candidate.compact_product,
                }
                for candidate in self.candidates
            ],
            "slots": [
                {
                    "slot_id": slot.slot_id,
                    "goal": slot.goal,
                    "selected_product_ids": slot.selected_product_ids,
                    "candidate_product_ids": slot.candidate_product_ids,
                    "rejected_product_ids": slot.rejected_product_ids,
                    "coverage_status": slot.coverage_status,
                    "reason": slot.reason,
                }
                for slot in self.slots
            ],
            "reflection_summary": self.reflection_summary,
            "trace_summary": self.trace_summary,
        }


class EvidenceCache(Protocol):
    def put_turn_evidence(self, bundle: EvidenceBundle) -> None:
        ...

    def get_latest_evidence(self, session_id: str) -> EvidenceBundle | None:
        ...

    def get_recent_evidence(self, session_id: str, limit: int = 20) -> list[EvidenceBundle]:
        ...

    def get_turn_evidence(self, session_id: str, turn_id: str) -> EvidenceBundle | None:
        ...

    def compact_recent(self, session_id: str, limit: int = 20) -> list[dict[str, Any]]:
        ...


class InMemoryEvidenceCache:
    """进程内短期证据缓存（InMemoryEvidenceCache）。

    用于最近若干轮 evidence bundle 的上下文辅助；缓存不是商品详情 source of truth。
    回答前仍必须按 product_id 回 DB 读取商品详情。
    """

    def __init__(
        self,
        *,
        ttl_seconds: int = 3600,
        recent_turns: int = 20,
        max_candidates_per_turn: int = 20,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.ttl_seconds = max(1, int(ttl_seconds))
        self.recent_turns = max(1, int(recent_turns))
        self.max_candidates_per_turn = max(1, int(max_candidates_per_turn))
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._by_session: dict[str, list[EvidenceBundle]] = {}

    def put_turn_evidence(self, bundle: EvidenceBundle) -> None:
        normalized = self._normalize_bundle(bundle)
        self._prune_session(normalized.session_id)
        bundles = [
            existing
            for existing in self._by_session.get(normalized.session_id, [])
            if existing.turn_id != normalized.turn_id
        ]
        bundles.append(normalized)
        bundles.sort(key=lambda item: item.created_at)
        self._by_session[normalized.session_id] = bundles[-self.recent_turns :]

    def get_latest_evidence(self, session_id: str) -> EvidenceBundle | None:
        self._prune_session(session_id)
        bundles = self._by_session.get(session_id, [])
        return deepcopy(bundles[-1]) if bundles else None

    def get_recent_evidence(self, session_id: str, limit: int = 20) -> list[EvidenceBundle]:
        self._prune_session(session_id)
        safe_limit = max(1, min(int(limit), self.recent_turns))
        return deepcopy(self._by_session.get(session_id, [])[-safe_limit:])

    def get_turn_evidence(self, session_id: str, turn_id: str) -> EvidenceBundle | None:
        self._prune_session(session_id)
        for bundle in self._by_session.get(session_id, []):
            if bundle.turn_id == turn_id:
                return deepcopy(bundle)
        return None

    def compact_recent(self, session_id: str, limit: int = 20) -> list[dict[str, Any]]:
        return [bundle.compact() for bundle in self.get_recent_evidence(session_id, limit=limit)]

    def _normalize_bundle(self, bundle: EvidenceBundle) -> EvidenceBundle:
        normalized = deepcopy(bundle)
        normalized.created_at = self._to_utc(normalized.created_at)
        normalized.candidates = normalized.candidates[: self.max_candidates_per_turn]
        normalized.candidate_product_ids = self._dedupe(normalized.candidate_product_ids)[: self.max_candidates_per_turn]
        normalized.selected_product_ids = self._dedupe(normalized.selected_product_ids)
        normalized.displayed_product_ids = self._dedupe(normalized.displayed_product_ids)
        normalized.rejected_product_ids = self._dedupe(normalized.rejected_product_ids)[: self.max_candidates_per_turn]
        return normalized

    def _prune_session(self, session_id: str) -> None:
        bundles = self._by_session.get(session_id)
        if not bundles:
            return
        expires_before = self._to_utc(self._now()) - timedelta(seconds=self.ttl_seconds)
        active = [bundle for bundle in bundles if self._to_utc(bundle.created_at) >= expires_before]
        self._by_session[session_id] = active[-self.recent_turns :]

    def _to_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _dedupe(self, values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            text = str(value or "").strip()
            if text and text not in result:
                result.append(text)
        return result
