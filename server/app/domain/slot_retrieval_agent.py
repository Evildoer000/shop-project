from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from app.domain.need_slot_schemas import AgentToolCall, NeedSlot, SlotCandidate, SlotSearchResult
from app.domain.product_search_tool import ProductSearchTool
from app.schemas import IntentPlan, QueryPlan


@dataclass
class SlotRetrievalAgentResult:
    slot: NeedSlot
    candidates: list[SlotCandidate] = field(default_factory=list)
    tool_calls: list[AgentToolCall] = field(default_factory=list)
    slot_result: dict[str, Any] = field(default_factory=dict)
    decision_steps: int = 0
    search_calls: int = 0
    repair_calls: int = 0
    termination_reason: str = ""


class SlotRetrievalAgent:
    MAX_DECISION_STEPS = 1
    MAX_ATTEMPTS_PER_SLOT = 1

    def __init__(
        self,
        search_tool: ProductSearchTool,
    ) -> None:
        self.search_tool = search_tool

    def run(
        self,
        slot: NeedSlot,
        plan: QueryPlan,
        intent_plan: IntentPlan,
    ) -> SlotRetrievalAgentResult:
        working_slot = slot.model_copy(deep=True)
        result = SlotRetrievalAgentResult(slot=working_slot)
        merged_candidates: dict[str, SlotCandidate] = {}

        result.decision_steps = 1
        query = working_slot.query.strip()
        if not query:
            result.termination_reason = "slot_agent_no_query"
        else:
            search_result, search_call = self._search_products(
                working_slot,
                plan,
                intent_plan,
                query,
                attempt_index=1,
                reason="initial_search",
            )
            result.tool_calls.append(search_call)
            result.search_calls = 1
            if search_result is None:
                result.termination_reason = "slot_agent_tool_failed"
            else:
                self._merge_slot_result(result, search_result)
                self._merge_candidates(merged_candidates, search_result.candidates)
                result.candidates = self._sorted_candidates(merged_candidates)
                result.slot_result["candidate_ids"] = [candidate.product_id for candidate in result.candidates]
                result.termination_reason = "slot_agent_completed" if result.candidates else "slot_agent_no_candidates"

        if not result.termination_reason:
            result.termination_reason = "slot_agent_completed"
        result.slot = working_slot
        result.candidates = self._sorted_candidates(merged_candidates)
        result.slot_result["candidate_ids"] = [candidate.product_id for candidate in result.candidates]
        result.slot_result["termination_reason"] = result.termination_reason
        result.slot_result["decision_steps"] = result.decision_steps
        result.slot_result["search_calls"] = result.search_calls
        result.slot_result["repair_calls"] = result.repair_calls
        return result

    def _search_products(
        self,
        slot: NeedSlot,
        plan: QueryPlan,
        intent_plan: IntentPlan,
        query: str,
        attempt_index: int,
        reason: str,
    ) -> tuple[SlotSearchResult | None, AgentToolCall]:
        started = time.perf_counter()
        call = AgentToolCall(
            action="search_products",
            slot_id=slot.slot_id,
            input_summary={"query": query, "attempt": attempt_index},
            reason=reason,
        )
        search_result: SlotSearchResult | None = None
        try:
            search_result = self.search_tool.search_query(
                slot=slot,
                base_plan=plan,
                intent_plan=intent_plan,
                query=query,
                attempt_index=attempt_index,
                reason=reason,
            )
            call.output_summary = self._search_output_summary(search_result)
        except Exception as exc:
            call.status = "failed"
            call.reason = str(exc)
        finally:
            call.duration_ms = round((time.perf_counter() - started) * 1000, 2)
        return search_result, call

    def _merge_slot_result(self, result: SlotRetrievalAgentResult, search_result: SlotSearchResult) -> None:
        current = result.slot_result
        current["query"] = search_result.query
        current["vector_query"] = search_result.vector_query
        current["keyword_query"] = search_result.keyword_query
        current["categories"] = search_result.categories
        current["category_resolution"] = search_result.category_resolution
        current["attempts"] = [*current.get("attempts", []), *search_result.attempts]
        current["counts"] = self._merge_counts(current.get("counts", {}), search_result.counts)

    def _merge_candidates(
        self,
        existing: dict[str, SlotCandidate],
        candidates: list[SlotCandidate],
    ) -> None:
        for candidate in candidates:
            current = existing.get(candidate.product_id)
            if current is None or candidate.rerank_score > current.rerank_score:
                existing[candidate.product_id] = candidate

    def _sorted_candidates(self, candidates: dict[str, SlotCandidate]) -> list[SlotCandidate]:
        return sorted(
            candidates.values(),
            key=lambda candidate: (candidate.rerank_score, candidate.rrf_score, candidate.keyword_score),
            reverse=True,
        )

    def _has_enough_signal(self, candidates: list[SlotCandidate], slot: NeedSlot) -> bool:
        if len(candidates) >= max(1, slot.min_candidates * 2):
            return True
        return len(candidates) >= slot.min_candidates and any(candidate.rerank_score >= 0.35 for candidate in candidates)

    def _search_output_summary(self, search_result: SlotSearchResult) -> dict[str, Any]:
        return {
            "candidate_count": len(search_result.candidates),
            "query": search_result.vector_query,
            "counts": search_result.counts,
            "attempts": search_result.attempts,
            "categories": search_result.categories,
            "category_resolution": search_result.category_resolution,
            "candidate_ids": [candidate.product_id for candidate in search_result.candidates],
        }

    def _merge_counts(self, left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
        keys = set(left) | set(right)
        merged: dict[str, int] = {}
        for key in keys:
            if key in {"before_structured_filter", "after_structured_filter"}:
                merged[key] = max(left.get(key, 0), right.get(key, 0))
            else:
                merged[key] = max(left.get(key, 0), right.get(key, 0))
        return merged

    def _compact(self, text: str) -> str:
        return " ".join(text.split())

    def _unique(self, values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            text = str(value).strip()
            if text and text not in result:
                result.append(text)
        return result
