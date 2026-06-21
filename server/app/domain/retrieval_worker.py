from __future__ import annotations

import time
from typing import Any

from app.domain.image_retrieval_worker import ImageRetrievalWorker
from app.domain.multi_need_retrieval_coordinator import MultiNeedRetrievalCoordinator
from app.domain.need_slot_schemas import AgentToolCall, MultiNeedState, NeedSlot, SlotCandidate
from app.domain.product_search_tool import ProductSearchTool
from app.domain.repair_worker import RepairPlan
from app.domain.single_retrieval_worker import SingleRetrievalEvidence, SingleRetrievalWorker
from app.schemas import IntentPlan, QueryPlan


class RetrievalWorker:
    def __init__(
        self,
        *,
        product_search_tool: ProductSearchTool,
        single_retrieval_worker: SingleRetrievalWorker,
        multi_need_coordinator: MultiNeedRetrievalCoordinator,
        image_retrieval_worker: ImageRetrievalWorker,
    ) -> None:
        self.product_search_tool = product_search_tool
        self.single_retrieval_worker = single_retrieval_worker
        self.multi_need_coordinator = multi_need_coordinator
        self.image_retrieval_worker = image_retrieval_worker

    def run_single_initial(
        self,
        original_query: str,
        intent_plan: IntentPlan,
        plan: QueryPlan,
    ) -> SingleRetrievalEvidence:
        return self.single_retrieval_worker.run(original_query, intent_plan, plan)

    async def run_multi_initial(
        self,
        original_query: str,
        intent_plan: IntentPlan,
        plan: QueryPlan,
        slots: list[NeedSlot],
    ) -> MultiNeedState:
        return await self.multi_need_coordinator.run(original_query, intent_plan, plan, slots)

    def run_image_initial(
        self,
        original_query: str,
        intent_plan: IntentPlan,
        plan: QueryPlan,
        image_path: str,
    ) -> Any:
        return self.image_retrieval_worker.run(
            original_query=original_query,
            intent_plan=intent_plan,
            plan=plan,
            image_path=image_path,
        )

    def run_single_repair(
        self,
        original_query: str,
        intent_plan: IntentPlan,
        plan: QueryPlan,
        repair_plan: RepairPlan,
    ) -> SingleRetrievalEvidence:
        slot = self._single_slot(original_query, intent_plan, plan)
        queries = repair_plan.queries_by_slot.get("single") or repair_plan.queries_by_slot.get(slot.slot_id) or []
        merged: dict[str, SlotCandidate] = {}
        vector_scores: dict[str, float] = {}
        keyword_scores: dict[str, float] = {}
        structured_products = []
        score_filtered_products = []
        hybrid_ranked_products = []
        counts: dict[str, int] = {}
        last_query = slot.query
        tool_calls = 0

        for attempt_index, query in enumerate(self._dedupe(queries)[:3], start=1):
            query = query.strip()
            if not query:
                continue
            last_query = query
            tool_calls += 1
            search_result = self.product_search_tool.search_query(
                slot=slot,
                base_plan=plan,
                intent_plan=intent_plan,
                query=query,
                attempt_index=attempt_index,
                reason="repair_search_executed",
                use_base_plan=True,
            )
            counts = self._merge_count_max(counts, search_result.counts)
            structured_products.extend(search_result.structured_products)
            score_filtered_products.extend(search_result.score_filtered_products)
            hybrid_ranked_products.extend(search_result.hybrid_ranked_products)
            vector_scores = self._merge_score_maps(vector_scores, search_result.vector_scores)
            keyword_scores = self._merge_score_maps(keyword_scores, search_result.keyword_scores)
            for candidate in search_result.candidates:
                current = merged.get(candidate.product_id)
                if current is None or candidate.rerank_score > current.rerank_score:
                    merged[candidate.product_id] = candidate

        candidates = self._sort_slot_candidates(list(merged.values()))
        evidence = SingleRetrievalEvidence(
            before_structured_filter=counts.get("before_structured_filter", 0),
            structured_products=self._dedupe_products(structured_products),
            vector_query=last_query,
            keyword_query=last_query,
            vector_scores=vector_scores,
            keyword_scores=keyword_scores,
            score_filtered_products=self._dedupe_products(score_filtered_products),
            hybrid_ranked_products=self._dedupe_products(hybrid_ranked_products),
            ranked=[(candidate.product, candidate.rerank_score) for candidate in candidates],
            rerank_query=last_query,
            tool_call_count=tool_calls,
        )
        if not candidates:
            evidence.failure_trigger = "repair_no_candidates"
        return evidence

    def run_multi_repair(
        self,
        state: MultiNeedState,
        repair_plan: RepairPlan,
    ) -> MultiNeedState:
        for slot_id in repair_plan.targets:
            slot = state.slot_by_id(slot_id)
            if slot is None:
                continue
            existing = {candidate.product_id: candidate for candidate in state.candidates_by_slot.get(slot_id, [])}
            for attempt_index, query in enumerate(self._dedupe(repair_plan.queries_by_slot.get(slot_id, []))[:3], start=1):
                started = time.perf_counter()
                call = AgentToolCall(
                    action="search_products",
                    slot_id=slot_id,
                    input_summary={"query": query, "attempt": attempt_index, "mode": "repair"},
                    reason="repair_search_executed",
                )
                try:
                    search_result = self.product_search_tool.search_query(
                        slot=slot,
                        base_plan=state.plan,
                        intent_plan=state.intent_plan,
                        query=query,
                        attempt_index=attempt_index,
                        reason="repair_search_executed",
                    )
                    call.output_summary = {
                        "candidate_count": len(search_result.candidates),
                        "candidate_ids": [candidate.product_id for candidate in search_result.candidates],
                        "counts": search_result.counts,
                    }
                    for candidate in search_result.candidates:
                        current = existing.get(candidate.product_id)
                        if current is None or candidate.rerank_score > current.rerank_score:
                            existing[candidate.product_id] = candidate
                except Exception as exc:
                    call.status = "failed"
                    call.reason = str(exc)
                finally:
                    call.duration_ms = round((time.perf_counter() - started) * 1000, 2)
                    state.tool_calls.append(call)
            state.candidates_by_slot[slot_id] = self._sort_slot_candidates(list(existing.values()))

        state.budgets.setdefault("internal_actions", []).append(
            {
                "action": "repair_search_executed",
                "targets": repair_plan.targets,
                "queries_by_slot": repair_plan.queries_by_slot,
            }
        )
        self.multi_need_coordinator.verify_coverage(state)
        state.final_signal = self.multi_need_coordinator._partial_or_no_product_signal(state, "repair evidence merged")
        state.termination_reason = state.final_signal.reason
        return state

    def _single_slot(self, original_query: str, intent_plan: IntentPlan, plan: QueryPlan) -> NeedSlot:
        query = intent_plan.vector_query or intent_plan.keyword_query or intent_plan.original_query or original_query
        return NeedSlot(
            slot_id="single",
            goal=query,
            product_type="",
            query=query,
            hard_constraints=list(plan.filters),
            soft_constraints=[*plan.preferences, *plan.scene],
            exclude_terms=list(plan.exclude),
            min_candidates=plan.retrieval_strategy.final_top_k,
        )

    def _sort_slot_candidates(self, candidates: list[SlotCandidate]) -> list[SlotCandidate]:
        return sorted(
            candidates,
            key=lambda candidate: (candidate.rerank_score, candidate.rrf_score, candidate.keyword_score),
            reverse=True,
        )

    def _merge_count_max(self, left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
        keys = set(left) | set(right)
        return {key: max(int(left.get(key, 0)), int(right.get(key, 0))) for key in keys}

    def _merge_score_maps(self, left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
        result = dict(left)
        for key, value in right.items():
            result[key] = max(result.get(key, 0.0), value)
        return result

    def _dedupe_products(self, products: list[Any]) -> list[Any]:
        result = []
        seen: set[str] = set()
        for product in products:
            product_id = getattr(product, "product_id", "")
            if product_id and product_id not in seen:
                result.append(product)
                seen.add(product_id)
        return result

    def _dedupe(self, values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            text = str(value or "").strip()
            if text and text not in result:
                result.append(text)
        return result
