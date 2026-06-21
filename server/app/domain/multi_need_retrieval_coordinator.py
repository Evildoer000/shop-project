from __future__ import annotations

import asyncio
from typing import Any

from app.db.session import get_sessionmaker
from app.domain.need_slot_schemas import FinalAnswerSignal, NeedSlot, SlotCoverageDecision, MultiNeedState
from app.domain.product_search_tool import ProductSearchTool
from app.domain.slot_retrieval_agent import SlotRetrievalAgent, SlotRetrievalAgentResult
from app.schemas import IntentPlan, QueryPlan
from app.services.product_repository import ProductRepository


class MultiNeedRetrievalCoordinator:
    MAX_SLOTS = 5
    MAX_CONCURRENT_SLOT_AGENTS = 5

    def __init__(
        self,
        search_tool: ProductSearchTool,
        slot_agent_cls: type[SlotRetrievalAgent] = SlotRetrievalAgent,
    ) -> None:
        self.search_tool = search_tool
        self.slot_agent_cls = slot_agent_cls

    async def run(
        self,
        original_query: str,
        intent_plan: IntentPlan,
        plan: QueryPlan,
        slots: list[NeedSlot] | None = None,
    ) -> MultiNeedState:
        state = MultiNeedState(
            original_query=original_query,
            intent_plan=intent_plan,
            plan=plan,
            global_constraints=self._global_constraints(plan),
            slots=(slots or [])[: self.MAX_SLOTS],
            budgets=self._initial_budgets(),
        )
        if not state.slots:
            state.termination_reason = "not_multi_need"
            return state

        semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_SLOT_AGENTS)
        tasks = [
            self._run_slot_agent_task(
                semaphore=semaphore,
                slot=slot,
                plan=plan,
                intent_plan=intent_plan,
            )
            for slot in state.slots
        ]
        state.budgets["slot_task_mode"] = "concurrent_slot_agents"
        state.budgets["parallel_slot_agents"] = len(tasks)
        state.budgets["coordinator"] = "MultiNeedRetrievalCoordinator"

        for slot_result in await asyncio.gather(*tasks):
            self._record_slot_agent_result(state, slot_result)

        self.verify_coverage(state)
        state.final_signal = self._partial_or_no_product_signal(state, "slot agents completed")
        state.termination_reason = state.final_signal.reason
        return state

    async def _run_slot_agent_task(
        self,
        semaphore: asyncio.Semaphore,
        slot: NeedSlot,
        plan: QueryPlan,
        intent_plan: IntentPlan,
    ) -> SlotRetrievalAgentResult:
        async with semaphore:
            return await asyncio.to_thread(
                self._run_slot_agent_in_isolated_context,
                slot.model_copy(deep=True),
                plan.model_copy(deep=True),
                intent_plan.model_copy(deep=True),
            )

    def _run_slot_agent_in_isolated_context(
        self,
        slot: NeedSlot,
        plan: QueryPlan,
        intent_plan: IntentPlan,
    ) -> SlotRetrievalAgentResult:
        self._ensure_thread_event_loop()
        repository = getattr(self.search_tool, "product_repository", None)
        current_db = getattr(repository, "db", None)
        retriever = getattr(self.search_tool, "retriever", None)
        reranker = getattr(self.search_tool, "reranker", None)

        if current_db is None or retriever is None or reranker is None:
            agent = self.slot_agent_cls(self.search_tool)
            return agent.run(slot, plan, intent_plan)

        db = get_sessionmaker()()
        try:
            product_repository = ProductRepository(db)
            search_tool = ProductSearchTool(product_repository, retriever, reranker)
            agent = self.slot_agent_cls(search_tool)
            return agent.run(slot, plan, intent_plan)
        finally:
            db.close()

    def _ensure_thread_event_loop(self) -> None:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
            return
        if loop.is_closed():
            asyncio.set_event_loop(asyncio.new_event_loop())

    def _record_slot_agent_result(
        self,
        state: MultiNeedState,
        result: SlotRetrievalAgentResult,
    ) -> None:
        slot_id = result.slot.slot_id
        for index, slot in enumerate(state.slots):
            if slot.slot_id == slot_id:
                state.slots[index] = result.slot
                break

        state.tool_calls.extend(result.tool_calls)
        state.candidates_by_slot[slot_id] = result.candidates
        state.budgets["decision_steps"] += result.decision_steps
        state.budgets["search_calls"] += result.search_calls
        state.budgets["search_attempts"] += len(result.slot_result.get("attempts") or [])
        state.budgets["repair_calls_by_slot"][slot_id] = result.repair_calls
        state.budgets["slot_results_by_slot"][slot_id] = result.slot_result
        state.budgets["slot_agent_results_by_slot"][slot_id] = {
            "slot_id": slot_id,
            "decision_steps": result.decision_steps,
            "search_calls": result.search_calls,
            "repair_calls": result.repair_calls,
            "termination_reason": result.termination_reason,
            "tool_call_count": len(result.tool_calls),
            "candidate_ids": [candidate.product_id for candidate in result.candidates],
        }

    def verify_coverage(self, state: MultiNeedState) -> None:
        for slot in state.slots:
            candidates = state.candidates_by_slot.get(slot.slot_id, [])
            slot_result = state.budgets.get("slot_results_by_slot", {}).get(slot.slot_id, {})
            notes: list[str] = []
            if slot_result.get("category_resolution"):
                notes.append(f"category_resolution={slot_result.get('category_resolution')}")
            if slot_result.get("categories"):
                notes.append(f"categories={slot_result.get('categories')}")
            if slot_result.get("termination_reason"):
                notes.append(f"termination_reason={slot_result.get('termination_reason')}")

            if len(candidates) >= slot.min_candidates:
                status = "covered"
                reason = f"SlotRetrievalAgent 检索得到 {len(candidates)} 个候选，等待最终 Corrective Agent 判断是否真正覆盖。"
            elif candidates:
                status = "weak"
                reason = "SlotRetrievalAgent 有候选但数量不足，等待最终 Corrective Agent 判断。"
            elif self._has_searched(state, slot.slot_id):
                status = "failed"
                reason = "SlotRetrievalAgent 初次检索没有候选，等待 Orchestrator 裁决是否进入统一 repair。"
            else:
                status = "pending"
                reason = "等待 SlotRetrievalAgent 检索。"

            slot.status = status  # type: ignore[assignment]
            state.coverage_by_slot[slot.slot_id] = SlotCoverageDecision(
                slot_id=slot.slot_id,
                status=status,  # type: ignore[arg-type]
                covered=status == "covered",
                reason=reason,
                candidate_ids=[candidate.product_id for candidate in candidates],
                raw_candidate_ids=[candidate.product_id for candidate in candidates],
                candidate_count=len(candidates),
                attempt_count=len(slot_result.get("attempts") or []),
                notes=notes,
            )

    def _partial_or_no_product_signal(self, state: MultiNeedState, reason: str) -> FinalAnswerSignal:
        required_slots = [slot for slot in state.slots if slot.need_type == "required"]
        covered = [
            slot
            for slot in required_slots
            if state.coverage_by_slot.get(slot.slot_id) and state.coverage_by_slot[slot.slot_id].covered
        ]
        if required_slots and len(covered) == len(required_slots):
            return FinalAnswerSignal(route="recommend", reason=reason)
        if covered:
            return FinalAnswerSignal(route="partial_recommend", reason=reason)
        return FinalAnswerSignal(route="no_product", reason=reason)

    def _initial_budgets(self) -> dict[str, Any]:
        return {
            "decision_steps": 0,
            "search_calls": 0,
            "search_attempts": 0,
            "repair_calls_by_slot": {},
            "slot_results_by_slot": {},
            "slot_agent_results_by_slot": {},
            "limits": {
                "max_slots": self.MAX_SLOTS,
                "max_concurrent_slot_agents": self.MAX_CONCURRENT_SLOT_AGENTS,
                "max_initial_search_attempts_per_slot": SlotRetrievalAgent.MAX_ATTEMPTS_PER_SLOT,
                "max_slot_agent_decision_steps": SlotRetrievalAgent.MAX_DECISION_STEPS,
            },
        }

    def _global_constraints(self, plan: QueryPlan) -> list[str]:
        constraints = []
        if plan.budget.max is not None:
            constraints.append(f"price <= {plan.budget.max:g}")
        constraints.extend(f"exclude {term}" for term in plan.exclude)
        return constraints

    def _has_searched(self, state: MultiNeedState, slot_id: str) -> bool:
        return any(call.action == "search_products" and call.slot_id == slot_id for call in state.tool_calls)
