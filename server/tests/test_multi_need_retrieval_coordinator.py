import asyncio
import time
from decimal import Decimal

from app.db.models import Product
from app.domain.multi_need_retrieval_coordinator import MultiNeedRetrievalCoordinator
from app.domain.need_slot_schemas import NeedSlot, SlotCandidate, SlotSearchResult
from app.schemas import IntentPlan, QueryPlan


class SlowSearchTool:
    MAX_ATTEMPTS_PER_SLOT = 3

    def __init__(self, delay_seconds: float = 0.2) -> None:
        self.delay_seconds = delay_seconds

    def query_variants(self, slot: NeedSlot) -> list[str]:
        return [slot.query]

    def search_query(
        self,
        slot: NeedSlot,
        base_plan: QueryPlan,
        intent_plan: IntentPlan,
        query: str,
        attempt_index: int,
        reason: str,
    ) -> SlotSearchResult:
        time.sleep(self.delay_seconds)
        product = Product(
            product_id=f"p_{slot.slot_id}",
            name=f"{slot.goal} 商品",
            category="测试类目",
            sub_category=slot.product_type,
            brand="Demo",
            price=Decimal("99"),
            stock=10,
            image_url="",
            description=slot.goal,
            specs={},
            ingredients_or_material="",
            suitable_for="",
            avoid_for="",
            tags=[slot.product_type],
            rating=Decimal("4.8"),
            sales=100,
            review_summary="",
            image_caption="",
            structured_attributes={},
        )
        return SlotSearchResult(
            slot_id=slot.slot_id,
            query=slot.query,
            vector_query=query,
            keyword_query=query,
            candidates=[
                SlotCandidate(
                    product=product,
                    product_id=product.product_id,
                    name=product.name,
                    category=product.category,
                    sub_category=product.sub_category,
                    price=float(product.price),
                    rerank_score=0.4,
                )
            ],
            attempts=[{"attempt": attempt_index, "query": query, "reason": reason}],
        )


class EventLoopCheckingSearchTool(SlowSearchTool):
    def search_query(
        self,
        slot: NeedSlot,
        base_plan: QueryPlan,
        intent_plan: IntentPlan,
        query: str,
        attempt_index: int,
        reason: str,
    ) -> SlotSearchResult:
        asyncio.get_event_loop()
        return super().search_query(slot, base_plan, intent_plan, query, attempt_index, reason)


def test_coordinator_runs_slot_agents_concurrently() -> None:
    coordinator = MultiNeedRetrievalCoordinator(SlowSearchTool(delay_seconds=0.2))
    assert not hasattr(coordinator, "detect_slots_with_trace")
    slots = [
        NeedSlot(slot_id="s1", goal="防晒", product_type="防晒", query="防晒"),
        NeedSlot(slot_id="s2", goal="鞋", product_type="徒步鞋", query="徒步鞋"),
    ]

    started = time.perf_counter()
    state = asyncio.run(
        coordinator.run(
            "旅行要防晒和鞋",
            IntentPlan(original_query="旅行要防晒和鞋", plan_type="multi_retrieval"),
            QueryPlan(),
            slots=slots,
        )
    )
    elapsed = time.perf_counter() - started

    assert elapsed < 0.35
    assert state.budgets["slot_task_mode"] == "concurrent_slot_agents"
    assert state.budgets["parallel_slot_agents"] == 2
    assert state.budgets["coordinator"] == "MultiNeedRetrievalCoordinator"
    assert state.budgets["search_calls"] == 2
    assert {call.slot_id for call in state.tool_calls} == {"s1", "s2"}


def test_coordinator_prepares_event_loop_for_threaded_slot_agents() -> None:
    coordinator = MultiNeedRetrievalCoordinator(EventLoopCheckingSearchTool(delay_seconds=0))
    slots = [
        NeedSlot(slot_id="s1", goal="笔记本电脑", product_type="笔记本电脑", query="笔记本电脑"),
        NeedSlot(slot_id="s2", goal="平板电脑", product_type="平板电脑", query="平板电脑"),
    ]

    state = asyncio.run(
        coordinator.run(
            "对比笔记本电脑和平板电脑",
            IntentPlan(original_query="对比笔记本电脑和平板电脑", plan_type="multi_retrieval"),
            QueryPlan(),
            slots=slots,
        )
    )

    assert all(call.status == "ok" for call in state.tool_calls)
    assert state.budgets["search_calls"] == 2
    assert state.coverage_by_slot["s1"].covered is True
    assert state.coverage_by_slot["s2"].covered is True
