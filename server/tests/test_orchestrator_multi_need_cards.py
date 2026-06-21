import asyncio
from decimal import Decimal

from app.db.models import Product
from app.domain.intent_planner import PlannerStreamEvent
from app.domain.image_retrieval_worker import ImageRetrievalEvidence
from app.domain.need_slot_schemas import MultiNeedSelection, MultiNeedState, NeedSlot, SlotCandidate
from app.domain.memory import ConversationContext, ConversationTurnView
from app.domain.orchestrator import EcommerceOrchestrator
from app.domain.task_lifecycle import TurnTaskState
from app.domain.input_processor import InputProcessor, NormalizedInput
from app.domain.retrieval_plan_builder import RetrievalPlanBuilder
from app.harness import BudgetManager, InMemoryEvidenceCache, TraceRecorder
from app.schemas import (
    ChatStreamRequest,
    ImageAttributes,
    IntentPlan,
    MULTI_NEED_PRODUCT_CARD_LIMIT,
    ProductCard,
    QueryBudget,
    QueryPlan,
    ReflectionResult,
)
from app.services.structured_llm import StructuredLlmValidationError


def make_product(product_id: str, name: str, price: str = "100") -> Product:
    return Product(
        product_id=product_id,
        name=name,
        category="服饰运动",
        sub_category="运动装备",
        brand="Demo",
        price=Decimal(price),
        stock=10,
        image_url="",
        description=name,
        specs={},
        ingredients_or_material="",
        suitable_for="",
        avoid_for="",
        tags=["运动"],
        rating=Decimal("4.8"),
        sales=100,
        review_summary="",
        image_caption="",
        structured_attributes={},
    )


def slot_candidate(product: Product, score: float = 0.8) -> SlotCandidate:
    return SlotCandidate(
        product=product,
        product_id=product.product_id,
        name=product.name,
        category=product.category,
        sub_category=product.sub_category,
        price=float(product.price),
        rerank_score=score,
    )


def test_multi_need_product_cards_include_final_combo_and_alternatives_deduped() -> None:
    primary = make_product("primary", "主推运动鞋")
    alternative = make_product("alternative", "备选运动鞋")
    state = MultiNeedState(
        original_query="配运动装备",
        intent_plan=IntentPlan(original_query="配运动装备", plan_type="multi_retrieval"),
        plan=QueryPlan(),
        slots=[NeedSlot(slot_id="s1", goal="运动鞋", product_type="运动鞋", query="运动鞋")],
        candidates_by_slot={"s1": [slot_candidate(primary), slot_candidate(alternative), slot_candidate(alternative)]},
    )
    selection = MultiNeedSelection(selected_by_slot={"s1": [slot_candidate(primary)]})
    reflection = ReflectionResult(
        passed_product_ids=["primary"],
        combo_summary={
            "final_combo_product_ids_by_slot": {"s1": ["primary"]},
            "alternative_product_ids_by_slot": {"s1": ["alternative", "alternative"]},
            "alternative_product_ids": ["alternative"],
        },
    )

    orchestrator = EcommerceOrchestrator.__new__(EcommerceOrchestrator)
    cards = orchestrator._card_candidates_from_reflection(state, selection, reflection)

    assert [candidate.product_id for candidate in cards] == ["primary", "alternative"]


def test_multi_need_product_cards_are_globally_limited() -> None:
    products = [make_product(f"p{index}", f"商品 {index}") for index in range(12)]
    state = MultiNeedState(
        original_query="配运动装备",
        intent_plan=IntentPlan(original_query="配运动装备", plan_type="multi_retrieval"),
        plan=QueryPlan(),
        slots=[NeedSlot(slot_id="s1", goal="运动鞋", product_type="运动鞋", query="运动鞋")],
        candidates_by_slot={"s1": [slot_candidate(product) for product in products]},
    )
    selection = MultiNeedSelection(selected_by_slot={"s1": [slot_candidate(products[0])]})
    reflection = ReflectionResult(
        passed_product_ids=["p0"],
        combo_summary={
            "final_combo_product_ids_by_slot": {"s1": ["p0"]},
            "alternative_product_ids_by_slot": {"s1": [product.product_id for product in products[1:]]},
            "alternative_product_ids": [product.product_id for product in products[1:]],
        },
    )

    orchestrator = EcommerceOrchestrator.__new__(EcommerceOrchestrator)
    cards = orchestrator._card_candidates_from_reflection(state, selection, reflection)

    assert len(cards) == MULTI_NEED_PRODUCT_CARD_LIMIT
    assert [candidate.product_id for candidate in cards][:2] == ["p0", "p1"]


class FakeCoordinator:
    def __init__(self, state: MultiNeedState) -> None:
        self.state = state

    async def run(self, *args, **kwargs) -> MultiNeedState:
        return self.state


class FakeRetrievalWorker:
    def __init__(self, state: MultiNeedState) -> None:
        self.state = state
        self.image_called = False

    async def run_multi_initial(self, *args, **kwargs) -> MultiNeedState:
        return self.state

    def run_multi_repair(self, state: MultiNeedState, repair_plan):
        return state

    def run_image_initial(self, *args, **kwargs) -> ImageRetrievalEvidence:
        self.image_called = True
        evidence = ImageRetrievalEvidence(
            before_structured_filter=0,
            structured_products=[],
            vector_query="<image>",
            keyword_query="",
            vector_scores={},
            keyword_scores={},
            score_filtered_products=[],
            hybrid_ranked_products=[],
            ranked=[],
            rerank_query="",
            tool_call_count=1,
            image_path=str(kwargs.get("image_path") or ""),
            max_image_score=0.0,
        )
        evidence.failure_trigger = "image_low_relevance"
        return evidence


class FakeCorrective:
    def __init__(self, reflection: ReflectionResult) -> None:
        self.reflection = reflection

    async def review_slots(self, *args, **kwargs) -> ReflectionResult:
        return self.reflection

    async def review(self, *args, **kwargs) -> ReflectionResult:
        return self.reflection


class FakeAnswerGenerator:
    async def stream_direct_text(self, *args, **kwargs):
        yield "answer"

    async def stream_text(self, *args, **kwargs):
        yield "answer"

    async def stream_multi_need_text(self, *args, **kwargs):
        yield "answer"

    def product_card(self, product: Product, plan: QueryPlan) -> ProductCard:
        return ProductCard(
            product_id=product.product_id,
            name=product.name,
            category=product.category,
            sub_category=product.sub_category,
            brand=product.brand,
            price=float(product.price),
            image_url=product.image_url,
            tags=product.tags[:6],
            rating=float(product.rating),
            reason="candidate card",
        )


class FakeTraceRun:
    async def span(self, *args, **kwargs) -> None:
        return None

    async def end(self, *args, **kwargs) -> None:
        return None


class FakeLangfuseTracer:
    def start_run(self, *args, **kwargs) -> FakeTraceRun:
        return FakeTraceRun()


class FakeMemoryManager:
    def build_context(self, *args, **kwargs) -> ConversationContext:
        return ConversationContext()


class FakeInputProcessor:
    def __init__(self, image_path) -> None:
        self.image_path = image_path

    def normalize(self, request: ChatStreamRequest) -> NormalizedInput:
        return NormalizedInput(text=request.message, image_path=self.image_path)


class UnavailableImageAttributeExtractor:
    async def extract(self, *args, **kwargs) -> ImageAttributes:
        return ImageAttributes(available=False, uncertainty_note="VLM unavailable in test.")


class FailingIntentPlanner:
    async def plan(self, *args, **kwargs) -> IntentPlan:
        raise StructuredLlmValidationError(
            "IntentPlanner returned invalid JSON.",
            errors=["missing field plan_type"],
            data={"foo": "bar"},
            content='{"foo":"bar"}',
        )

    async def stream_plan_with_summary(self, *args, **kwargs):
        if False:
            yield None
        raise StructuredLlmValidationError(
            "IntentPlanner returned invalid JSON.",
            errors=["missing field plan_type"],
            data={"foo": "bar"},
            content='{"foo":"bar"}',
        )


class FakeStreamingIntentPlanner:
    async def stream_plan_with_summary(self, query: str, context: dict) -> None:
        yield PlannerStreamEvent(kind="summary_delta", content="我会先理解需求，再选择合适路径。")
        yield PlannerStreamEvent(
            kind="plan",
            intent_plan=IntentPlan(
                original_query=query,
                summary="我会先理解需求，再选择合适路径。",
                plan_type="direct_answer",
                plan_reason="本轮是能力说明，不需要商品库证据。",
            ),
        )


class FakeReferenceIntentPlanner:
    async def stream_plan_with_summary(self, query: str, context: dict) -> None:
        yield PlannerStreamEvent(kind="summary_delta", content="我会复用上一轮商品证据进行对比。")
        yield PlannerStreamEvent(
            kind="plan",
            intent_plan=IntentPlan(
                original_query=query,
                summary="我会复用上一轮商品证据进行对比。",
                plan_type="direct_answer",
                referenced_product_ids=["p1", "p2"],
                plan_reason="当前问题引用上一轮商品，可基于上下文商品证据回答。",
            ),
        )


class FakeImageIntentPlanner:
    async def stream_plan_with_summary(self, query: str, context: dict) -> None:
        yield PlannerStreamEvent(kind="summary_delta", content="我会结合图片理解找相似商品。")
        yield PlannerStreamEvent(
            kind="plan",
            intent_plan=IntentPlan(
                original_query=query,
                summary="我会结合图片理解找相似商品。",
                plan_type="single_retrieval",
                vector_query="图片相似商品",
                keyword_query="图片 相似 商品",
                plan_reason="用户上传图片并要求找相似商品。",
            ),
        )

    async def plan(self, query: str, context: dict) -> IntentPlan:
        return IntentPlan(
            original_query=query,
            plan_type="single_retrieval",
            vector_query="图片相似商品",
            keyword_query="图片 相似 商品",
        )


class PlannerShouldNotBeCalled:
    async def stream_plan_with_summary(self, *args, **kwargs):
        raise AssertionError("Planner should not be called for pure image fast path.")
        if False:
            yield None

    async def plan(self, *args, **kwargs):
        raise AssertionError("Planner should not be called for pure image fast path.")


class FakeReferenceMemoryManager:
    def build_context(self, *args, **kwargs) -> ConversationContext:
        return ConversationContext(
            recent_turns=[
                ConversationTurnView(
                    turn_id=1,
                    user_message="推荐两款防晒",
                    assistant_message="推荐了欧莱雅和理肤泉。",
                    route="recommend",
                    product_ids=["p1", "p2"],
                )
            ]
        )


class FakeProductRepository:
    def __init__(self, products: list[Product]) -> None:
        self.products = {product.product_id: product for product in products}

    def get_by_ids(self, product_ids: list[str]) -> list[Product]:
        return [self.products[product_id] for product_id in product_ids if product_id in self.products]


def test_stream_stops_with_failure_text_when_intent_planner_retry_exhausted() -> None:
    orchestrator = EcommerceOrchestrator.__new__(EcommerceOrchestrator)
    orchestrator.input_processor = InputProcessor()
    orchestrator.intent_planner = FailingIntentPlanner()
    orchestrator.memory_manager = FakeMemoryManager()
    orchestrator.langfuse_tracer = FakeLangfuseTracer()
    orchestrator.budget_manager = BudgetManager()
    orchestrator.evidence_cache = InMemoryEvidenceCache()
    orchestrator.trace_recorder = TraceRecorder()
    memory_updates: list[dict] = []
    orchestrator._schedule_memory_update = lambda **kwargs: memory_updates.append(kwargs)

    events = asyncio.run(
        _collect(
            orchestrator.stream(
                ChatStreamRequest(user_id="u1", session_id="s1", message="你是谁？")
            )
        )
    )

    trace_event = next(event for event in events if event["type"] == "decision_trace")
    token_event = next(event for event in events if event["type"] == "token")

    assert trace_event["trace"]["route"] == "planner_failed"
    assert trace_event["trace"]["failure_stage"] == "intent_planning"
    assert trace_event["trace"]["task_status"] == "failed"
    assert "停止检索流程" in token_event["content"]
    assert events[-1]["type"] == "done"
    assert memory_updates[0]["route"] == "planner_failed"


def test_stream_emits_planner_agent_update_before_answer_token_and_sanitizes_trace() -> None:
    orchestrator = EcommerceOrchestrator.__new__(EcommerceOrchestrator)
    orchestrator.input_processor = InputProcessor()
    orchestrator.intent_planner = FakeStreamingIntentPlanner()
    orchestrator.memory_manager = FakeMemoryManager()
    orchestrator.langfuse_tracer = FakeLangfuseTracer()
    orchestrator.budget_manager = BudgetManager()
    orchestrator.evidence_cache = InMemoryEvidenceCache()
    orchestrator.trace_recorder = TraceRecorder()
    orchestrator.answer_generator = FakeAnswerGenerator()
    orchestrator._schedule_memory_update = lambda **kwargs: None

    events = asyncio.run(
        _collect(
            orchestrator.stream(
                ChatStreamRequest(user_id="u1", session_id="s1", message="你能做什么？")
            )
        )
    )

    agent_index = next(index for index, event in enumerate(events) if event["type"] == "agent_update")
    token_index = next(index for index, event in enumerate(events) if event["type"] == "token")
    trace_event = next(event for event in events if event["type"] == "decision_trace")

    assert agent_index < token_index
    assert events[agent_index]["stage"] == "planner"
    assert "理解需求" in events[agent_index]["title"]
    assert not _contains_any_key(
        trace_event["trace"],
        {"original_query", "vector_query", "keyword_query", "query", "normalized_query", "message"},
    )


def test_stream_context_reference_emits_product_cards() -> None:
    p1 = make_product("p1", "欧莱雅防晒", "170")
    p2 = make_product("p2", "理肤泉防晒", "268")
    orchestrator = EcommerceOrchestrator.__new__(EcommerceOrchestrator)
    orchestrator.input_processor = InputProcessor()
    orchestrator.intent_planner = FakeReferenceIntentPlanner()
    orchestrator.memory_manager = FakeReferenceMemoryManager()
    orchestrator.product_repository = FakeProductRepository([p1, p2])
    orchestrator.langfuse_tracer = FakeLangfuseTracer()
    orchestrator.budget_manager = BudgetManager()
    orchestrator.evidence_cache = InMemoryEvidenceCache()
    orchestrator.trace_recorder = TraceRecorder()
    orchestrator.answer_generator = FakeAnswerGenerator()
    orchestrator._schedule_memory_update = lambda **kwargs: None

    events = asyncio.run(
        _collect(
            orchestrator.stream(
                ChatStreamRequest(user_id="u1", session_id="s1", message="对比这两款防晒")
            )
        )
    )

    cards_event = next(event for event in events if event["type"] == "product_cards")
    trace_event = next(event for event in events if event["type"] == "decision_trace")

    assert trace_event["trace"]["retrieval_summary"]["answer_mode"] == "context_evidence"
    assert [product["product_id"] for product in cards_event["products"]] == ["p1", "p2"]
    assert [product["name"] for product in cards_event["products"]] == ["欧莱雅防晒", "理肤泉防晒"]


def test_vlm_unavailable_still_runs_image_retrieval(tmp_path) -> None:
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"fake image")
    state = MultiNeedState(original_query="", intent_plan=IntentPlan(), plan=QueryPlan())
    retrieval_worker = FakeRetrievalWorker(state)
    orchestrator = EcommerceOrchestrator.__new__(EcommerceOrchestrator)
    orchestrator.input_processor = FakeInputProcessor(image_path)
    orchestrator.image_attribute_extractor = UnavailableImageAttributeExtractor()
    orchestrator.intent_planner = FakeImageIntentPlanner()
    orchestrator.memory_manager = FakeMemoryManager()
    orchestrator.langfuse_tracer = FakeLangfuseTracer()
    orchestrator.budget_manager = BudgetManager()
    orchestrator.evidence_cache = InMemoryEvidenceCache()
    orchestrator.trace_recorder = TraceRecorder()
    orchestrator.retrieval_plan_builder = RetrievalPlanBuilder()
    orchestrator.retrieval_worker = retrieval_worker
    orchestrator.corrective_agent = FakeCorrective(
        ReflectionResult(
            has_passed_products=False,
            reason="图片候选不足。",
            fallback_plan="no_product",
        )
    )
    orchestrator.answer_generator = FakeAnswerGenerator()
    orchestrator._schedule_memory_update = lambda **kwargs: None

    events = asyncio.run(
        _collect(
            orchestrator.stream(
                ChatStreamRequest(
                    user_id="u1",
                    session_id="s1",
                    message="帮我找相似款",
                    image_id="img1",
                )
            )
        )
    )

    assert retrieval_worker.image_called
    first_agent_index = next(index for index, event in enumerate(events) if event["type"] == "agent_update")
    image_trace_index = next(
        index
        for index, event in enumerate(events)
        if event["type"] == "trace" and event["stage"] == "image_attribute_extraction"
    )
    assert first_agent_index < image_trace_index
    assert events[first_agent_index]["title"] == "理解图片"
    assert any(event["type"] == "trace" and event["stage"] == "image_attribute_extraction" for event in events)
    assert any(event["type"] == "trace" and event["stage"] == "image_retrieval_worker_execution" for event in events)
    trace = next(event["trace"] for event in events if event["type"] == "decision_trace")
    assert trace["route"] == "no_product"
    assert trace["image_attributes"]["available"] is False


def test_pure_image_fast_path_skips_planner(tmp_path) -> None:
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"fake image")
    state = MultiNeedState(original_query="", intent_plan=IntentPlan(), plan=QueryPlan())
    retrieval_worker = FakeRetrievalWorker(state)
    orchestrator = EcommerceOrchestrator.__new__(EcommerceOrchestrator)
    orchestrator.input_processor = FakeInputProcessor(image_path)
    orchestrator.image_attribute_extractor = UnavailableImageAttributeExtractor()
    orchestrator.intent_planner = PlannerShouldNotBeCalled()
    orchestrator.memory_manager = FakeMemoryManager()
    orchestrator.langfuse_tracer = FakeLangfuseTracer()
    orchestrator.budget_manager = BudgetManager()
    orchestrator.evidence_cache = InMemoryEvidenceCache()
    orchestrator.trace_recorder = TraceRecorder()
    orchestrator.retrieval_plan_builder = RetrievalPlanBuilder()
    orchestrator.retrieval_worker = retrieval_worker
    orchestrator.corrective_agent = FakeCorrective(
        ReflectionResult(
            has_passed_products=False,
            reason="图片候选不足。",
            fallback_plan="no_product",
        )
    )
    orchestrator.answer_generator = FakeAnswerGenerator()
    orchestrator._schedule_memory_update = lambda **kwargs: None

    events = asyncio.run(
        _collect(
            orchestrator.stream(
                ChatStreamRequest(
                    user_id="u1",
                    session_id="s1",
                    message="",
                    image_id="img1",
                )
            )
        )
    )

    assert retrieval_worker.image_called
    intent_trace = next(event for event in events if event["type"] == "trace" and event["stage"] == "intent_planning")
    assert intent_trace["content"] == "Orchestrator 已构造图片快路径计划"
    assert any(event["type"] == "trace" and event["stage"] == "image_retrieval_worker_execution" for event in events)


def test_stream_multi_need_emits_product_cards_for_over_budget_combo() -> None:
    shoe = make_product("shoe", "跑鞋", "899")
    shorts = make_product("shorts", "运动短裤", "149")
    state = MultiNeedState(
        original_query="总预算1000，运动鞋和运动服装",
        intent_plan=IntentPlan(
            original_query="总预算1000，运动鞋和运动服装",
            plan_type="multi_retrieval",
            budget_max=1000,
            budget_scope="total",
        ),
        plan=QueryPlan(budget=QueryBudget(max=1000)),
        slots=[
            NeedSlot(slot_id="s1", goal="运动鞋", product_type="运动鞋", query="运动鞋"),
            NeedSlot(slot_id="s2", goal="运动服装", product_type="运动服装", query="运动服装"),
        ],
        candidates_by_slot={"s1": [slot_candidate(shoe)], "s2": [slot_candidate(shorts)]},
        budgets={"search_calls": 2, "decision_steps": 2, "slot_results_by_slot": {}},
        termination_reason="slot agents completed",
    )
    reflection = ReflectionResult(
        has_passed_products=True,
        reason="完整组合超预算。",
        passed_product_ids=["shoe", "shorts"],
        slot_coverage=[
            {"slot_id": "s1", "status": "covered", "selected_product_ids": ["shoe"], "rejected_product_ids": [], "reason": "ok"},
            {"slot_id": "s2", "status": "covered", "selected_product_ids": ["shorts"], "rejected_product_ids": [], "reason": "ok"},
        ],
        combo_summary={
            "status": "over_budget",
            "final_combo_product_ids_by_slot": {"s1": ["shoe"], "s2": ["shorts"]},
            "selected_product_ids_by_slot": {"s1": ["shoe"], "s2": ["shorts"]},
        },
    )
    orchestrator = EcommerceOrchestrator.__new__(EcommerceOrchestrator)
    orchestrator.multi_need_coordinator = FakeCoordinator(state)
    orchestrator.retrieval_worker = FakeRetrievalWorker(state)
    orchestrator.corrective_agent = FakeCorrective(reflection)
    orchestrator.answer_generator = FakeAnswerGenerator()
    orchestrator.budget_manager = BudgetManager()
    orchestrator.trace_recorder = TraceRecorder()
    orchestrator.evidence_cache = InMemoryEvidenceCache()
    orchestrator._schedule_memory_update = lambda **kwargs: None
    task = TurnTaskState(user_id="u1", session_id="s1")
    request = ChatStreamRequest(user_id="u1", session_id="s1", message=state.original_query)

    events = asyncio.run(
        _collect(
            orchestrator._stream_multi_need(
                request=request,
                query=state.original_query,
                intent_plan=state.intent_plan,
                plan=state.plan,
                slots=state.slots,
                task=task,
                trace_run=FakeTraceRun(),
            )
        )
    )

    cards_event = next(event for event in events if event["type"] == "product_cards")
    assert [product["product_id"] for product in cards_event["products"]] == ["shoe", "shorts"]
    trace_event = next(event for event in events if event["type"] == "decision_trace")
    assert trace_event["trace"]["route"] == "over_budget_combo"


def test_multi_need_optional_slot_missing_does_not_downgrade_route() -> None:
    orchestrator = EcommerceOrchestrator.__new__(EcommerceOrchestrator)
    reflection = ReflectionResult(
        has_passed_products=True,
        passed_product_ids=["foundation"],
        slot_coverage=[
            {"slot_id": "required_makeup", "status": "covered", "selected_product_ids": ["foundation"], "reason": "ok"},
            {"slot_id": "optional_brow", "status": "missing", "selected_product_ids": [], "reason": "optional missing"},
        ],
        combo_summary={
            "status": "missing_required",
            "missing_required_slot_ids": [],
            "selected_product_ids": ["foundation"],
        },
    )

    assert orchestrator._final_route_from_reflection(reflection, "multi_retrieval") == "recommend"


def test_multi_need_required_slot_missing_downgrades_route() -> None:
    orchestrator = EcommerceOrchestrator.__new__(EcommerceOrchestrator)
    reflection = ReflectionResult(
        has_passed_products=True,
        passed_product_ids=["foundation"],
        combo_summary={
            "status": "missing_required",
            "missing_required_slot_ids": ["required_lip"],
            "selected_product_ids": ["foundation"],
        },
    )

    assert orchestrator._final_route_from_reflection(reflection, "multi_retrieval") == "partial_recommend"


def test_multi_need_inferred_missing_required_slot_does_not_downgrade_route() -> None:
    orchestrator = EcommerceOrchestrator.__new__(EcommerceOrchestrator)
    state = MultiNeedState(
        original_query="露营拍照和徒步都要用，推荐一套轻量户外装备。",
        intent_plan=IntentPlan(original_query="露营拍照和徒步都要用，推荐一套轻量户外装备。", plan_type="multi_retrieval"),
        plan=QueryPlan(),
        slots=[
            NeedSlot(slot_id="s1", need_type="required", goal="露营帐篷", product_type="帐篷", query="轻量帐篷"),
            NeedSlot(slot_id="s2", need_type="required", goal="徒步背包", product_type="背包", query="轻量徒步背包"),
        ],
    )
    reflection = ReflectionResult(
        has_passed_products=True,
        passed_product_ids=["backpack"],
        combo_summary={
            "status": "missing_required",
            "missing_required_slot_ids": ["s1"],
            "selected_product_ids": ["backpack"],
        },
    )

    assert (
        orchestrator._final_route_from_reflection(reflection, "multi_retrieval", multi_need_state=state)
        == "recommend"
    )


def test_multi_need_explicit_missing_required_slot_still_downgrades_route() -> None:
    orchestrator = EcommerceOrchestrator.__new__(EcommerceOrchestrator)
    state = MultiNeedState(
        original_query="买粉底和口红",
        intent_plan=IntentPlan(original_query="买粉底和口红", plan_type="multi_retrieval"),
        plan=QueryPlan(),
        slots=[
            NeedSlot(slot_id="s1", need_type="required", goal="粉底", product_type="粉底", query="粉底"),
            NeedSlot(slot_id="s2", need_type="required", goal="口红", product_type="口红", query="口红"),
        ],
    )
    reflection = ReflectionResult(
        has_passed_products=True,
        passed_product_ids=["foundation"],
        combo_summary={
            "status": "missing_required",
            "missing_required_slot_ids": ["s2"],
            "selected_product_ids": ["foundation"],
        },
    )

    assert (
        orchestrator._final_route_from_reflection(reflection, "multi_retrieval", multi_need_state=state)
        == "partial_recommend"
    )


def test_previous_evidence_reference_does_not_skip_retrieval_for_multi_plan() -> None:
    orchestrator = EcommerceOrchestrator.__new__(EcommerceOrchestrator)
    orchestrator.evidence_cache = InMemoryEvidenceCache()
    task = TurnTaskState(user_id="u1", session_id="s1")
    context = ConversationContext(
        recent_turns=[
            ConversationTurnView(
                turn_id=1,
                user_message="配新手化妆品",
                assistant_message="找到粉底液，也提到唇釉候选。",
                route="partial_recommend",
                product_ids=["p1"],
            )
        ]
    )
    intent_plan = IntentPlan(
        original_query="唇釉也可以",
        plan_type="multi_retrieval",
        referenced_product_ids=["p1"],
    )

    decision = orchestrator.decide_previous_evidence_answer(
        task,
        intent_plan,
        context,
        [make_product("p1", "粉底液")],
    )

    assert decision.approved is False
    assert decision.decision_summary["plan_allows_direct_context_answer"] is False


async def _collect(generator):
    return [event async for event in generator]


def _contains_any_key(value, keys: set[str]) -> bool:
    if isinstance(value, dict):
        return any(key in keys or _contains_any_key(item, keys) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_any_key(item, keys) for item in value)
    return False
