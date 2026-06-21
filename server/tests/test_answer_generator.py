import asyncio
import json
from decimal import Decimal

from app.db.models import Product
from app.domain.answer_generator import AnswerGenerator
from app.domain.need_slot_schemas import MultiNeedSelection, MultiNeedState, NeedSlot, SlotCandidate, SlotCoverageDecision
from app.schemas import IntentPlan, QueryPlan


class RecordingLlmClient:
    def __init__(self) -> None:
        self.system_prompt = ""
        self.user_prompt = ""

    async def generate_required(self, system_prompt: str, user_prompt: str) -> str:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return "LLM 多需求回答"


def make_product() -> Product:
    return Product(
        product_id="laptop",
        name="轻薄笔记本电脑",
        category="数码电子",
        sub_category="笔记本电脑",
        brand="Demo",
        price=Decimal("6999"),
        stock=10,
        image_url="",
        description="适合办公学习的轻薄笔记本电脑",
        specs={},
        ingredients_or_material="",
        suitable_for="",
        avoid_for="",
        tags=["笔记本电脑"],
        rating=Decimal("4.8"),
        sales=100,
        review_summary="性能稳定，便携。",
        image_caption="",
        structured_attributes={},
    )


def make_product_with_id(product_id: str, name: str, price: str = "10") -> Product:
    product = make_product()
    product.product_id = product_id
    product.name = name
    product.price = Decimal(price)
    product.review_summary = f"{name} 评价摘要"
    return product


def test_single_answer_treats_all_ranked_products_as_formal_recommendations() -> None:
    products = [
        make_product_with_id("drink1", "甜味饮料 A"),
        make_product_with_id("drink2", "甜味饮料 B"),
        make_product_with_id("drink3", "甜味饮料 C"),
        make_product_with_id("drink4", "甜味饮料 D"),
    ]
    llm_client = RecordingLlmClient()
    generator = AnswerGenerator()
    generator.llm_client = llm_client

    asyncio.run(
        _collect(
            generator.stream_text(
                QueryPlan(preferences=["甜"]),
                [(product, 1.0 - index * 0.1) for index, product in enumerate(products)],
            )
        )
    )
    payload = json.loads(llm_client.user_prompt)

    assert payload["final_recommendation_order"] == ["drink1", "drink2", "drink3", "drink4"]
    assert [product["product_id"] for product in payload["final_products"]] == ["drink1", "drink2", "drink3", "drink4"]
    assert "推荐全部 4 个" in payload["instruction"]
    assert "不要只挑 2-3 个" in llm_client.system_prompt
    assert "候补" in llm_client.system_prompt


def test_multi_need_answer_uses_corrective_slot_coverage() -> None:
    product = make_product()
    near_miss = make_product()
    near_miss.product_id = "tablet"
    near_miss.name = "轻办公平板电脑"
    near_miss.sub_category = "平板电脑"
    state = MultiNeedState(
        original_query="对比笔记本电脑和主机",
        intent_plan=IntentPlan(original_query="对比笔记本电脑和主机"),
        plan=QueryPlan(),
        slots=[
            NeedSlot(slot_id="s1", goal="笔记本电脑", product_type="笔记本电脑", query="笔记本电脑"),
            NeedSlot(slot_id="s2", goal="主机", product_type="主机", query="主机"),
        ],
        coverage_by_slot={
            "s1": SlotCoverageDecision(slot_id="s1", status="covered", covered=True),
            "s2": SlotCoverageDecision(slot_id="s2", status="covered", covered=True),
        },
        candidates_by_slot={
            "s2": [
                SlotCandidate(
                    product=near_miss,
                    product_id=near_miss.product_id,
                    name=near_miss.name,
                    category=near_miss.category,
                    sub_category=near_miss.sub_category,
                    price=float(near_miss.price),
                    rerank_score=0.6,
                )
            ]
        },
    )
    selection = MultiNeedSelection(
        route="partial_recommend",
        selected_by_slot={
            "s1": [
                SlotCandidate(
                    product=product,
                    product_id=product.product_id,
                    name=product.name,
                    category=product.category,
                    sub_category=product.sub_category,
                    price=float(product.price),
                    rerank_score=0.8,
                )
            ]
        },
    )
    llm_client = RecordingLlmClient()
    generator = AnswerGenerator()
    generator.llm_client = llm_client

    text = "".join(
        asyncio.run(
            _collect(
                generator.stream_multi_need_text(
                    state,
                    selection,
                    "partial_recommend",
                    "笔记本电脑可推荐，主机缺失。",
                    [
                        {"slot_id": "s1", "status": "covered", "selected_product_ids": ["laptop"]},
                        {"slot_id": "s2", "status": "missing", "selected_product_ids": []},
                    ],
                )
            )
        )
    )
    payload = json.loads(llm_client.user_prompt)

    assert text == "LLM 多需求回答"
    assert payload["final_products_by_slot"]["s1"]["corrective_coverage"]["status"] == "covered"
    assert "s2" not in payload["final_products_by_slot"]
    assert payload["missing_slots"][0]["slot_id"] == "s2"
    assert payload["near_miss_suggestions"][0]["slot_id"] == "s2"
    near_miss = payload["near_miss_suggestions"][0]["suggestions"][0]
    assert near_miss["product_id"] == "tablet"
    assert "price" not in near_miss
    assert "rating" not in near_miss
    assert "description" not in near_miss


def test_multi_need_answer_includes_alternatives_separately_from_near_miss() -> None:
    product = make_product()
    alternative = make_product()
    alternative.product_id = "alt_laptop"
    alternative.name = "同类轻薄笔记本电脑"
    state = MultiNeedState(
        original_query="推荐笔记本电脑组合",
        intent_plan=IntentPlan(original_query="推荐笔记本电脑组合"),
        plan=QueryPlan(),
        slots=[NeedSlot(slot_id="s1", goal="笔记本电脑", product_type="笔记本电脑", query="笔记本电脑")],
        candidates_by_slot={
            "s1": [
                SlotCandidate(
                    product=product,
                    product_id=product.product_id,
                    name=product.name,
                    category=product.category,
                    sub_category=product.sub_category,
                    price=float(product.price),
                    rerank_score=0.9,
                ),
                SlotCandidate(
                    product=alternative,
                    product_id=alternative.product_id,
                    name=alternative.name,
                    category=alternative.category,
                    sub_category=alternative.sub_category,
                    price=float(alternative.price),
                    rerank_score=0.7,
                ),
            ]
        },
    )
    selection = MultiNeedSelection(
        route="recommend",
        selected_by_slot={
            "s1": [
                SlotCandidate(
                    product=product,
                    product_id=product.product_id,
                    name=product.name,
                    category=product.category,
                    sub_category=product.sub_category,
                    price=float(product.price),
                    rerank_score=0.9,
                )
            ]
        },
    )
    llm_client = RecordingLlmClient()
    generator = AnswerGenerator()
    generator.llm_client = llm_client

    asyncio.run(
        _collect(
            generator.stream_multi_need_text(
                state,
                selection,
                "recommend",
                "主推和备选都语义通过。",
                [{"slot_id": "s1", "status": "covered", "selected_product_ids": ["laptop", "alt_laptop"]}],
                [],
                {
                    "final_combo_product_ids_by_slot": {"s1": ["laptop"]},
                    "alternative_product_ids_by_slot": {"s1": ["alt_laptop"]},
                },
            )
        )
    )
    payload = json.loads(llm_client.user_prompt)

    assert payload["final_products_by_slot"]["s1"]["products"][0]["product_id"] == "laptop"
    assert payload["alternatives_by_slot"]["s1"]["products"][0]["product_id"] == "alt_laptop"
    assert payload["near_miss_suggestions"] == []


def test_multi_need_answer_keeps_over_budget_combo_out_of_formal_recommendations() -> None:
    product = make_product()
    state = MultiNeedState(
        original_query="总预算500，配笔记本电脑组合",
        intent_plan=IntentPlan(original_query="总预算500，配笔记本电脑组合", budget_scope="total", budget_max=500),
        plan=QueryPlan(),
        slots=[NeedSlot(slot_id="s1", goal="笔记本电脑", product_type="笔记本电脑", query="笔记本电脑")],
    )
    selection = MultiNeedSelection(
        route="over_budget_combo",
        selected_by_slot={
            "s1": [
                SlotCandidate(
                    product=product,
                    product_id=product.product_id,
                    name=product.name,
                    category=product.category,
                    sub_category=product.sub_category,
                    price=float(product.price),
                    rerank_score=0.9,
                )
            ]
        },
    )
    llm_client = RecordingLlmClient()
    generator = AnswerGenerator()
    generator.llm_client = llm_client

    asyncio.run(
        _collect(
            generator.stream_multi_need_text(
                state,
                selection,
                "over_budget_combo",
                "完整组合超预算。",
                [{"slot_id": "s1", "status": "covered", "selected_product_ids": ["laptop"]}],
                [],
                {
                    "status": "over_budget",
                    "budget_scope": "total",
                    "budget_max": 500,
                    "total_price": 6999,
                    "over_budget_amount": 6499,
                    "final_combo_product_ids_by_slot": {"s1": ["laptop"]},
                },
            )
        )
    )
    payload = json.loads(llm_client.user_prompt)

    assert payload["final_products_by_slot"] == {}
    assert payload["over_budget_combo_by_slot"]["s1"]["products"][0]["product_id"] == "laptop"
    assert "answer_evidence_brief" in payload
    assert "当前组合是否为本轮通过候选中的最低完整组合：是" in payload["answer_evidence_brief"]
    assert "本轮是否发现预算内完整组合：否" in payload["answer_evidence_brief"]
    assert "最终回复用户时不要输出" in payload["answer_evidence_brief"]
    assert "最低完整组合候选" in llm_client.system_prompt
    assert "不要输出 product_id" in llm_client.system_prompt
    assert "不得暗示当前已经存在低价替代" in llm_client.system_prompt


def test_multi_need_answer_marks_inferred_slots_without_user_attribution() -> None:
    product = make_product()
    inferred_candidate = make_product()
    inferred_candidate.product_id = "spray_like"
    inferred_candidate.name = "相近补涂产品"
    state = MultiNeedState(
        original_query="去海边玩，要准备的防晒单品有哪些？预算500，帮我规划一下",
        intent_plan=IntentPlan(original_query="去海边玩，要准备的防晒单品有哪些？预算500，帮我规划一下", budget_scope="total", budget_max=500),
        plan=QueryPlan(),
        slots=[
            NeedSlot(slot_id="s1", goal="防晒霜", product_type="防晒霜", query="防晒霜"),
            NeedSlot(slot_id="s2", goal="防晒喷雾", product_type="防晒喷雾", query="防晒喷雾"),
        ],
        candidates_by_slot={
            "s2": [
                SlotCandidate(
                    product=inferred_candidate,
                    product_id=inferred_candidate.product_id,
                    name=inferred_candidate.name,
                    category=inferred_candidate.category,
                    sub_category=inferred_candidate.sub_category,
                    price=float(inferred_candidate.price),
                    rerank_score=0.6,
                )
            ]
        },
    )
    selection = MultiNeedSelection(
        route="partial_recommend",
        selected_by_slot={
            "s1": [
                SlotCandidate(
                    product=product,
                    product_id=product.product_id,
                    name=product.name,
                    category=product.category,
                    sub_category=product.sub_category,
                    price=float(product.price),
                    rerank_score=0.9,
                )
            ]
        },
    )
    llm_client = RecordingLlmClient()
    generator = AnswerGenerator()
    generator.llm_client = llm_client

    asyncio.run(
        _collect(
            generator.stream_multi_need_text(
                state,
                selection,
                "partial_recommend",
                "防晒霜可推荐，补涂方向缺失。",
                [
                    {"slot_id": "s1", "status": "covered", "selected_product_ids": ["laptop"]},
                    {"slot_id": "s2", "status": "missing", "selected_product_ids": []},
                ],
            )
        )
    )
    payload = json.loads(llm_client.user_prompt)

    assert payload["final_products_by_slot"]["s1"]["source_attribution"]["source"] == "planner_inferred"
    assert payload["missing_slots"][0]["source_attribution"]["source"] == "planner_inferred"
    assert payload["near_miss_suggestions"][0]["source_attribution"]["source"] == "planner_inferred"
    assert "original_query 才是用户原话" in llm_client.system_prompt
    assert "planner_inferred" in llm_client.system_prompt
    assert "不要写成“您提到的 X”" in llm_client.system_prompt


def test_slot_source_attribution_marks_explicit_slot_terms() -> None:
    generator = AnswerGenerator()

    explicit = generator._slot_source_attribution("笔记本电脑", "笔记本电脑", "对比笔记本电脑和主机")
    inferred = generator._slot_source_attribution("防晒喷雾", "防晒喷雾", "去海边玩，要准备的防晒单品有哪些")

    assert explicit["source"] == "user_explicit"
    assert explicit["matched_terms"] == ["笔记本电脑"]
    assert inferred["source"] == "planner_inferred"
    assert inferred["matched_terms"] == []


async def _collect(stream):
    return [chunk async for chunk in stream]
