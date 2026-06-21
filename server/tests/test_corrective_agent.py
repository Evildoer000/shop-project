import asyncio
import json
from decimal import Decimal

from app.db.models import Product
from app.domain.corrective_agent import CorrectiveAgentController
from app.domain.need_slot_schemas import MultiNeedState, NeedSlot, SlotCandidate, SlotCoverageDecision
from app.schemas import IntentPlan, QueryBudget, QueryPlan


class StaticLlmClient:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    async def generate(self, system_prompt: str, user_prompt: str, response_format: dict | None = None) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response_format": response_format,
            }
        )
        return json.dumps(self.payload, ensure_ascii=False)

    async def generate_required(self, system_prompt: str, user_prompt: str, response_format: dict | None = None) -> str:
        return await self.generate(system_prompt, user_prompt, response_format)

    def is_configured(self) -> bool:
        return True


def make_product(product_id: str, name: str, price: str = "100", category: str = "服饰运动", sub_category: str = "跑步鞋") -> Product:
    return Product(
        product_id=product_id,
        name=name,
        category=category,
        sub_category=sub_category,
        brand="Demo",
        price=Decimal(price),
        stock=10,
        image_url="",
        description=name,
        specs={},
        ingredients_or_material="",
        suitable_for="",
        avoid_for="",
        tags=[sub_category],
        rating=Decimal("4.8"),
        sales=100,
        review_summary=name,
        image_caption="",
        structured_attributes={},
    )


def make_candidate(product: Product, score: float = 0.5) -> SlotCandidate:
    return SlotCandidate(
        product=product,
        product_id=product.product_id,
        name=product.name,
        category=product.category,
        sub_category=product.sub_category,
        price=float(product.price),
        vector_score=0.3,
        keyword_score=0.2,
        rrf_score=0.03,
        rerank_score=score,
    )


def test_single_reflection_can_fallback_to_direct_answer_without_route() -> None:
    product = make_product("p1", "错误候选")
    payload = {
        "reason": "用户是在问助手身份，不应推荐商品。",
        "fallback_plan": "direct_answer",
        "passed_product_ids": [],
        "rejected_products": [{"product_id": "p1", "reason": "非商品需求。"}],
    }

    result = asyncio.run(
        CorrectiveAgentController(StaticLlmClient(payload)).review(
            "你是谁？",
            IntentPlan(original_query="你是谁？", plan_type="direct_answer"),
            QueryPlan(),
            [(product, 0.9)],
            {},
            {},
        )
    )

    assert result.has_passed_products is False
    assert result.fallback_plan == "direct_answer"
    assert result.passed_product_ids == []


def test_single_reflection_passed_products_force_fallback_none() -> None:
    product = make_product("p1", "防晒霜", sub_category="防晒")
    payload = {
        "reason": "候选商品匹配防晒需求。",
        "fallback_plan": "no_product",
        "passed_product_ids": ["p1"],
        "rejected_products": [],
    }

    result = asyncio.run(
        CorrectiveAgentController(StaticLlmClient(payload)).review(
            "推荐防晒",
            IntentPlan(original_query="推荐防晒", plan_type="single_retrieval", vector_query="防晒"),
            QueryPlan(),
            [(product, 0.9)],
            {"p1": 0.4},
            {"p1": 0.3},
        )
    )

    assert result.has_passed_products is True
    assert result.passed_product_ids == ["p1"]
    assert result.fallback_plan == "none"


def test_single_corrective_prompt_keeps_old_semantic_guardrails() -> None:
    product = make_product("p1", "唇釉", sub_category="唇妆")
    payload = {
        "reason": "候选商品属于同一商品族。",
        "fallback_plan": "none",
        "passed_product_ids": ["p1"],
        "rejected_products": [],
    }
    llm_client = StaticLlmClient(payload)

    asyncio.run(
        CorrectiveAgentController(llm_client).review(
            "新手化妆可以要唇妆",
            IntentPlan(original_query="新手化妆可以要唇妆", plan_type="single_retrieval"),
            QueryPlan(),
            [(product, 0.9)],
            {},
            {},
        )
    )

    system_prompt = llm_client.calls[0]["system_prompt"]
    assert "证据反射 Worker Agent" in system_prompt
    assert "只输出 JSON object" in system_prompt
    assert "商品形态、商品族、核心功能" in system_prompt
    assert "这些例子是语义审核原则，不是固定映射表" in system_prompt
    assert "fallback_plan 只是给 Orchestrator 的反射建议，不是 final_route" in system_prompt


def test_multi_need_reflection_builds_over_budget_combo_summary() -> None:
    shoe = make_product("shoe", "跑鞋", "899", sub_category="跑步鞋")
    pants = make_product("pants", "运动短裤", "149", sub_category="运动短裤")
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
        candidates_by_slot={
            "s1": [make_candidate(shoe, 0.9)],
            "s2": [make_candidate(pants, 0.8)],
        },
        coverage_by_slot={
            "s1": SlotCoverageDecision(slot_id="s1", status="covered", covered=True),
            "s2": SlotCoverageDecision(slot_id="s2", status="covered", covered=True),
        },
    )
    payload = {
        "reason": "两个 required slot 都有语义通过商品。",
        "fallback_plan": "none",
        "slot_coverage": [
            {"slot_id": "s1", "status": "covered", "selected_product_ids": ["shoe"], "rejected_product_ids": [], "reason": "匹配运动鞋。"},
            {"slot_id": "s2", "status": "covered", "selected_product_ids": ["pants"], "rejected_product_ids": [], "reason": "匹配运动服装。"},
        ],
        "passed_product_ids": ["shoe", "pants"],
        "rejected_products": [],
    }

    result = asyncio.run(
        CorrectiveAgentController(StaticLlmClient(payload)).review_slots(
            state.original_query,
            state.intent_plan,
            state.plan,
            state,
        )
    )

    assert result.passed_product_ids == ["shoe", "pants"]
    assert result.combo_summary["status"] == "over_budget"
    assert result.combo_summary["total_price"] == 1048
    assert result.combo_summary["over_budget_amount"] == 48
    assert result.fallback_plan == "none"


def test_multi_need_corrective_prompt_keeps_slot_guardrails() -> None:
    lip = make_product("lip", "唇釉", "99", category="美妆护肤", sub_category="唇妆")
    state = MultiNeedState(
        original_query="新手化妆，要唇妆",
        intent_plan=IntentPlan(original_query="新手化妆，要唇妆", plan_type="multi_retrieval"),
        plan=QueryPlan(),
        slots=[NeedSlot(slot_id="s1", goal="唇妆", product_type="唇妆", query="唇妆")],
        candidates_by_slot={"s1": [make_candidate(lip, 0.9)]},
        coverage_by_slot={"s1": SlotCoverageDecision(slot_id="s1", status="covered", covered=True)},
    )
    payload = {
        "reason": "唇釉可覆盖唇妆 slot。",
        "fallback_plan": "none",
        "slot_coverage": [
            {"slot_id": "s1", "status": "covered", "selected_product_ids": ["lip"], "rejected_product_ids": [], "reason": "匹配唇妆。"},
        ],
        "passed_product_ids": ["lip"],
        "rejected_products": [],
    }
    llm_client = StaticLlmClient(payload)

    asyncio.run(
        CorrectiveAgentController(llm_client).review_slots(
            state.original_query,
            state.intent_plan,
            state.plan,
            state,
        )
    )

    system_prompt = llm_client.calls[0]["system_prompt"]
    assert "每个 slot 只按自己的 goal / product_type / query" in system_prompt
    assert "partial 是合法证据状态" in system_prompt
    assert "passed_product_ids 必须是所有 slot_coverage selected_product_ids 的去重并集" in system_prompt
    assert "唇妆 slot 可以包含唇釉/口红" in system_prompt
