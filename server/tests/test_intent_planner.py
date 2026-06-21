import asyncio
import json

import pytest

from app.domain.intent_planner import IntentPlanner
from app.services.structured_llm import StructuredLlmValidationError


class StaticLlmClient:
    def __init__(self, payload: dict | str, stream_chunks: list[str] | None = None) -> None:
        self.payload = payload
        self.stream_chunks = stream_chunks
        self.calls: list[dict] = []

    async def generate(self, system_prompt: str, user_prompt: str, response_format: dict | None = None) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response_format": response_format,
            }
        )
        if isinstance(self.payload, str):
            return self.payload
        return json.dumps(self.payload, ensure_ascii=False)

    async def generate_required(self, system_prompt: str, user_prompt: str, response_format: dict | None = None) -> str:
        return await self.generate(system_prompt, user_prompt, response_format)

    async def generate_stream_required(self, system_prompt: str, user_prompt: str, response_format: dict | None = None):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response_format": response_format,
            }
        )
        chunks = self.stream_chunks
        if chunks is None:
            content = self.payload if isinstance(self.payload, str) else json.dumps(self.payload, ensure_ascii=False)
            chunks = [content]
        for chunk in chunks:
            yield chunk

    def is_configured(self) -> bool:
        return True


def base_payload(**overrides) -> dict:
    payload = {
        "original_query": "帮我推荐防晒，200以内",
        "plan_type": "single_retrieval",
        "vector_query": "防晒 200以内",
        "keyword_query": "防晒 200以内",
        "budget_min": None,
        "budget_max": 200,
        "budget_scope": "per_item",
        "need_slots": [],
        "referenced_product_ids": [],
        "profile_lookup": {"requested": False, "query": "", "reason": ""},
        "plan_reason": "用户明确要推荐防晒。",
    }
    payload.update(overrides)
    return payload


def test_intent_planner_direct_answer_contract() -> None:
    payload = base_payload(
        original_query="你是谁？",
        plan_type="direct_answer",
        vector_query="",
        keyword_query="",
        budget_max=None,
        budget_scope="unknown",
        plan_reason="用户询问助手身份，不需要商品证据。",
    )

    plan = asyncio.run(IntentPlanner(StaticLlmClient(payload)).plan("你是谁？", {}))

    assert plan.plan_type == "direct_answer"
    assert plan.vector_query == ""
    assert plan.keyword_query == ""
    assert plan.need_slots == []


def test_intent_planner_streams_summary_then_parses_json() -> None:
    summary = "我会先理解商品需求，再筛选匹配候选。"
    tagged_output = (
        f"<summary>{summary}</summary>"
        "<json>"
        + json.dumps(base_payload(summary=summary), ensure_ascii=False)
        + "</json>"
    )

    events = asyncio.run(
        _collect_stream(
            IntentPlanner(
                StaticLlmClient({}, stream_chunks=[tagged_output[:20], tagged_output[20:45], tagged_output[45:]])
            ).stream_plan_with_summary("帮我推荐防晒，200以内", {})
        )
    )

    assert "".join(event.content for event in events if event.kind == "summary_delta") == summary
    plan = next(event.intent_plan for event in events if event.kind == "plan")
    assert plan is not None
    assert plan.summary == summary
    assert plan.plan_type == "single_retrieval"


def test_intent_planner_stream_prompt_contains_compositional_need_contract() -> None:
    summary = "我会先判断是单品需求还是组合任务。"
    tagged_output = f"<summary>{summary}</summary><json>{json.dumps(base_payload(summary=summary), ensure_ascii=False)}</json>"

    client = StaticLlmClient({}, stream_chunks=[tagged_output])

    asyncio.run(_collect_stream(IntentPlanner(client).stream_plan_with_summary("推荐适合新手的电脑", {})))

    system_prompt = client.calls[0]["system_prompt"]
    assert "新手、入门、适合春天、通勤、旅行、预算、轻薄、好看等词本身只是约束" in system_prompt
    assert "单一商品目标 + 多个约束仍是 single_retrieval" in system_prompt
    assert "组合任务/生活任务/搭配任务/装备清单应输出 multi_retrieval" in system_prompt
    assert "泛数量表达不等于多槽" in system_prompt
    assert "如果「套装」修饰的是上位品类、生活任务或搭配目标" in system_prompt
    assert "只有「装备」或场景词并不等于多槽" in system_prompt
    assert "场景词不能直接变成 required slot" in system_prompt
    assert "Planner 自己推断的补充件、配饰、拍照/露营/通勤衍生件应 optional" in system_prompt
    assert "不要把「新手化妆品套装」「春天穿搭套装」「开学装备」这类组合目标本身当成一个 slot" in system_prompt
    assert "「推荐适合新手的电脑」=> single_retrieval" in system_prompt
    assert "「根据我平时偏好，选几件日常通勤穿的衣服」=> single_retrieval" in system_prompt
    assert "「新手化妆品套装」=> multi_retrieval" in system_prompt
    assert "「户外露营装备推荐下」=> single_retrieval" in system_prompt
    assert "不要拆帐篷/相机" in system_prompt
    assert "profile_lookup.requested=true" in system_prompt


def test_intent_planner_accepts_minimal_direct_answer_payload() -> None:
    plan = asyncio.run(IntentPlanner(StaticLlmClient({"plan_type": "direct_answer"})).plan("你是谁？", {}))

    assert plan.original_query == "你是谁？"
    assert plan.plan_type == "direct_answer"
    assert plan.vector_query == ""
    assert plan.keyword_query == ""
    assert plan.budget_min is None
    assert plan.budget_max is None
    assert plan.budget_scope == "unknown"
    assert plan.need_slots == []
    assert plan.referenced_product_ids == []
    assert plan.profile_lookup.requested is False


def test_intent_planner_prompt_contains_context_inheritance_guardrails() -> None:
    client = StaticLlmClient({"plan_type": "direct_answer"})

    asyncio.run(
        IntentPlanner(client).plan(
            "你是谁",
            {
                "recent_turns": [
                    {
                        "user": "帮我买防晒和面霜，预算 500",
                        "product_ids": ["p1", "p2"],
                        "rewrite": {"need_slots": [{"slot_id": "s1", "goal": "防晒"}]},
                    }
                ]
            },
        )
    )

    system_prompt = client.calls[0]["system_prompt"]
    user_prompt = client.calls[0]["user_prompt"]
    assert "当前 query 永远优先" in system_prompt
    assert "不能机械继承上一轮商品目标、预算、need_slots 或 product_ids" in system_prompt
    assert "必须输出 plan_type=direct_answer" in system_prompt
    assert "当前问题是询问助手身份，不能继承上一轮防晒和面霜的商品计划" in user_prompt
    assert "referenced_product_ids" in user_prompt


def test_intent_planner_plan_prompt_contains_compositional_need_contract() -> None:
    client = StaticLlmClient({"plan_type": "single_retrieval"})

    asyncio.run(IntentPlanner(client).plan("推荐适合新手的电脑", {}))

    system_prompt = client.calls[0]["system_prompt"]
    assert "新手、入门、适合春天、通勤、旅行、预算、轻薄、好看等词本身只是约束" in system_prompt
    assert "单一商品目标 + 多个约束仍是 single_retrieval" in system_prompt
    assert "组合任务/生活任务/搭配任务/装备清单应输出 multi_retrieval" in system_prompt
    assert "泛数量表达不等于多槽" in system_prompt
    assert "套装边界：如果用户是在找现成售卖的单一套装 SKU" in system_prompt
    assert "如果「套装」修饰的是上位品类、生活任务或搭配目标" in system_prompt
    assert "「适合春天的穿搭套装」=> multi_retrieval" in system_prompt
    assert "「户外露营装备推荐下」=> single_retrieval" in system_prompt
    assert "配件只能 optional" in system_prompt
    assert "profile_lookup.requested=true" in system_prompt


def test_intent_planner_normalizes_direct_answer_shape_variation() -> None:
    payload = base_payload(
        original_query="你是谁？",
        plan_type="direct_answer",
        vector_query="你是谁",
        keyword_query="你是谁",
        budget_max=None,
        budget_scope="unknown",
        need_slots=[
            {"slot_id": "s1", "goal": "助手身份", "product_type": "助手", "query": "你是谁"},
        ],
        profile_lookup={"requested": "false", "query": "", "reason": ""},
        plan_reason="用户询问助手身份，不需要商品证据。",
    )

    plan = asyncio.run(IntentPlanner(StaticLlmClient(payload)).plan("你是谁", {}))

    assert plan.original_query == "你是谁"
    assert plan.plan_type == "direct_answer"
    assert plan.vector_query == ""
    assert plan.keyword_query == ""
    assert plan.need_slots == []
    assert plan.profile_lookup.requested is False


def test_intent_planner_accepts_minimal_single_retrieval_payload() -> None:
    plan = asyncio.run(IntentPlanner(StaticLlmClient({"plan_type": "single_retrieval"})).plan("推荐耳机", {}))

    assert plan.plan_type == "single_retrieval"
    assert plan.vector_query == "推荐耳机"
    assert plan.keyword_query == "推荐耳机"
    assert plan.need_slots == []


def test_intent_planner_clarify_contract() -> None:
    payload = base_payload(
        original_query="推荐点东西吧",
        plan_type="clarify",
        vector_query="",
        keyword_query="",
        budget_max=None,
        budget_scope="unknown",
        plan_reason="用户没有给出商品目标。",
    )

    plan = asyncio.run(IntentPlanner(StaticLlmClient(payload)).plan("推荐点东西吧", {}))

    assert plan.plan_type == "clarify"
    assert plan.vector_query == ""
    assert plan.keyword_query == ""


def test_intent_planner_budget_min_and_max() -> None:
    payload = base_payload(
        original_query="预算300到500，推荐耳机",
        vector_query="预算300到500 耳机",
        keyword_query="耳机 预算300到500",
        budget_min=300,
        budget_max=500,
    )

    plan = asyncio.run(IntentPlanner(StaticLlmClient(payload)).plan("预算300到500，推荐耳机", {}))

    assert plan.plan_type == "single_retrieval"
    assert plan.budget_min == 300
    assert plan.budget_max == 500


def test_intent_planner_multi_retrieval_slots_and_total_budget() -> None:
    payload = base_payload(
        original_query="总预算1000，买运动鞋和运动服装",
        plan_type="multi_retrieval",
        vector_query="运动鞋 运动服装 总预算1000",
        keyword_query="运动鞋 运动服装 总预算1000",
        budget_max=1000,
        budget_scope="total",
        need_slots=[
            {"slot_id": "s1", "need_type": "required", "goal": "运动鞋", "product_type": "运动鞋", "query": "运动鞋", "soft_constraints": [], "exclude_terms": [], "min_candidates": 1},
            {"slot_id": "s2", "need_type": "required", "goal": "运动服装", "product_type": "运动服装", "query": "运动服装", "soft_constraints": [], "exclude_terms": [], "min_candidates": 1},
        ],
    )

    plan = asyncio.run(IntentPlanner(StaticLlmClient(payload)).plan("总预算1000，买运动鞋和运动服装", {}))

    assert plan.plan_type == "multi_retrieval"
    assert plan.budget_scope == "total"
    assert [slot.goal for slot in plan.need_slots] == ["运动鞋", "运动服装"]


def test_intent_planner_context_reference_and_profile_lookup() -> None:
    payload = base_payload(
        original_query="这两个哪个更适合跑步？",
        plan_type="direct_answer",
        vector_query="",
        keyword_query="",
        budget_max=None,
        budget_scope="unknown",
        referenced_product_ids=["p1", "p2"],
        profile_lookup={"requested": True, "query": "跑步偏好", "reason": "用户询问个人适合度。"},
    )

    plan = asyncio.run(
        IntentPlanner(StaticLlmClient(payload)).plan(
            "这两个哪个更适合跑步？",
            {"recent_turns": [{"product_ids": ["p1", "p2"]}]},
        )
    )

    assert plan.referenced_product_ids == ["p1", "p2"]
    assert plan.profile_lookup.requested is True


def test_intent_planner_retries_then_raises_invalid_json() -> None:
    client = StaticLlmClient("not json")

    with pytest.raises(StructuredLlmValidationError):
        asyncio.run(IntentPlanner(client).plan("你是谁", {}))

    assert len(client.calls) == 2
    assert "输出不是可解析的 JSON object" in client.calls[1]["user_prompt"]


async def _collect_stream(generator):
    return [event async for event in generator]
