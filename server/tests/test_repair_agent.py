import asyncio
import json

from app.domain.need_slot_schemas import NeedSlot
from app.domain.repair_worker import RepairAgent
from app.schemas import IntentPlan, QueryPlan, ReflectionResult, RepairHint


class StaticLlmClient:
    def __init__(self, payload: dict | None = None, configured: bool = True) -> None:
        self.payload = payload or {}
        self.configured = configured
        self.calls: list[dict] = []

    async def generate(self, system_prompt: str, user_prompt: str, response_format: dict | None = None) -> str:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt, "response_format": response_format})
        return json.dumps(self.payload, ensure_ascii=False)

    async def generate_required(self, system_prompt: str, user_prompt: str, response_format: dict | None = None) -> str:
        return await self.generate(system_prompt, user_prompt, response_format)

    def is_configured(self) -> bool:
        return self.configured


def test_repair_agent_generates_plan_without_search_tool_dependency() -> None:
    llm = StaticLlmClient(
        {
            "targets": ["single"],
            "queries_by_slot": {"single": ["type-c 数据线 充电头 套装"]},
            "reason": "保留商品形态重新检索。",
        }
    )
    agent = RepairAgent(llm)

    plan = asyncio.run(
        agent.plan_repair(
            original_query="买根type-c数据线和充电头",
            intent_plan=IntentPlan(original_query="买根type-c数据线和充电头"),
            plan=QueryPlan(),
            slots=[NeedSlot(slot_id="single", goal="type-c 数据线", query="type-c 数据线")],
            trigger="corrective_rejected_candidates",
            reflection_result=ReflectionResult(
                has_passed_products=False,
                repair_hint=RepairHint(repairable=True, target_slot_ids=["single"], failure_type="constraint_mismatch"),
            ),
            previous_candidates={"single": []},
        )
    )

    assert plan.targets == ["single"]
    assert plan.queries_by_slot == {"single": ["type-c 数据线 充电头 套装"]}
    assert plan.used_llm is True
    assert "ProductSearchTool" not in llm.calls[0]["system_prompt"]


def test_repair_agent_falls_back_to_deterministic_queries_when_llm_unconfigured() -> None:
    agent = RepairAgent(StaticLlmClient(configured=False))

    plan = asyncio.run(
        agent.plan_repair(
            original_query="推荐防晒",
            intent_plan=IntentPlan(original_query="推荐防晒", vector_query="清爽防晒"),
            plan=QueryPlan(),
            slots=[NeedSlot(slot_id="single", goal="防晒", product_type="防晒", query="清爽防晒")],
            trigger="score_filter_empty",
            reflection_result=ReflectionResult(
                has_passed_products=False,
                repair_hint=RepairHint(repairable=True, target_slot_ids=["single"], missing_terms=["户外"]),
            ),
            previous_candidates={"single": []},
        )
    )

    assert plan.used_llm is False
    assert plan.targets == ["single"]
    assert "清爽防晒" in plan.queries_by_slot["single"][0]
