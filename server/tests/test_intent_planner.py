import asyncio
import json

import pytest

from app.domain.intent_planner import IntentPlanner
from app.domain.supervisor.validators import (
    IntentPlanContractError,
    validate_intent_plan_contract,
)
from app.services.structured_llm import StructuredLlmValidationError


class StaticLlmClient:
    def __init__(self, payload: dict | str, stream_chunks: list[str] | None = None) -> None:
        self.payload = payload
        self.stream_chunks = stream_chunks
        self.calls: list[dict] = []

    async def generate_required(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: dict | None = None,
        **kwargs,
    ) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response_format": response_format,
                **kwargs,
            }
        )
        return (
            self.payload
            if isinstance(self.payload, str)
            else json.dumps(self.payload, ensure_ascii=False)
        )

    async def generate_stream_required(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: dict | None = None,
        **kwargs,
    ):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response_format": response_format,
                **kwargs,
            }
        )
        chunks = self.stream_chunks
        if chunks is None:
            chunks = [
                self.payload
                if isinstance(self.payload, str)
                else json.dumps(self.payload, ensure_ascii=False)
            ]
        for chunk in chunks:
            yield chunk

    def is_configured(self) -> bool:
        return True


def route_basis(
    *,
    target_clarity: str = "explicit_product",
    external_information_need: str = "none",
    trigger_text: str = "",
    product_family: str = "防晒霜",
) -> dict:
    return {
        "target_clarity": target_clarity,
        "external_information_need": external_information_need,
        "trigger_text": trigger_text,
        "product_family": product_family,
        "reason": "测试路由依据",
    }


def intent_payload(
    intent_id: str,
    goal: str,
    *,
    intent_type: str = "product_recommendation",
    priority: str = "required",
    route: dict | None = None,
    constraints: list[dict] | None = None,
    product_needs: list[dict] | None = None,
    referenced_product_ids: list[str] | None = None,
) -> dict:
    return {
        "intent_id": intent_id,
        "intent_type": intent_type,
        "priority": priority,
        "goal": goal,
        "resolved_query": goal,
        "constraints": constraints or [],
        "context_reference_keys": [
            f"product:{product_id}" for product_id in (referenced_product_ids or [])
        ],
        "product_needs": product_needs or [],
        "uncertainties": [],
        "route_basis": route or route_basis(),
    }


def task_payload(
    task_id: str,
    capability: str,
    intent_id: str,
    *,
    depends_on: list[str] | None = None,
    optional_upstream_task_ids: list[str] | None = None,
    parameters: dict | None = None,
) -> dict:
    resolved_parameters = dict(parameters or {})
    if capability == "knowledge_research":
        resolved_parameters.setdefault(
            "knowledge_mode",
            "concept_bridge"
            if resolved_parameters.get("trigger_type") == "knowledge_bridge"
            else "knowledge_answer",
        )
    return {
        "task_id": task_id,
        "capability": capability,
        "intent_id": intent_id,
        "objective": f"执行 {intent_id} 的 {capability}",
        "reason": "用户目标需要该专家能力",
        "depends_on": depends_on or [],
        "optional_upstream_task_ids": optional_upstream_task_ids or [],
        "parameters": resolved_parameters,
    }


def plan_payload(intents: list[dict], tasks: list[dict], *, summary: str = "已拆分任务") -> dict:
    return {
        "schema_version": "3.0",
        "summary": summary,
        "intents": intents,
        "task_proposals": tasks,
    }


def trusted_context(*product_ids: str) -> dict:
    return {
        "trusted_product_references": {
            "groups": [
                {
                    "key": f"product:{product_id}",
                    "label": product_id,
                    "product_ids": [product_id],
                    "products": [{"product_id": product_id}],
                }
                for product_id in product_ids
            ],
            "trusted_product_ids": list(product_ids),
            "suggested_keys": [],
        }
    }


def single_product_payload() -> dict:
    intent = intent_payload(
        "i1",
        "推荐150元以内适合油皮通勤的防晒霜",
        constraints=[
            {
                "name": "budget_max",
                "value": 150,
                "strength": "hard",
                "source": "current_query",
                "reason": "用户明确预算",
            },
            {
                "name": "skin_type",
                "value": "油皮",
                "strength": "hard",
                "source": "current_query",
                "reason": "用户明确肤质",
            },
        ],
        product_needs=[
            {
                "need_id": "n1",
                "priority": "required",
                "goal": "清爽通勤防晒霜",
                "product_type": "防晒霜",
                "constraints": [],
                "exclusions": ["厚重"],
            }
        ],
    )
    task = task_payload(
        "t1",
        "single_product_recommendation",
        "i1",
        parameters={"product_need_ids": ["n1"]},
    )
    return plan_payload([intent], [task], summary="检索一款油皮通勤防晒")


def test_parses_single_intent_v3_without_global_retrieval_fields() -> None:
    plan = asyncio.run(
        IntentPlanner(StaticLlmClient(single_product_payload())).plan(
            "我是油皮，预算150以内，推荐夏天通勤不闷的防晒霜",
            {},
        )
    )

    validate_intent_plan_contract(plan)
    assert plan.schema_version == "3.0"
    assert [item.intent_id for item in plan.intents] == ["i1"]
    assert plan.intents[0].constraints[0].value == 150
    assert plan.task_proposals[0].capability == "single_product_recommendation"
    assert "plan_type" not in plan.model_dump()
    assert "vector_query" not in plan.model_dump()
    assert "need_slots" not in plan.model_dump()


def test_context_only_recommendation_is_executable_without_retrieval_task() -> None:
    intent = intent_payload(
        "i_context",
        "在刚才推荐的面部补水商品里选两个含芦荟的",
        route=route_basis(target_clarity="context_product", product_family="面部补水商品"),
        referenced_product_ids=["p_face_1", "p_face_2", "p_face_3"],
    )
    intent["recommendation_policy"] = {
        "candidate_source": "context_only",
        "reference_policy": "eligible",
        "requested_count": 2,
        "count_mode": "exact",
        "reason": "用户明确限定在上一轮商品中筛选两个",
    }
    payload = plan_payload([intent], [])

    plan = asyncio.run(
        IntentPlanner(StaticLlmClient(payload)).plan(
            "就在刚才那些里选两个",
            trusted_context("p_face_1", "p_face_2", "p_face_3"),
        )
    )

    validate_intent_plan_contract(plan)
    policy = plan.intents[0].recommendation_policy
    assert policy.candidate_source == "context_only"
    assert policy.reference_policy == "eligible"
    assert policy.requested_count == 2
    assert policy.count_mode == "exact"
    assert plan.task_proposals == []


def test_streams_summary_and_preserves_three_intents_in_user_order() -> None:
    intents = [
        intent_payload(
            "i_skin",
            "为敏感肌判断适合买什么商品",
            route=route_basis(
                target_clarity="vague_effect_or_use",
                external_information_need="knowledge_bridge",
                trigger_text="敏感肌",
                product_family="",
            ),
        ),
        intent_payload(
            "i_throat",
            "为喉咙不舒服判断适合买什么商品",
            route=route_basis(
                target_clarity="vague_effect_or_use",
                external_information_need="knowledge_bridge",
                trigger_text="喉咙不舒服",
                product_family="",
            ),
        ),
        intent_payload(
            "i_outfit",
            "下周去海边从头到脚搭配一套",
            product_needs=[
                {
                    "need_id": "n_hat",
                    "priority": "required",
                    "goal": "海边遮阳帽",
                    "product_type": "遮阳帽",
                    "constraints": [],
                    "exclusions": [],
                },
                {
                    "need_id": "n_top",
                    "priority": "required",
                    "goal": "海边透气上装",
                    "product_type": "短袖上衣",
                    "constraints": [],
                    "exclusions": [],
                },
                {
                    "need_id": "n_shoes",
                    "priority": "required",
                    "goal": "海边轻便鞋",
                    "product_type": "凉鞋",
                    "constraints": [],
                    "exclusions": [],
                },
            ],
        ),
    ]
    tasks = [
        task_payload(
            "t_skin_knowledge",
            "knowledge_research",
            "i_skin",
            parameters={
                "query": "敏感肌适用商品类型",
                "trigger_type": "knowledge_bridge",
                "trigger_text": "敏感肌",
            },
        ),
        task_payload(
            "t_throat_knowledge",
            "knowledge_research",
            "i_throat",
            parameters={
                "query": "喉咙不舒服可购买商品类型",
                "trigger_type": "knowledge_bridge",
                "trigger_text": "喉咙不舒服",
            },
        ),
        task_payload(
            "t_outfit",
            "multi_product_bundle",
            "i_outfit",
            parameters={"product_need_ids": ["n_hat", "n_top", "n_shoes"]},
        ),
    ]
    summary = "拆成两个知识桥任务和一个海边穿搭任务"
    content = (
        f"<summary>{summary}</summary>"
        f"<json>{json.dumps(plan_payload(intents, tasks, summary=summary), ensure_ascii=False)}</json>"
    )
    planner = IntentPlanner(
        StaticLlmClient({}, stream_chunks=[content[:31], content[31:97], content[97:]])
    )

    events = asyncio.run(
        _collect_stream(
            planner.stream_plan_with_summary(
                "我是敏感肌帮我看看买什么；我喉咙不舒服帮我看看买什么；下周去海边帮我从头到脚搭一套",
                {},
            )
        )
    )

    assert "".join(
        item.content for item in events if item.kind == "summary_delta"
    ) == "正在识别意图、上下文引用和任务依赖。"
    plan = next(item.intent_plan for item in events if item.kind == "plan")
    assert plan is not None
    assert [item.intent_id for item in plan.intents] == ["i_skin", "i_throat", "i_outfit"]
    assert [item.task_id for item in plan.task_proposals] == [
        "t_skin_knowledge",
        "t_throat_knowledge",
        "t_outfit",
    ]
    assert [item.capability for item in plan.task_proposals[:2]] == [
        "knowledge_research",
        "knowledge_research",
    ]
    validate_intent_plan_contract(plan)


def test_mixed_request_keeps_constraints_and_dependencies_intent_scoped() -> None:
    payload = plan_payload(
        [
            intent_payload("i_recommend", "推荐油皮防晒"),
            intent_payload(
                "i_explain",
                "解释为什么清爽不闷",
                intent_type="shopping_knowledge",
                route=route_basis(
                    target_clarity="not_applicable",
                    external_information_need="explicit_web",
                    trigger_text="解释",
                    product_family="",
                ),
            ),
            intent_payload(
                "i_compare",
                "与上一款防晒比较",
                intent_type="product_comparison",
                route=route_basis(target_clarity="context_product"),
                referenced_product_ids=["p_previous"],
            ),
        ],
        [
            task_payload("t_recommend", "single_product_recommendation", "i_recommend"),
            task_payload(
                "t_explain",
                "knowledge_research",
                "i_explain",
                depends_on=["t_recommend"],
                parameters={
                    "query": "防晒清爽不闷原理",
                    "trigger_type": "explicit_web_request",
                    "trigger_text": "解释",
                },
            ),
            task_payload(
                "t_compare",
                "comparison",
                "i_compare",
                depends_on=["t_recommend"],
                parameters={},
            ),
        ],
    )

    plan = asyncio.run(
        IntentPlanner(StaticLlmClient(payload)).plan(
            "混合请求",
            trusted_context("p_previous"),
        )
    )

    validate_intent_plan_contract(plan)
    by_id = {task.task_id: task for task in plan.task_proposals}
    assert by_id["t_explain"].intent_id == "i_explain"
    assert by_id["t_explain"].depends_on == ["t_recommend"]
    assert by_id["t_compare"].intent_id == "i_compare"
    assert plan.intents[2].referenced_product_ids == ["p_previous"]


def test_prompt_defines_v3_boundaries_and_excludes_supervisor_capabilities() -> None:
    client = StaticLlmClient(single_product_payload())

    asyncio.run(IntentPlanner(client).plan("推荐防晒", {}))

    system_prompt = client.calls[0]["system_prompt"]
    user_payload = json.loads(client.calls[0]["user_prompt"])
    assert "顶层只允许 schema_version、summary、intents、task_proposals" in system_prompt
    assert "相同 capability 可以出现多次" in system_prompt
    assert "optional_upstream_task_ids" in system_prompt
    assert "optional_context_from" not in user_payload["required_output"].get(
        "$defs", {}
    ).get("AgentTaskProposal", {}).get("properties", {})
    assert "latest:item:2、turn:403、product:p_xxx" in system_prompt
    assert "不得提案 EvidenceVerifier、BundleOptimizer、Repair、AnswerGenerator" in system_prompt
    assert "检索 Agent 自己生成 RetrievalPlan" in system_prompt
    assert "不得重建或修改其它 intent" not in system_prompt
    capabilities = {
        item["capability"] for item in user_payload["available_agent_capabilities"]
    }
    assert "single_product_recommendation" in capabilities
    assert "comparison" in capabilities
    assert "repair" not in capabilities
    assert "answer_generation" not in capabilities


def test_policy_replan_feedback_requests_a_complete_v3_replacement() -> None:
    client = StaticLlmClient(single_product_payload())
    context = {
        "replan_attempt": 1,
        "previous_intent_plan": plan_payload([], []),
        "policy_evaluation": {
            "approved": False,
            "errors": ["no required intent has an executable task branch"],
        },
    }

    asyncio.run(IntentPlanner(client).plan("推荐防晒", context))

    assert "策略拒绝后的受限重规划" in client.calls[0]["system_prompt"]
    prompt_context = json.loads(client.calls[0]["user_prompt"])["context"]
    assert prompt_context["replan_attempt"] == 1
    assert prompt_context["policy_evaluation"]["approved"] is False


def test_knowledge_revision_prompt_is_limited_to_one_target_intent() -> None:
    client = StaticLlmClient(single_product_payload())
    context = {
        "intent_revision": {
            "intent_id": "i1",
            "target_intent": single_product_payload()["intents"][0],
            "knowledge_result": {"candidate_concepts": ["物理防晒霜"]},
        }
    }

    asyncio.run(IntentPlanner(client).plan("敏感肌适合买什么", context))

    prompt = client.calls[0]["system_prompt"]
    assert "本次只能输出 target_intent 对应的一个 intent" in prompt
    assert "不得重建或修改其它 intent" in prompt
    assert "已完成的 knowledge_research 不得再次提案" in prompt


def test_rejects_removed_global_route_fields() -> None:
    payload = single_product_payload()
    payload["execution_mode"] = "single_product"
    payload["plan_type"] = "single_retrieval"
    client = StaticLlmClient(payload)

    with pytest.raises(StructuredLlmValidationError) as error:
        asyncio.run(IntentPlanner(client).plan("推荐防晒", {}))

    assert "V3 top level contains removed fields" in " ".join(error.value.errors)
    assert len(client.calls) == 2


def test_contract_rejects_one_task_bound_to_multiple_intents() -> None:
    payload = plan_payload(
        [intent_payload("i1", "推荐防晒"), intent_payload("i2", "推荐耳机")],
        [task_payload("t1", "single_product_recommendation", "i1")],
    )
    payload["task_proposals"][0].pop("intent_id")
    payload["task_proposals"][0]["intent_ids"] = ["i1", "i2"]

    with pytest.raises(StructuredLlmValidationError) as error:
        asyncio.run(IntentPlanner(StaticLlmClient(payload)).plan("两个任务", {}))

    assert "scalar intent_id" in " ".join(error.value.errors)


def test_rejects_duplicate_task_ids_before_compilation() -> None:
    payload = plan_payload(
        [intent_payload("i1", "推荐防晒"), intent_payload("i2", "推荐耳机")],
        [
            task_payload("t1", "single_product_recommendation", "i1"),
            task_payload("t1", "single_product_recommendation", "i2"),
        ],
    )

    with pytest.raises(StructuredLlmValidationError) as error:
        asyncio.run(IntentPlanner(StaticLlmClient(payload)).plan("两个任务", {}))

    assert "duplicate task_id" in " ".join(error.value.errors)


def test_moves_trusted_product_reference_out_of_optional_upstream_tasks() -> None:
    intent = intent_payload(
        "i_product",
        "结合刚才第二个商品补充资料并对比",
        intent_type="product_qa",
        route=route_basis(target_clarity="context_product"),
    )
    intent["context_reference_keys"] = ["latest:item:2"]
    payload = plan_payload(
        [intent],
        [
            task_payload(
                "task_knowledge_001",
                "knowledge_research",
                "i_product",
                parameters={"query": "补充商品资料", "knowledge_mode": "product_evidence"},
            ),
            task_payload(
                "task_comparison_001",
                "comparison",
                "i_product",
                optional_upstream_task_ids=[
                    "task_knowledge_001",
                    "latest:item:2",
                ],
            ),
        ],
    )
    context = trusted_context("p_context_1", "p_context_2")
    context["trusted_product_references"]["groups"].append(
        {
            "key": "latest:item:2",
            "label": "最近一次展示商品中的第 2 个",
            "product_ids": ["p_context_2"],
            "products": [{"product_id": "p_context_2"}],
        }
    )
    planner = IntentPlanner(StaticLlmClient(payload))

    plan = asyncio.run(planner.plan("继续比较刚才第二个", context))

    assert plan.intents[0].context_reference_keys == ["latest:item:2"]
    assert plan.intents[0].referenced_product_ids == ["p_context_2"]
    assert plan.task_proposals[1].optional_upstream_task_ids == [
        "task_knowledge_001"
    ]
    assert planner.last_validation_attempts[0]["normalizations"][0][
        "moved_reference_keys"
    ] == ["latest:item:2"]


def test_does_not_swallow_unknown_reference_like_upstream_value() -> None:
    payload = plan_payload(
        [intent_payload("i1", "介绍刚才的商品", intent_type="product_qa")],
        [
            task_payload(
                "task_knowledge_001",
                "knowledge_research",
                "i1",
                optional_upstream_task_ids=["latest:item:99"],
                parameters={"query": "介绍商品", "knowledge_mode": "product_evidence"},
            )
        ],
    )

    with pytest.raises(StructuredLlmValidationError) as error:
        asyncio.run(
            IntentPlanner(StaticLlmClient(payload)).plan(
                "介绍刚才第九十九个商品",
                trusted_context("p_context_1"),
            )
        )

    joined = " ".join(error.value.errors)
    assert "not trusted context keys" in joined
    assert "unknown dependencies" in joined


def test_rejects_optional_self_dependency_before_graph_compilation() -> None:
    payload = plan_payload(
        [intent_payload("i1", "推荐防晒")],
        [
            task_payload(
                "task_001",
                "single_product_recommendation",
                "i1",
                optional_upstream_task_ids=["task_001"],
            )
        ],
    )

    with pytest.raises(StructuredLlmValidationError) as error:
        asyncio.run(IntentPlanner(StaticLlmClient(payload)).plan("推荐防晒", {}))

    assert "task task_001 cannot depend on itself" in " ".join(error.value.errors)


def test_rejects_cycle_combining_hard_and_optional_upstream_edges() -> None:
    payload = plan_payload(
        [
            intent_payload("i1", "推荐防晒"),
            intent_payload("i2", "解释防晒原理", intent_type="shopping_knowledge"),
        ],
        [
            task_payload(
                "task_recommend_001",
                "single_product_recommendation",
                "i1",
                optional_upstream_task_ids=["task_knowledge_001"],
            ),
            task_payload(
                "task_knowledge_001",
                "knowledge_research",
                "i2",
                depends_on=["task_recommend_001"],
                parameters={"query": "防晒原理", "knowledge_mode": "knowledge_answer"},
            ),
        ],
    )

    with pytest.raises(StructuredLlmValidationError) as error:
        asyncio.run(IntentPlanner(StaticLlmClient(payload)).plan("推荐并解释防晒", {}))

    assert "task dependency cycle detected" in " ".join(error.value.errors)


def test_accepts_valid_hard_and_optional_upstream_edges() -> None:
    intent = intent_payload("i1", "补充资料后对比商品", intent_type="product_comparison")
    payload = plan_payload(
        [intent],
        [
            task_payload(
                "task_local_001",
                "knowledge_research",
                "i1",
                parameters={"query": "商品参数", "knowledge_mode": "product_evidence"},
            ),
            task_payload(
                "task_extra_001",
                "knowledge_research",
                "i1",
                parameters={"query": "选购知识", "knowledge_mode": "knowledge_answer"},
            ),
            task_payload(
                "task_compare_001",
                "comparison",
                "i1",
                depends_on=["task_local_001"],
                optional_upstream_task_ids=["task_extra_001"],
            ),
        ],
    )

    plan = asyncio.run(IntentPlanner(StaticLlmClient(payload)).plan("补充资料后对比", {}))

    validate_intent_plan_contract(plan)
    assert plan.task_proposals[-1].depends_on == ["task_local_001"]
    assert plan.task_proposals[-1].optional_upstream_task_ids == ["task_extra_001"]


async def _collect_stream(generator):
    return [event async for event in generator]
