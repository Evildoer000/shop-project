import asyncio
from collections import defaultdict

import pytest

from app.domain.agents import (
    AgentExecutionContext,
    AgentExecutor,
    AgentResult,
    BusinessAgentHandlers,
    BusinessAgentServices,
)
from app.domain.supervisor.agent_registry import build_foundation_agent_registry
from app.domain.supervisor.flow import SupervisorFlow
from app.domain.supervisor.policy_gate import SupervisorPolicyGate
from app.domain.supervisor.prompts import build_default_prompt_registry
from app.domain.supervisor.supervisor import SupervisorPlanCompiler
from app.domain.supervisor.task_graph import TaskGraph, TaskGraphNode
from app.domain.supervisor.validators import (
    IntentPlanContractError,
    validate_intent_plan_contract,
)
from app.harness.tool_registry import ToolRegistry
from app.schemas import (
    AgentTaskParameters,
    AgentTaskProposal,
    IntentConstraint,
    IntentItem,
    IntentPlanV3,
    IntentProductNeed,
    IntentRouteBasis,
    RecommendationPolicy,
    ReflectionResult,
    RepairHint,
)


def make_intent(
    intent_id: str,
    goal: str,
    *,
    intent_type: str = "product_recommendation",
    priority: str = "required",
    target_clarity: str = "explicit_product",
    external_need: str = "none",
    trigger_text: str = "",
    product_family: str = "商品",
    needs: list[IntentProductNeed] | None = None,
    references: list[str] | None = None,
) -> IntentItem:
    return IntentItem(
        intent_id=intent_id,
        intent_type=intent_type,
        priority=priority,
        goal=goal,
        resolved_query=goal,
        product_needs=needs or [],
        referenced_product_ids=references or [],
        route_basis=IntentRouteBasis(
            target_clarity=target_clarity,
            external_information_need=external_need,
            trigger_text=trigger_text,
            product_family=product_family,
            reason="测试路由依据",
        ),
    )


def make_task(
    task_id: str,
    capability: str,
    intent_id: str,
    *,
    depends_on: list[str] | None = None,
    optional_context_from: list[str] | None = None,
    parameters: AgentTaskParameters | None = None,
) -> AgentTaskProposal:
    return AgentTaskProposal(
        task_id=task_id,
        capability=capability,
        intent_ids=[intent_id],
        objective=f"执行 {intent_id} 的 {capability}",
        reason="用户目标需要该能力",
        depends_on=depends_on or [],
        optional_context_from=optional_context_from or [],
        parameters=parameters or AgentTaskParameters(),
    )


def single_plan(*, intent_id: str = "i1", task_id: str = "t1") -> IntentPlanV3:
    intent = make_intent(intent_id, "推荐150元以内适合油皮通勤的防晒霜", product_family="防晒霜")
    intent.constraints.append(
        IntentConstraint(
            name="budget_max",
            value=150,
            strength="hard",
            source="current_query",
        )
    )
    return IntentPlanV3(
        original_query=intent.goal,
        normalized_query=intent.goal,
        summary="单商品推荐",
        intents=[intent],
        task_proposals=[
            make_task(task_id, "single_product_recommendation", intent_id)
        ],
    )


def two_independent_recommendations() -> IntentPlanV3:
    intents = [
        make_intent("i_skin", "推荐敏感肌面霜", product_family="面霜"),
        make_intent("i_audio", "推荐通勤降噪耳机", product_family="耳机"),
    ]
    return IntentPlanV3(
        original_query="推荐敏感肌面霜；再推荐通勤降噪耳机",
        normalized_query="推荐敏感肌面霜；再推荐通勤降噪耳机",
        summary="两个独立推荐任务",
        intents=intents,
        task_proposals=[
            make_task("t_skin", "single_product_recommendation", "i_skin"),
            make_task("t_audio", "single_product_recommendation", "i_audio"),
        ],
    )


def bundle_plan() -> IntentPlanV3:
    needs = [
        IntentProductNeed(
            need_id="n_hat",
            goal="海边遮阳帽",
            product_type="遮阳帽",
        ),
        IntentProductNeed(
            need_id="n_top",
            goal="海边透气上装",
            product_type="短袖上衣",
        ),
        IntentProductNeed(
            need_id="n_shoes",
            goal="海边轻便鞋",
            product_type="凉鞋",
        ),
    ]
    intent = make_intent(
        "i_outfit",
        "下周去海边从头到脚搭配一套",
        product_family="海边穿搭",
        needs=needs,
    )
    return IntentPlanV3(
        original_query=intent.goal,
        normalized_query=intent.goal,
        summary="海边多商品搭配",
        intents=[intent],
        task_proposals=[
            make_task(
                "t_outfit",
                "multi_product_bundle",
                "i_outfit",
                parameters=AgentTaskParameters(
                    product_need_ids=[item.need_id for item in needs]
                ),
            )
        ],
    )


def three_independent_plan() -> IntentPlanV3:
    intents = [
        make_intent(
            "i_skin",
            "为敏感肌判断适合买什么商品",
            target_clarity="vague_effect_or_use",
            external_need="knowledge_bridge",
            trigger_text="敏感肌",
            product_family="",
        ),
        make_intent(
            "i_throat",
            "为喉咙不舒服判断适合买什么商品",
            target_clarity="vague_effect_or_use",
            external_need="knowledge_bridge",
            trigger_text="喉咙不舒服",
            product_family="",
        ),
        bundle_plan().intents[0],
    ]
    tasks = [
        make_task(
            "t_skin_knowledge",
            "knowledge_research",
            "i_skin",
            parameters=AgentTaskParameters(
                query="敏感肌适用商品类型",
                trigger_type="knowledge_bridge",
                trigger_text="敏感肌",
            ),
        ),
        make_task(
            "t_throat_knowledge",
            "knowledge_research",
            "i_throat",
            parameters=AgentTaskParameters(
                query="喉咙不舒服可购买商品类型",
                trigger_type="knowledge_bridge",
                trigger_text="喉咙不舒服",
            ),
        ),
        bundle_plan().task_proposals[0],
    ]
    return IntentPlanV3(
        original_query="敏感肌买什么；喉咙不舒服买什么；海边从头到脚搭一套",
        normalized_query="敏感肌买什么；喉咙不舒服买什么；海边从头到脚搭一套",
        summary="三个独立意图",
        intents=intents,
        task_proposals=tasks,
    )


def test_compiles_each_intent_to_an_independent_branch() -> None:
    plan = three_independent_plan()

    validate_intent_plan_contract(plan)
    graph = SupervisorPlanCompiler().compile(plan, turn_id="turn_three")

    assert graph.metadata["intent_order"] == ["i_skin", "i_throat", "i_outfit"]
    assert graph.require_node("proposal:t_skin_knowledge").depends_on == ["system:policy_gate"]
    assert graph.require_node("proposal:t_throat_knowledge").depends_on == ["system:policy_gate"]
    assert graph.require_node("proposal:t_outfit").depends_on == ["system:policy_gate"]
    assert graph.require_node("proposal:t_skin_knowledge").agent_id == "product_knowledge_agent"
    assert graph.require_node("proposal:t_throat_knowledge").agent_id == "product_knowledge_agent"
    assert graph.require_node("proposal:t_skin_knowledge").node_id != graph.require_node(
        "proposal:t_throat_knowledge"
    ).node_id
    answer = graph.require_node("system:answer_generation")
    assert answer.dependency_policy == "all_terminal"
    assert answer.metadata["intent_order"] == ["i_skin", "i_throat", "i_outfit"]
    assert set(graph.metadata["branch_terminal_nodes"]) == {
        "i_skin",
        "i_throat",
        "i_outfit",
    }


def test_same_capability_tasks_are_not_merged() -> None:
    plan = three_independent_plan()
    graph = SupervisorPlanCompiler().compile(plan, turn_id="turn_same_capability")

    knowledge_nodes = [
        node for node in graph.nodes if node.capability == "knowledge_research"
    ]

    assert [node.node_id for node in knowledge_nodes] == [
        "proposal:t_skin_knowledge",
        "proposal:t_throat_knowledge",
    ]
    assert [node.intent_ids for node in knowledge_nodes] == [["i_skin"], ["i_throat"]]


def test_multi_product_task_expands_to_parallel_slots_merge_verifier_and_optimizer() -> None:
    graph = SupervisorPlanCompiler().compile(bundle_plan(), turn_id="turn_bundle")

    dispatch = graph.require_node("proposal:t_outfit")
    slots = [
        graph.require_node(f"proposal:t_outfit:slot:{need_id}")
        for need_id in ("n_hat", "n_top", "n_shoes")
    ]
    merge = graph.require_node("proposal:t_outfit:merge")
    verifier = graph.require_node("system:verify:i_outfit")
    optimizer = graph.require_node("system:optimize:i_outfit")

    assert all(node.depends_on == [dispatch.node_id] for node in slots)
    assert merge.depends_on == [node.node_id for node in slots]
    assert verifier.depends_on == [merge.node_id]
    assert optimizer.depends_on == [verifier.node_id]
    assert graph.metadata["branch_terminal_nodes"]["i_outfit"] == optimizer.node_id


def test_context_only_recommendation_compiles_directly_to_verifier() -> None:
    intent = make_intent(
        "i_context",
        "从上一轮商品里选两个含芦荟的",
        target_clarity="context_product",
        product_family="面部补水商品",
        references=["p1", "p2", "p3"],
    )
    intent.recommendation_policy = RecommendationPolicy(
        candidate_source="context_only",
        reference_policy="eligible",
        requested_count=2,
        count_mode="exact",
    )
    plan = IntentPlanV3(
        original_query=intent.goal,
        normalized_query=intent.goal,
        intents=[intent],
        task_proposals=[],
    )

    compiler = SupervisorPlanCompiler()
    evaluation = compiler.evaluate(plan)
    graph = compiler.compile_evaluated(
        plan,
        evaluation=evaluation,
        turn_id="turn_context_only",
    )

    assert evaluation.approved is True
    assert evaluation.intent_statuses[0].status == "ready"
    verifier = graph.require_node("system:verify:i_context")
    assert verifier.depends_on == ["system:policy_gate"]
    assert verifier.metadata["candidate_source"] == "context_only"
    assert graph.metadata["branch_terminal_nodes"]["i_context"] == verifier.node_id


def test_invalid_route_rejects_only_its_intent_when_another_required_branch_is_valid() -> None:
    valid = make_intent("i_valid", "推荐防晒", product_family="防晒霜")
    invalid = make_intent("i_invalid", "推荐耳机", product_family="耳机")
    plan = IntentPlanV3(
        original_query="推荐防晒和耳机",
        normalized_query="推荐防晒和耳机",
        intents=[valid, invalid],
        task_proposals=[
            make_task("t_valid", "single_product_recommendation", "i_valid"),
            make_task(
                "t_invalid",
                "knowledge_research",
                "i_invalid",
                parameters=AgentTaskParameters(query="耳机"),
            ),
        ],
    )

    evaluation = SupervisorPlanCompiler().evaluate(plan)
    graph = SupervisorPlanCompiler().compile_evaluated(
        plan,
        evaluation=evaluation,
        turn_id="turn_partial_policy",
    )

    assert evaluation.approved is True
    assert evaluation.selections == {
        "t_valid": "single_product_recommendation_agent"
    }
    assert evaluation.uncovered_required_intent_ids == ["i_invalid"]
    assert graph.get_node("proposal:t_valid") is not None
    assert graph.get_node("proposal:t_invalid") is None
    assert graph.metadata["intent_statuses"]["i_invalid"]["status"] == "unsupported"


def test_hard_dependency_rejection_does_not_remove_independent_sibling() -> None:
    intents = [
        make_intent("i_valid", "推荐防晒", product_family="防晒霜"),
        make_intent("i_bad", "解释不存在的研究路由", product_family="耳机"),
        make_intent(
            "i_compare",
            "对比研究结果",
            intent_type="product_comparison",
            target_clarity="context_product",
            references=["p1", "p2"],
        ),
    ]
    plan = IntentPlanV3(
        original_query="混合请求",
        normalized_query="混合请求",
        intents=intents,
        task_proposals=[
            make_task("t_valid", "single_product_recommendation", "i_valid"),
            make_task(
                "t_bad",
                "knowledge_research",
                "i_bad",
                parameters=AgentTaskParameters(query="耳机"),
            ),
            make_task(
                "t_descendant",
                "comparison",
                "i_compare",
                depends_on=["t_bad"],
                parameters=AgentTaskParameters(referenced_product_ids=["p1", "p2"]),
            ),
        ],
    )

    evaluation = SupervisorPlanCompiler().evaluate(plan)

    assert evaluation.approved is True
    assert set(evaluation.selections) == {"t_valid"}
    decisions = {item.subject_id: item for item in evaluation.decisions}
    assert decisions["t_bad"].decision == "reject_route_policy"
    assert decisions["t_descendant"].decision == "reject_hard_dependency"


def test_contract_rejects_cross_intent_task_coalescing() -> None:
    plan = two_independent_recommendations()
    plan.task_proposals[0].intent_ids = ["i_skin", "i_audio"]
    plan.task_proposals = [plan.task_proposals[0]]

    with pytest.raises(IntentPlanContractError, match="exactly one intent"):
        validate_intent_plan_contract(plan)


def test_registry_and_prompts_keep_agent_tools_inside_the_expert_boundary() -> None:
    registry = build_foundation_agent_registry()
    prompts = build_default_prompt_registry()

    assert registry.select_for_capability("single_product_recommendation").manifest.allowed_tools == (
        "product_search",
        "image_search",
        "image_understanding",
    )
    assert registry.select_for_capability("multi_product_bundle").manifest.allowed_tools == (
        "image_understanding",
    )
    assert registry.select_for_capability("slot_product_retrieval").manifest.allowed_tools == (
        "product_search",
    )
    assert registry.select_for_capability("comparison").manifest.allowed_tools == (
        "product_detail",
    )
    assert registry.select_for_capability("knowledge_research").manifest.agent_id == (
        "product_knowledge_agent"
    )
    assert registry.select_for_capability("knowledge_research").manifest.allowed_tools == (
        "product_detail",
        "web_search",
    )
    assert registry.select_for_capability("evidence_verification").manifest.allowed_tools == (
        "product_detail",
    )
    assert "不得调用网页搜索或平台搜索" in prompts.require(
        "comparison_agent"
    ).system_template
    assert "product_detail" in prompts.require(
        "product_knowledge_agent"
    ).system_template


def test_executor_starts_independent_ready_tasks_concurrently() -> None:
    graph = TaskGraph(
        graph_id="parallel",
        turn_id="parallel",
        nodes=[
            TaskGraphNode(
                node_id="n1",
                task_id="parallel:n1",
                agent_id="product_knowledge_agent",
                capability="knowledge_research",
                intent_ids=["i1"],
            ),
            TaskGraphNode(
                node_id="n2",
                task_id="parallel:n2",
                agent_id="product_knowledge_agent",
                capability="knowledge_research",
                intent_ids=["i2"],
            ),
        ],
        entry_node_ids=["n1", "n2"],
        terminal_node_ids=["n1", "n2"],
    )
    both_started = asyncio.Event()
    started: set[str] = set()

    async def handler(context: AgentExecutionContext) -> AgentResult:
        started.add(context.node.node_id)
        if len(started) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=0.5)
        return AgentResult.success({"node_id": context.node.node_id})

    executor = AgentExecutor(
        agent_registry=build_foundation_agent_registry(),
        tool_registry=ToolRegistry(),
        handlers={"knowledge_research": handler},
    )
    context = AgentExecutionContext(
        node=graph.nodes[0],
        graph=graph,
        query="两个独立知识任务",
        user_id="u1",
        session_id="s1",
        turn_id="parallel",
        intent_plan=IntentPlanV3(),
    )

    report = asyncio.run(executor.execute(graph, base_context=context))

    assert started == {"n1", "n2"}
    assert report.completed is True
    assert all(node.status == "succeeded" for node in graph.nodes)


class OrderedAnswerGenerator:
    async def stream_direct_text(self, query: str, *args, **kwargs):
        yield f"<{query}>"


def test_answer_generator_emits_intents_in_original_order() -> None:
    intents = [
        make_intent(
            "i1",
            "第一个问题",
            intent_type="social_chat",
            target_clarity="not_applicable",
            product_family="",
        ),
        make_intent(
            "i2",
            "第二个问题",
            intent_type="social_chat",
            target_clarity="not_applicable",
            product_family="",
        ),
        make_intent(
            "i3",
            "第三个问题",
            intent_type="social_chat",
            target_clarity="not_applicable",
            product_family="",
        ),
    ]
    plan = IntentPlanV3(
        original_query="三个问题",
        normalized_query="三个问题",
        intents=intents,
        task_proposals=[],
    )
    graph = SupervisorPlanCompiler().compile(plan, turn_id="answer_order")
    answer = graph.require_node("system:answer_generation")
    context = AgentExecutionContext(
        node=answer,
        graph=graph,
        query=plan.original_query,
        user_id="u1",
        session_id="s1",
        turn_id="answer_order",
        intent_plan=plan,
        artifacts={"intent_plan": plan},
        services=BusinessAgentServices(answer_generator=OrderedAnswerGenerator()),
    )

    result = asyncio.run(
        BusinessAgentHandlers(context.services).answer_generation(context)
    )

    text = result.artifacts["answer_text"]
    assert text.index("### 1. 第一个问题") < text.index("<第一个问题>")
    assert text.index("<第一个问题>") < text.index("### 2. 第二个问题")
    assert text.index("<第二个问题>") < text.index("### 3. 第三个问题")
    assert text.endswith("<第三个问题>")


def test_quality_repair_retries_only_the_failed_intent_branch() -> None:
    plan = two_independent_recommendations()
    flow = SupervisorFlow(
        services=BusinessAgentServices(),
        tool_registry=ToolRegistry(),
        span_recorder=None,
        budget_manager=None,
    )
    retrieval_calls: dict[str, int] = defaultdict(int)
    verifier_calls: dict[str, int] = defaultdict(int)

    async def retrieval(context: AgentExecutionContext) -> AgentResult:
        intent_id = context.node.intent_ids[0]
        retrieval_calls[intent_id] += 1
        product_id = f"p_{intent_id}_{retrieval_calls[intent_id]}"
        return AgentResult.success(
            {"candidate_ids": [product_id], "candidate_count": 1},
            artifacts={"single_evidence": {"product_id": product_id}},
        )

    async def verifier(context: AgentExecutionContext) -> AgentResult:
        intent_id = context.node.intent_ids[0]
        verifier_calls[intent_id] += 1
        repairable = intent_id == "i_skin" and verifier_calls[intent_id] == 1
        reflection = ReflectionResult(
            has_passed_products=not repairable,
            passed_product_ids=[] if repairable else [f"p_{intent_id}_1"],
            fallback_plan="no_product" if repairable else "none",
            repair_hint=RepairHint(
                repairable=repairable,
                target_slot_ids=["single"] if repairable else [],
                reason="首轮候选不匹配" if repairable else "",
            ),
        )
        return AgentResult.success(
            {"repairable": repairable},
            artifacts={"reflection": reflection},
        )

    async def success(context: AgentExecutionContext) -> AgentResult:
        return AgentResult.success({"capability": context.node.capability})

    async def answer(context: AgentExecutionContext) -> AgentResult:
        return AgentResult.success(
            {"route": "multi_intent"},
            artifacts={
                "answer_text": "完成",
                "answer_tokens": ["完成"],
                "answer_cards": [],
                "answer_product_ids": [],
                "answer_route": "multi_intent",
                "answer_branches": [],
            },
        )

    flow.executor.handlers.update(
        {
            "single_product_recommendation": retrieval,
            "evidence_verification": verifier,
            "repair": success,
            "answer_generation": answer,
            "memory_distillation": success,
        }
    )

    events = asyncio.run(_collect_flow(flow, plan))

    assert retrieval_calls == {"i_skin": 2, "i_audio": 1}
    assert verifier_calls == {"i_skin": 2, "i_audio": 1}
    assert any(
        item.get("supervisor_decision") == "branch_quality_repair_scheduled"
        for item in events
    )
    history = flow.last_result.graph.metadata["repair_history"]
    assert len(history) == 1
    assert history[0]["kind"] == "evidence_quality"
    assert history[0]["intent_id"] == "i_skin"
    assert flow.last_result.report.failed_node_ids == []


class RevisionPlanner:
    def __init__(self, revised_plan: IntentPlanV3) -> None:
        self.revised_plan = revised_plan
        self.calls: list[tuple[str, dict]] = []

    async def plan(self, query: str, context: dict) -> IntentPlanV3:
        self.calls.append((query, context))
        return self.revised_plan


def test_knowledge_bridge_replans_only_its_own_intent() -> None:
    bridge = make_intent(
        "i_bridge",
        "敏感肌适合买什么",
        target_clarity="vague_effect_or_use",
        external_need="knowledge_bridge",
        trigger_text="敏感肌",
        product_family="",
    )
    regular = make_intent("i_regular", "推荐通勤耳机", product_family="耳机")
    initial = IntentPlanV3(
        original_query="敏感肌适合买什么；推荐通勤耳机",
        normalized_query="敏感肌适合买什么；推荐通勤耳机",
        intents=[bridge, regular],
        task_proposals=[
            make_task(
                "t_bridge",
                "knowledge_research",
                "i_bridge",
                parameters=AgentTaskParameters(
                    query="敏感肌适用商品类型",
                    trigger_type="knowledge_bridge",
                    trigger_text="敏感肌",
                ),
            ),
            make_task("t_regular", "single_product_recommendation", "i_regular"),
        ],
    )
    revised_intent = make_intent(
        "i_bridge",
        bridge.goal,
        target_clarity="explicit_product",
        product_family="敏感肌面霜",
    )
    revised = IntentPlanV3(
        original_query=bridge.goal,
        normalized_query="推荐敏感肌面霜",
        intents=[revised_intent],
        task_proposals=[
            make_task("t_bridge_recommend", "single_product_recommendation", "i_bridge")
        ],
    )
    planner = RevisionPlanner(revised)
    flow = SupervisorFlow(
        services=BusinessAgentServices(intent_planner=planner),
        tool_registry=ToolRegistry(),
        span_recorder=None,
        budget_manager=None,
    )
    retrieval_calls: list[str] = []

    async def knowledge(context: AgentExecutionContext) -> AgentResult:
        return AgentResult.success(
            {"available": True},
            artifacts={
                "knowledge_research": {"candidate_concepts": ["敏感肌面霜"]}
            },
        )

    async def retrieval(context: AgentExecutionContext) -> AgentResult:
        retrieval_calls.append(context.node.intent_ids[0])
        return AgentResult.success(
            {"candidate_ids": [], "candidate_count": 0},
            artifacts={"single_evidence": {}},
        )

    async def verifier(context: AgentExecutionContext) -> AgentResult:
        reflection = ReflectionResult(
            has_passed_products=False,
            fallback_plan="no_product",
        )
        return AgentResult.success({}, artifacts={"reflection": reflection})

    async def answer(context: AgentExecutionContext) -> AgentResult:
        return AgentResult.success(
            {"route": "multi_intent"},
            artifacts={
                "answer_text": "完成",
                "answer_tokens": ["完成"],
                "answer_cards": [],
                "answer_product_ids": [],
                "answer_route": "multi_intent",
                "answer_branches": [],
            },
        )

    async def success(context: AgentExecutionContext) -> AgentResult:
        return AgentResult.success({})

    flow.executor.handlers.update(
        {
            "knowledge_research": knowledge,
            "single_product_recommendation": retrieval,
            "evidence_verification": verifier,
            "answer_generation": answer,
            "memory_distillation": success,
        }
    )

    asyncio.run(_collect_flow(flow, initial))

    assert len(planner.calls) == 1
    assert planner.calls[0][0] == bridge.resolved_query
    assert planner.calls[0][1]["intent_revision"]["intent_id"] == "i_bridge"
    assert retrieval_calls.count("i_regular") == 1
    assert retrieval_calls.count("i_bridge") == 1
    revisions = flow.last_result.graph.metadata["intent_revisions"]
    assert len(revisions) == 1
    assert revisions[0]["intent_id"] == "i_bridge"
    assert flow.last_result.graph.metadata["intent_order"] == ["i_bridge", "i_regular"]


async def _collect_flow(flow: SupervisorFlow, plan: IntentPlanV3) -> list[dict]:
    return [
        event
        async for event in flow.run(
            query=plan.original_query,
            user_id="u1",
            session_id="s1",
            turn_id="turn_flow",
            intent_plan=plan,
        )
    ]
