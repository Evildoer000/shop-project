import pytest

from app.domain.supervisor.agent_registry import AgentRegistry, build_foundation_agent_registry
from app.domain.supervisor.capability_catalog import build_default_capability_catalog
from app.domain.supervisor.policy_gate import SupervisorPolicyGate
from app.domain.supervisor.prompts import build_default_prompt_registry
from app.domain.supervisor.supervisor import SupervisorPlanCompiler
from app.domain.supervisor.validators import IntentPlanContractError, validate_intent_plan_contract
from app.schemas import (
    AgentTaskProposal,
    ClarificationProposal,
    ContextRequest,
    IntentConstraint,
    IntentConstraintSet,
    IntentItem,
    IntentPlan,
    IntentQueryRewrite,
    ResearchRequest,
    RewriteNeedSlot,
)
from app.domain.agents.contracts import EvidenceRef


def test_compound_intent_compiles_to_parallel_and_dependent_agent_graph() -> None:
    plan = _compound_plan()

    validate_intent_plan_contract(plan)
    graph = SupervisorPlanCompiler().compile(plan, turn_id="turn_compound")

    assert graph.topological_order()[0] == "system:intent_understanding"
    profile = graph.require_node("proposal:p_profile")
    recommendation = graph.require_node("proposal:p_single")
    comparison = graph.require_node("proposal:p_compare")
    knowledge = graph.require_node("proposal:p_knowledge")
    verifier = graph.require_node("system:evidence_verification")
    answer = graph.require_node("system:answer_generation")
    memory = graph.require_node("system:memory_distillation")

    assert profile.depends_on == ["system:intent_understanding"]
    assert recommendation.depends_on == ["system:intent_understanding"]
    assert comparison.depends_on == ["proposal:p_single"]
    assert knowledge.depends_on == ["proposal:p_single"]
    assert set(verifier.depends_on) == {
        "proposal:p_profile",
        "proposal:p_single",
        "proposal:p_compare",
        "proposal:p_knowledge",
    }
    assert answer.depends_on == ["system:evidence_verification", "proposal:p_profile"]
    assert memory.depends_on == ["system:answer_generation"]
    assert memory.asynchronous is True
    assert "system:repair" not in {node.node_id for node in graph.nodes}


def test_clarification_is_a_terminal_business_branch_without_answer_generator() -> None:
    plan = IntentPlan(
        original_query="推荐点东西",
        normalized_query="推荐点东西",
        primary_intent="product_recommendation",
        intents=[IntentItem(intent_id="i1", intent_type="product_recommendation", goal="获得商品推荐")],
        execution_mode="clarify",
        clarification=ClarificationProposal(
            required=True,
            blocking=True,
            missing_fields=["商品目标"],
            question_goal="确认用户想购买的商品类型",
            reason="没有可执行商品目标",
        ),
        agent_proposals=[
            AgentTaskProposal(
                proposal_id="p_clarify",
                capability="clarification",
                intent_ids=["i1"],
                reason="需要补齐商品目标",
            )
        ],
        plan_type="clarify",
    )

    graph = SupervisorPlanCompiler().compile(plan, turn_id="turn_clarify")

    node_ids = {node.node_id for node in graph.nodes}
    assert "proposal:p_clarify" in node_ids
    assert "system:answer_generation" not in node_ids
    assert graph.require_node("system:memory_distillation").depends_on == ["proposal:p_clarify"]


def test_planner_cannot_propose_supervisor_managed_capabilities() -> None:
    plan = _single_product_plan()
    plan.agent_proposals.append(
        AgentTaskProposal(
            proposal_id="p_repair",
            capability="repair",
            intent_ids=["i1"],
            reason="Planner tries to force repair",
        )
    )

    with pytest.raises(IntentPlanContractError, match="Supervisor-managed capability repair"):
        validate_intent_plan_contract(plan)


def test_long_term_profile_cannot_create_hard_constraints() -> None:
    plan = _single_product_plan()
    plan.constraints.items.append(
        IntentConstraint(
            name="brand",
            value="某品牌",
            strength="hard",
            source="long_term_profile",
        )
    )

    with pytest.raises(IntentPlanContractError, match="long_term_profile constraints must be soft"):
        validate_intent_plan_contract(plan)


def test_policy_rejects_required_proposal_without_enabled_agent() -> None:
    catalog = build_default_capability_catalog()
    registry = AgentRegistry(catalog)

    evaluation = SupervisorPolicyGate(registry, catalog).evaluate(_single_product_plan())

    assert evaluation.approved is False
    assert evaluation.decisions[0].approved is False
    assert "No enabled Agent" in evaluation.decisions[0].reason


def test_runtime_repair_only_rewires_failed_node_dependents() -> None:
    compiler = SupervisorPlanCompiler()
    graph = compiler.compile(_single_product_plan(), turn_id="turn_repair")
    retrieval = graph.require_node("proposal:p_single")
    retrieval.status = "failed"

    retry_node_ids = compiler.schedule_repair(
        graph,
        [retrieval.node_id],
        reason="candidate evidence did not satisfy the product goal",
    )

    assert retry_node_ids == ["runtime:retry:proposal:p_single:2"]
    repair = graph.require_node("runtime:repair:1")
    retry = graph.require_node(retry_node_ids[0])
    verifier = graph.require_node("system:evidence_verification")
    assert repair.depends_on == ["proposal:p_single"]
    assert repair.dependency_policy == "all_terminal"
    assert repair in graph.ready_nodes()
    assert retry.depends_on == ["runtime:repair:1"]
    assert retry.attempt == 2
    assert verifier.depends_on == [retry.node_id]


def test_task_graph_preserves_retry_dependencies_and_separates_repair_cycle_from_attempt() -> None:
    compiler = SupervisorPlanCompiler()
    graph = compiler.compile(_compound_plan(), turn_id="turn_repair_dependencies")
    retrieval = graph.require_node("proposal:p_single")
    retrieval.status = "failed"
    comparison = graph.require_node("proposal:p_compare")
    comparison.status = "failed"

    retry_ids = compiler.schedule_repair(
        graph,
        [retrieval.node_id, comparison.node_id],
        reason="both evidence-producing branches failed",
    )

    repair = graph.require_node("runtime:repair:1")
    retry_retrieval = graph.require_node("runtime:retry:proposal:p_single:2")
    retry_comparison = graph.require_node("runtime:retry:proposal:p_compare:2")
    assert retry_ids == [retry_retrieval.node_id, retry_comparison.node_id]
    assert repair.attempt == 1
    assert repair.metadata["repair_cycle"] == 1
    assert "runtime:retry:proposal:p_single:2" in retry_comparison.depends_on
    assert "runtime:repair:1" in retry_retrieval.depends_on


def test_task_graph_marks_failed_dependency_as_blocked() -> None:
    compiler = SupervisorPlanCompiler()
    graph = compiler.compile(_single_product_plan(), turn_id="turn_blocked")
    retrieval = graph.require_node("proposal:p_single")
    retrieval.status = "failed"

    assert [node.node_id for node in graph.blocked_nodes()] == ["system:evidence_verification"]
    assert graph.mark_blocked_nodes() == ["system:evidence_verification"]
    assert graph.require_node("system:evidence_verification").status == "skipped"


def test_agent_registry_separates_business_agents_from_atomic_tools() -> None:
    registry = build_foundation_agent_registry()

    single = registry.select_for_capability("single_product_recommendation", execution_mode="single_product")
    assert single is not None
    assert single.manifest.agent_id == "single_product_recommendation_agent"
    assert set(single.manifest.allowed_tools) == {
        "product_search",
        "image_search",
        "image_understanding",
        "product_detail",
    }
    assert registry.select_for_capability("single_product_recommendation", execution_mode="multi_product") is None


def test_prompt_registry_builds_all_foundation_prompts() -> None:
    prompts = build_default_prompt_registry()

    assert prompts.require("evidence_verifier_agent").output_contract["type"] == "EvidenceVerificationResult"
    assert prompts.require("slot_retrieval_agent").output_contract["type"] == "SlotRetrievalEvidence"
    assert prompts.require("bundle_optimizer").output_contract["type"] == "BundleOptimizationResult"
    assert len(prompts.describe()) == 14


def test_multi_product_compiles_parallel_slot_retrieval_and_merge_boundary() -> None:
    graph = SupervisorPlanCompiler().compile(_multi_product_plan(), turn_id="turn_bundle")

    dispatch = graph.require_node("proposal:p_multi")
    slot_a = graph.require_node("proposal:p_multi:slot:s1")
    slot_b = graph.require_node("proposal:p_multi:slot:s2")
    merge = graph.require_node("proposal:p_multi:merge")
    verifier = graph.require_node("system:evidence_verification")

    assert slot_a.agent_id == "slot_retrieval_agent"
    assert slot_b.agent_id == "slot_retrieval_agent"
    assert slot_a.depends_on == [dispatch.node_id]
    assert slot_b.depends_on == [dispatch.node_id]
    assert set(merge.depends_on) == {slot_a.node_id, slot_b.node_id}
    assert verifier.depends_on == [merge.node_id]
    assert merge.metadata["phase"] == "merge_slot_evidence"


def test_required_commerce_research_is_covered_by_explicit_tool_allowlist() -> None:
    plan = _single_product_plan()
    plan.research_requests = [
        ResearchRequest(
            request_id="r_xhs",
            intent_id="i1",
            mode="social_content",
            platforms=["xiaohongshu"],
            query="油皮防晒真实使用评价",
            reason="用户明确要求查看小红书评价",
            required=True,
        )
    ]
    plan.agent_proposals.append(
        AgentTaskProposal(
            proposal_id="p_commerce",
            capability="commerce_research",
            intent_ids=["i1"],
            reason="查询获批平台的外部评价",
        )
    )

    evaluation = SupervisorPolicyGate(build_foundation_agent_registry()).evaluate(plan)

    assert evaluation.approved is True
    research_decision = next(item for item in evaluation.decisions if item.subject_id == "r_xhs")
    assert research_decision.selected_agent_id == "commerce_research_agent"
    assert research_decision.details["required_tool"] == "commerce_search"


def test_commerce_research_rejects_platforms_outside_project_allowlist() -> None:
    plan = _single_product_plan()
    plan.research_requests = [
        ResearchRequest(
            request_id="r_other",
            intent_id="i1",
            mode="marketplace",
            platforms=["jd"],
            query="防晒",
            reason="测试不受支持的平台",
        )
    ]

    with pytest.raises(IntentPlanContractError, match="unsupported commerce platforms"):
        validate_intent_plan_contract(plan)


def test_knowledge_follow_up_is_supervisor_approved_from_item_level_evidence() -> None:
    plan = _knowledge_bridge_plan()
    gate = SupervisorPolicyGate(build_foundation_agent_registry())
    evidence = EvidenceRef(
        evidence_id="web:0",
        source_type="web_search",
        source_id="https://example.test/aloe",
        claim="芦荟胶选购资料",
        summary="资料明确提到芦荟胶这一商品概念。",
    )

    decision = gate.approve_knowledge_follow_up(
        plan,
        knowledge_artifact={
            "candidate_concepts": ["芦荟胶", "消炎"],
            "concept_proposals": [
                {
                    "concept": "芦荟胶",
                    "supporting_evidence_ids": ["web:0"],
                },
                {
                    "concept": "消炎",
                    "supporting_evidence_ids": [],
                },
            ],
            "risks": ["不能将知识资料表述为医疗结论"],
        },
        evidence_refs=[evidence],
    )

    assert decision.approved is True
    assert decision.decision == "approve_knowledge_to_retrieval"
    assert decision.approved_concepts == ["芦荟胶"]
    assert decision.supporting_evidence_ids == ["web:0"]
    assert "芦荟胶" in decision.approved_query
    assert decision.rejected_concepts[0]["concept"] == "消炎"
    assert decision.risk_flags


def test_knowledge_follow_up_rejects_untraceable_concepts() -> None:
    plan = _knowledge_bridge_plan()
    gate = SupervisorPolicyGate(build_foundation_agent_registry())

    decision = gate.approve_knowledge_follow_up(
        plan,
        knowledge_artifact={
            "candidate_concepts": ["芦荟胶"],
            "concept_proposals": [
                {"concept": "芦荟胶", "supporting_evidence_ids": ["web:missing"]}
            ],
        },
        evidence_refs=[],
    )

    assert decision.approved is False
    assert decision.decision == "reject_knowledge_to_retrieval"
    assert decision.approved_query == ""
    assert "不存在" in decision.rejected_concepts[0]["reason"]


def test_knowledge_only_route_never_turns_into_product_retrieval() -> None:
    plan = _knowledge_bridge_plan(knowledge_only=True)
    gate = SupervisorPolicyGate(build_foundation_agent_registry())

    decision = gate.approve_knowledge_follow_up(
        plan,
        knowledge_artifact={
            "candidate_concepts": ["防晒霜"],
            "concept_proposals": [
                {"concept": "防晒霜", "supporting_evidence_ids": ["web:0"]}
            ],
        },
        evidence_refs=[
            EvidenceRef(
                evidence_id="web:0",
                source_type="web_search",
                source_id="https://example.test/sunscreen",
            )
        ],
    )

    assert decision.approved is False
    assert decision.decision == "not_applicable"


def _knowledge_bridge_plan(*, knowledge_only: bool = False) -> IntentPlan:
    intent_type = "shopping_knowledge" if knowledge_only else "product_recommendation"
    return IntentPlan(
        original_query="想要消炎的东西",
        normalized_query="根据资料寻找适合的商品概念",
        primary_intent=intent_type,
        intents=[IntentItem(intent_id="i1", intent_type=intent_type, goal="解释并判断可购买的商品概念")],
        execution_mode="context_evidence",
        research_requests=[
            ResearchRequest(
                request_id="r_web",
                intent_id="i1",
                mode="web_general",
                query="消炎相关的可购买商品概念",
                reason="用户目标模糊，需要外部知识桥接",
            )
        ],
        agent_proposals=[
            AgentTaskProposal(
                proposal_id="p_knowledge",
                capability="knowledge_research",
                intent_ids=["i1"],
                reason="先研究商品概念",
            )
        ],
        plan_type="single_retrieval",
    )


def _single_product_plan() -> IntentPlan:
    return IntentPlan(
        original_query="150以内适合油皮通勤的防晒",
        normalized_query="推荐150元以内适合油皮通勤的防晒霜",
        primary_intent="product_recommendation",
        intents=[
            IntentItem(
                intent_id="i1",
                intent_type="product_recommendation",
                goal="推荐适合油皮通勤的防晒霜",
                query_rewrite=IntentQueryRewrite(
                    semantic_query="适合油皮夏天通勤且清爽的防晒霜",
                    keyword_query="防晒霜 油皮 清爽 通勤 150元以内",
                ),
            )
        ],
        execution_mode="single_product",
        constraints=IntentConstraintSet(budget_max=150, budget_scope="per_item"),
        agent_proposals=[
            AgentTaskProposal(
                proposal_id="p_single",
                capability="single_product_recommendation",
                intent_ids=["i1"],
                reason="只有一个商品目标",
            )
        ],
        plan_type="single_retrieval",
        vector_query="适合油皮夏天通勤且清爽的防晒霜",
        keyword_query="防晒霜 油皮 清爽 通勤 150元以内",
        budget_max=150,
        budget_scope="per_item",
    )


def _compound_plan() -> IntentPlan:
    return IntentPlan(
        original_query="按我平时偏好推荐一款150以内的油皮通勤防晒，再和刚才那款对比，解释为什么不闷并看看小红书评价",
        normalized_query="按长期偏好推荐150元以内油皮通勤防晒，与上一款对比并解释清爽依据，补充小红书评价",
        primary_intent="product_recommendation",
        intents=[
            IntentItem(
                intent_id="i1",
                intent_type="product_recommendation",
                goal="推荐油皮通勤防晒",
                query_rewrite=IntentQueryRewrite(
                    semantic_query="适合油皮夏天通勤且清爽不闷的防晒霜",
                    keyword_query="防晒霜 油皮 清爽 通勤 150元以内",
                ),
            ),
            IntentItem(
                intent_id="i2",
                intent_type="product_comparison",
                goal="将新推荐与上一轮防晒进行对比",
                depends_on=["i1"],
                referenced_product_ids=["p_recent_001"],
            ),
            IntentItem(
                intent_id="i3",
                intent_type="shopping_knowledge",
                goal="解释防晒清爽不闷的证据依据",
                depends_on=["i1"],
            ),
        ],
        execution_mode="single_product",
        constraints=IntentConstraintSet(
            budget_max=150,
            budget_scope="per_item",
            items=[
                IntentConstraint(name="skin_type", value="油皮", strength="hard"),
                IntentConstraint(name="texture", value="清爽不闷", strength="soft"),
            ],
        ),
        context_requests=[
            ContextRequest(
                request_id="ctx_profile",
                usage="ranking_only",
                query="防晒和肤感偏好",
                reason="用户明确要求按照平时偏好推荐",
            )
        ],
        research_requests=[
            ResearchRequest(
                request_id="r_xhs",
                intent_id="i2",
                mode="social_content",
                platforms=["xiaohongshu"],
                query="两款防晒的真实使用评价",
                reason="用户明确要求查看小红书评价",
            )
        ],
        agent_proposals=[
            AgentTaskProposal(
                proposal_id="p_profile",
                capability="profile_preference",
                intent_ids=["i1"],
                reason="需要个性化排序",
            ),
            AgentTaskProposal(
                proposal_id="p_single",
                capability="single_product_recommendation",
                intent_ids=["i1"],
                reason="只有一个新商品目标",
            ),
            AgentTaskProposal(
                proposal_id="p_compare",
                capability="comparison",
                intent_ids=["i2"],
                reason="用户要求对比",
                depends_on=["p_single"],
            ),
            AgentTaskProposal(
                proposal_id="p_knowledge",
                capability="knowledge_research",
                intent_ids=["i3"],
                reason="用户要求解释肤感原理",
                depends_on=["p_single"],
            ),
        ],
        need_slots=[
            RewriteNeedSlot(
                slot_id="s1",
                intent_id="i1",
                goal="油皮通勤防晒",
                product_type="防晒霜",
                query="油皮通勤防晒霜",
                semantic_query="适合油皮夏天通勤且清爽不闷的防晒霜",
                keyword_query="防晒霜 油皮 清爽 通勤 150元以内",
            )
        ],
        referenced_product_ids=["p_recent_001"],
        plan_type="single_retrieval",
    )


def _multi_product_plan() -> IntentPlan:
    return IntentPlan(
        original_query="帮我配一套通勤穿搭",
        normalized_query="帮我配一套通勤穿搭",
        primary_intent="product_recommendation",
        intents=[
            IntentItem(
                intent_id="i1",
                intent_type="product_recommendation",
                goal="完成通勤穿搭组合",
            )
        ],
        execution_mode="multi_product",
        need_slots=[
            RewriteNeedSlot(
                slot_id="s1",
                intent_id="i1",
                goal="通勤上装",
                product_type="通勤上装",
                query="通勤上装",
            ),
            RewriteNeedSlot(
                slot_id="s2",
                intent_id="i1",
                goal="通勤下装",
                product_type="通勤下装",
                query="通勤下装",
            ),
        ],
        agent_proposals=[
            AgentTaskProposal(
                proposal_id="p_multi",
                capability="multi_product_bundle",
                intent_ids=["i1"],
                reason="用户要求配齐一套穿搭",
            )
        ],
        plan_type="multi_retrieval",
    )
