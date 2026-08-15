import asyncio

import pytest

from app.domain.agents import AgentExecutionContext, AgentResult, BusinessAgentServices, ExecutionReport
from app.domain.supervisor.agent_registry import AgentRegistry, build_foundation_agent_registry
from app.domain.supervisor.capability_catalog import build_default_capability_catalog
from app.domain.supervisor.policy_gate import SupervisorPolicyGate
from app.domain.supervisor.prompts import build_default_prompt_registry
from app.domain.supervisor.supervisor import (
    SupervisorPlanCompiler,
    SupervisorPolicyRejectedError,
)
from app.domain.supervisor.flow import SupervisorFlow
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
    IntentRouteBasis,
    ResearchRequest,
    RewriteNeedSlot,
)
from app.domain.agents.contracts import EvidenceRef
from app.harness.span_recorder import SpanRecorder
from app.harness.tool_registry import ToolRegistry


def test_compound_intent_compiles_to_parallel_and_dependent_agent_graph() -> None:
    plan = _compound_plan()

    validate_intent_plan_contract(plan)
    graph = SupervisorPlanCompiler().compile(plan, turn_id="turn_compound")

    assert graph.topological_order()[0] == "system:intent_understanding"
    policy = graph.require_node("system:policy_gate")
    profile = graph.require_node("proposal:p_profile")
    recommendation = graph.require_node("proposal:p_single")
    comparison = graph.require_node("proposal:p_compare")
    knowledge = graph.require_node("proposal:p_knowledge")
    verifier = graph.require_node("system:evidence_verification")
    answer = graph.require_node("system:answer_generation")
    memory = graph.require_node("system:memory_distillation")

    assert policy.depends_on == ["system:intent_understanding"]
    assert policy.agent_id == "supervisor_policy_gate"
    assert policy.metadata["implementation"] == "deterministic_code"
    assert profile.depends_on == ["system:policy_gate"]
    assert profile.required is False
    assert recommendation.depends_on == ["system:policy_gate"]
    assert recommendation.input_refs == ["proposal:p_profile"]
    assert comparison.depends_on == ["proposal:p_single", "system:policy_gate"]
    assert comparison.input_refs == ["proposal:p_commerce"]
    assert knowledge.depends_on == ["proposal:p_single", "system:policy_gate"]
    assert verifier.depends_on == ["proposal:p_single"]
    assert set(verifier.input_refs) == {
        "proposal:p_profile",
        "proposal:p_compare",
        "proposal:p_knowledge",
        "proposal:p_commerce",
    }
    assert answer.depends_on == ["system:evidence_verification"]
    assert answer.input_refs == ["proposal:p_profile"]
    assert memory.depends_on == ["system:answer_generation"]
    assert memory.asynchronous is True
    assert "system:repair" not in {node.node_id for node in graph.nodes}


def test_ranking_only_profile_failure_does_not_block_product_recommendation() -> None:
    graph = SupervisorPlanCompiler().compile(_compound_plan(), turn_id="turn_optional_profile")
    profile = graph.require_node("proposal:p_profile")
    recommendation = graph.require_node("proposal:p_single")

    profile.status = "skipped"

    assert profile.required is False
    assert recommendation in graph.ready_nodes()


def test_supervisor_trace_aggregates_tool_calls_from_execution_report() -> None:
    flow = SupervisorFlow(
        services=BusinessAgentServices(),
        tool_registry=ToolRegistry(),
        span_recorder=None,
        budget_manager=None,
    )
    plan = _compound_plan()
    graph = flow.compiler.compile(plan, turn_id="turn_tool_trace")
    recommendation = graph.require_node("proposal:p_single")
    report = ExecutionReport(graph=graph)
    report.results[recommendation.node_id] = AgentResult.success(
        {"candidate_count": 3},
        tool_calls=[
            {
                "tool": "product_search",
                "operation": "single_initial",
                "agent_id": recommendation.agent_id,
                "task_id": recommendation.task_id,
                "attempt": 1,
                "status": "succeeded",
            }
        ],
    )

    trace = flow._trace(graph, plan, report, "recommend", None)

    assert trace["task"]["tool_call_count"] == 1
    assert trace["trace_schema_version"] == "v2"
    assert trace["run_id"] == ""
    assert trace["tool_calls"] == [
        {
            "call_id": f"{recommendation.task_id}:tool:1",
            "node_id": recommendation.node_id,
            "capability": recommendation.capability,
            "tool": "product_search",
            "operation": "single_initial",
            "agent_id": recommendation.agent_id,
            "task_id": recommendation.task_id,
            "attempt": 1,
            "status": "succeeded",
        }
    ]


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
    assert graph.require_node("proposal:p_clarify").depends_on == ["system:policy_gate"]
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

    with pytest.raises(SupervisorPolicyRejectedError) as error:
        SupervisorPlanCompiler(registry=registry, catalog=catalog).compile(
            _single_product_plan(),
            turn_id="turn_policy_rejected",
        )
    assert error.value.errors == evaluation.errors


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
    assert retry.depends_on == ["runtime:repair:1", "system:policy_gate"]
    assert retry.attempt == 2
    assert retry.task_id == "turn_repair:runtime:retry:proposal:p_single:2"
    assert verifier.depends_on == [retry.node_id]
    old_handoff = next(
        item
        for item in graph.handoffs
        if item.from_node_id == "proposal:p_single"
        and item.to_node_id == verifier.node_id
        and item.required
    )
    retry_handoff = next(
        item
        for item in graph.handoffs
        if item.from_node_id == retry.node_id
        and item.to_node_id == verifier.node_id
        and item.required
    )
    assert old_handoff.status == "superseded"
    assert retry_handoff.status == "planned"


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
    assert retry_retrieval.task_id != retrieval.task_id
    assert retry_comparison.task_id != comparison.task_id
    assert retry_retrieval.task_id != retry_comparison.task_id
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
    policy = registry.select_for_capability("policy_gate", execution_mode="single_product")
    assert policy is not None
    assert policy.manifest.allowed_tools == ()
    assert policy.manifest.supervisor_managed is True


def test_profile_intent_refinement_requires_a_second_policy_gate() -> None:
    plan = _compound_plan()
    plan.context_requests[0] = plan.context_requests[0].model_copy(
        update={"usage": "intent_refinement"}
    )

    graph = SupervisorPlanCompiler().compile(plan, turn_id="turn_refine_policy")

    profile = graph.require_node("proposal:p_profile")
    refinement = graph.require_node("system:intent_refinement")
    refined_policy = graph.require_node("system:policy_gate:refinement")
    assert profile.depends_on == ["system:policy_gate"]
    assert refinement.depends_on == ["system:policy_gate"]
    assert refinement.input_refs == [profile.node_id]
    assert refined_policy.depends_on == [refinement.node_id]
    assert refined_policy.capability == "policy_gate"
    for node_id in (
        "proposal:p_single",
        "proposal:p_compare",
        "proposal:p_knowledge",
        "proposal:p_commerce",
    ):
        assert refined_policy.node_id in graph.require_node(node_id).depends_on


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
    plan.original_query = "推荐油皮防晒，并看看小红书评价"
    plan.normalized_query = plan.original_query
    plan.intents[0] = plan.intents[0].model_copy(
        update={
            "route_basis": IntentRouteBasis(
                target_clarity="explicit_product",
                local_catalog_status="sufficient",
                external_information_need="explicit_platform",
                trigger_text="小红书",
                product_family="防晒霜",
                reason="用户明确要求平台评价。",
            )
        }
    )
    plan.research_requests = [
        ResearchRequest(
            request_id="r_xhs",
            intent_id="i1",
            mode="social_content",
            consumer_capability="commerce_research",
            trigger_type="explicit_platform_request",
            trigger_text="小红书",
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
            consumer_capability="commerce_research",
            trigger_type="explicit_platform_request",
            trigger_text="京东",
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
                    "concept_type": "product_type",
                    "supporting_evidence_ids": ["web:0"],
                },
                {
                    "concept": "消炎",
                    "concept_type": "efficacy",
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
    assert decision.approved_query == "芦荟胶"
    assert "消炎" not in decision.approved_query
    assert decision.verification_goal == "想要消炎的东西"
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
                {
                    "concept": "芦荟胶",
                    "concept_type": "product_type",
                    "supporting_evidence_ids": ["web:missing"],
                }
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
                {
                    "concept": "防晒霜",
                    "concept_type": "product_type",
                    "supporting_evidence_ids": ["web:0"],
                }
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


def test_knowledge_follow_up_rejects_efficacy_even_when_evidence_mentions_it() -> None:
    decision = SupervisorPolicyGate(build_foundation_agent_registry()).approve_knowledge_follow_up(
        _knowledge_bridge_plan(),
        knowledge_artifact={
            "candidate_concepts": ["消炎"],
            "concept_proposals": [
                {
                    "concept": "消炎",
                    "concept_type": "efficacy",
                    "supporting_evidence_ids": ["web:0"],
                }
            ],
        },
        evidence_refs=[
            EvidenceRef(
                evidence_id="web:0",
                source_type="web_search",
                source_id="https://example.test/efficacy",
                claim="消炎相关资料",
                summary="网页讨论了消炎功效，但没有给出可购买商品类别。",
            )
        ],
    )

    assert decision.approved is False
    assert decision.approved_query == ""
    assert decision.rejected_concepts[0]["concept_type"] == "efficacy"
    assert "不是可直接检索的商品品类" in decision.rejected_concepts[0]["reason"]


def test_knowledge_flow_materializes_policy_and_retrieval_lineage() -> None:
    async def scenario() -> None:
        recorder = _memory_span_recorder("turn_knowledge_flow")
        intent_span = recorder.start_span(
            "intent_planning",
            agent_id="intent_understanding_agent",
            task_id="turn_knowledge_flow:intent_understanding",
            span_type="agent",
        )
        intent_payload = recorder.finish_span(intent_span)
        flow = SupervisorFlow(
            services=BusinessAgentServices(),
            tool_registry=ToolRegistry(),
            span_recorder=recorder,
            budget_manager=None,
        )
        calls: list[str] = []

        async def knowledge(context):
            calls.append("knowledge")
            evidence = EvidenceRef(
                evidence_id="web:0",
                source_type="web_search",
                source_id="https://example.test/aloe",
                claim="芦荟胶选购资料",
                summary="资料明确提到芦荟胶这一商品类别。",
            )
            artifact = {
                "candidate_concepts": ["芦荟胶", "消炎"],
                "concept_proposals": [
                    {
                        "concept": "芦荟胶",
                        "concept_type": "product_type",
                        "supporting_evidence_ids": ["web:0"],
                    },
                    {
                        "concept": "消炎",
                        "concept_type": "efficacy",
                        "supporting_evidence_ids": ["web:0"],
                    },
                ],
                "risks": ["功效目标只能由后续证据校验"],
            }
            return AgentResult.success(
                {"concept_count": 2},
                evidence=[evidence],
                artifacts={"knowledge_research": artifact},
            )

        async def retrieval(context):
            calls.append("retrieval")
            assert context.node.metadata["query_override"] == "芦荟胶"
            assert context.artifact("knowledge_retrieval_plan").vector_query == "芦荟胶"
            return AgentResult.success(
                {"query": "芦荟胶"},
                evidence=[
                    EvidenceRef(
                        evidence_id="product:p_aloe",
                        source_type="local_product_catalog",
                        source_id="p_aloe",
                        product_id="p_aloe",
                        claim="芦荟胶",
                        verified=True,
                    )
                ],
            )

        async def verifier(context):
            calls.append("verifier")
            reflection = ReflectionResult(
                has_passed_products=False,
                reason="测试证据边界已执行。",
                fallback_plan="direct_answer",
            )
            return AgentResult.success(
                {"status": "verified"},
                artifacts={"reflection": reflection},
            )

        async def answer(context):
            calls.append("answer")
            return AgentResult.success(
                {"route": "direct_answer"},
                artifacts={
                    "answer_text": "已完成受控知识转检索。",
                    "answer_tokens": ["已完成受控知识转检索。"],
                    "answer_product_ids": [],
                    "answer_route": "direct_answer",
                },
            )

        async def memory(context):
            calls.append("memory")
            return AgentResult.success({"scheduled": True})

        flow.executor.handlers.update(
            {
                "knowledge_research": knowledge,
                "single_product_recommendation": retrieval,
                "evidence_verification": verifier,
                "answer_generation": answer,
                "memory_distillation": memory,
            }
        )
        events = []
        async for event in flow.run(
            query="想要消炎的东西",
            user_id="u1",
            session_id="s1",
            turn_id="turn_knowledge_flow",
            intent_plan=_knowledge_bridge_plan(),
            intent_span_key=intent_payload["span_key"],
        ):
            events.append(event)

        result = flow.last_result
        assert result is not None
        graph = result.graph
        policy = graph.require_node("runtime:policy_gate:knowledge_follow_up")
        retrieval_node = graph.require_node("runtime:knowledge_recommendation")
        assert policy.depends_on == ["proposal:p_knowledge"]
        assert policy.metadata["approved"] is True
        assert policy.metadata["approved_query"] == "芦荟胶"
        assert retrieval_node.depends_on == [policy.node_id]
        assert graph.all_terminal()
        assert calls == ["knowledge", "retrieval", "verifier", "answer", "memory"]
        assert any(event.get("type") == "timing_update" for event in events)

        spans_by_task = {span["task_id"]: span for span in recorder.spans}
        knowledge_span = spans_by_task["turn_knowledge_flow:p_knowledge"]
        dynamic_policy_span = spans_by_task["turn_knowledge_flow:policy_gate:knowledge_follow_up"]
        retrieval_span = spans_by_task["turn_knowledge_flow:knowledge_recommendation"]
        policy_handoff_span = next(
            span
            for span in recorder.spans
            if span["span_type"] == "handoff"
            and span["input_summary"].get("from_node_id") == policy.node_id
            and span["input_summary"].get("to_node_id") == retrieval_node.node_id
        )
        assert dynamic_policy_span["span_type"] == "policy"
        assert dynamic_policy_span["parent_span_key"] == knowledge_span["span_key"]
        assert policy_handoff_span["parent_span_key"] == dynamic_policy_span["span_key"]
        assert retrieval_span["parent_span_key"] == policy_handoff_span["span_key"]

    asyncio.run(scenario())


def test_knowledge_follow_up_policy_is_idempotent_and_records_final_rejection() -> None:
    recorder = _memory_span_recorder("turn_knowledge_policy")
    flow = SupervisorFlow(
        services=BusinessAgentServices(),
        tool_registry=ToolRegistry(),
        span_recorder=recorder,
        budget_manager=None,
    )
    plan = _knowledge_bridge_plan()
    graph = flow.compiler.compile(plan, turn_id="turn_knowledge_policy")
    knowledge_node = graph.require_node("proposal:p_knowledge")
    knowledge_node.status = "succeeded"
    context = AgentExecutionContext(
        node=knowledge_node,
        graph=graph,
        query=plan.original_query,
        user_id="u1",
        session_id="s1",
        turn_id=graph.turn_id,
        intent_plan=plan,
        artifacts={
            "intent_plan": plan,
            "knowledge_research": {
                "concept_proposals": [
                    {
                        "concept": "芦荟胶",
                        "concept_type": "product_type",
                        "supporting_evidence_ids": ["web:0"],
                    }
                ]
            },
            f"evidence:{knowledge_node.node_id}": [
                EvidenceRef(
                    evidence_id="web:0",
                    source_type="web_search",
                    source_id="https://example.test/aloe",
                    claim="芦荟胶选购资料",
                    summary="资料明确提到芦荟胶这一商品类别。",
                )
            ],
        },
    )
    # Remove the verifier to force a deterministic materialization rejection.
    graph.nodes = [node for node in graph.nodes if node.capability != "evidence_verification"]
    answer = graph.get_node("system:answer_generation")
    if answer is not None:
        answer.depends_on = [knowledge_node.node_id]
    graph.refresh_terminal_nodes()

    result = flow._knowledge_follow_up_decision(graph, context)

    assert result is not None
    decision, span = result
    assert decision.approved is False
    assert decision.decision == "reject_knowledge_to_retrieval"
    policy = graph.require_node("runtime:policy_gate:knowledge_follow_up")
    assert policy.metadata["approved"] is False
    assert policy.metadata["decision"] == "reject_knowledge_to_retrieval"
    assert span is not None
    assert span["output_summary"]["approved"] is False
    assert span["termination_reason"] == "policy_rejected"
    assert flow._knowledge_follow_up_decision(graph, context) is None
    assert len(
        [node for node in graph.nodes if node.node_id == "runtime:policy_gate:knowledge_follow_up"]
    ) == 1
    assert len(
        [item for item in recorder.spans if item["task_id"] == "turn_knowledge_policy:policy_gate:knowledge_follow_up"]
    ) == 1


def _knowledge_bridge_plan(*, knowledge_only: bool = False) -> IntentPlan:
    intent_type = "shopping_knowledge" if knowledge_only else "product_recommendation"
    return IntentPlan(
        original_query="想要消炎的东西",
        normalized_query="根据资料寻找适合的商品概念",
        primary_intent=intent_type,
        intents=[
            IntentItem(
                intent_id="i1",
                intent_type=intent_type,
                goal="解释并判断可购买的商品概念",
                route_basis=IntentRouteBasis(
                    target_clarity="vague_effect_or_use",
                    local_catalog_status="insufficient",
                    external_information_need="knowledge_bridge",
                    trigger_text="消炎",
                    product_family="",
                    reason="当前只有功效目标，商品族不明确。",
                ),
            )
        ],
        execution_mode="context_evidence",
        research_requests=[
            ResearchRequest(
                request_id="r_web",
                intent_id="i1",
                mode="web_general",
                consumer_capability="knowledge_research",
                trigger_type="knowledge_bridge",
                trigger_text="消炎",
                query="消炎相关的可购买商品概念",
                local_catalog_gap="当前商品目录没有明确对应的可购买商品族。",
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
                route_basis=IntentRouteBasis(
                    target_clarity="explicit_product",
                    local_catalog_status="sufficient",
                    external_information_need="none",
                    product_family="防晒霜",
                    reason="当前请求已经明确商品目标。",
                ),
            ),
            IntentItem(
                intent_id="i2",
                intent_type="product_comparison",
                goal="将新推荐与上一轮防晒进行对比",
                depends_on=["i1"],
                referenced_product_ids=["p_recent_001"],
                route_basis=IntentRouteBasis(
                    target_clarity="context_product",
                    local_catalog_status="sufficient",
                    external_information_need="explicit_platform",
                    trigger_text="小红书",
                    product_family="防晒霜",
                    reason="用户明确要求查看小红书评价。",
                ),
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
                consumer_capability="commerce_research",
                trigger_type="explicit_platform_request",
                trigger_text="小红书",
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
                optional_context_from=["p_commerce"],
            ),
            AgentTaskProposal(
                proposal_id="p_knowledge",
                capability="knowledge_research",
                intent_ids=["i3"],
                reason="用户要求解释肤感原理",
                depends_on=["p_single"],
            ),
            AgentTaskProposal(
                proposal_id="p_commerce",
                capability="commerce_research",
                intent_ids=["i2"],
                reason="用户明确要求查看小红书评价。",
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


def _memory_span_recorder(turn_id: str) -> SpanRecorder:
    recorder = SpanRecorder()
    recorder._persist_run = lambda: None  # type: ignore[method-assign]
    recorder._persist_span = lambda *args, **kwargs: None  # type: ignore[method-assign]
    recorder.start_run(
        user_id="u1",
        session_id="s1",
        turn_id=turn_id,
        query_summary="test",
    )
    return recorder
