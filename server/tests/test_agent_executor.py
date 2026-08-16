import asyncio

from app.domain.agents.contracts import AgentExecutionContext, AgentResult, ExecutionReport
from app.domain.agents.executor import AgentExecutor
from app.harness.span_recorder import SpanRecorder
from app.domain.supervisor.agent_registry import build_foundation_agent_registry
from app.domain.supervisor.task_graph import TaskGraph, TaskGraphNode
from app.schemas import IntentPlan
from app.harness.tool_registry import ToolRegistry


def test_executor_passes_dependency_artifacts_and_enforces_agent_tools() -> None:
    registry = build_foundation_agent_registry()
    tools = ToolRegistry()
    profile_tool = object()
    tools.register("profile_lookup", profile_tool, description="test profile tool")
    calls: list[str] = []

    async def profile(context: AgentExecutionContext) -> AgentResult:
        assert context.tool("profile_lookup") is profile_tool
        calls.append("profile")
        return AgentResult.success({"profile": "soft preference"})

    async def recommendation(context: AgentExecutionContext) -> AgentResult:
        calls.append("recommendation")
        assert context.output_from("profile") == {"profile": "soft preference"}
        return AgentResult.success({"used_profile": True})

    recorder = _memory_span_recorder("t_executor")
    executor = AgentExecutor(
        agent_registry=registry,
        tool_registry=tools,
        span_recorder=recorder,
        handlers={
            "profile_preference": profile,
            "single_product_recommendation": recommendation,
        },
    )
    graph = TaskGraph(
        graph_id="g_executor",
        turn_id="t_executor",
        nodes=[
            TaskGraphNode(
                node_id="profile",
                task_id="t_executor:profile",
                agent_id="profile_preference_agent",
                capability="profile_preference",
            ),
            TaskGraphNode(
                node_id="recommendation",
                task_id="t_executor:recommendation",
                agent_id="single_product_recommendation_agent",
                capability="single_product_recommendation",
                input_refs=["profile"],
            ),
        ],
    )
    context = AgentExecutionContext(
        node=graph.nodes[0],
        graph=graph,
        query="推荐防晒",
        user_id="u1",
        session_id="s1",
        turn_id="t_executor",
        intent_plan=IntentPlan(original_query="推荐防晒"),
    )

    report = asyncio.run(executor.execute(graph, base_context=context))

    assert report.succeeded is True
    assert calls == ["profile", "recommendation"]
    assert report.artifacts["profile"] == {"profile": "soft preference"}
    assert report.artifacts["recommendation"] == {"used_profile": True}
    handoff = graph.handoffs_to("recommendation")[0]
    assert handoff.required is False
    assert handoff.status == "consumed"
    handoff_span = next(
        span
        for span in recorder.spans
        if span["span_type"] == "handoff"
        and span["input_summary"].get("handoff_id") == handoff.handoff_id
    )
    recommendation_span = next(
        span
        for span in recorder.spans
        if span["task_id"] == "t_executor:recommendation" and span["span_type"] == "agent"
    )
    assert handoff_span["parent_span_key"] == next(
        span["span_key"]
        for span in recorder.spans
        if span["task_id"] == "t_executor:profile" and span["span_type"] == "agent"
    )
    assert recommendation_span["parent_span_key"] == handoff_span["span_key"]


def test_optional_profile_failure_degrades_handoff_without_blocking_recommendation() -> None:
    calls: list[str] = []

    async def profile(context: AgentExecutionContext) -> AgentResult:
        calls.append("profile")
        return AgentResult.failure_result("profile_unavailable", "profile service unavailable")

    async def recommendation(context: AgentExecutionContext) -> AgentResult:
        calls.append("recommendation")
        assert context.output_from("profile") == {}
        return AgentResult.success({"used_profile": False})

    recorder = _memory_span_recorder("t_optional_profile")
    executor = AgentExecutor(
        agent_registry=build_foundation_agent_registry(),
        tool_registry=ToolRegistry(),
        span_recorder=recorder,
        handlers={
            "profile_preference": profile,
            "single_product_recommendation": recommendation,
        },
    )
    graph = TaskGraph(
        graph_id="g_optional_profile",
        turn_id="t_optional_profile",
        nodes=[
            TaskGraphNode(
                node_id="profile",
                task_id="t_optional_profile:profile",
                agent_id="profile_preference_agent",
                capability="profile_preference",
                required=False,
            ),
            TaskGraphNode(
                node_id="recommendation",
                task_id="t_optional_profile:recommendation",
                agent_id="single_product_recommendation_agent",
                capability="single_product_recommendation",
                input_refs=["profile"],
            ),
        ],
    )
    context = AgentExecutionContext(
        node=graph.nodes[0],
        graph=graph,
        query="推荐防晒",
        user_id="u1",
        session_id="s1",
        turn_id=graph.turn_id,
        intent_plan=IntentPlan(original_query="推荐防晒"),
    )

    report = asyncio.run(executor.execute(graph, base_context=context))

    assert report.succeeded is True
    assert report.failed_node_ids == []
    assert report.blocked_node_ids == ["profile"]
    assert calls == ["profile", "recommendation"]
    assert graph.require_node("profile").status == "skipped"
    assert graph.require_node("profile").metadata["skipped_reason"] == "optional_branch_failed"
    assert graph.require_node("recommendation").status == "succeeded"
    handoff = graph.handoffs_to("recommendation")[0]
    assert handoff.required is False
    assert handoff.status == "unavailable"
    handoff_span = next(
        span
        for span in recorder.spans
        if span["span_type"] == "handoff"
        and span["input_summary"].get("handoff_id") == handoff.handoff_id
    )
    recommendation_span = next(
        span
        for span in recorder.spans
        if span["task_id"] == "t_optional_profile:recommendation"
        and span["span_type"] == "agent"
    )
    assert handoff_span["status"] == "degraded"
    assert recommendation_span["parent_span_key"] == handoff_span["span_key"]


def test_executor_runs_independent_nodes_in_the_same_batch() -> None:
    registry = build_foundation_agent_registry()
    tools = ToolRegistry()
    order: list[str] = []

    async def handler(context: AgentExecutionContext) -> AgentResult:
        order.append(context.node.node_id)
        await asyncio.sleep(0)
        return AgentResult.success({"node": context.node.node_id})

    executor = AgentExecutor(
        agent_registry=registry,
        tool_registry=tools,
        handlers={"single_product_recommendation": handler},
    )
    graph = TaskGraph(
        graph_id="g_parallel",
        turn_id="t_parallel",
        nodes=[
            TaskGraphNode(
                node_id="a",
                task_id="t_parallel:a",
                agent_id="single_product_recommendation_agent",
                capability="single_product_recommendation",
            ),
            TaskGraphNode(
                node_id="b",
                task_id="t_parallel:b",
                agent_id="single_product_recommendation_agent",
                capability="single_product_recommendation",
            ),
        ],
    )
    context = AgentExecutionContext(
        node=graph.nodes[0],
        graph=graph,
        query="推荐耳机",
        user_id="u1",
        session_id="s1",
        turn_id="t_parallel",
        intent_plan=IntentPlan(original_query="推荐耳机"),
    )

    report = asyncio.run(executor.execute(graph, base_context=context))

    assert report.succeeded is True
    assert set(order) == {"a", "b"}


def test_node_cannot_expand_manifest_tool_allowlist() -> None:
    class ForbiddenWebTool:
        called = False

        def search(self) -> None:
            self.called = True

    web_tool = ForbiddenWebTool()
    tools = ToolRegistry()
    tools.register("web_search", web_tool, networked=True)

    async def malicious_handler(context: AgentExecutionContext) -> AgentResult:
        context.tool("web_search").search()
        return AgentResult.success()

    executor = AgentExecutor(
        agent_registry=build_foundation_agent_registry(),
        tool_registry=tools,
        handlers={"single_product_recommendation": malicious_handler},
    )
    graph, context = _single_node_graph_context("tool_escalation")
    graph.nodes[0].metadata["allowed_tools"] = ["web_search"]

    report = asyncio.run(executor.execute(graph, base_context=context))

    result = report.results["tool_escalation"]
    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.error_type == "ToolAccessDenied"
    assert "not allowlisted" in result.failure.message
    assert web_tool.called is False
    assert graph.nodes[0].metadata["allowed_tools"] == []


def test_executor_rejects_agent_capability_mismatch_before_handler_call() -> None:
    called = False

    async def handler(context: AgentExecutionContext) -> AgentResult:
        nonlocal called
        called = True
        return AgentResult.success()

    recorder = _memory_span_recorder("t_mismatch")
    executor = AgentExecutor(
        agent_registry=build_foundation_agent_registry(),
        tool_registry=ToolRegistry(),
        span_recorder=recorder,
        handlers={"single_product_recommendation": handler},
    )
    graph = TaskGraph(
        graph_id="g_mismatch",
        turn_id="t_mismatch",
        nodes=[
            TaskGraphNode(
                node_id="mismatch",
                task_id="t_mismatch:mismatch",
                agent_id="profile_preference_agent",
                capability="single_product_recommendation",
            )
        ],
    )
    context = AgentExecutionContext(
        node=graph.nodes[0],
        graph=graph,
        query="推荐商品",
        user_id="u1",
        session_id="s1",
        turn_id=graph.turn_id,
        intent_plan=IntentPlan(original_query="推荐商品"),
    )

    report = asyncio.run(executor.execute(graph, base_context=context))

    result = report.results["mismatch"]
    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.error_type == "agent_capability_mismatch"
    assert called is False
    spans = [span for span in recorder.spans if span["task_id"] == "t_mismatch:mismatch"]
    assert len(spans) == 1
    assert spans[0]["status"] == "failed"
    assert spans[0]["error_type"] == "agent_capability_mismatch"
    assert spans[0]["termination_reason"] == "configuration_error"


def test_executor_records_unregistered_agent_as_attributable_failure_span() -> None:
    recorder = _memory_span_recorder("t_unregistered")
    executor = AgentExecutor(
        agent_registry=build_foundation_agent_registry(),
        tool_registry=ToolRegistry(),
        span_recorder=recorder,
    )
    graph = TaskGraph(
        graph_id="g_unregistered",
        turn_id="t_unregistered",
        nodes=[
            TaskGraphNode(
                node_id="unregistered",
                task_id="t_unregistered:unregistered",
                agent_id="missing_agent",
                capability="single_product_recommendation",
            )
        ],
    )
    context = AgentExecutionContext(
        node=graph.nodes[0],
        graph=graph,
        query="推荐商品",
        user_id="u1",
        session_id="s1",
        turn_id=graph.turn_id,
        intent_plan=IntentPlan(original_query="推荐商品"),
    )

    report = asyncio.run(executor.execute(graph, base_context=context))

    assert report.results["unregistered"].failure.error_type == "agent_not_registered"
    spans = [span for span in recorder.spans if span["task_id"] == "t_unregistered:unregistered"]
    assert len(spans) == 1
    assert spans[0]["agent_id"] == "missing_agent"
    assert spans[0]["termination_reason"] == "configuration_error"


def test_executor_blocks_required_dependents_after_failure() -> None:
    registry = build_foundation_agent_registry()
    executor = AgentExecutor(
        agent_registry=registry,
        tool_registry=ToolRegistry(),
        handlers={
            "single_product_recommendation": lambda context: AgentResult.failure_result(
                "tool_failed", "search unavailable"
            ),
        },
    )
    graph = TaskGraph(
        graph_id="g_failure",
        turn_id="t_failure",
        nodes=[
            TaskGraphNode(
                node_id="failed",
                task_id="t_failure:failed",
                agent_id="single_product_recommendation_agent",
                capability="single_product_recommendation",
            ),
            TaskGraphNode(
                node_id="dependent",
                task_id="t_failure:dependent",
                agent_id="single_product_recommendation_agent",
                capability="single_product_recommendation",
                depends_on=["failed"],
            ),
        ],
    )
    context = AgentExecutionContext(
        node=graph.nodes[0],
        graph=graph,
        query="推荐耳机",
        user_id="u1",
        session_id="s1",
        turn_id="t_failure",
        intent_plan=IntentPlan(original_query="推荐耳机"),
    )

    report = asyncio.run(executor.execute(graph, base_context=context))

    assert report.succeeded is False
    assert report.failed_node_ids == ["failed"]
    assert graph.require_node("dependent").status == "skipped"
    assert report.blocked_node_ids == ["dependent"]


def test_executor_closing_stream_cancels_running_agent_and_finishes_span() -> None:
    async def scenario() -> None:
        registry = build_foundation_agent_registry()
        started = asyncio.Event()
        stopped = asyncio.Event()

        async def handler(context: AgentExecutionContext) -> AgentResult:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
            return AgentResult.success()

        recorder = _memory_span_recorder("t_cancel")
        executor = AgentExecutor(
            agent_registry=registry,
            tool_registry=ToolRegistry(),
            span_recorder=recorder,
            handlers={"single_product_recommendation": handler},
        )
        graph, context = _single_node_graph_context("cancel")
        report = ExecutionReport(graph=graph)
        stream = executor.execute_stream(graph, base_context=context, report=report)

        event = await stream.__anext__()
        assert event.kind == "node_started"
        await asyncio.wait_for(started.wait(), timeout=1)
        await stream.aclose()
        await asyncio.wait_for(stopped.wait(), timeout=1)

        node = graph.require_node("cancel")
        assert node.status == "cancelled"
        assert report.failed_node_ids == ["cancel"]
        assert report.results["cancel"].status == "cancelled"
        assert graph.metadata["termination_reason"] == "executor_stream_closed"
        agent_spans = [span for span in recorder.spans if span["span_type"] == "agent"]
        assert len(agent_spans) == 1
        assert agent_spans[0]["status"] == "cancelled"
        assert agent_spans[0]["termination_reason"] == "cancelled"

    asyncio.run(scenario())


def test_executor_closing_on_start_event_accounts_for_not_yet_started_agent() -> None:
    async def scenario() -> None:
        registry = build_foundation_agent_registry()

        async def handler(context: AgentExecutionContext) -> AgentResult:
            await asyncio.Event().wait()
            return AgentResult.success()

        recorder = _memory_span_recorder("t_early_cancel")
        executor = AgentExecutor(
            agent_registry=registry,
            tool_registry=ToolRegistry(),
            span_recorder=recorder,
            handlers={"single_product_recommendation": handler},
        )
        graph, context = _single_node_graph_context("early_cancel")
        report = ExecutionReport(graph=graph)
        stream = executor.execute_stream(graph, base_context=context, report=report)

        assert (await stream.__anext__()).kind == "node_started"
        await stream.aclose()

        assert graph.require_node("early_cancel").status == "cancelled"
        assert report.results["early_cancel"].status == "cancelled"
        agent_spans = [span for span in recorder.spans if span["span_type"] == "agent"]
        assert len(agent_spans) == 1
        assert agent_spans[0]["status"] == "cancelled"

    asyncio.run(scenario())


def test_span_recorder_finish_operations_are_idempotent() -> None:
    recorder = _memory_span_recorder("t_idempotent")
    span = recorder.start_span("work", agent_id="test_agent", task_id="t_idempotent:work")

    first_span = recorder.finish_span(span, termination_reason="completed")
    second_span = recorder.finish_span(span, status="failed", termination_reason="duplicate")
    first_run = recorder.finish_run(route="direct_answer", termination_reason="completed")
    second_run = recorder.finish_run(
        route="failed",
        status="failed",
        termination_reason="duplicate",
    )

    assert first_span == second_span
    assert len([item for item in recorder.spans if item["span_key"] == span.span_key]) == 1
    assert first_run == second_run
    assert second_run["summary"]["status"] == "succeeded"
    assert second_run["summary"]["termination_reason"] == "completed"
    assert len([item for item in recorder.spans if item["span_type"] == "run"]) == 1


def _single_node_graph_context(node_id: str) -> tuple[TaskGraph, AgentExecutionContext]:
    graph = TaskGraph(
        graph_id=f"g_{node_id}",
        turn_id=f"t_{node_id}",
        nodes=[
            TaskGraphNode(
                node_id=node_id,
                task_id=f"t_{node_id}:{node_id}",
                agent_id="single_product_recommendation_agent",
                capability="single_product_recommendation",
            )
        ],
    )
    context = AgentExecutionContext(
        node=graph.nodes[0],
        graph=graph,
        query="推荐商品",
        user_id="u1",
        session_id="s1",
        turn_id=graph.turn_id,
        intent_plan=IntentPlan(original_query="推荐商品"),
    )
    return graph, context


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
