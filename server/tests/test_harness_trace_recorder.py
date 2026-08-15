from app.domain.task_lifecycle import OrchestratorDecision, TurnTaskState
from app.harness.trace_recorder import TraceRecorder
from app.schemas import DecisionTrace


def test_trace_recorder_stage_filters_empty_details() -> None:
    recorder = TraceRecorder()

    stage = recorder.stage("intent_planning", "passed", "ok", product_ids=[], count=2, note="")

    assert stage == {
        "name": "intent_planning",
        "status": "passed",
        "reason": "ok",
        "details": {"count": 2},
    }


def test_trace_recorder_finish_trace_writes_task_and_agent_path() -> None:
    task = TurnTaskState(user_id="u1", session_id="s1")
    task.planner_proposal = {"plan_type": "single_retrieval", "plan_reason": "needs product evidence"}
    task.add_decision(OrchestratorDecision(decision="execution_path", selected="single_retrieval", reason="approved"))
    trace = DecisionTrace(
        route="recommend",
        retrieval_summary={"reflection_result": {"fallback_plan": "none", "passed_product_ids": ["p1"]}},
    )

    TraceRecorder().finish_trace(trace, task, route="recommend")

    assert trace.task_status == "succeeded"
    assert trace.task["execution_path"] == "single_retrieval"
    assert trace.task["final_route"] == "recommend"
    assert trace.orchestrator_decisions[0]["decision"] == "execution_path"
    assert [item["node"] for item in trace.agent_path] == [
        "IntentPlanner",
        "Orchestrator",
        "SingleRetrievalWorker",
        "CorrectiveAgent",
        "AnswerGenerator",
    ]


def test_decision_trace_v2_preserves_execution_lineage_fields() -> None:
    trace = DecisionTrace(
        run_id="run_123",
        agent_path=[{"node_id": "proposal:p1", "agent_id": "single_product_recommendation_agent"}],
        tool_calls=[
            {
                "call_id": "turn:p1:tool:1",
                "tool": "product_search",
                "operation": "single_initial",
                "status": "succeeded",
            }
        ],
        handoffs=[
            {
                "handoff_id": "handoff:planner->p1:hard",
                "from_agent_id": "intent_understanding_agent",
                "to_agent_id": "single_product_recommendation_agent",
                "required": True,
                "status": "consumed",
            }
        ],
        failed_node_ids=["proposal:failed"],
        blocked_node_ids=["system:answer_generation"],
    )

    payload = trace.model_dump()

    assert payload["trace_schema_version"] == "v2"
    assert payload["run_id"] == "run_123"
    assert payload["agent_path"][0]["node_id"] == "proposal:p1"
    assert payload["tool_calls"][0]["tool"] == "product_search"
    assert payload["handoffs"][0]["status"] == "consumed"
    assert payload["failed_node_ids"] == ["proposal:failed"]
    assert payload["blocked_node_ids"] == ["system:answer_generation"]
