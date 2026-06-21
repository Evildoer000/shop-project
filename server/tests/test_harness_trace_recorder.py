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
