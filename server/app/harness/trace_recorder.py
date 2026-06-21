from __future__ import annotations

from typing import Any

from app.domain.task_lifecycle import TurnTaskState
from app.schemas import DecisionTrace


class TraceRecorder:
    """决策链记录器（TraceRecorder）。

    归属 Harness Infrastructure，统一生成 stage、task snapshot、agent_path 和 failure trace 的公共结构。
    Orchestrator 仍传入语义 summary、裁决结果和最终 route。
    """

    def stage(self, name: str, status: str, reason: str, **details: Any) -> dict[str, Any]:
        clean_details = {key: value for key, value in details.items() if value not in (None, "", [])}
        return {"name": name, "status": status, "reason": reason, "details": clean_details}

    def finish_trace(
        self,
        trace: DecisionTrace,
        task: TurnTaskState,
        *,
        route: str,
        agent_path: list[dict[str, Any]] | None = None,
    ) -> None:
        task.mark_succeeded(route)
        self.apply_task_snapshot(trace, task)
        trace.agent_path = agent_path if agent_path is not None else self.agent_path(trace, task)

    def apply_failure_trace(self, trace: DecisionTrace, task: TurnTaskState) -> None:
        self.apply_task_snapshot(trace, task)
        trace.agent_path = self.agent_path(trace, task)

    def apply_task_snapshot(self, trace: DecisionTrace, task: TurnTaskState) -> None:
        trace.planner_proposal = dict(task.planner_proposal)
        trace.orchestrator_decisions = [decision.model_dump() for decision in task.orchestrator_decisions]
        trace.task = {
            "status": task.status,
            "execution_path": task.execution_path,
            "final_route": task.final_route,
            "budget": task.budget.model_dump(),
        }
        if task.failure_type:
            trace.task["failure_type"] = task.failure_type
        if task.failure_reason:
            trace.task["failure_reason"] = task.failure_reason
        trace.task_status = task.status

    def agent_path(self, trace: DecisionTrace, task: TurnTaskState) -> list[dict[str, Any]]:
        path = [
            {
                "node": "IntentPlanner",
                "role": "Planner Agent",
                "output": task.planner_proposal.get("plan_type"),
                "summary": task.planner_proposal.get("plan_reason", ""),
            }
        ]
        for decision in task.orchestrator_decisions:
            path.append(
                {
                    "node": "Orchestrator",
                    "role": "Control Plane",
                    "decision": decision.decision,
                    "selected": decision.selected,
                    "approved": decision.approved,
                    "internal_decision": decision.internal_decision,
                    "reason": decision.reason,
                }
            )
        if task.execution_path == "single_retrieval":
            path.append({"node": "SingleRetrievalWorker", "role": "Retrieval Worker", "output": "evidence"})
        if task.execution_path == "multi_retrieval":
            path.append({"node": "MultiNeedRetrievalCoordinator", "role": "Retrieval Worker", "output": "slot evidence"})
        if task.execution_path == "image_retrieval":
            path.append({"node": "ImageRetrievalWorker", "role": "Retrieval Worker", "output": "image evidence"})
        if "reflection_result" in trace.retrieval_summary:
            reflection = trace.retrieval_summary.get("reflection_result", {})
            path.append(
                {
                    "node": "CorrectiveAgent",
                    "role": "Worker Agent",
                    "output": "reflection_result",
                    "fallback_plan": reflection.get("fallback_plan"),
                    "passed_product_ids": reflection.get("passed_product_ids"),
                }
            )
        path.append({"node": "AnswerGenerator", "role": "Worker Agent", "input_final_route": trace.route})
        return path
