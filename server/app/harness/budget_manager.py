from __future__ import annotations

from app.domain.task_lifecycle import TurnTaskState


class BudgetManager:
    """本轮执行预算管理器（BudgetManager）。

    归属 Harness Infrastructure，只负责读写 TurnTaskState 的预算账本和预算检查。
    Orchestrator 仍决定是否执行 planner/tool/repair/final_route。
    """

    def record_planner_call(self, task: TurnTaskState, count: int = 1) -> None:
        self._increment("planner_call_count", task, count)

    def record_corrective_call(self, task: TurnTaskState, count: int = 1) -> None:
        self._increment("corrective_call_count", task, count)

    def record_answer_call(self, task: TurnTaskState, count: int = 1) -> None:
        self._increment("answer_call_count", task, count)

    def record_tool_call(self, task: TurnTaskState, count: int = 1) -> None:
        self._increment("tool_call_count", task, count)

    def record_repair_attempt(self, task: TurnTaskState, count: int = 1) -> None:
        self._increment("repair_attempt_count", task, count)

    def can_call_planner(self, task: TurnTaskState) -> bool:
        return task.budget.can_call_planner()

    def can_call_tool(self, task: TurnTaskState) -> bool:
        return task.budget.can_call_tool()

    def can_repair(self, task: TurnTaskState) -> bool:
        return task.budget.can_repair()

    def budget_snapshot(self, task: TurnTaskState) -> dict:
        return task.budget.model_dump()

    def repair_snapshot(self, task: TurnTaskState) -> dict[str, int]:
        return {
            "repair_attempt_count": task.budget.repair_attempt_count,
            "max": task.budget.max_repair_attempts,
        }

    def _increment(self, field_name: str, task: TurnTaskState, count: int) -> None:
        if count <= 0:
            return
        setattr(task.budget, field_name, int(getattr(task.budget, field_name)) + count)
