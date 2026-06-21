from app.domain.task_lifecycle import TurnTaskState
from app.harness.budget_manager import BudgetManager


def test_budget_manager_records_counts_and_checks_repair_limit() -> None:
    task = TurnTaskState(user_id="u1", session_id="s1")
    manager = BudgetManager()

    manager.record_planner_call(task)
    manager.record_tool_call(task, 3)
    manager.record_corrective_call(task)
    manager.record_answer_call(task)
    manager.record_repair_attempt(task, 2)

    assert task.budget.planner_call_count == 1
    assert task.budget.tool_call_count == 3
    assert task.budget.corrective_call_count == 1
    assert task.budget.answer_call_count == 1
    assert manager.can_call_planner(task)
    assert not manager.can_repair(task)
    assert manager.repair_snapshot(task) == {"repair_attempt_count": 2, "max": 2}
