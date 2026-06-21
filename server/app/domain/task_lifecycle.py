from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


TurnTaskStatus = Literal["pending", "running", "succeeded", "failed", "cancelled", "timeout"]
StepStatus = Literal["pending", "running", "succeeded", "skipped", "failed", "cancelled", "timeout"]
FailureType = Literal[
    "llm_failed",
    "tool_failed",
    "validation_failed",
    "timeout",
    "cancelled",
    "budget_exceeded",
    "unexpected_exception",
]


class ExecutionBudget(BaseModel):
    planner_call_count: int = 0
    corrective_call_count: int = 0
    answer_call_count: int = 0
    tool_call_count: int = 0
    repair_attempt_count: int = 0

    max_planner_calls: int = 2
    max_corrective_calls: int = 3
    max_answer_calls: int = 1
    max_tool_calls: int = 20
    max_repair_attempts: int = 2

    def can_call_planner(self) -> bool:
        return self.planner_call_count < self.max_planner_calls

    def can_call_tool(self) -> bool:
        return self.tool_call_count < self.max_tool_calls

    def can_repair(self) -> bool:
        return self.repair_attempt_count < self.max_repair_attempts


class StepRecord(BaseModel):
    step_name: str
    status: StepStatus = "pending"
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    input_summary: dict = Field(default_factory=dict)
    output_summary: dict = Field(default_factory=dict)
    decision_summary: dict = Field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None


class OrchestratorDecision(BaseModel):
    decision: str
    approved: bool | None = None
    selected: str = ""
    internal_decision: str = ""
    reason: str = ""
    proposal_summary: dict = Field(default_factory=dict)
    decision_summary: dict = Field(default_factory=dict)


class TurnTaskState(BaseModel):
    turn_id: str = ""
    user_id: str
    session_id: str
    status: TurnTaskStatus = "running"
    execution_path: str = ""
    final_route: str | None = None
    failure_type: str | None = None
    failure_reason: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    budget: ExecutionBudget = Field(default_factory=ExecutionBudget)
    steps: list[StepRecord] = Field(default_factory=list)
    planner_proposal: dict = Field(default_factory=dict)
    orchestrator_decisions: list[OrchestratorDecision] = Field(default_factory=list)

    def add_step(
        self,
        step_name: str,
        status: StepStatus,
        *,
        input_summary: dict | None = None,
        output_summary: dict | None = None,
        decision_summary: dict | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        self.steps.append(
            StepRecord(
                step_name=step_name,
                status=status,
                finished_at=datetime.now(timezone.utc),
                input_summary=input_summary or {},
                output_summary=output_summary or {},
                decision_summary=decision_summary or {},
                error_type=error_type,
                error_message=error_message,
            )
        )

    def add_decision(self, decision: OrchestratorDecision) -> OrchestratorDecision:
        self.orchestrator_decisions.append(decision)
        if decision.decision == "execution_path" and decision.selected:
            self.execution_path = decision.selected
        if decision.decision == "final_route" and decision.selected:
            self.final_route = decision.selected
        return decision

    def mark_succeeded(self, final_route: str) -> None:
        self.status = "succeeded"
        self.final_route = final_route
        self.finished_at = datetime.now(timezone.utc)

    def mark_failed(self, failure_type: FailureType, reason: str) -> None:
        self.status = failure_type if failure_type in {"timeout", "cancelled"} else "failed"
        self.failure_type = failure_type
        self.failure_reason = reason
        self.finished_at = datetime.now(timezone.utc)
