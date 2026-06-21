from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.db.models import Product
from app.schemas import IntentPlan, QueryPlan


SlotNeedType = Literal["required", "optional"]
SlotStatus = Literal["pending", "covered", "weak", "failed"]
AgentAction = Literal[
    "search_products",
    "repair_plan_generated",
    "repair_search_executed",
    "evidence_merged",
    "relax_soft_constraints",
    "broaden_category",
    "ask_clarification",
    "final_answer",
]


class NeedSlot(BaseModel):
    slot_id: str
    need_type: SlotNeedType = "required"
    goal: str
    product_type: str = ""
    query: str
    hard_constraints: list[str] = Field(default_factory=list)
    soft_constraints: list[str] = Field(default_factory=list)
    exclude_terms: list[str] = Field(default_factory=list)
    min_candidates: int = 1
    status: SlotStatus = "pending"


class SlotCandidate(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    product: Product = Field(exclude=True)
    product_id: str
    name: str
    category: str
    sub_category: str | None = None
    price: float
    vector_score: float = 0.0
    keyword_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    coverage_reason: str = ""


class SlotSearchResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    slot_id: str
    query: str
    vector_query: str
    keyword_query: str
    candidates: list[SlotCandidate] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    attempts: list[dict] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    category_resolution: str = "unknown"
    rejected_candidates: list[dict] = Field(default_factory=list)
    structured_products: list[Product] = Field(default_factory=list, exclude=True)
    score_filtered_products: list[Product] = Field(default_factory=list, exclude=True)
    hybrid_ranked_products: list[Product] = Field(default_factory=list, exclude=True)
    vector_scores: dict[str, float] = Field(default_factory=dict)
    keyword_scores: dict[str, float] = Field(default_factory=dict)


class SlotCoverageDecision(BaseModel):
    slot_id: str
    status: SlotStatus
    covered: bool = False
    reason: str = ""
    candidate_ids: list[str] = Field(default_factory=list)
    raw_candidate_ids: list[str] = Field(default_factory=list)
    candidate_count: int = 0
    attempt_count: int = 0
    notes: list[str] = Field(default_factory=list)


class AgentToolCall(BaseModel):
    action: str
    slot_id: str | None = None
    input_summary: dict = Field(default_factory=dict)
    output_summary: dict = Field(default_factory=dict)
    status: str = "ok"
    reason: str = ""
    duration_ms: float = 0.0


class FinalAnswerSignal(BaseModel):
    route: Literal["recommend", "partial_recommend", "over_budget_combo", "no_product", "clarify"] = "recommend"
    reason: str = ""


class MultiNeedSelection(BaseModel):
    selected_by_slot: dict[str, list[SlotCandidate]] = Field(default_factory=dict)
    rejected_candidates: list[dict] = Field(default_factory=list)
    route: Literal["recommend", "partial_recommend", "over_budget_combo", "no_product", "clarify"] = "no_product"
    reason: str = ""

    @property
    def flat_candidates(self) -> list[SlotCandidate]:
        result: list[SlotCandidate] = []
        for candidates in self.selected_by_slot.values():
            result.extend(candidates)
        return result


class MultiNeedState(BaseModel):
    original_query: str
    intent_plan: IntentPlan
    plan: QueryPlan
    global_constraints: list[str] = Field(default_factory=list)
    slots: list[NeedSlot] = Field(default_factory=list)
    tool_calls: list[AgentToolCall] = Field(default_factory=list)
    candidates_by_slot: dict[str, list[SlotCandidate]] = Field(default_factory=dict)
    coverage_by_slot: dict[str, SlotCoverageDecision] = Field(default_factory=dict)
    budgets: dict = Field(default_factory=dict)
    termination_reason: str = ""
    clarification_question: str | None = None
    final_signal: FinalAnswerSignal | None = None

    def slot_by_id(self, slot_id: str) -> NeedSlot | None:
        for slot in self.slots:
            if slot.slot_id == slot_id:
                return slot
        return None
