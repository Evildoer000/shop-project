from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


SINGLE_RECOMMENDATION_LIMIT = 5
SINGLE_RETRIEVAL_REVIEW_LIMIT = 10
MULTI_NEED_PRIMARY_PER_SLOT = 1
MULTI_NEED_ALTERNATIVES_PER_SLOT = 2
MULTI_NEED_PRODUCT_CARD_LIMIT = 10


class ChatStreamRequest(BaseModel):
    user_id: str = Field(default="demo_user")
    session_id: str = Field(default="demo_session")
    message: str
    image_id: str | None = None


class ImageUploadResponse(BaseModel):
    image_id: str
    image_url: str
    bytes: int


class EventReportRequest(BaseModel):
    user_id: str
    session_id: str = "default"
    event_type: Literal[
        "impression",
        "click",
        "view",
        "detail_view",
        "cart_add",
        "cart_remove",
        "buy",
        "favorite",
        "dismiss",
    ]
    product_id: str
    turn_id: int | None = None
    position: int | None = None
    context: dict = Field(default_factory=dict)


class EventReportResponse(BaseModel):
    ok: bool = True
    event_id: int


class CartItemResponse(BaseModel):
    product_id: str
    name: str
    category: str
    sub_category: str | None = None
    brand: str
    price: float                              # 用 float 避免 Pydantic 把 Decimal 序列化成字符串导致 Android 端崩溃
    image_url: str
    quantity: int
    rating: float = 0.0
    sku: dict = Field(default_factory=dict)   # 选中规格 (例: {"尺码": "40 码", "款型": "男款"})


class CartResponse(BaseModel):
    user_id: str
    session_id: str
    items: list[CartItemResponse] = Field(default_factory=list)
    total_quantity: int = 0
    total_price: float = 0.0


class QueryBudget(BaseModel):
    min: float | None = None
    max: float | None = None


class QueryRetrievalStrategy(BaseModel):
    use_vector: bool = True
    use_keyword: bool = True
    vector_top_k: int = 12
    keyword_top_k: int = 12
    hybrid_top_k: int = 20
    candidate_limit: int = 200
    final_top_k: int = SINGLE_RECOMMENDATION_LIMIT


class QueryPlan(BaseModel):
    intent: str = "recommendation"
    categories: list[str] = Field(default_factory=list)
    scene: list[str] = Field(default_factory=list)
    budget: QueryBudget = Field(default_factory=QueryBudget)
    preferences: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    filters: list[str] = Field(default_factory=list)
    retrieval_strategy: QueryRetrievalStrategy = Field(default_factory=QueryRetrievalStrategy)
    compare_targets: list[str] = Field(default_factory=list)
    cart_action: str | None = None
    need_clarification: bool = False
    clarification_question: str | None = None


class RewriteNeedSlot(BaseModel):
    slot_id: str
    need_type: Literal["required", "optional"] = "required"
    goal: str
    product_type: str = ""
    query: str
    soft_constraints: list[str] = Field(default_factory=list)
    exclude_terms: list[str] = Field(default_factory=list)
    min_candidates: int = 1


PlanType = Literal[
    "direct_answer",
    "clarify",
    "single_retrieval",
    "multi_retrieval",
    "image_retrieval",
]


class ProfileLookupProposal(BaseModel):
    requested: bool = False
    query: str = ""
    reason: str = ""


class IntentPlan(BaseModel):
    original_query: str = ""
    summary: str = ""
    plan_type: PlanType = "single_retrieval"
    vector_query: str = ""
    keyword_query: str = ""
    budget_min: float | None = None
    budget_max: float | None = None
    budget_scope: Literal["per_item", "total", "unknown"] = "unknown"
    need_slots: list[RewriteNeedSlot] = Field(default_factory=list)
    referenced_product_ids: list[str] = Field(default_factory=list)
    profile_lookup: ProfileLookupProposal = Field(default_factory=ProfileLookupProposal)
    plan_reason: str = ""


class ImageAttributes(BaseModel):
    available: bool = False
    category_guess: str = ""
    product_type_guess: str = ""
    colors: list[str] = Field(default_factory=list)
    style_tags: list[str] = Field(default_factory=list)
    material_guess: str = ""
    occasion_tags: list[str] = Field(default_factory=list)
    retrieval_query: str = ""
    confidence: float = 0.0
    uncertainty_note: str = ""


FallbackPlan = Literal["none", "direct_answer", "clarify", "no_product"]


class RepairHint(BaseModel):
    repairable: bool = False
    target_slot_ids: list[str] = Field(default_factory=list)
    failure_type: str = ""
    missing_terms: list[str] = Field(default_factory=list)
    avoid_terms: list[str] = Field(default_factory=list)
    reason: str = ""


class ReflectionResult(BaseModel):
    has_passed_products: bool = True
    reason: str = ""
    used_llm: bool = False
    passed_product_ids: list[str] = Field(default_factory=list)
    rejected_products: list[dict] = Field(default_factory=list)
    slot_coverage: list[dict] = Field(default_factory=list)
    combo_summary: dict = Field(default_factory=dict)
    fallback_plan: FallbackPlan = "none"
    repair_hint: RepairHint = Field(default_factory=RepairHint)


class ProductResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    product_id: str
    name: str
    category: str
    sub_category: str | None = None
    brand: str
    price: Decimal
    stock: int | None = None
    image_url: str
    description: str
    specs: dict
    ingredients_or_material: str
    suitable_for: str
    avoid_for: str
    tags: list[str]
    rating: Decimal
    sales: int | None = None
    review_summary: str
    image_caption: str
    structured_attributes: dict


class ProductCard(BaseModel):
    product_id: str
    name: str
    category: str
    sub_category: str | None = None
    brand: str
    price: float
    image_url: str
    tags: list[str]
    rating: float
    reason: str


class RecommendationCard(BaseModel):
    product_id: str
    name: str
    category: str
    sub_category: str | None = None
    brand: str
    price: float
    image_url: str
    tags: list[str] = Field(default_factory=list)
    rating: float = 0.0
    reason: str = ""
    score: float = 0.0


class RecommendationResponse(BaseModel):
    products: list[RecommendationCard] = Field(default_factory=list)
    stage: str = "cold"
    total_events: int = 0


class DecisionTrace(BaseModel):
    query_understanding: dict = Field(default_factory=dict)
    image_attributes: dict = Field(default_factory=dict)
    memory_used: list[str] = Field(default_factory=list)
    filters: list[str] = Field(default_factory=list)
    retrieval_summary: dict = Field(default_factory=dict)
    multi_need_trace: dict = Field(default_factory=dict)
    agent_path: list[dict] = Field(default_factory=list)
    planner_proposal: dict = Field(default_factory=dict)
    orchestrator_decisions: list[dict] = Field(default_factory=list)
    task: dict = Field(default_factory=dict)
    task_status: str = ""
    route: str = ""
    failure_stage: str = ""
    failure_reason: str = ""
    candidate_counts: dict = Field(default_factory=dict)
    stages: list[dict] = Field(default_factory=list)
    rerank_factors: list[str] = Field(default_factory=list)
    final_reason: str = ""
