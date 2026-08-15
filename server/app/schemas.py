from __future__ import annotations

from datetime import datetime
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


class LoginRequest(BaseModel):
    phone: str = Field(min_length=5, max_length=32)
    password: str = Field(min_length=1, max_length=64)


class LoginResponse(BaseModel):
    ok: bool = True
    user_id: str
    phone: str
    display_name: str


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
    intent_id: str = ""
    need_type: Literal["required", "optional"] = "required"
    goal: str
    product_type: str = ""
    query: str
    semantic_query: str = ""
    keyword_query: str = ""
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
    usage: Literal["intent_refinement", "ranking_only", "answer_personalization"] = "ranking_only"
    reason: str = ""


IntentType = Literal[
    "social_chat",
    "product_recommendation",
    "product_comparison",
    "product_qa",
    "shopping_knowledge",
    "cart_action",
]

ExecutionMode = Literal[
    "direct",
    "clarify",
    "context_evidence",
    "single_product",
    "multi_product",
]

InputModality = Literal["text", "image", "audio"]

AgentCapability = Literal[
    "intent_understanding",
    "policy_gate",
    "profile_preference",
    "clarification",
    "single_product_recommendation",
    "multi_product_bundle",
    "slot_product_retrieval",
    "commerce_research",
    "comparison",
    "knowledge_research",
    "evidence_verification",
    "bundle_optimization",
    "repair",
    "answer_generation",
    "memory_distillation",
]


class IntentQueryRewrite(BaseModel):
    semantic_query: str = ""
    keyword_query: str = ""


class IntentRouteBasis(BaseModel):
    """Planner-observed facts used by deterministic routing policy."""

    target_clarity: Literal[
        "not_applicable",
        "explicit_product",
        "context_product",
        "vague_effect_or_use",
    ] = "not_applicable"
    local_catalog_status: Literal["sufficient", "insufficient", "unknown"] = "unknown"
    external_information_need: Literal[
        "none",
        "explicit_web",
        "explicit_platform",
        "freshness_required",
        "knowledge_bridge",
    ] = "none"
    trigger_text: str = ""
    product_family: str = ""
    reason: str = ""


class IntentItem(BaseModel):
    intent_id: str
    intent_type: IntentType
    goal: str
    depends_on: list[str] = Field(default_factory=list)
    query_rewrite: IntentQueryRewrite = Field(default_factory=IntentQueryRewrite)
    referenced_product_ids: list[str] = Field(default_factory=list)
    route_basis: IntentRouteBasis = Field(default_factory=IntentRouteBasis)


class IntentConstraint(BaseModel):
    name: str
    value: str | float | int | bool | list[str]
    strength: Literal["hard", "soft"] = "hard"
    source: Literal[
        "current_query",
        "recent_turn",
        "session_summary",
        "image_inference",
        "long_term_profile",
    ] = "current_query"
    reason: str = ""


class IntentConstraintSet(BaseModel):
    budget_min: float | None = None
    budget_max: float | None = None
    budget_scope: Literal["per_item", "total", "unknown"] = "unknown"
    items: list[IntentConstraint] = Field(default_factory=list)


class ContextRequest(BaseModel):
    request_id: str
    context_type: Literal["long_term_profile"] = "long_term_profile"
    usage: Literal["intent_refinement", "ranking_only", "answer_personalization"] = "ranking_only"
    query: str = ""
    reason: str
    required: bool = False


class ClarificationProposal(BaseModel):
    required: bool = False
    blocking: bool = False
    missing_fields: list[str] = Field(default_factory=list)
    ambiguity: str = ""
    question_goal: str = ""
    reason: str = ""


class ResearchRequest(BaseModel):
    request_id: str
    intent_id: str
    mode: Literal["web_general", "marketplace", "social_content"]
    consumer_capability: Literal["knowledge_research", "comparison", "commerce_research"]
    trigger_type: Literal[
        "explicit_web_request",
        "explicit_platform_request",
        "freshness_required",
        "knowledge_bridge",
    ]
    trigger_text: str
    platforms: list[str] = Field(default_factory=list)
    query: str = ""
    freshness: Literal["any", "recent", "realtime"] = "recent"
    local_catalog_gap: str = ""
    reason: str
    required: bool = False


class AgentTaskProposal(BaseModel):
    proposal_id: str
    capability: AgentCapability
    intent_ids: list[str] = Field(default_factory=list)
    reason: str
    # The Supervisor derives core/auxiliary status from the execution contract.
    # Planner output may narrow a branch but cannot promote an auxiliary branch.
    required: bool = False
    depends_on: list[str] = Field(default_factory=list)
    optional_context_from: list[str] = Field(default_factory=list)
    expected_output_schema: str = ""


class IntentUncertainty(BaseModel):
    field: str
    description: str
    blocking: bool = False
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class IntentPlan(BaseModel):
    schema_version: str = "2.0"
    original_query: str = ""
    normalized_query: str = ""
    summary: str = ""
    primary_intent: IntentType = "product_recommendation"
    intents: list[IntentItem] = Field(default_factory=list)
    execution_mode: ExecutionMode = "single_product"
    input_modalities: list[InputModality] = Field(default_factory=lambda: ["text"])
    constraints: IntentConstraintSet = Field(default_factory=IntentConstraintSet)
    context_requests: list[ContextRequest] = Field(default_factory=list)
    clarification: ClarificationProposal = Field(default_factory=ClarificationProposal)
    research_requests: list[ResearchRequest] = Field(default_factory=list)
    agent_proposals: list[AgentTaskProposal] = Field(default_factory=list)
    uncertainties: list[IntentUncertainty] = Field(default_factory=list)

    # Retrieval projection consumed by the current retrieval workers. The Supervisor
    # will own this projection once the execution layer is connected.
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


class ProductSubCategorySummary(BaseModel):
    name: str
    count: int = 0


class ProductCategorySummary(BaseModel):
    name: str
    count: int = 0
    sub_categories: list[ProductSubCategorySummary] = Field(default_factory=list)


class ProductCategoriesResponse(BaseModel):
    categories: list[ProductCategorySummary] = Field(default_factory=list)


class ProductCatalogResponse(BaseModel):
    products: list[RecommendationCard] = Field(default_factory=list)
    total: int = 0
    page: int = 1
    page_size: int = 24


class ChatSessionSummary(BaseModel):
    session_id: str
    title: str
    last_message: str
    turn_count: int
    updated_at: datetime | None = None


class ChatSessionListResponse(BaseModel):
    sessions: list[ChatSessionSummary] = Field(default_factory=list)


class ChatSessionTurn(BaseModel):
    turn_id: int
    user_message: str
    assistant_message: str
    route: str = ""
    product_ids: list[str] = Field(default_factory=list)
    products: list[RecommendationCard] = Field(default_factory=list)
    rewrite_summary: dict = Field(default_factory=dict)
    trace_summary: dict = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ChatSessionDetailResponse(BaseModel):
    session_id: str
    turns: list[ChatSessionTurn] = Field(default_factory=list)


class AgentRunSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    run_id: str
    user_id: str
    session_id: str
    turn_id: str
    query_summary: str = ""
    route: str = ""
    plan_type: str = ""
    status: str = "running"
    termination_reason: str = ""
    total_latency_ms: float | None = None
    first_token_latency_ms: float | None = None
    product_ids: list[str] = Field(default_factory=list)
    evaluation_summary: dict = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class AgentRunSpanResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    span_id: int
    run_id: str
    span_key: str | None = None
    parent_span_key: str | None = None
    task_id: str = ""
    agent_id: str = ""
    span_type: str = "stage"
    attempt: int = 1
    sequence: int = 0
    trace_schema_version: str = "v2"
    name: str
    label: str = ""
    agent: str = ""
    status: str = "running"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: float | None = None
    input_summary: dict = Field(default_factory=dict)
    output_summary: dict = Field(default_factory=dict)
    metrics: dict = Field(default_factory=dict)
    error_type: str = ""
    error_message: str = ""
    termination_reason: str = ""


class AgentRunConversation(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    turn_id: int
    user_message: str
    assistant_message: str = ""
    route: str = ""
    product_ids: list[str] = Field(default_factory=list)
    rewrite_summary: dict = Field(default_factory=dict)
    trace_summary: dict = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class AgentRunListResponse(BaseModel):
    runs: list[AgentRunSummary] = Field(default_factory=list)
    total: int = 0
    limit: int = 50
    offset: int = 0


class AgentRunDetailResponse(BaseModel):
    run: AgentRunSummary
    spans: list[AgentRunSpanResponse] = Field(default_factory=list)
    conversation: AgentRunConversation | None = None


class DecisionTrace(BaseModel):
    trace_schema_version: str = "v2"
    run_id: str = ""
    query_understanding: dict = Field(default_factory=dict)
    image_attributes: dict = Field(default_factory=dict)
    memory_used: list[str] = Field(default_factory=list)
    filters: list[str] = Field(default_factory=list)
    retrieval_summary: dict = Field(default_factory=dict)
    multi_need_trace: dict = Field(default_factory=dict)
    agent_path: list[dict] = Field(default_factory=list)
    tool_calls: list[dict] = Field(default_factory=list)
    handoffs: list[dict] = Field(default_factory=list)
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
    failed_node_ids: list[str] = Field(default_factory=list)
    blocked_node_ids: list[str] = Field(default_factory=list)
    final_reason: str = ""
