from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
]

InputModality = Literal["text", "image", "audio"]
IntentPriority = Literal["required", "optional"]

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


class PlannerStrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IntentRouteBasis(PlannerStrictModel):
    """Planner-observed facts used by deterministic routing policy."""

    target_clarity: Literal[
        "not_applicable",
        "explicit_product",
        "context_product",
        "vague_effect_or_use",
    ] = "not_applicable"
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


class IntentBudget(PlannerStrictModel):
    minimum: float | None = Field(default=None, ge=0)
    maximum: float | None = Field(default=None, ge=0)
    scope: Literal["per_item", "total", "unknown"] = "unknown"
    currency: Literal["CNY"] = "CNY"


class IntentConstraint(PlannerStrictModel):
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


class IntentConstraintSet(PlannerStrictModel):
    budget_min: float | None = None
    budget_max: float | None = None
    budget_scope: Literal["per_item", "total", "unknown"] = "unknown"
    items: list[IntentConstraint] = Field(default_factory=list)


class IntentUncertainty(PlannerStrictModel):
    field: str
    description: str
    blocking: bool = False
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class IntentProductNeed(PlannerStrictModel):
    """A business product need; retrieval-engine details are deliberately absent."""

    need_id: str
    priority: IntentPriority = "required"
    goal: str
    product_type: str = ""
    constraints: list[IntentConstraint] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)


class RecommendationPolicy(PlannerStrictModel):
    """Intent-scoped rules for assembling and selecting recommendation candidates."""

    candidate_source: Literal[
        "context_only",
        "context_plus_new",
        "new_only",
    ] = "new_only"
    reference_policy: Literal[
        "none",
        "must_include",
        "eligible",
        "comparison_only",
    ] = "none"
    requested_count: int | None = Field(default=None, ge=1, le=50)
    count_mode: Literal[
        "exact",
        "at_most",
        "at_least",
        "unknown",
    ] = "unknown"
    reason: str = ""


class IntentItem(PlannerStrictModel):
    intent_id: str
    intent_type: IntentType
    priority: IntentPriority = "required"
    goal: str
    resolved_query: str
    budget: IntentBudget | None = None
    constraints: list[IntentConstraint] = Field(default_factory=list)
    context_reference_keys: list[str] = Field(default_factory=list)
    referenced_product_ids: list[str] = Field(default_factory=list)
    recommendation_policy: RecommendationPolicy = Field(
        default_factory=RecommendationPolicy
    )
    product_needs: list[IntentProductNeed] = Field(default_factory=list)
    uncertainties: list[IntentUncertainty] = Field(default_factory=list)
    route_basis: IntentRouteBasis = Field(default_factory=IntentRouteBasis)


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
    consumer_capability: Literal["knowledge_research", "commerce_research"]
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


class AgentTaskParameters(PlannerStrictModel):
    """Typed business inputs passed to one proposed Agent task."""

    query: str = ""
    freshness: Literal["any", "recent", "realtime"] = "recent"
    platforms: list[Literal["taobao", "douyin_ec", "xiaohongshu"]] = Field(default_factory=list)
    trigger_type: Literal[
        "none",
        "explicit_web_request",
        "explicit_platform_request",
        "freshness_required",
        "knowledge_bridge",
    ] = "none"
    trigger_text: str = ""
    profile_usage: Literal[
        "intent_refinement",
        "ranking_only",
        "answer_personalization",
    ] = "ranking_only"
    missing_fields: list[str] = Field(default_factory=list)
    question_goal: str = ""
    product_need_ids: list[str] = Field(default_factory=list)
    referenced_product_ids: list[str] = Field(default_factory=list)
    comparison_dimensions: list[str] = Field(default_factory=list)
    knowledge_mode: Literal[
        "knowledge_answer",
        "product_evidence",
        "concept_bridge",
    ] = "knowledge_answer"


class KnowledgeTaskParameters(PlannerStrictModel):
    query: str
    freshness: Literal["any", "recent", "realtime"] = "recent"
    trigger_type: Literal[
        "none",
        "explicit_web_request",
        "freshness_required",
        "knowledge_bridge",
    ] = "none"
    trigger_text: str = ""
    knowledge_mode: Literal[
        "knowledge_answer",
        "product_evidence",
        "concept_bridge",
    ]


class ComparisonTaskParameters(PlannerStrictModel):
    comparison_dimensions: list[str] = Field(default_factory=list)


class RecommendationTaskParameters(PlannerStrictModel):
    query: str = ""
    product_need_ids: list[str] = Field(default_factory=list)


class BundleTaskParameters(RecommendationTaskParameters):
    pass


class CommerceTaskParameters(PlannerStrictModel):
    query: str
    freshness: Literal["any", "recent", "realtime"] = "recent"
    platforms: list[Literal["taobao", "douyin_ec", "xiaohongshu"]] = Field(
        default_factory=list
    )
    trigger_type: Literal["explicit_platform_request"] = "explicit_platform_request"
    trigger_text: str = ""


class ClarificationTaskParameters(PlannerStrictModel):
    missing_fields: list[str] = Field(default_factory=list)
    question_goal: str = ""


class ProfileTaskParameters(PlannerStrictModel):
    query: str
    profile_usage: Literal[
        "intent_refinement",
        "ranking_only",
        "answer_personalization",
    ] = "ranking_only"


PlannerTaskParameters = (
    KnowledgeTaskParameters
    | ComparisonTaskParameters
    | RecommendationTaskParameters
    | BundleTaskParameters
    | CommerceTaskParameters
    | ClarificationTaskParameters
    | ProfileTaskParameters
)


class AgentTaskProposal(PlannerStrictModel):
    task_id: str
    capability: AgentCapability
    intent_id: str = ""
    # Kept out of serialized Planner output while older runtime adapters migrate.
    intent_ids: list[str] = Field(default_factory=list, exclude=True)
    objective: str
    reason: str
    depends_on: list[str] = Field(default_factory=list)
    optional_upstream_task_ids: list[str] = Field(default_factory=list)
    parameters: PlannerTaskParameters

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, obj: Any) -> Any:
        if isinstance(obj, dict):
            obj = dict(obj)
            legacy_ids = obj.get("intent_ids")
            if not obj.get("intent_id") and isinstance(legacy_ids, list) and legacy_ids:
                obj["intent_id"] = str(legacy_ids[0])
            parameters = obj.get("parameters")
            capability = obj.get("capability")
            if isinstance(parameters, AgentTaskParameters):
                parameters = parameters.model_dump(exclude_defaults=True)
            if parameters is None:
                parameters = {}
            if capability == "knowledge_research" and isinstance(parameters, dict):
                parameters.setdefault(
                    "knowledge_mode",
                    "concept_bridge"
                    if parameters.get("trigger_type") == "knowledge_bridge"
                    else "knowledge_answer",
                )
            if isinstance(parameters, dict):
                parameter_model = {
                    "knowledge_research": KnowledgeTaskParameters,
                    "comparison": ComparisonTaskParameters,
                    "single_product_recommendation": RecommendationTaskParameters,
                    "multi_product_bundle": BundleTaskParameters,
                    "commerce_research": CommerceTaskParameters,
                    "clarification": ClarificationTaskParameters,
                    "profile_preference": ProfileTaskParameters,
                }.get(capability, AgentTaskParameters)
                obj["parameters"] = parameter_model.model_validate(parameters)
        return obj

    def model_post_init(self, __context: Any) -> None:
        if not self.intent_ids and self.intent_id:
            self.intent_ids = [self.intent_id]
        elif self.intent_ids and not self.intent_id:
            self.intent_id = self.intent_ids[0]


class IntentPlanV3(PlannerStrictModel):
    """Planner proposal consumed by the Supervisor; no global route or retrieval plan."""

    schema_version: Literal["3.0"] = "3.0"
    original_query: str = Field(default="", exclude=True)
    normalized_query: str = Field(default="", exclude=True)
    summary: str = ""
    intents: list[IntentItem] = Field(default_factory=list)
    task_proposals: list[AgentTaskProposal] = Field(default_factory=list)
    trusted_context_product_ids: list[str] = Field(default_factory=list, exclude=True)
    references_resolved: bool = Field(default=False, exclude=True)


class RetrievalIntentPlan(BaseModel):
    """Task-local projection built inside a retrieval Agent, never by the planner LLM."""

    original_query: str = ""
    normalized_query: str = ""
    summary: str = ""
    constraints: IntentConstraintSet = Field(default_factory=IntentConstraintSet)
    plan_type: PlanType = "single_retrieval"
    vector_query: str = ""
    keyword_query: str = ""
    budget_min: float | None = None
    budget_max: float | None = None
    budget_scope: Literal["per_item", "total", "unknown"] = "unknown"
    need_slots: list[RewriteNeedSlot] = Field(default_factory=list)
    referenced_product_ids: list[str] = Field(default_factory=list)
    recommendation_policy: RecommendationPolicy = Field(
        default_factory=RecommendationPolicy
    )
    profile_lookup: ProfileLookupProposal = Field(default_factory=ProfileLookupProposal)
    plan_reason: str = ""


# Existing retrieval workers use this import name. It now explicitly points to
# the task-local projection rather than the Supervisor's V3 planning contract.
IntentPlan = RetrievalIntentPlan


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
    verified_product_ids: list[str] = Field(default_factory=list)
    passed_product_ids: list[str] = Field(default_factory=list)
    selected_product_ids: list[str] = Field(default_factory=list)
    quantity_status: Literal[
        "met",
        "partial",
        "unavailable",
        "constraint_conflict",
        "not_applicable",
    ] = "not_applicable"
    reference_decisions: list[dict] = Field(default_factory=list)
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
