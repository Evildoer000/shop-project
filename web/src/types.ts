export type Role = "user" | "assistant";

export type TraceLog = {
  stage: string;
  content: string;
};

export type AgentUpdate = {
  stage: string;
  title: string;
  contentDelta: string;
  done: boolean;
};

export type ProductCard = {
  product_id: string;
  name: string;
  category: string;
  sub_category?: string | null;
  brand: string;
  price: number;
  image_url: string;
  tags: string[];
  rating: number;
  reason: string;
  score?: number;
};

export type ProductDetail = {
  product_id: string;
  name: string;
  category: string;
  sub_category?: string | null;
  brand: string;
  price: string;
  stock?: number | null;
  image_url: string;
  description: string;
  specs: Record<string, unknown>;
  ingredients_or_material: string;
  suitable_for: string;
  avoid_for: string;
  tags: string[];
  rating: string;
  sales?: number | null;
  review_summary: string;
  image_caption: string;
  structured_attributes: Record<string, unknown>;
};

export type ImageUploadResponse = {
  image_id: string;
  image_url: string;
  bytes: number;
};

export type LoginRequest = {
  phone: string;
  password: string;
};

export type LoginResponse = {
  ok: boolean;
  user_id: string;
  phone: string;
  display_name: string;
};

export type EventType =
  | "impression"
  | "click"
  | "view"
  | "detail_view"
  | "cart_add"
  | "cart_remove"
  | "buy"
  | "favorite"
  | "dismiss";

export type EventReportRequest = {
  user_id: string;
  session_id?: string;
  event_type: EventType;
  product_id: string;
  turn_id?: number | null;
  position?: number | null;
  context?: Record<string, unknown>;
};

export type EventReportResponse = {
  ok: boolean;
  event_id: number;
};

export type CartItem = {
  product_id: string;
  name: string;
  category: string;
  sub_category?: string | null;
  brand: string;
  price: number;
  image_url: string;
  quantity: number;
  rating: number;
  sku: Record<string, string>;
};

export type CartResponse = {
  user_id: string;
  session_id: string;
  items: CartItem[];
  total_quantity: number;
  total_price: number;
};

export type RecommendationResponse = {
  products: ProductCard[];
  stage: string;
  total_events: number;
};

export type ProductSubCategorySummary = {
  name: string;
  count: number;
};

export type ProductCategorySummary = {
  name: string;
  count: number;
  sub_categories: ProductSubCategorySummary[];
};

export type ProductCategoriesResponse = {
  categories: ProductCategorySummary[];
};

export type ProductCatalogResponse = {
  products: ProductCard[];
  total: number;
  page: number;
  page_size: number;
};

export type ProductCatalogParams = {
  category?: string;
  subCategory?: string;
  q?: string;
  page?: number;
  pageSize?: number;
  sort?: string;
};

export type ChatSessionSummary = {
  session_id: string;
  title: string;
  last_message: string;
  turn_count: number;
  updated_at?: string | null;
};

export type ChatSessionListResponse = {
  sessions: ChatSessionSummary[];
};

export type ChatSessionTurn = {
  turn_id: number;
  user_message: string;
  assistant_message: string;
  route: string;
  product_ids: string[];
  products: ProductCard[];
  rewrite_summary: Record<string, unknown>;
  trace_summary: Record<string, unknown>;
  created_at?: string | null;
  updated_at?: string | null;
};

export type ChatSessionDetailResponse = {
  session_id: string;
  turns: ChatSessionTurn[];
};

export type DecisionTrace = {
  query_understanding?: Record<string, unknown>;
  image_attributes?: Record<string, unknown>;
  memory_used?: string[];
  filters?: string[];
  retrieval_summary?: Record<string, unknown>;
  agent_path?: Record<string, unknown>[];
  planner_proposal?: Record<string, unknown>;
  orchestrator_decisions?: Record<string, unknown>[];
  task?: Record<string, unknown>;
  task_status?: string;
  route?: string;
  failure_stage?: string;
  failure_reason?: string;
  candidate_counts?: Record<string, unknown>;
  stages?: Record<string, unknown>[];
  rerank_factors?: string[];
  final_reason?: string;
  [key: string]: unknown;
};

export type TimingSpan = {
  run_id: string;
  name: string;
  label: string;
  agent: string;
  status: string;
  started_at?: string;
  finished_at?: string;
  duration_ms: number;
  input_summary: Record<string, unknown>;
  output_summary: Record<string, unknown>;
  metrics: Record<string, unknown>;
  error_type?: string;
  error_message?: string;
};

export type TimingSummary = {
  status: string;
  termination_reason?: string;
  route?: string;
  plan_type?: string;
  total_latency_ms?: number | null;
  first_token_latency_ms?: number | null;
  completed_spans: number;
  root_span_key?: string | null;
  trace_schema_version?: string;
};

export type AgentRunSummary = {
  run_id: string;
  user_id: string;
  session_id: string;
  turn_id: string;
  query_summary: string;
  route: string;
  plan_type: string;
  status: string;
  termination_reason?: string;
  total_latency_ms?: number | null;
  first_token_latency_ms?: number | null;
  product_ids: string[];
  evaluation_summary: Record<string, unknown>;
  created_at?: string | null;
  updated_at?: string | null;
};

export type AgentRunSpan = TimingSpan & {
  span_id: number;
  span_key?: string | null;
  parent_span_key?: string | null;
  task_id: string;
  agent_id: string;
  span_type: string;
  attempt: number;
  sequence: number;
  trace_schema_version: string;
  termination_reason?: string;
};

export type AgentRunConversation = {
  turn_id: number;
  user_message: string;
  assistant_message: string;
  route: string;
  product_ids: string[];
  rewrite_summary: Record<string, unknown>;
  trace_summary: Record<string, unknown>;
  created_at?: string | null;
  updated_at?: string | null;
};

export type AgentRunListResponse = {
  runs: AgentRunSummary[];
  total: number;
  limit: number;
  offset: number;
};

export type AgentRunDetailResponse = {
  run: AgentRunSummary;
  spans: AgentRunSpan[];
  conversation?: AgentRunConversation | null;
};

export type RuleEvaluationCheck = {
  key: string;
  label: string;
  value: unknown;
  status: "passed" | "warning" | "failed" | "skipped" | string;
  description: string;
};

export type RuleEvaluationSummary = {
  checks?: RuleEvaluationCheck[];
  raw?: Record<string, unknown>;
};

export type StreamEvent =
  | { type: "trace"; stage: string; content: string; [key: string]: unknown }
  | { type: "decision_trace"; trace: DecisionTrace }
  | { type: "token"; content: string }
  | { type: "agent_update"; stage: string; title: string; content_delta: string; done: boolean }
  | { type: "timing_update"; run_id: string; span?: TimingSpan; summary?: TimingSummary; evaluation?: RuleEvaluationSummary }
  | { type: "product_cards"; products: ProductCard[] }
  | { type: "error"; message?: string; error?: string; stage?: string }
  | { type: "done" }
  | { type: string; [key: string]: unknown };

export type ChatMessage = {
  id: string;
  role: Role;
  content: string;
  attachedImageUrl?: string | null;
  isStreaming?: boolean;
  isError?: boolean;
  traceLogs?: TraceLog[];
  decisionTrace?: DecisionTrace | null;
  timingSpans?: TimingSpan[];
  timingSummary?: TimingSummary | null;
  ruleEvaluation?: RuleEvaluationSummary | null;
  agentUpdates?: AgentUpdate[];
  products?: ProductCard[];
};

export type HealthStatus = {
  status: string;
};
