import { ChevronDown, ListChecks, Layers3 } from "lucide-react";
import type { DecisionTrace } from "../types";

type TracePanelProps = {
  trace?: DecisionTrace | null;
  title?: string;
  compact?: boolean;
};

const fieldLabels: Record<string, string> = {
  route: "route（最终路径）",
  stages: "stages（阶段数）",
  reason: "reason（最终原因）",
  retrieval_summary: "retrieval_summary（召回与校验摘要）",
  planner_proposal: "planner_proposal（意图规划结果）",
  candidate_counts: "candidate_counts（候选数量统计）",
  plan_type: "plan_type（计划类型）",
  plan_reason: "plan_reason（计划原因）",
  vector_query: "vector_query（向量检索语义 query）",
  keyword_query: "keyword_query（关键词检索 query）",
  budget_min: "budget_min（最低预算）",
  budget_max: "budget_max（最高预算）",
  budget_scope: "budget_scope（预算范围）",
  need_slots: "need_slots（拆分后的需求槽位）",
  referenced_product_ids: "referenced_product_ids（引用历史商品）",
  passed_product_ids: "passed_product_ids（校验通过商品）",
  rejected_products: "rejected_products（被拒绝商品）",
  fallback_plan: "fallback_plan（兜底策略）",
  reflection_result: "reflection_result（证据校验结果）",
  after_rerank: "after_rerank（重排后数量）",
  after_corrective: "after_corrective（校验后数量）",
  route_reason: "route_reason（路径原因）",
};

export function TracePanel({ trace, title = "决策轨迹", compact = false }: TracePanelProps) {
  if (!trace) {
    return (
      <section className="panel">
        <div className="panel-title">
          <ListChecks size={16} />
          <span>{title}</span>
        </div>
        <div className="empty-hint">还没有决策轨迹。</div>
      </section>
    );
  }

  const route = stringValue(trace.route);
  const reason = stringValue(trace.final_reason ?? trace.failure_reason);
  const retrieval = asRecord(trace.retrieval_summary);
  const planner = asRecord(trace.planner_proposal);
  const candidateCounts = asRecord(trace.candidate_counts);
  const stageCount = Array.isArray(trace.stages) ? trace.stages.length : 0;

  return (
    <section className={`panel ${compact ? "panel-compact" : ""}`}>
      <div className="panel-title">
        <Layers3 size={16} />
        <span>{title}</span>
      </div>

      <div className="trace-summary-grid">
        <TraceMetric label={fieldLabels.route} value={route || "-"} />
        <TraceMetric label={fieldLabels.stages} value={String(stageCount)} />
        <TraceMetric label={fieldLabels.reason} value={reason || "-"} />
      </div>

      <div className="trace-block">
        <div className="trace-block-title">{fieldLabels.retrieval_summary}</div>
        <TracePairs entries={retrieval} />
      </div>

      <div className="trace-block">
        <div className="trace-block-title">{fieldLabels.planner_proposal}</div>
        <TracePairs entries={planner} />
      </div>

      <div className="trace-block">
        <div className="trace-block-title">{fieldLabels.candidate_counts}</div>
        <TracePairs entries={candidateCounts} />
      </div>

      <details className="trace-details">
        <summary>
          <span>原始 JSON</span>
          <ChevronDown size={14} />
        </summary>
        <pre>{JSON.stringify(trace, null, 2)}</pre>
      </details>
    </section>
  );
}

function TraceMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="trace-metric">
      <div className="trace-metric-label">{label}</div>
      <div className="trace-metric-value">{value}</div>
    </div>
  );
}

function TracePairs({ entries }: { entries: Record<string, unknown> }) {
  const pairs = Object.entries(entries);
  if (pairs.length === 0) {
    return <div className="empty-hint">暂无</div>;
  }
  return (
    <div className="kv-grid">
      {pairs.map(([key, value]) => (
        <div key={key} className="kv-row">
          <div className="kv-key">{fieldLabels[key] ?? key}</div>
          <div className="kv-value">{formatValue(value)}</div>
        </div>
      ))}
    </div>
  );
}

function formatValue(value: unknown): string {
  if (value == null) return "-";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function asRecord(value: unknown): Record<string, unknown> {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return {};
}

function stringValue(value: unknown) {
  return typeof value === "string" ? value : "";
}
