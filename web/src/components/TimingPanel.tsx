import { Activity, AlertTriangle, CheckCircle2, Clock3, MinusCircle } from "lucide-react";
import type { RuleEvaluationCheck, RuleEvaluationSummary, TimingSpan, TimingSummary } from "../types";

type TimingPanelProps = {
  spans?: TimingSpan[];
  summary?: TimingSummary | null;
  evaluation?: RuleEvaluationSummary | null;
};

const metricLabels: Record<string, string> = {
  recall_count: "召回数量",
  candidate_count: "候选数量",
  passed_product_count: "通过数量",
  rejected_product_count: "拒绝数量",
  corrective_pass_rate: "校验通过率",
  tool_call_count: "工具调用",
  need_slot_count: "需求槽位",
  budget_extracted: "识别预算",
  first_token_latency_ms: "首 token",
  search_calls: "检索调用",
  slot_count: "槽位数",
  plan_type: "计划类型",
};

export function TimingPanel({ spans = [], summary, evaluation }: TimingPanelProps) {
  const checks = evaluation?.checks ?? [];
  if (!summary && spans.length === 0 && checks.length === 0) {
    return (
      <section className="panel timing-panel">
        <div className="panel-title">
          <Clock3 size={16} />
          <span>链路耗时</span>
        </div>
        <div className="empty-hint">还没有链路耗时。发送消息后会实时显示每个节点。</div>
      </section>
    );
  }

  return (
    <section className="panel timing-panel">
      <div className="panel-title">
        <Activity size={16} />
        <span>链路耗时</span>
      </div>

      <div className="timing-summary-grid">
        <TimingMetric label="总耗时" value={formatMs(summary?.total_latency_ms)} />
        <TimingMetric label="首 token" value={formatMs(summary?.first_token_latency_ms)} />
        <TimingMetric label="节点数" value={String(summary?.completed_spans ?? spans.length)} />
        <TimingMetric label="路径" value={summary?.route || summary?.plan_type || "-"} />
      </div>

      {checks.length > 0 ? (
        <div className="timing-block">
          <div className="trace-block-title">规则检查</div>
          <div className="rule-check-list">
            {checks.map((check) => (
              <RuleCheckRow key={check.key} check={check} />
            ))}
          </div>
        </div>
      ) : null}

      <div className="timing-block">
        <div className="trace-block-title">节点耗时</div>
        <div className="span-list">
          {spans.map((span, index) => (
            <div key={`${span.name}-${index}`} className={`span-row span-${span.status}`}>
              <div className="span-main">
                <strong>{span.label || span.name}</strong>
                <span>{span.agent || span.name}</span>
              </div>
              <div className="span-side">
                <b>{formatMs(span.duration_ms)}</b>
                <small>{statusText(span.status)}</small>
              </div>
              {Object.keys(span.metrics ?? {}).length > 0 ? (
                <div className="span-metrics">
                  {Object.entries(span.metrics).slice(0, 5).map(([key, value]) => (
                    <span key={key} className="metric-chip">
                      {metricLabels[key] ?? key}: {formatValue(value)}
                    </span>
                  ))}
                </div>
              ) : null}
              {(Object.keys(span.input_summary ?? {}).length > 0 || Object.keys(span.output_summary ?? {}).length > 0) ? (
                <details className="span-details">
                  <summary>查看输入/输出摘要</summary>
                  <KeyValueBlock title="输入" value={span.input_summary} />
                  <KeyValueBlock title="输出" value={span.output_summary} />
                </details>
              ) : null}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function TimingMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="trace-metric">
      <div className="trace-metric-label">{label}</div>
      <div className="trace-metric-value">{value}</div>
    </div>
  );
}

function RuleCheckRow({ check }: { check: RuleEvaluationCheck }) {
  const Icon = check.status === "passed" ? CheckCircle2 : check.status === "skipped" ? MinusCircle : AlertTriangle;
  return (
    <div className={`rule-check rule-${check.status}`}>
      <Icon size={15} />
      <div>
        <strong>{check.label}</strong>
        <span>{formatValue(check.value)}</span>
        <p>{check.description}</p>
      </div>
    </div>
  );
}

function KeyValueBlock({ title, value }: { title: string; value: Record<string, unknown> }) {
  const entries = Object.entries(value ?? {});
  if (entries.length === 0) return null;
  return (
    <div className="span-kv-block">
      <strong>{title}</strong>
      {entries.slice(0, 8).map(([key, item]) => (
        <div key={key}>
          <span>{metricLabels[key] ?? key}</span>
          <em>{formatValue(item)}</em>
        </div>
      ))}
    </div>
  );
}

function formatMs(value?: number | null) {
  if (value == null || !Number.isFinite(Number(value))) return "-";
  const number = Number(value);
  return number >= 1000 ? `${(number / 1000).toFixed(2)}s` : `${number.toFixed(0)}ms`;
}

function statusText(status: string) {
  if (status === "succeeded") return "完成";
  if (status === "failed") return "失败";
  if (status === "skipped") return "跳过";
  return status || "-";
}

function formatValue(value: unknown): string {
  if (value == null) return "-";
  if (typeof value === "boolean") return value ? "是" : "否";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}
