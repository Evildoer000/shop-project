import { ArrowRight, ChevronDown, GitBranch, ListChecks, Layers3, Wrench } from "lucide-react";
import type { AgentHandoff, AgentToolCallTrace, DecisionTrace } from "../types";

type TracePanelProps = {
  trace?: DecisionTrace | null;
  handoffs?: AgentHandoff[];
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
  task_status: "task_status（任务状态）",
  handoff_count: "handoff_count（交接数量）",
  tool_call_count: "tool_call_count（工具调用数量）",
  input_refs: "input_refs（软上下文来源）",
  depends_on: "depends_on（硬依赖来源）",
  required: "required（是否必须成功）",
  skipped_reason: "skipped_reason（跳过原因）",
};

export function TracePanel({ trace, handoffs: liveHandoffs, title = "决策轨迹", compact = false }: TracePanelProps) {
  const handoffs = trace?.handoffs?.length ? trace.handoffs : liveHandoffs ?? [];
  if (!trace && handoffs.length === 0) {
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

  const resolvedTrace = trace ?? {};
  const route = stringValue(resolvedTrace.route);
  const reason = stringValue(resolvedTrace.final_reason ?? resolvedTrace.failure_reason);
  const retrieval = asRecord(resolvedTrace.retrieval_summary);
  const planner = asRecord(resolvedTrace.planner_proposal);
  const candidateCounts = asRecord(resolvedTrace.candidate_counts);
  const stageCount = Array.isArray(resolvedTrace.stages) ? resolvedTrace.stages.length : 0;
  const task = asRecord(resolvedTrace.task);
  const agentPath = Array.isArray(resolvedTrace.agent_path) ? resolvedTrace.agent_path : [];
  const toolCalls = Array.isArray(resolvedTrace.tool_calls) ? resolvedTrace.tool_calls : [];

  return (
    <section className={`panel ${compact ? "panel-compact" : ""}`}>
      <div className="panel-title">
        <Layers3 size={16} />
        <span>{title}</span>
      </div>

      <div className="trace-summary-grid">
        <TraceMetric label={fieldLabels.route} value={route || "执行中"} />
        <TraceMetric label={fieldLabels.stages} value={String(stageCount)} />
        <TraceMetric label={fieldLabels.reason} value={reason || stringValue(resolvedTrace.task_status) || "-"} />
      </div>

      {handoffs.length > 0 ? (
        <div className="trace-block">
          <div className="trace-block-title">Agent 交接（Handoff）</div>
          <HandoffList handoffs={handoffs} />
        </div>
      ) : null}

      {agentPath.length > 0 ? (
        <div className="trace-block">
          <div className="trace-block-title trace-block-title-icon">
            <GitBranch size={14} />
            <span>执行节点（Agent Path）</span>
          </div>
          <AgentPathList nodes={agentPath} />
        </div>
      ) : null}

      {toolCalls.length > 0 ? (
        <div className="trace-block">
          <div className="trace-block-title trace-block-title-icon">
            <Wrench size={14} />
            <span>工具调用（Tool Calls）</span>
          </div>
          <ToolCallList calls={toolCalls} />
        </div>
      ) : null}

      {Object.keys(task).length > 0 ? (
        <div className="trace-block">
          <div className="trace-block-title">任务图摘要</div>
          <TracePairs entries={task} />
        </div>
      ) : null}

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
        <pre>{JSON.stringify(trace ?? { handoffs }, null, 2)}</pre>
      </details>
    </section>
  );
}

function AgentPathList({ nodes }: { nodes: Record<string, unknown>[] }) {
  return (
    <div className="trace-node-list">
      {nodes.map((node, index) => {
        const nodeId = stringValue(node.node_id ?? node.node) || `node-${index + 1}`;
        const agentId = stringValue(node.agent_id ?? node.node) || "unknown_agent";
        const capability = stringValue(node.capability ?? node.role);
        const status = stringValue(node.status) || "succeeded";
        const dependencies = stringList(node.depends_on);
        const inputs = stringList(node.input_refs);
        const attempt = numberValue(node.attempt, 1);
        return (
          <div className="trace-node-row" key={`${nodeId}-${index}`}>
            <div className="trace-node-index">{index + 1}</div>
            <div className="trace-node-main">
              <div className="trace-node-heading">
                <strong>{agentLabel(agentId)}</strong>
                <span className={`trace-status trace-status-${status}`}>{statusLabel(status)}</span>
              </div>
              <code>{capability || nodeId}</code>
              <div className="trace-node-meta">
                {dependencies.length > 0 ? <span>硬依赖：{dependencies.join("、")}</span> : null}
                {inputs.length > 0 ? <span>软上下文：{inputs.join("、")}</span> : null}
                {attempt > 1 ? <span>第 {attempt} 次执行</span> : null}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function ToolCallList({ calls }: { calls: AgentToolCallTrace[] }) {
  return (
    <div className="tool-call-list">
      {calls.map((call, index) => {
        const result = asRecord(call.result);
        const status = call.status || "unknown";
        return (
          <div className="tool-call-row" key={call.call_id || `${call.node_id || "tool"}-${index}`}>
            <div className="tool-call-heading">
              <div>
                <strong>{toolLabel(call.tool)}</strong>
                <code>{call.operation}</code>
              </div>
              <span className={`trace-status trace-status-${status}`}>{statusLabel(status)}</span>
            </div>
            <div className="tool-call-meta">
              <span>调用方：{agentLabel(call.agent_id || "unknown_agent")}</span>
              {call.attempt && call.attempt > 1 ? <span>第 {call.attempt} 次</span> : null}
              {result.candidate_count != null ? <span>候选：{formatValue(result.candidate_count)}</span> : null}
              {result.item_count != null ? <span>结果：{formatValue(result.item_count)}</span> : null}
              {result.latency_ms != null ? <span>耗时：{formatValue(result.latency_ms)} ms</span> : null}
              {call.error_type ? <span className="tool-call-error">{call.error_type}</span> : null}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function HandoffList({ handoffs }: { handoffs: AgentHandoff[] }) {
  return (
    <div className="handoff-list">
      {handoffs.map((handoff) => (
        <div className={`handoff-row handoff-${handoff.status}`} key={handoff.handoff_id}>
          <div className="handoff-route">
            <strong>{agentLabel(handoff.from_agent_id)}</strong>
            <ArrowRight size={14} />
            <strong>{agentLabel(handoff.to_agent_id)}</strong>
          </div>
          <div className="handoff-meta">
            <span>{handoff.required ? "硬依赖" : "软上下文"}</span>
            <span>{handoffTypeLabel(handoff.handoff_type)}</span>
            <b>{handoffStatusLabel(handoff.status)}</b>
          </div>
        </div>
      ))}
    </div>
  );
}

function agentLabel(agentId: string) {
  const labels: Record<string, string> = {
    intent_understanding_agent: "意图理解 Agent",
    supervisor_policy_gate: "Supervisor 策略门",
    profile_preference_agent: "画像偏好 Agent",
    single_product_recommendation_agent: "单商品推荐 Agent",
    multi_product_bundle_agent: "多商品组合 Agent",
    slot_retrieval_agent: "槽位检索 Agent",
    commerce_research_agent: "平台研究 Agent",
    comparison_agent: "商品对比 Agent",
    product_knowledge_agent: "商品信息与知识补充 Agent",
    evidence_verifier_agent: "证据校验 Agent",
    repair_agent: "修复 Agent",
    bundle_optimizer: "组合优化 Agent",
    answer_generator: "答案生成 Agent",
    memory_distillation_agent: "记忆蒸馏 Agent",
  };
  return labels[agentId] ?? agentId;
}

function handoffTypeLabel(value: string) {
  const labels: Record<string, string> = {
    intent_plan_proposal: "意图计划提案",
    approved_task: "获批任务",
    profile_context: "画像上下文",
    evidence_submission: "证据提交",
    verified_evidence: "已校验证据",
    failure_diagnostic: "失败诊断",
    repair_instruction: "修复指令",
    answer_context: "回答上下文",
    completed_turn: "完整轮次",
    policy_decision: "策略决策",
    approved_retrieval_query: "获批检索词",
    task_result: "任务结果",
  };
  return labels[value] ?? value;
}

function handoffStatusLabel(value: string) {
  const labels: Record<string, string> = {
    planned: "等待上游",
    ready: "可交接",
    accepted: "已接收",
    consumed: "已消费",
    unavailable: "软依赖不可用",
    blocked: "硬依赖阻断",
    superseded: "已被重试替代",
  };
  return labels[value] ?? value;
}

function statusLabel(value: string) {
  const labels: Record<string, string> = {
    pending: "等待",
    running: "执行中",
    succeeded: "成功",
    failed: "失败",
    timeout: "超时",
    cancelled: "已取消",
    skipped: "已跳过",
    degraded: "降级",
    unknown: "未知",
  };
  return labels[value] ?? value;
}

function toolLabel(value: string) {
  const labels: Record<string, string> = {
    product_search: "商品检索",
    image_search: "图片检索",
    image_understanding: "图片理解",
    product_detail: "商品详情",
    profile_lookup: "长期画像查询",
    commerce_search: "外部平台检索",
    commerce_product_detail: "外部商品详情",
    commerce_reviews: "外部评论查询",
    web_search: "联网搜索",
  };
  return labels[value] ?? value;
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

function stringList(value: unknown): string[] {
  return Array.isArray(value) ? value.map(stringValue).filter(Boolean) : [];
}

function numberValue(value: unknown, fallback: number) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}
