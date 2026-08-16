import {
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  CircleDot,
  GitBranch,
  Network,
  ShieldCheck,
  XCircle,
} from "lucide-react";
import type { AgentRunSpan } from "../types";

type AgentExecutionPanelProps = {
  spans: AgentRunSpan[];
  selectedSpanKey: string;
  onSelect: (spanKey: string) => void;
};

type OwnerKind = "controller" | "policy" | "agent";

type HandoffView = {
  key: string;
  fromAgentId: string;
  toAgentId: string;
  fromTaskId: string;
  toTaskId: string;
  required: boolean;
  handoffType: string;
  status: string;
  sequence: number;
};

type ExecutionGroup = {
  key: string;
  span: AgentRunSpan;
  kind: OwnerKind;
  businessIndex: number | null;
  operations: AgentRunSpan[];
  incoming: HandoffView[];
  outgoing: HandoffView[];
};

export function AgentExecutionPanel({ spans, selectedSpanKey, onSelect }: AgentExecutionPanelProps) {
  const { groups, handoffs } = buildAgentExecution(spans);
  const businessAgentCount = groups.filter((group) => group.kind === "agent").length;
  const llmCount = spans.filter((span) => span.span_type === "llm").length;
  const toolCount = spans.filter((span) => span.span_type === "tool").length;

  return (
    <section className="panel execution-panel agent-execution-panel">
      <div className="panel-title">
        <GitBranch size={16} />
        <span>Agent 执行路线</span>
        <small>
          {businessAgentCount} 个业务 Agent · {llmCount} 次 LLM · {toolCount} 次 Tool
        </small>
      </div>

      <div className="agent-run-stats" aria-label="执行统计">
        <ExecutionStat value={businessAgentCount} label="业务 Agent" />
        <ExecutionStat value={groups.filter((group) => group.kind === "controller").length} label="控制器" />
        <ExecutionStat value={groups.filter((group) => group.kind === "policy").length} label="代码策略门" />
        <ExecutionStat value={llmCount} label="模型调用" />
        <ExecutionStat value={toolCount} label="工具调用" />
      </div>

      {handoffs.length > 0 ? (
        <div className="agent-routing-block">
          <div className="agent-section-heading">调度关系</div>
          <div className="agent-routing-list">
            {handoffs.map((handoff) => (
              <div className="agent-routing-row" key={handoff.key}>
                <span>{agentLabel(handoff.fromAgentId)}</span>
                <ArrowRight size={13} />
                <span>{agentLabel(handoff.toAgentId)}</span>
                <b className={handoff.required ? "dependency-hard" : "dependency-soft"}>
                  {handoff.required ? "硬依赖" : "软上下文"}
                </b>
                <em>{handoffTypeLabel(handoff.handoffType)}</em>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      <div className="agent-execution-list">
        {groups.length === 0 ? (
          <div className="empty-hint">这次请求没有可归组的 Agent 执行记录。</div>
        ) : (
          groups.map((group) => (
            <AgentExecutionGroup
              key={group.key}
              group={group}
              selectedSpanKey={selectedSpanKey}
              onSelect={onSelect}
            />
          ))
        )}
      </div>
    </section>
  );
}

function ExecutionStat({ value, label }: { value: number; label: string }) {
  return (
    <div className="agent-run-stat">
      <strong>{value}</strong>
      <span>{label}</span>
    </div>
  );
}

function AgentExecutionGroup({
  group,
  selectedSpanKey,
  onSelect,
}: {
  group: ExecutionGroup;
  selectedSpanKey: string;
  onSelect: (spanKey: string) => void;
}) {
  const selected = group.key === selectedSpanKey;
  const relationSummary = buildRelationSummary(group);
  return (
    <article className={`agent-execution-group agent-kind-${group.kind}`}>
      <button
        type="button"
        className={`agent-execution-header ${selected ? "selected" : ""}`}
        onClick={() => onSelect(group.key)}
      >
        <span className="agent-order-marker">
          {group.kind === "agent" ? group.businessIndex : group.kind === "controller" ? <Network size={16} /> : <ShieldCheck size={16} />}
        </span>
        <span className="agent-execution-identity">
          <span className="agent-execution-titleline">
            <strong>{agentLabel(group.span.agent_id || group.span.agent)}</strong>
            <b className={`agent-kind-badge agent-kind-badge-${group.kind}`}>{ownerKindLabel(group.kind)}</b>
          </span>
          <span>{agentRole(group.span.agent_id || group.span.agent)}</span>
        </span>
        <span className="agent-execution-result">
          <span>
            <SpanStatusIcon status={group.span.status} />
            {statusLabel(group.span.status)}
          </span>
          <code>{formatMs(group.span.duration_ms)}</code>
        </span>
      </button>

      {relationSummary.length > 0 ? (
        <div className="agent-relation-summary">
          {relationSummary.map((item) => (
            <span key={item}>{item}</span>
          ))}
        </div>
      ) : null}

      <div className="agent-internal-section">
        <div className="agent-section-heading">
          内部执行
          <span>{group.operations.length} 项</span>
        </div>
        {group.operations.length === 0 ? (
          <div className="agent-no-operations">{ownerCompletionSummary(group.span)}</div>
        ) : (
          <div className="agent-operation-list">
            {group.operations.map((operation) => {
              const key = spanKey(operation);
              return (
                <button
                  type="button"
                  className={`agent-operation-row ${key === selectedSpanKey ? "selected" : ""}`}
                  key={key}
                  onClick={() => onSelect(key)}
                >
                  <span className={`operation-kind operation-kind-${operation.span_type}`}>
                    {operationKindLabel(operation.span_type)}
                  </span>
                  <span className="agent-operation-main">
                    <strong>{operationTitle(operation)}</strong>
                    <span>{operationSummary(operation)}</span>
                  </span>
                  <span className="agent-operation-status">
                    <SpanStatusIcon status={operation.status} />
                    <code>{formatMs(operation.duration_ms)}</code>
                  </span>
                </button>
              );
            })}
          </div>
        )}
      </div>
    </article>
  );
}

function buildAgentExecution(spans: AgentRunSpan[]) {
  const ordered = [...spans].sort(compareSpans);
  const spanByKey = new Map(ordered.map((span) => [spanKey(span), span]));
  const ownerSpans = ordered.filter(isOwnerSpan);
  let businessIndex = 0;
  const groups: ExecutionGroup[] = ownerSpans.map((span) => {
    const kind = ownerKind(span);
    return {
      key: spanKey(span),
      span,
      kind,
      businessIndex: kind === "agent" ? ++businessIndex : null,
      operations: [],
      incoming: [],
      outgoing: [],
    };
  });
  const groupByKey = new Map(groups.map((group) => [group.key, group]));
  const handoffs = ordered.filter((span) => span.span_type === "handoff").map(toHandoffView);

  for (const span of ordered) {
    if (isOwnerSpan(span) || span.span_type === "handoff") continue;
    const owner = findOwnerGroup(span, spanByKey, groupByKey, groups);
    owner?.operations.push(span);
  }

  for (const handoff of handoffs) {
    const source = findGroupForEndpoint(groups, handoff.fromTaskId, handoff.fromAgentId);
    const target = findGroupForEndpoint(groups, handoff.toTaskId, handoff.toAgentId);
    source?.outgoing.push(handoff);
    target?.incoming.push(handoff);
  }

  for (const group of groups) {
    group.operations.sort(compareSpans);
    group.incoming.sort((left, right) => left.sequence - right.sequence);
    group.outgoing.sort((left, right) => left.sequence - right.sequence);
  }

  return { groups, handoffs: handoffs.sort((left, right) => left.sequence - right.sequence) };
}

function findOwnerGroup(
  span: AgentRunSpan,
  spanByKey: Map<string, AgentRunSpan>,
  groupByKey: Map<string, ExecutionGroup>,
  groups: ExecutionGroup[],
) {
  let parentKey = span.parent_span_key || "";
  const visited = new Set<string>();
  while (parentKey && !visited.has(parentKey)) {
    visited.add(parentKey);
    const direct = groupByKey.get(parentKey);
    if (direct) return direct;
    const parent = spanByKey.get(parentKey);
    parentKey = parent?.parent_span_key || "";
  }

  const sameAgent = [...groups]
    .reverse()
    .find(
      (group) =>
        group.span.agent_id === span.agent_id &&
        (group.span.sequence ?? 0) <= (span.sequence ?? 0),
    );
  return sameAgent || groups.find((group) => group.kind === "controller");
}

function findGroupForEndpoint(groups: ExecutionGroup[], taskId: string, agentId: string) {
  return (
    groups.find((group) => taskId && group.span.task_id === taskId) ||
    groups.find((group) => agentId && (group.span.agent_id || group.span.agent) === agentId)
  );
}

function toHandoffView(span: AgentRunSpan): HandoffView {
  const input = asRecord(span.input_summary);
  const output = asRecord(span.output_summary);
  return {
    key: spanKey(span),
    fromAgentId: stringValue(input.from_agent_id) || "unknown_source",
    toAgentId: stringValue(input.to_agent_id) || "unknown_target",
    fromTaskId: stringValue(input.from_task_id),
    toTaskId: stringValue(input.to_task_id),
    required: Boolean(input.required),
    handoffType: stringValue(input.handoff_type) || span.name.replace(/^handoff:/, ""),
    status: stringValue(output.handoff_status) || span.status,
    sequence: span.sequence ?? 0,
  };
}

function buildRelationSummary(group: ExecutionGroup) {
  const values: string[] = [];
  for (const handoff of group.incoming) {
    values.push(`输入：${agentLabel(handoff.fromAgentId)}（${handoff.required ? "硬依赖" : "软上下文"}）`);
  }
  for (const handoff of group.outgoing) {
    values.push(`输出：${agentLabel(handoff.toAgentId)}（${handoff.required ? "必须交付" : "可选上下文"}）`);
  }
  return [...new Set(values)];
}

function isOwnerSpan(span: AgentRunSpan) {
  return span.span_type === "run" || span.span_type === "agent" || span.span_type === "policy";
}

function ownerKind(span: AgentRunSpan): OwnerKind {
  if (span.span_type === "run") return "controller";
  if (span.span_type === "policy") return "policy";
  return "agent";
}

function ownerKindLabel(kind: OwnerKind) {
  if (kind === "controller") return "总控";
  if (kind === "policy") return "代码策略";
  return "业务 Agent";
}

function operationKindLabel(kind: string) {
  if (kind === "llm") return "LLM";
  if (kind === "tool") return "Tool";
  if (kind === "stage") return "阶段";
  return kind || "步骤";
}

function operationTitle(span: AgentRunSpan) {
  const name = span.name.toLowerCase();
  if (name.includes("intent_planner")) return "调用大模型生成结构化意图计划";
  if (name.includes("comparison_agent.compare")) return "调用大模型生成多维商品对比";
  if (name.includes("corrective_agent.review_auxiliary")) return "调用大模型审核辅助证据";
  if (name.includes("corrective_agent")) return "调用大模型校验商品证据";
  if (name.includes("answer_generator") && name.includes("direct_text")) return "调用大模型生成解释或对比回答";
  if (name.includes("answer_generator")) return "调用大模型生成商品推荐回答";
  if (name.includes("repair_agent")) return "调用大模型生成局部修复计划";
  if (name === "input_normalize") return "标准化客户端输入";
  if (name === "memory_context_load") return "读取最近对话与会话摘要";
  if (span.span_type === "tool") return toolOperationTitle(span);
  return span.label || span.name;
}

function toolOperationTitle(span: AgentRunSpan) {
  const name = span.name.toLowerCase();
  if (name.includes("product_detail")) return "读取指定商品详情";
  if (name.includes("profile_lookup")) return "查询用户长期画像";
  if (name.includes("product_search")) return "检索本地商品库";
  if (name.includes("image_understanding")) return "理解图片中的商品属性";
  if (name.includes("image_search")) return "检索相似商品图片";
  if (name.includes("commerce_search")) return "检索外部电商平台";
  if (name.includes("commerce_product_detail")) return "读取外部商品详情";
  if (name.includes("commerce_reviews")) return "读取外部商品评论";
  if (name.includes("web_search")) return "搜索联网资料";
  return span.label || span.name;
}

function operationSummary(span: AgentRunSpan) {
  const input = asRecord(span.input_summary);
  const output = asRecord(span.output_summary);
  const metrics = asRecord(span.metrics);
  const details: string[] = [];

  if (span.span_type === "llm") {
    const usage = asRecord(output.usage);
    const model = stringValue(input.model);
    const totalTokens = numberValue(usage.total_tokens ?? metrics.total_tokens);
    if (model) details.push(`模型 ${model}`);
    if (totalTokens != null) details.push(`Token ${totalTokens}`);
  } else if (span.span_type === "tool") {
    const requested = numberValue(input.product_count ?? input.limit);
    const returned = numberValue(output.count ?? output.item_count ?? output.candidate_count);
    if (requested != null) details.push(`请求 ${requested}`);
    if (returned != null) details.push(`返回 ${returned}`);
    if (metrics.networked === true) details.push("网络 Tool");
    if (metrics.networked === false) details.push("本地 Tool");
  } else if (span.name === "memory_context_load") {
    const recent = numberValue(output.recent_turn_count);
    if (recent != null) details.push(`最近对话 ${recent} 轮`);
    details.push(output.long_term_profile_loaded ? "已读取长期画像" : "未读取长期画像");
  } else if (span.name === "input_normalize") {
    const modalities = Array.isArray(output.input_modalities) ? output.input_modalities.join(" + ") : "";
    if (modalities) details.push(`输入 ${modalities}`);
  }

  if (span.error_type) details.push(span.error_type);
  return details.join(" · ") || span.termination_reason || "执行完成";
}

function ownerCompletionSummary(span: AgentRunSpan) {
  if (span.termination_reason === "scheduled") return "异步记忆任务已提交，不阻塞本轮回答。";
  if (span.span_type === "policy") return "使用确定性代码完成路线、依赖和 Tool 权限审批。";
  return span.termination_reason ? `结束原因：${span.termination_reason}` : "该 Agent 未产生额外 LLM 或 Tool 调用。";
}

function agentLabel(agentId: string) {
  const labels: Record<string, string> = {
    EcommerceOrchestrator: "EcommerceOrchestrator",
    intent_understanding_agent: "意图理解 Agent",
    supervisor_policy_gate: "Supervisor 策略门",
    profile_preference_agent: "画像偏好 Agent",
    clarification_agent: "澄清问题 Agent",
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
  return labels[agentId] || agentId || "未知执行主体";
}

function agentRole(agentId: string) {
  const roles: Record<string, string> = {
    EcommerceOrchestrator: "接收请求并编排整条 Agent 链路",
    intent_understanding_agent: "结合当前消息和会话上下文生成结构化意图提案",
    supervisor_policy_gate: "用确定性规则审批路线、依赖和工具权限",
    profile_preference_agent: "按审批读取长期偏好，并作为软上下文提供给下游",
    clarification_agent: "生成阻塞执行所需的最小澄清问题",
    single_product_recommendation_agent: "围绕一个商品目标执行检索与候选整理",
    multi_product_bundle_agent: "拆分并汇总多个商品需求槽位",
    slot_retrieval_agent: "独立检索一个商品槽位",
    commerce_research_agent: "获取淘宝、抖音电商或小红书的外部证据",
    comparison_agent: "只基于已提供商品证据执行多维对比",
    product_knowledge_agent: "读取商品详情并补充有证据的商品知识",
    evidence_verifier_agent: "校验证据能否支撑用户需求和后续回答",
    repair_agent: "针对失败节点生成局部修复计划",
    bundle_optimizer: "在多商品证据通过后完成预算与组合优化",
    answer_generator: "把已校验证据组织成最终客户端回答",
    memory_distillation_agent: "回答结束后异步更新会话摘要和记忆候选",
  };
  return roles[agentId] || "执行本节点负责的业务任务";
}

function handoffTypeLabel(value: string) {
  const labels: Record<string, string> = {
    intent_plan_proposal: "意图提案",
    approved_task: "获批任务",
    profile_context: "画像上下文",
    evidence_submission: "证据提交",
    verified_evidence: "已校验证据",
    failure_diagnostic: "失败诊断",
    repair_instruction: "修复指令",
    completed_turn: "完整轮次",
  };
  return labels[value] || value;
}

function SpanStatusIcon({ status }: { status: string }) {
  if (status === "succeeded") return <CheckCircle2 className="status-icon status-success" size={15} />;
  if (status === "failed" || status === "cancelled" || status === "timeout") {
    return <XCircle className="status-icon status-failed" size={15} />;
  }
  if (status === "running") return <CircleDot className="status-icon status-running" size={15} />;
  return <AlertTriangle className="status-icon status-warning" size={15} />;
}

function statusLabel(status: string) {
  const labels: Record<string, string> = {
    succeeded: "成功",
    failed: "失败",
    degraded: "降级",
    running: "执行中",
    cancelled: "已取消",
    timeout: "超时",
    skipped: "已跳过",
  };
  return labels[status] || status || "未知";
}

function compareSpans(left: AgentRunSpan, right: AgentRunSpan) {
  const sequenceDiff = (left.sequence ?? 0) - (right.sequence ?? 0);
  return sequenceDiff || left.span_id - right.span_id;
}

function spanKey(span: AgentRunSpan) {
  return span.span_key || String(span.span_id);
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

function stringValue(value: unknown) {
  return typeof value === "string" ? value : "";
}

function numberValue(value: unknown): number | null {
  if (value == null || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatMs(value?: number | null) {
  if (value == null || !Number.isFinite(Number(value))) return "-";
  const number = Number(value);
  return number >= 1000 ? `${(number / 1000).toFixed(2)}s` : `${number.toFixed(0)}ms`;
}
