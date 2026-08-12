import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  ChevronRight,
  Clipboard,
  Download,
  GitBranch,
  Loader2,
  RefreshCw,
  Search,
  XCircle,
} from "lucide-react";
import { getAgentRun, getAgentRuns } from "../lib/api";
import { useAppIdentity } from "../lib/app-state";
import type { AgentRunDetailResponse, AgentRunSpan, AgentRunSummary } from "../types";

type TreeNode = {
  span: AgentRunSpan;
  children: TreeNode[];
};

export function ObservabilityPage() {
  const { userId } = useAppIdentity();
  const [runs, setRuns] = useState<AgentRunSummary[]>([]);
  const [selectedRunId, setSelectedRunId] = useState("");
  const [detail, setDetail] = useState<AgentRunDetailResponse | null>(null);
  const [loadingRuns, setLoadingRuns] = useState(true);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState("");
  const [scope, setScope] = useState<"current" | "all">("current");
  const [notice, setNotice] = useState("");

  useEffect(() => {
    void loadRuns();
  }, [userId, status, scope]);

  useEffect(() => {
    if (!selectedRunId && runs.length > 0) {
      setSelectedRunId(runs[0].run_id);
      return;
    }
    if (selectedRunId && !runs.some((run) => run.run_id === selectedRunId)) {
      setSelectedRunId(runs[0]?.run_id ?? "");
    }
  }, [runs, selectedRunId]);

  useEffect(() => {
    if (selectedRunId) void loadDetail(selectedRunId);
    else setDetail(null);
  }, [selectedRunId, userId, scope]);

  async function loadRuns() {
    setLoadingRuns(true);
    setError("");
    try {
      const response = await getAgentRuns(scope === "current" ? userId : null, { status, limit: 100 });
      setRuns(response.runs);
    } catch (err) {
      setError(err instanceof Error ? err.message : "历史运行加载失败");
    } finally {
      setLoadingRuns(false);
    }
  }

  async function loadDetail(runId: string) {
    setLoadingDetail(true);
    setError("");
    try {
      setDetail(await getAgentRun(scope === "current" ? userId : null, runId));
    } catch (err) {
      setError(err instanceof Error ? err.message : "运行详情加载失败");
      setDetail(null);
    } finally {
      setLoadingDetail(false);
    }
  }

  const filteredRuns = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return runs;
    return runs.filter((run) =>
      [run.run_id, run.query_summary, run.route, run.plan_type, run.session_id]
        .join(" ")
        .toLowerCase()
        .includes(normalized),
    );
  }, [query, runs]);

  const tree = useMemo(() => buildSpanTree(detail?.spans ?? []), [detail?.spans]);

  async function copyJson() {
    if (!detail) return;
    await navigator.clipboard.writeText(JSON.stringify(detail, null, 2));
    showNotice("已复制本次运行的完整 JSON");
  }

  function downloadJson() {
    if (!detail) return;
    const blob = new Blob([JSON.stringify(detail, null, 2)], { type: "application/json;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${detail.run.run_id}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
    showNotice("已下载本次运行 JSON");
  }

  function showNotice(message: string) {
    setNotice(message);
    window.setTimeout(() => setNotice(""), 2200);
  }

  return (
    <div className="page-stack observability-page">
      <section className="page-heading page-heading-row">
        <div>
          <p className="eyebrow">Agent Observability</p>
          <h1>运行观测</h1>
          <p>查看历史请求的规划、检索、校验、修复和回答链路。</p>
        </div>
        <button className="button button-secondary" type="button" onClick={() => void loadRuns()} disabled={loadingRuns}>
          {loadingRuns ? <Loader2 className="spin" size={16} /> : <RefreshCw size={16} />}
          刷新列表
        </button>
      </section>

      {notice ? <div className="notice-line">{notice}</div> : null}
      {error ? <div className="error-line">{error}</div> : null}

      <section className="observability-layout">
        <aside className="panel run-list-panel">
          <div className="run-list-toolbar">
            <div className="panel-title">
              <Activity size={16} />
              <span>历史请求</span>
              <em>{runs.length}</em>
            </div>
            <div className="run-list-filters">
              <select className="select-input" value={scope} onChange={(event) => setScope(event.target.value as "current" | "all")}>
                <option value="current">当前用户</option>
                <option value="all">全部用户</option>
              </select>
              <select className="select-input" value={status} onChange={(event) => setStatus(event.target.value)}>
                <option value="">全部状态</option>
                <option value="succeeded">成功</option>
                <option value="failed">失败</option>
                <option value="degraded">降级</option>
              </select>
            </div>
          </div>

          <label className="observability-search">
            <Search size={15} />
            <input
              className="text-input"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="搜索问题、run_id、路径"
            />
          </label>

          {loadingRuns ? (
            <div className="loading-inline">
              <Loader2 className="spin" size={18} />
              正在加载历史请求...
            </div>
          ) : filteredRuns.length === 0 ? (
            <div className="empty-hint run-list-empty">没有符合条件的历史请求。</div>
          ) : (
            <div className="run-list">
              {filteredRuns.map((run) => (
                <button
                  key={run.run_id}
                  className={`run-list-item ${run.run_id === selectedRunId ? "selected" : ""}`}
                  type="button"
                  onClick={() => setSelectedRunId(run.run_id)}
                >
                  <div className="run-item-topline">
                    <StatusIcon status={run.status} />
                    <strong>{run.query_summary || "图片检索"}</strong>
                    <ChevronRight size={15} />
                  </div>
                  <div className="run-item-meta">
                    <span>{run.plan_type || "未记录计划"}</span>
                    <span>{run.route || "未记录路径"}</span>
                    <span>{formatMs(run.total_latency_ms)}</span>
                  </div>
                  <div className="run-item-id">{run.run_id}</div>
                  <div className="run-item-date">{formatDate(run.created_at)}</div>
                </button>
              ))}
            </div>
          )}
        </aside>

        <main className="observability-detail">
          {loadingDetail ? (
            <section className="panel loading-panel">
              <Loader2 className="spin" size={22} />
              正在加载运行详情...
            </section>
          ) : detail ? (
            <RunDetail detail={detail} tree={tree} onCopy={copyJson} onDownload={downloadJson} />
          ) : (
            <section className="panel empty-page">
              <GitBranch size={34} />
              <h2>选择一次历史请求</h2>
              <p>左侧选择请求后，这里会展示完整执行链路。</p>
            </section>
          )}
        </main>
      </section>
    </div>
  );
}

function RunDetail({
  detail,
  tree,
  onCopy,
  onDownload,
}: {
  detail: AgentRunDetailResponse;
  tree: TreeNode[];
  onCopy: () => void;
  onDownload: () => void;
}) {
  const [selectedSpanKey, setSelectedSpanKey] = useState<string>(
    detail.spans[0]?.span_key || String(detail.spans[0]?.span_id || ""),
  );
  const selectedSpan =
    detail.spans.find((span) => span.span_key === selectedSpanKey) ||
    detail.spans.find((span) => String(span.span_id) === selectedSpanKey) ||
    detail.spans[0];

  useEffect(() => {
    setSelectedSpanKey(detail.spans[0]?.span_key || String(detail.spans[0]?.span_id || ""));
  }, [detail.run.run_id, detail.spans]);

  return (
    <div className="page-stack">
      <section className="panel run-overview">
        <div className="run-detail-heading">
          <div>
            <p className="eyebrow">Run Detail</p>
            <h2>{detail.run.query_summary || "图片检索请求"}</h2>
            <code>{detail.run.run_id}</code>
          </div>
          <div className="run-detail-actions">
            <button className="icon-button" type="button" onClick={onCopy} title="复制运行 JSON">
              <Clipboard size={17} />
            </button>
            <button className="icon-button" type="button" onClick={onDownload} title="下载运行 JSON">
              <Download size={17} />
            </button>
          </div>
        </div>
        <div className="run-overview-grid">
          <OverviewMetric label="状态" value={statusLabel(detail.run.status)} />
          <OverviewMetric label="结束原因" value={detail.run.termination_reason || "正常完成"} />
          <OverviewMetric label="计划类型" value={detail.run.plan_type || "-"} />
          <OverviewMetric label="最终路径" value={detail.run.route || "-"} />
          <OverviewMetric label="总耗时" value={formatMs(detail.run.total_latency_ms)} />
          <OverviewMetric label="首 token" value={formatMs(detail.run.first_token_latency_ms)} />
          <OverviewMetric label="商品数" value={String(detail.run.product_ids?.length ?? 0)} />
        </div>
        <div className="run-context-grid">
          <ContextValue label="用户" value={detail.run.user_id} />
          <ContextValue label="会话" value={detail.run.session_id} />
          <ContextValue label="任务" value={detail.run.turn_id} />
          <ContextValue label="创建时间" value={formatDate(detail.run.created_at)} />
        </div>
      </section>

      {detail.conversation ? (
        <section className="panel conversation-debug">
          <div className="panel-title">
            <Search size={16} />
            <span>请求结果</span>
          </div>
          <div className="debug-message debug-user">
            <strong>用户问题</strong>
            <p>{detail.conversation.user_message}</p>
          </div>
          <div className="debug-message">
            <strong>助手回答</strong>
            <p>{detail.conversation.assistant_message || "没有保存回答文本"}</p>
          </div>
          <DebugObject title="需求规划结果" value={detail.conversation.rewrite_summary} />
          <DebugObject title="决策轨迹摘要" value={detail.conversation.trace_summary} />
        </section>
      ) : null}

      <section className="panel execution-panel">
        <div className="panel-title">
          <GitBranch size={16} />
          <span>执行树</span>
          <small>{detail.spans.length} 个节点，按父子关系展示</small>
        </div>
        <div className="execution-tree">
          {tree.length === 0 ? (
            <div className="empty-hint">这次请求没有保存 span。</div>
          ) : (
            tree.map((node) => (
              <TreeRow
                key={node.span.span_key || node.span.span_id}
                node={node}
                depth={0}
                selectedSpanKey={selectedSpanKey}
                onSelect={setSelectedSpanKey}
              />
            ))
          )}
        </div>
      </section>

      {selectedSpan ? <SpanDetail span={selectedSpan} /> : null}
    </div>
  );
}

function TreeRow({
  node,
  depth,
  selectedSpanKey,
  onSelect,
}: {
  node: TreeNode;
  depth: number;
  selectedSpanKey: string;
  onSelect: (spanKey: string) => void;
}) {
  const key = node.span.span_key || String(node.span.span_id);
  return (
    <div className="tree-node">
      <button
        type="button"
        className={`tree-row ${key === selectedSpanKey ? "selected" : ""}`}
        style={{ "--tree-depth": depth } as React.CSSProperties}
        onClick={() => onSelect(key)}
      >
        <span className="tree-branch">{depth > 0 ? "└" : "•"}</span>
        <StatusIcon status={node.span.status} />
        <span className="tree-name">{node.span.label || node.span.name}</span>
        <span className="tree-agent">{node.span.agent_id || node.span.agent || "-"}</span>
        <span className="tree-duration">{formatMs(node.span.duration_ms)}</span>
        {node.span.attempt > 1 ? <span className="tree-attempt">第 {node.span.attempt} 次</span> : null}
      </button>
      {node.children.map((child) => (
        <TreeRow
          key={child.span.span_key || child.span.span_id}
          node={child}
          depth={depth + 1}
          selectedSpanKey={selectedSpanKey}
          onSelect={onSelect}
        />
      ))}
    </div>
  );
}

function SpanDetail({ span }: { span: AgentRunSpan }) {
  return (
    <section className="panel span-detail-panel">
      <div className="span-detail-heading">
        <div>
          <div className="panel-title">
            <Activity size={16} />
            <span>节点详情</span>
          </div>
          <h3>{span.label || span.name}</h3>
        </div>
        <div className="span-detail-status">
          <StatusIcon status={span.status} />
          {statusLabel(span.status)}
        </div>
      </div>
      <div className="span-meta-grid">
        <ContextValue label="Agent" value={span.agent_id || span.agent || "-"} />
        <ContextValue label="节点类型" value={span.span_type || "-"} />
        <ContextValue label="任务 ID" value={span.task_id || "-"} />
        <ContextValue label="父节点" value={span.parent_span_key || "根节点"} />
        <ContextValue label="序号" value={String(span.sequence ?? 0)} />
        <ContextValue label="耗时" value={formatMs(span.duration_ms)} />
        <ContextValue label="开始时间" value={formatDate(span.started_at)} />
        <ContextValue label="结束原因" value={span.termination_reason || "正常完成"} />
      </div>
      {span.error_message || span.error_type ? (
        <div className="span-error-box">
          <AlertTriangle size={16} />
          <span>{span.error_type ? `${span.error_type}: ` : ""}{span.error_message || "节点执行失败"}</span>
        </div>
      ) : null}
      <DebugObject title="输入摘要" value={span.input_summary} />
      <DebugObject title="输出摘要" value={span.output_summary} />
      <DebugObject title="指标" value={span.metrics} />
    </section>
  );
}

function DebugObject({ title, value }: { title: string; value: Record<string, unknown> }) {
  const entries = Object.entries(value ?? {});
  return (
    <div className="debug-object">
      <strong>{title}</strong>
      {entries.length === 0 ? (
        <span className="empty-hint">暂无</span>
      ) : (
        <div className="debug-object-grid">
          {entries.map(([key, item]) => (
            <div className="debug-object-row" key={key}>
              <span>{fieldLabel(key)}</span>
              <code>{formatValue(item)}</code>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function OverviewMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="trace-metric">
      <div className="trace-metric-label">{label}</div>
      <div className="trace-metric-value">{value}</div>
    </div>
  );
}

function ContextValue({ label, value }: { label: string; value: string }) {
  return (
    <div className="context-value">
      <span>{label}</span>
      <code title={value}>{value || "-"}</code>
    </div>
  );
}

function StatusIcon({ status }: { status: string }) {
  if (status === "succeeded") return <CheckCircle2 className="status-icon status-success" size={15} />;
  if (status === "failed") return <XCircle className="status-icon status-failed" size={15} />;
  return <AlertTriangle className="status-icon status-warning" size={15} />;
}

function buildSpanTree(spans: AgentRunSpan[]): TreeNode[] {
  const nodes = new Map<string, TreeNode>();
  const roots: TreeNode[] = [];
  const ordered = [...spans].sort((left, right) => {
    const sequenceDiff = (left.sequence ?? 0) - (right.sequence ?? 0);
    return sequenceDiff || left.span_id - right.span_id;
  });

  for (const span of ordered) {
    const key = span.span_key || String(span.span_id);
    nodes.set(key, { span, children: [] });
  }
  for (const span of ordered) {
    const key = span.span_key || String(span.span_id);
    const node = nodes.get(key);
    if (!node) continue;
    const parent = span.parent_span_key ? nodes.get(span.parent_span_key) : null;
    if (parent && parent !== node) parent.children.push(node);
    else roots.push(node);
  }
  return roots;
}

function fieldLabel(key: string) {
  const labels: Record<string, string> = {
    plan_type: "计划类型",
    plan_reason: "计划原因",
    vector_query: "向量查询",
    keyword_query: "关键词查询",
    query: "查询",
    candidate_count: "候选数量",
    route: "路径",
    status: "状态",
    fallback_plan: "兜底策略",
    repair_attempt: "修复次数",
  };
  return labels[key] || key;
}

function formatValue(value: unknown): string {
  if (value == null) return "-";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function formatMs(value?: number | null) {
  if (value == null || !Number.isFinite(Number(value))) return "-";
  const number = Number(value);
  return number >= 1000 ? `${(number / 1000).toFixed(2)}s` : `${number.toFixed(0)}ms`;
}

function formatDate(value?: string | null) {
  if (!value) return "未知时间";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "未知时间" : date.toLocaleString("zh-CN");
}

function statusLabel(status: string) {
  return { succeeded: "成功", failed: "失败", degraded: "降级", running: "执行中" }[status] || status || "未知";
}
