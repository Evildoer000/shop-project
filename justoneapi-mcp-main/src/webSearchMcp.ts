import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";

type McpConnection = {
  client: Client;
  transport: StreamableHTTPClientTransport;
};

type WebSearchInput = {
  query: string;
  count?: number;
  freshness?: string;
};

type WebSearchClientOptions = {
  endpoint: string;
  token: string;
  toolName?: string;
  timeoutMs?: number;
};

type NormalizationOptions = {
  query: string;
  count: number;
  toolName: string;
  latencyMs?: number;
};

const DATE_RANGE_PATTERN = /^\d{4}-\d{2}-\d{2}to\d{4}-\d{2}-\d{2}$/;
const PROVIDER_FRESHNESS = new Set(["pd", "pw", "pm", "py"]);
const ITEM_LIST_KEYS = [
  "items",
  "results",
  "references",
  "web_pages",
  "webPages",
  "pages",
  "documents",
  "list",
];
const NESTED_RESULT_KEYS = [
  "data",
  "result",
  "output",
  "payload",
  "response",
  "search_results",
  "structuredContent",
];

export class BaiduWebSearchMcpClient {
  private readonly endpoint: string;
  private readonly token: string;
  private readonly toolName: string;
  private readonly timeoutMs: number;
  private connection: McpConnection | null = null;
  private connecting: Promise<McpConnection> | null = null;

  constructor(options: WebSearchClientOptions) {
    this.endpoint = options.endpoint.trim();
    this.token = options.token.trim();
    this.toolName = options.toolName?.trim() || "web_search";
    this.timeoutMs = clamp(options.timeoutMs ?? 30_000, 1_000, 120_000);
  }

  get configured(): boolean {
    return Boolean(this.endpoint && this.token);
  }

  get configuredToolName(): string {
    return this.toolName;
  }

  async search(input: WebSearchInput): Promise<Record<string, unknown>> {
    const query = String(input.query || "").trim().slice(0, 500);
    const count = clamp(input.count ?? 8, 1, 20);
    if (!query) {
      return unavailable("empty_query", "搜索词为空。", { query, count });
    }
    if (!this.configured) {
      return unavailable("web_search_not_configured", "百度联网搜索 MCP 尚未配置。", {
        query,
        count,
      });
    }

    const started = performance.now();
    let connection: McpConnection | null = null;
    try {
      connection = await this.getConnection();
      const freshness = mapWebSearchFreshness(input.freshness);
      const args: Record<string, unknown> = { query, count };
      if (freshness) args.freshness = freshness;

      const result = await connection.client.callTool(
        { name: this.toolName, arguments: args },
        undefined,
        { timeout: this.timeoutMs, maxTotalTimeout: this.timeoutMs },
      );
      return normalizeMcpWebSearchResult(result, {
        query,
        count,
        toolName: this.toolName,
        latencyMs: roundedMs(started),
      });
    } catch (error) {
      if (connection) await this.discardConnection(connection);
      return unavailable("web_search_mcp_failed", safeErrorMessage(error, this.token), {
        query,
        count,
        tool_name: this.toolName,
        latency_ms: roundedMs(started),
      });
    }
  }

  private async getConnection(): Promise<McpConnection> {
    if (this.connection) return this.connection;
    if (this.connecting) return this.connecting;

    this.connecting = this.openConnection();
    try {
      this.connection = await this.connecting;
      return this.connection;
    } finally {
      this.connecting = null;
    }
  }

  private async openConnection(): Promise<McpConnection> {
    const client = new Client({ name: "shop-agent-web-search-bridge", version: "1.0.0" });
    const transport = new StreamableHTTPClientTransport(new URL(this.endpoint), {
      requestInit: {
        headers: {
          Authorization: `Bearer ${this.token}`,
        },
      },
      reconnectionOptions: {
        maxReconnectionDelay: 5_000,
        initialReconnectionDelay: 500,
        reconnectionDelayGrowFactor: 2,
        maxRetries: 1,
      },
    });
    try {
      await withTimeout(client.connect(transport), this.timeoutMs, "MCP initialization timed out");
      return { client, transport };
    } catch (error) {
      await transport.close().catch(() => undefined);
      throw error;
    }
  }

  private async discardConnection(connection: McpConnection): Promise<void> {
    if (this.connection === connection) this.connection = null;
    await connection.client.close().catch(() => undefined);
  }
}

export function mapWebSearchFreshness(value: unknown): string | undefined {
  const freshness = String(value || "").trim().toLowerCase();
  if (!freshness || freshness === "any") return undefined;
  if (freshness === "recent") return "pm";
  if (freshness === "realtime") return "pd";
  if (PROVIDER_FRESHNESS.has(freshness) || DATE_RANGE_PATTERN.test(freshness)) return freshness;
  return undefined;
}

export function normalizeMcpWebSearchResult(
  result: unknown,
  options: NormalizationOptions,
): Record<string, unknown> {
  const root = asRecord(result);
  const payloads: unknown[] = [];
  if (root?.structuredContent != null) payloads.push(root.structuredContent);

  const content = Array.isArray(root?.content) ? root.content : [];
  for (const block of content) {
    const item = asRecord(block);
    if (!item) continue;
    if (item.type === "text" && typeof item.text === "string") {
      const parsed = parseJsonText(item.text);
      if (parsed != null) payloads.push(parsed);
    } else if (item.type === "resource_link") {
      payloads.push({ items: [item] });
    } else if (item.type === "resource") {
      const resource = asRecord(item.resource);
      if (resource?.text && typeof resource.text === "string") {
        const parsed = parseJsonText(resource.text);
        if (parsed != null) payloads.push(parsed);
      }
    }
  }

  const normalizedItems: Record<string, unknown>[] = [];
  const seen = new Set<string>();
  for (const payload of payloads) {
    for (const candidate of findItems(payload)) {
      const normalized = normalizeItem(candidate);
      if (!normalized) continue;
      const key = String(normalized.url || normalized.source_id || "");
      if (!key || seen.has(key)) continue;
      seen.add(key);
      normalizedItems.push(normalized);
      if (normalizedItems.length >= options.count) break;
    }
    if (normalizedItems.length >= options.count) break;
  }

  if (root?.isError === true) {
    return unavailable("web_search_mcp_tool_error", textContentMessage(content), {
      query: options.query,
      count: options.count,
      tool_name: options.toolName,
      latency_ms: options.latencyMs,
    });
  }
  if (!normalizedItems.length) {
    return unavailable("web_search_empty_result", "百度搜索未返回可解析的网页结果。", {
      query: options.query,
      count: options.count,
      tool_name: options.toolName,
      latency_ms: options.latencyMs,
    });
  }

  return {
    available: true,
    provider: "baidu_web_search_mcp",
    query: options.query,
    count: normalizedItems.length,
    requested_count: options.count,
    tool_name: options.toolName,
    items: normalizedItems,
    ...(options.latencyMs == null ? {} : { latency_ms: options.latencyMs }),
  };
}

function findItems(value: unknown, depth = 0): Record<string, unknown>[] {
  if (depth > 7 || value == null) return [];
  if (Array.isArray(value)) return value.filter(isRecord);
  const record = asRecord(value);
  if (!record) return [];

  for (const key of ITEM_LIST_KEYS) {
    const nested = record[key];
    if (Array.isArray(nested)) return nested.filter(isRecord);
  }
  for (const key of NESTED_RESULT_KEYS) {
    const raw = record[key];
    const parsed = typeof raw === "string" ? parseJsonText(raw) : raw;
    const nested = findItems(parsed, depth + 1);
    if (nested.length) return nested;
  }
  return looksLikeWebResult(record) ? [record] : [];
}

function normalizeItem(item: Record<string, unknown>): Record<string, unknown> | null {
  const url = firstString(item, ["url", "link", "href", "uri", "source_url", "sourceUrl", "page_url"]);
  const sourceId = url || firstString(item, ["source_id", "id", "doc_id", "document_id"]);
  const title = firstString(item, ["title", "name", "page_title"]);
  const snippet = firstString(item, ["snippet", "summary", "description", "content", "abstract"]);
  if (!sourceId || (!title && !snippet)) return null;

  const publishedAt = firstString(item, ["published_at", "publish_time", "date", "time"]);
  const normalized: Record<string, unknown> = {
    title: title || sourceId,
    snippet,
    source_id: sourceId,
  };
  if (url) normalized.url = url;
  if (publishedAt) normalized.published_at = publishedAt;
  return normalized;
}

function looksLikeWebResult(value: Record<string, unknown>): boolean {
  return Boolean(
    firstString(value, ["url", "link", "href", "uri", "source_url", "source_id", "id"]),
  );
}

function firstString(value: Record<string, unknown>, keys: string[]): string {
  for (const key of keys) {
    const candidate = value[key];
    if (typeof candidate === "string" && candidate.trim()) return candidate.trim();
  }
  return "";
}

function parseJsonText(value: string): unknown | null {
  const trimmed = value.trim();
  const withoutFence = trimmed
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/\s*```$/, "")
    .trim();
  if (!withoutFence) return null;
  try {
    return JSON.parse(withoutFence);
  } catch {
    return null;
  }
}

function textContentMessage(content: unknown[]): string {
  const messages = content
    .map(asRecord)
    .filter((item): item is Record<string, unknown> => Boolean(item))
    .filter((item) => item.type === "text" && typeof item.text === "string")
    .map((item) => String(item.text).trim())
    .filter(Boolean);
  return messages.join(" ").slice(0, 500) || "百度搜索 MCP 返回了工具错误。";
}

function unavailable(
  code: string,
  message: string,
  details: Record<string, unknown>,
): Record<string, unknown> {
  return {
    available: false,
    provider: "baidu_web_search_mcp",
    items: [],
    ...details,
    error: { code, message },
  };
}

function safeErrorMessage(error: unknown, token: string): string {
  const raw = error instanceof Error ? error.message : String(error || "web_search_mcp_failed");
  return (token ? raw.replaceAll(token, "[redacted]") : raw).slice(0, 500);
}

function clamp(value: unknown, min: number, max: number): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return min;
  return Math.max(min, Math.min(max, Math.trunc(parsed)));
}

function roundedMs(started: number): number {
  return Math.round((performance.now() - started) * 100) / 100;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return isRecord(value) ? value : null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

async function withTimeout<T>(promise: Promise<T>, timeoutMs: number, message: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error(message)), timeoutMs);
  });
  try {
    return await Promise.race([promise, timeout]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}
