import { createServer, IncomingMessage, ServerResponse } from "node:http";
import { CatalogManager } from "./catalog/manager.js";
import { FileCatalogStore } from "./node/fileCatalogStore.js";
import { callEndpoint } from "./tools/callEndpoint.js";
import { loadNodeConfig } from "./config.js";
import { bundledCatalog } from "./generated/bundledCatalog.js";
import { stderrLogger } from "./common/logger.js";
import { RuntimeContext } from "./common/runtime.js";
import { BaiduWebSearchMcpClient } from "./webSearchMcp.js";

// This bridge deliberately exposes high-level operations only.  The native MCP
// server still owns discovery, schemas and generic endpoint calling, but no
// business Agent can submit an arbitrary endpoint id through this HTTP API.
const ALLOWED_ENDPOINTS = {
  search: {
    taobao: "taobao.search_item_list_v1",
    douyin_ec: "douyin_ec.search_item_list_v1",
    xiaohongshu: "xiaohongshu.search_note_v2",
  },
  detail: {
    taobao: "taobao.get_item_detail_v4",
    douyin_ec: "douyin_ec.get_item_detail_v2",
    xiaohongshu: "xiaohongshu.get_note_detail_v6",
  },
  reviews: {
    taobao: "taobao.get_item_comment_v3",
    douyin_ec: "douyin_ec.get_item_comments_v1",
    xiaohongshu: "xiaohongshu.get_note_comment_v2",
  },
} as const;

type Platform = keyof typeof ALLOWED_ENDPOINTS.search;
type Operation = keyof typeof ALLOWED_ENDPOINTS;

const config = loadNodeConfig();
const catalogManager = new CatalogManager(
  new FileCatalogStore(config.catalogCacheDir),
  bundledCatalog,
  config
);
const runtime: RuntimeContext = {
  transport: "stdio",
  config,
  catalogManager,
  logger: stderrLogger,
  getToken: () => process.env.JUSTONEAPI_TOKEN?.trim() || null,
  isAdmin: () => true,
};

const webSearchClient = new BaiduWebSearchMcpClient({
  endpoint: process.env.BAIDU_WEB_SEARCH_MCP_URL || "",
  token: process.env.BAIDU_WEB_SEARCH_MCP_TOKEN || "",
  toolName: process.env.BAIDU_WEB_SEARCH_MCP_TOOL_NAME || "web_search",
  timeoutMs: Number(process.env.BAIDU_WEB_SEARCH_MCP_TIMEOUT_MS || 30_000),
});

const port = Number(process.env.BRIDGE_PORT || 8787);

createServer(async (request, response) => {
  try {
    if (request.method === "GET" && request.url === "/health") {
      sendJson(response, 200, {
        ok: true,
        token_configured: Boolean(runtime.getToken()),
        allowed_platforms: Object.keys(ALLOWED_ENDPOINTS.search),
        web_search_configured: webSearchClient.configured,
        web_search_tool: webSearchClient.configuredToolName,
      });
      return;
    }
    if (request.method === "POST" && request.url === "/v1/web/search") {
      const input = await readJson(request);
      const result = await webSearchClient.search({
        query: String(input.query || ""),
        count: clamp(input.count ?? input.limit ?? 8, 1, 20),
        freshness: String(input.freshness || ""),
      });
      sendJson(response, 200, {
        ...result,
        operation: "search",
        collected_at: new Date().toISOString(),
      });
      return;
    }
    if (request.method !== "POST" || !request.url?.startsWith("/v1/commerce/")) {
      sendJson(response, 404, { ok: false, error: "not_found" });
      return;
    }
    const operation = request.url.slice("/v1/commerce/".length) as Operation;
    if (!(operation in ALLOWED_ENDPOINTS)) {
      sendJson(response, 404, { available: false, error: { code: "operation_not_allowed" } });
      return;
    }
    const input = await readJson(request);
    const platform = String(input.platform || "").trim().toLowerCase() as Platform;
    if (!(platform in ALLOWED_ENDPOINTS[operation])) {
      sendJson(response, 400, {
        available: false,
        error: { code: "platform_not_allowed", platform },
      });
      return;
    }

    const endpointId = ALLOWED_ENDPOINTS[operation][platform];
    const params = paramsFor(operation, platform, input);
    const result = await callEndpoint(
      { endpoint_id: endpointId, params, max_items: clamp(input.limit ?? 20, 1, 100) },
      runtime
    );
    const normalizedResult = result as EndpointResult;
    const success = isSuccessfulResult(normalizedResult);
    sendJson(response, 200, {
      available: success,
      operation,
      platform,
      endpoint_id: endpointId,
      collected_at: new Date().toISOString(),
      items: success ? extractItems(normalizedResult.data) : [],
      ...(success
        ? {}
        : {
            error: normalizedResult.error ?? {
              code: "upstream_error",
              message: normalizedResult.message,
            },
          }),
      result,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "commerce_bridge_error";
    sendJson(response, 200, {
      available: false,
      items: [],
      error: { code: "commerce_bridge_error", message },
    });
  }
}).listen(port, "0.0.0.0", () => {
  stderrLogger.info("commerce_bridge_started", {
    port,
    token_configured: Boolean(runtime.getToken()),
    allowed_platforms: Object.keys(ALLOWED_ENDPOINTS.search),
    web_search_configured: webSearchClient.configured,
    web_search_tool: webSearchClient.configuredToolName,
  });
});

function paramsFor(operation: Operation, platform: Platform, input: Record<string, unknown>) {
  if (operation === "search") {
    const query = String(input.query || "").trim();
    if (!query) throw new Error("query is required");
    if (platform === "taobao") {
      return {
        keyword: query,
        page: clamp(input.page ?? 1, 1, 100),
        ...(input.min_price != null ? { start_price: String(input.min_price) } : {}),
        ...(input.max_price != null ? { end_price: String(input.max_price) } : {}),
      };
    }
    if (platform === "douyin_ec") {
      return { keyword: query, page: String(clamp(input.page ?? 1, 1, 100)) };
    }
    return { keyword: query, page: clamp(input.page ?? 1, 1, 100) };
  }

  const productId = String(input.product_id || input.note_id || "").trim();
  if (!productId) throw new Error("product_id is required");
  if (platform === "xiaohongshu") {
    return {
      note_id: productId,
      ...(operation === "reviews" ? { page: clamp(input.page ?? 1, 1, 100) } : {}),
    };
  }
  return {
    item_id: productId,
    ...(operation === "reviews" ? { page: clamp(input.page ?? 1, 1, 100) } : {}),
  };
}

function clamp(value: unknown, min: number, max: number): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return min;
  return Math.max(min, Math.min(max, Math.trunc(parsed)));
}

type EndpointResult = {
  success?: unknown;
  data?: unknown;
  error?: unknown;
  message?: unknown;
};

function isSuccessfulResult(value: EndpointResult): value is EndpointResult & { success: true } {
  return value?.success === true;
}

function extractItems(value: unknown, depth = 0): Record<string, unknown>[] {
  if (depth > 6 || value == null) return [];
  if (Array.isArray(value)) {
    return value.filter(isRecord);
  }
  if (!isRecord(value)) return [];

  for (const key of [
    "items",
    "item_list",
    "products",
    "product_list",
    "notes",
    "note_list",
    "comments",
    "comment_list",
    "results",
    "list",
  ]) {
    const nested = value[key];
    if (Array.isArray(nested)) return nested.filter(isRecord);
  }
  for (const key of ["data", "result", "result_data", "payload"]) {
    const nested = extractItems(value[key], depth + 1);
    if (nested.length) return nested;
  }

  // Detail endpoints commonly return one object rather than a list. Keep it
  // as one normalized item only when it carries a recognizable external ID.
  return externalId(value) ? [value] : [];
}

function externalId(value: Record<string, unknown>): string {
  for (const key of ["item_id", "itemId", "product_id", "productId", "note_id", "noteId", "id"]) {
    const candidate = String(value[key] ?? "").trim();
    if (candidate) return candidate;
  }
  return "";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

async function readJson(request: IncomingMessage): Promise<Record<string, unknown>> {
  const chunks: Buffer[] = [];
  for await (const chunk of request) chunks.push(Buffer.from(chunk));
  const value: unknown = JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("JSON object required");
  return value as Record<string, unknown>;
}

function sendJson(response: ServerResponse, status: number, body: unknown): void {
  response.statusCode = status;
  response.setHeader("content-type", "application/json; charset=utf-8");
  response.end(JSON.stringify(body));
}
