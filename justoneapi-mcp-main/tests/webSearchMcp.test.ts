import { describe, expect, it } from "vitest";
import {
  mapWebSearchFreshness,
  normalizeMcpWebSearchResult,
} from "../src/webSearchMcp.js";

describe("Baidu web-search MCP adapter", () => {
  it("maps the internal freshness vocabulary to Baidu values", () => {
    expect(mapWebSearchFreshness("any")).toBeUndefined();
    expect(mapWebSearchFreshness("recent")).toBe("pm");
    expect(mapWebSearchFreshness("realtime")).toBe("pd");
    expect(mapWebSearchFreshness("2026-08-01to2026-08-05")).toBe("2026-08-01to2026-08-05");
    expect(mapWebSearchFreshness("unsupported")).toBeUndefined();
  });

  it("normalizes structured MCP results to the backend evidence contract", () => {
    const result = normalizeMcpWebSearchResult(
      {
        structuredContent: {
          results: [
            {
              title: "网页标题",
              url: "https://example.com/item",
              snippet: "网页摘要",
            },
          ],
        },
      },
      { query: "测试", count: 8, toolName: "web_search", latencyMs: 12.3 },
    );

    expect(result.available).toBe(true);
    expect(result.items).toEqual([
      {
        title: "网页标题",
        snippet: "网页摘要",
        source_id: "https://example.com/item",
        url: "https://example.com/item",
      },
    ]);
  });

  it("parses JSON returned in an MCP text content block", () => {
    const result = normalizeMcpWebSearchResult(
      {
        content: [
          {
            type: "text",
            text: '{"items":[{"name":"文本结果","link":"https://example.com/text","description":"文本摘要"}]}',
          },
        ],
      },
      { query: "测试", count: 1, toolName: "web_search" },
    );

    expect(result.available).toBe(true);
    expect(result.items).toEqual([
      {
        title: "文本结果",
        snippet: "文本摘要",
        source_id: "https://example.com/text",
        url: "https://example.com/text",
      },
    ]);
  });

  it("does not turn an MCP error into evidence", () => {
    const result = normalizeMcpWebSearchResult(
      {
        isError: true,
        content: [{ type: "text", text: "upstream unavailable" }],
      },
      { query: "测试", count: 8, toolName: "web_search" },
    );

    expect(result.available).toBe(false);
    expect(result.items).toEqual([]);
    expect(result.error).toEqual({
      code: "web_search_mcp_tool_error",
      message: "upstream unavailable",
    });
  });
});
