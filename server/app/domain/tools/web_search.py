from __future__ import annotations

import time
from typing import Any

import httpx


class WebSearchTool:
    """Optional general web-search adapter.

    Until a Baidu API/MCP is supplied, this deliberately returns a structured
    unavailable result.  It never fabricates search evidence.
    """

    def __init__(self, *, endpoint: str = "", api_key: str = "", timeout_seconds: float = 15.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    async def search(
        self,
        *,
        query: str,
        freshness: str = "recent",
        limit: int = 8,
    ) -> dict[str, Any]:
        normalized_query = str(query or "").strip()[:500]
        if not normalized_query:
            return {
                "available": False,
                "items": [],
                "error": {"code": "empty_query", "message": "搜索词为空。"},
            }
        if not self.endpoint:
            return {
                "available": False,
                "query": normalized_query,
                "items": [],
                "error": {
                    "code": "web_search_not_configured",
                    "message": "联网搜索 Tool 尚未配置，不能把模型常识当作联网证据。",
                },
            }

        started = time.perf_counter()
        headers = {"authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    self.endpoint,
                    headers=headers,
                    json={
                        "query": normalized_query,
                        "freshness": freshness,
                        "limit": max(1, min(int(limit), 20)),
                    },
                )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("web search response must be an object")
            data.setdefault("available", True)
            data.setdefault("query", normalized_query)
            data.setdefault("latency_ms", round((time.perf_counter() - started) * 1000, 2))
            return data
        except (httpx.HTTPError, ValueError) as exc:
            return {
                "available": False,
                "query": normalized_query,
                "items": [],
                "error": {"code": "web_search_failed", "message": str(exc)},
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
