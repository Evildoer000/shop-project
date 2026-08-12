from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx


ALLOWED_COMMERCE_PLATFORMS = frozenset({"taobao", "douyin_ec", "xiaohongshu"})


class CommerceToolError(RuntimeError):
    """A controlled failure from the commerce MCP boundary."""


class CommerceMcpTransport(Protocol):
    async def call(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class UnavailableCommerceTransport:
    reason: str = "commerce_mcp_not_configured"

    async def call(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "available": False,
            "operation": operation,
            "items": [],
            "error": {"code": self.reason, "message": "外部电商 MCP 尚未配置。"},
        }


class HttpCommerceMcpTransport:
    """HTTP client for the optional JustOneAPI high-level bridge.

    The bridge owns the native MCP process and endpoint allowlist.  The
    backend never receives a generic `call_endpoint` capability.
    """

    def __init__(self, base_url: str, *, timeout_seconds: float = 15.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def call(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.base_url:
            return await UnavailableCommerceTransport().call(operation, payload)
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    f"{self.base_url}/v1/commerce/{operation}",
                    json=payload,
                )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise CommerceToolError("commerce_mcp_invalid_response")
            data = _normalize_response(data, operation=operation)
            data.setdefault("latency_ms", round((time.perf_counter() - started) * 1000, 2))
            return data
        except (httpx.HTTPError, ValueError, CommerceToolError) as exc:
            return {
                "available": False,
                "operation": operation,
                "items": [],
                "error": {"code": "commerce_mcp_unavailable", "message": str(exc)},
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }


class _CommerceToolBase:
    def __init__(self, transport: CommerceMcpTransport | None = None) -> None:
        self.transport = transport or UnavailableCommerceTransport()

    def _platform(self, platform: str) -> str:
        value = str(platform or "").strip().lower()
        if value not in ALLOWED_COMMERCE_PLATFORMS:
            raise CommerceToolError(f"unsupported_commerce_platform:{value or 'empty'}")
        return value


class CommerceResearchTool(_CommerceToolBase):
    """Search external marketplace/content sources through a fixed contract."""

    async def search(
        self,
        *,
        platform: str,
        query: str,
        limit: int = 10,
        page: int = 1,
    ) -> dict[str, Any]:
        return await self.transport.call(
            "search",
            {
                "platform": self._platform(platform),
                "query": str(query or "").strip()[:300],
                "limit": max(1, min(int(limit), 20)),
                "page": max(1, int(page)),
            },
        )


class CommerceProductDetailTool(_CommerceToolBase):
    async def get(self, *, platform: str, product_id: str) -> dict[str, Any]:
        return await self.transport.call(
            "detail",
            {"platform": self._platform(platform), "product_id": str(product_id or "").strip()[:128]},
        )


class CommerceReviewsTool(_CommerceToolBase):
    async def list(self, *, platform: str, product_id: str, page: int = 1) -> dict[str, Any]:
        return await self.transport.call(
            "reviews",
            {
                "platform": self._platform(platform),
                "product_id": str(product_id or "").strip()[:128],
                "page": max(1, int(page)),
            },
        )


def _normalize_response(data: dict[str, Any], *, operation: str) -> dict[str, Any]:
    """Defensively normalize bridge and older nested JustOneAPI responses."""
    result = data.get("result")
    result_success = result.get("success") if isinstance(result, dict) else None
    available = bool(data.get("available")) and result_success is not False
    items = _extract_items(data)
    normalized = {
        **data,
        "available": available,
        "operation": str(data.get("operation") or operation),
        "items": items if available else [],
    }
    if not available and "error" not in normalized:
        normalized["error"] = (
            result.get("error")
            if isinstance(result, dict) and isinstance(result.get("error"), dict)
            else {"code": "commerce_mcp_unavailable", "message": "外部接口未返回可用结果。"}
        )
    return normalized


def _extract_items(value: Any, *, depth: int = 0) -> list[dict[str, Any]]:
    if depth > 7:
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if not isinstance(value, dict):
        return []
    for key in (
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
    ):
        nested = value.get(key)
        if isinstance(nested, list):
            return [item for item in nested if isinstance(item, dict)]
    for key in ("data", "result", "result_data", "payload", "raw"):
        nested = _extract_items(value.get(key), depth=depth + 1)
        if nested:
            return nested
    if _external_id(value):
        return [value]
    return []


def _external_id(value: dict[str, Any]) -> str:
    for key in ("item_id", "itemId", "product_id", "productId", "note_id", "noteId", "id"):
        candidate = str(value.get(key) or "").strip()
        if candidate:
            return candidate
    return ""
