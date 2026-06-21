from __future__ import annotations

import httpx

from app.core.config import get_settings
from app.db.models import Product


class HybridReranker:
    """Use the existing hybrid/RRF order as the final ranking without remote API calls."""

    def rerank(
        self,
        query: str,
        products: list[Product],
        top_k: int = 5,
    ) -> list[tuple[Product, float]]:
        if not products:
            return []
        limit = min(top_k, len(products))
        denominator = max(len(products) + 1, 1)
        return [
            (product, round(1.0 - index / denominator, 4))
            for index, product in enumerate(products[:limit])
        ]


class RemoteCrossEncoderReranker:
    def __init__(self) -> None:
        self.settings = get_settings()

    def rerank(
        self,
        query: str,
        products: list[Product],
        top_k: int = 5,
    ) -> list[tuple[Product, float]]:
        if not products:
            return []
        if not self._is_configured():
            raise RuntimeError("Rerank API 未配置，请设置 RERANK_API_KEY、RERANK_BASE_URL 和 RERANK_MODEL。")

        documents = [product.search_text() for product in products]
        payload = {
            "model": self.settings.rerank_model,
            "input": {
                "query": query,
                "documents": documents,
            },
            "parameters": {
                "return_documents": False,
                "top_n": min(top_k, len(products)),
            },
        }
        headers = {
            "Authorization": f"Bearer {self.settings.rerank_api_key}",
            "Content-Type": "application/json",
        }
        url = self.settings.rerank_base_url.rstrip('/')
        try:
            with httpx.Client(timeout=self.settings.rerank_timeout_seconds) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            raise RuntimeError(f"Rerank API 调用失败: {self.settings.rerank_model}") from exc

        scores = self._parse_scores(data, len(products))
        ranked = [(products[index], score) for index, score in scores]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:top_k]

    def _is_configured(self) -> bool:
        return bool(
            self.settings.rerank_api_key
            and self.settings.rerank_base_url
            and self.settings.rerank_model
        )

    def _parse_scores(self, data: object, product_count: int) -> list[tuple[int, float]]:
        if not isinstance(data, dict):
            raise RuntimeError("Rerank API 返回格式无效。")

        # DashScope native API 把 results 包在 output 里，先剥一层
        if isinstance(data.get("output"), dict):
            data = data["output"]

        if isinstance(data.get("results"), list):
            scores: list[tuple[int, float]] = []
            for item in data["results"]:
                if not isinstance(item, dict):
                    continue
                index = int(item.get("index", len(scores)))
                score = item.get("relevance_score", item.get("score", 0.0))
                if 0 <= index < product_count:
                    scores.append((index, float(score)))
            if scores:
                return scores

        if isinstance(data.get("scores"), list):
            return [
                (index, float(score))
                for index, score in enumerate(data["scores"])
                if index < product_count
            ]

        raise RuntimeError("Rerank API 返回中缺少 results 或 scores。")


def build_reranker() -> object:
    settings = get_settings()
    backend = (settings.rerank_backend or "hybrid").strip().lower()
    if backend == "remote":
        return RemoteCrossEncoderReranker()
    if backend in {"hybrid", "none", "disabled", "off"}:
        return HybridReranker()
    raise RuntimeError(f"未知 RERANK_BACKEND: {settings.rerank_backend}")
