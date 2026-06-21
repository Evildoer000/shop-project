from __future__ import annotations

from pathlib import Path

from app.core.config import get_settings
from app.db.models import Product
from app.services.embedding_client import EmbeddingClient


class ImageIndexUnavailable(RuntimeError):
    """图片向量索引不可用（ImageIndexUnavailable）。"""


class ImageRetriever:
    """图片向量召回服务。

    输入是本地图片路径和可售商品候选池，输出仅包含候选池内商品的相似度分数。
    """

    def __init__(self, embedding_client: EmbeddingClient | None = None) -> None:
        self.settings = get_settings()
        self.embedding_client = embedding_client or EmbeddingClient()
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from pymilvus import MilvusClient

            self._client = MilvusClient(
                uri=self.settings.milvus_uri,
                token=self.settings.milvus_token,
            )
        return self._client

    def retrieve(self, image_path: str | Path | None, candidates: list[Product], top_k: int = 12) -> dict[str, float]:
        if image_path is None or not candidates:
            return {}

        collection_name = self.settings.image_milvus_collection
        if not self._has_collection(collection_name):
            raise ImageIndexUnavailable(f"Milvus image collection '{collection_name}' is not available.")

        vector = self.embedding_client.embed_image(image_path)
        allowed_ids = {product.product_id for product in candidates}
        try:
            results = self.client.search(
                collection_name=collection_name,
                data=[vector],
                output_fields=["product_id"],
                limit=max(top_k * 3, top_k),
            )
        except Exception as exc:
            if self._is_missing_collection_error(exc, collection_name):
                raise ImageIndexUnavailable(f"Milvus image collection '{collection_name}' is not available.") from exc
            raise
        if not results:
            return {}

        scores: dict[str, float] = {}
        for hit in results[0]:
            entity = hit.get("entity", {}) or {}
            product_id = entity.get("product_id")
            if product_id in allowed_ids and product_id not in scores:
                scores[product_id] = float(hit.get("distance", 0.0))
            if len(scores) >= top_k:
                break
        return scores

    def _has_collection(self, collection_name: str) -> bool:
        try:
            return bool(self.client.has_collection(collection_name))
        except Exception as exc:
            if self._is_missing_collection_error(exc, collection_name):
                return False
            raise

    def _is_missing_collection_error(self, exc: Exception, collection_name: str) -> bool:
        message = str(exc).lower()
        return "collection not found" in message and collection_name.lower() in message
