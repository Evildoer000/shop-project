from __future__ import annotations

from threading import Lock
from typing import Any

from app.core.config import get_settings
from app.db.models import Product
from app.services.embedding_client import EmbeddingClient


class LlamaIndexMilvusRetriever:
    _index_cache: dict[tuple[str, str, str, int], Any] = {}
    _cache_lock = Lock()

    def __init__(self) -> None:
        self.settings = get_settings()
        self.embedding_client = EmbeddingClient()

    def index_products(self, products: list[Product], overwrite: bool = True) -> None:
        from llama_index.core import Settings, StorageContext, VectorStoreIndex
        from llama_index.core.embeddings import BaseEmbedding
        from llama_index.core.schema import TextNode
        from llama_index.vector_stores.milvus import MilvusVectorStore

        nodes = [
            TextNode(
                text=product.search_text(),
                metadata={
                    "product_id": product.product_id,
                    "category": product.category,
                    "brand": product.brand,
                    "price": float(product.price),
                    "tags": product.tags,
                },
            )
            for product in products
        ]
        if not nodes:
            return

        Settings.embed_model = self._remote_embedding_model(BaseEmbedding)
        vector_store = MilvusVectorStore(
            uri=self.settings.milvus_uri,
            token=self.settings.milvus_token,
            collection_name=self.settings.text_milvus_collection or self.settings.milvus_collection,
            dim=self.settings.embedding_dim,
            overwrite=overwrite,
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex(nodes, storage_context=storage_context, embed_model=Settings.embed_model)

    def retrieve(
        self,
        query: str,
        allowed_products: list[Product],
        top_k: int = 12,
    ) -> dict[str, float]:
        if not query or not allowed_products:
            return {}
        return self._retrieve_from_milvus(query, {p.product_id for p in allowed_products}, top_k)

    def _retrieve_from_milvus(self, query: str, allowed_ids: set[str], top_k: int) -> dict[str, float]:
        index = self._text_index()
        retriever = index.as_retriever(similarity_top_k=max(top_k * 4, 20))
        results = retriever.retrieve(query)
        scores: dict[str, float] = {}
        for item in results:
            product_id = item.node.metadata.get("product_id")
            if product_id in allowed_ids:
                scores[product_id] = float(item.score or 0)
            if len(scores) >= top_k:
                break
        return scores

    def _text_index(self) -> Any:
        from llama_index.core import Settings, VectorStoreIndex
        from llama_index.core.embeddings import BaseEmbedding
        from llama_index.vector_stores.milvus import MilvusVectorStore

        collection_name = self.settings.text_milvus_collection or self.settings.milvus_collection
        cache_key = (
            self.settings.milvus_uri,
            self.settings.milvus_token or "",
            collection_name,
            int(self.settings.embedding_dim),
        )
        with self._cache_lock:
            cached = self._index_cache.get(cache_key)
            if cached is not None:
                return cached
            embed_model = self._remote_embedding_model(BaseEmbedding)
            Settings.embed_model = embed_model
            vector_store = MilvusVectorStore(
                uri=self.settings.milvus_uri,
                token=self.settings.milvus_token,
                collection_name=collection_name,
                dim=self.settings.embedding_dim,
                overwrite=False,
            )
            index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
            self._index_cache[cache_key] = index
            return index

    def _remote_embedding_model(self, base_embedding_class: type):
        dim = self.settings.embedding_dim

        class RemoteEmbedding(base_embedding_class):
            embed_dim: int = dim

            def _get_query_embedding(self, query: str) -> list[float]:
                return EmbeddingClient().embed(query)

            async def _aget_query_embedding(self, query: str) -> list[float]:
                return self._get_query_embedding(query)

            def _get_text_embedding(self, text: str) -> list[float]:
                return EmbeddingClient().embed(text)

        return RemoteEmbedding()
