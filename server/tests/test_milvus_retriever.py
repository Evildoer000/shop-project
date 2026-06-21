from types import SimpleNamespace

import pytest

from app.rag.llamaindex_milvus import LlamaIndexMilvusRetriever


def test_retrieve_propagates_remote_embedding_or_milvus_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    retriever = LlamaIndexMilvusRetriever()

    def broken_retrieve(query: str, allowed_ids: set[str], top_k: int) -> dict[str, float]:
        raise RuntimeError("remote embedding unavailable")

    monkeypatch.setattr(retriever, "_retrieve_from_milvus", broken_retrieve)

    with pytest.raises(RuntimeError, match="remote embedding unavailable"):
        retriever.retrieve("防晒", [SimpleNamespace(product_id="p1")])  # type: ignore[list-item]
