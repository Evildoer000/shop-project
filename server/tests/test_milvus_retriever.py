from types import SimpleNamespace

import pytest

from app.rag.llamaindex_milvus import LlamaIndexMilvusRetriever
from app.services.embedding_client import EmbeddingClient


def stub_embedding_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = SimpleNamespace(embedding_dim=2)
    monkeypatch.setattr("app.rag.llamaindex_milvus.get_settings", lambda: settings)
    monkeypatch.setattr(EmbeddingClient, "__init__", lambda self: setattr(self, "settings", settings))


def test_retrieve_propagates_remote_embedding_or_milvus_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_embedding_settings(monkeypatch)
    retriever = LlamaIndexMilvusRetriever()

    def broken_retrieve(query: str, allowed_ids: set[str], top_k: int) -> dict[str, float]:
        raise RuntimeError("remote embedding unavailable")

    monkeypatch.setattr(retriever, "_retrieve_from_milvus", broken_retrieve)

    with pytest.raises(RuntimeError, match="remote embedding unavailable"):
        retriever.retrieve("防晒", [SimpleNamespace(product_id="p1")])  # type: ignore[list-item]


def test_query_embedding_never_uses_hash_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def remote_required(self: EmbeddingClient, text: str) -> list[float]:
        calls.append(text)
        return [0.1, 0.2]

    def forbidden_fallback(self: EmbeddingClient, text: str) -> list[float]:
        raise AssertionError("hash-capable embed() must not be used for a remote-built Milvus index")

    monkeypatch.setattr(EmbeddingClient, "embed_remote_required", remote_required)
    monkeypatch.setattr(EmbeddingClient, "embed", forbidden_fallback)
    stub_embedding_settings(monkeypatch)
    retriever = LlamaIndexMilvusRetriever()
    model = retriever._remote_embedding_model(object)

    assert model._get_query_embedding("semantic sunscreen") == [0.1, 0.2]
    assert calls == ["semantic sunscreen"]
