from __future__ import annotations

import pytest

from app.core.config import Settings
from app.services.embedding_client import EmbeddingClient


def test_embed_remote_required_never_uses_hash_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.embedding_client.get_settings",
        lambda: Settings(
            embedding_api_key="sk-test",
            embedding_base_url="https://embedding.example/v1",
            embedding_model="text-embedding-test",
            embedding_dim=3,
        ),
    )
    client = EmbeddingClient()
    monkeypatch.setattr(
        client,
        "_remote_embedding",
        lambda text: (_ for _ in ()).throw(RuntimeError("quota exhausted")),
    )
    monkeypatch.setattr(
        client,
        "_hash_embedding",
        lambda text: pytest.fail("strict indexing must not use hash fallback"),
    )

    with pytest.raises(RuntimeError, match="quota exhausted"):
        client.embed_remote_required("防晒霜")


def test_embed_remote_required_validates_dimension(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.embedding_client.get_settings",
        lambda: Settings(
            embedding_api_key="sk-test",
            embedding_base_url="https://embedding.example/v1",
            embedding_model="text-embedding-test",
            embedding_dim=3,
        ),
    )
    client = EmbeddingClient()
    monkeypatch.setattr(client, "_remote_embedding", lambda text: [1.0, 0.0])

    with pytest.raises(RuntimeError, match="dimension mismatch"):
        client.embed_remote_required("防晒霜")


def test_embed_remote_required_returns_remote_vector(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.embedding_client.get_settings",
        lambda: Settings(
            embedding_api_key="sk-test",
            embedding_base_url="https://embedding.example/v1",
            embedding_model="text-embedding-test",
            embedding_dim=3,
        ),
    )
    client = EmbeddingClient()
    monkeypatch.setattr(client, "_remote_embedding", lambda text: [1.0, 0.0, 0.0])

    assert client.embed_remote_required("防晒霜") == [1.0, 0.0, 0.0]
