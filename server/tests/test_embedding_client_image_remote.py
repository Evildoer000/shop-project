from __future__ import annotations

import sys
from types import SimpleNamespace

from app.core.config import Settings
from app.services.embedding_client import EmbeddingClient


def test_embed_image_uses_dashscope_multimodal_embedding(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"fake image bytes")
    calls = []

    class FakeMultiModalEmbedding:
        @staticmethod
        def call(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                status_code=200,
                output={"embeddings": [{"type": "image", "embedding": [1.0, 0.0, 0.0]}]},
            )

    monkeypatch.setitem(
        sys.modules,
        "dashscope",
        SimpleNamespace(MultiModalEmbedding=FakeMultiModalEmbedding),
    )
    monkeypatch.setattr(
        "app.services.embedding_client.get_settings",
        lambda: Settings(
            dashscope_api_key="sk-test",
            image_embedding_api_key=None,
            image_embedding_backend="dashscope",
            image_embedding_model="tongyi-embedding-vision-flash-2026-03-06",
            image_embedding_dim=3,
        ),
    )

    embedding = EmbeddingClient().embed_image(image_path)

    assert embedding == [1.0, 0.0, 0.0]
    assert calls[0]["api_key"] == "sk-test"
    assert calls[0]["model"] == "tongyi-embedding-vision-flash-2026-03-06"
    assert calls[0]["input"][0]["image"].startswith("data:image/jpeg;base64,")


def test_embed_image_skips_dashscope_when_not_configured(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"fake image bytes")
    monkeypatch.setattr(
        "app.services.embedding_client.get_settings",
        lambda: Settings(
            dashscope_api_key=None,
            image_embedding_api_key=None,
            image_embedding_backend="dashscope",
            image_embedding_model="",
            image_embedding_dim=4,
        ),
    )

    embedding = EmbeddingClient().embed_image(image_path)

    assert len(embedding) == 4
