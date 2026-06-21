from __future__ import annotations

from app.core.config import Settings
from scripts import index_image_vectors


def test_docker_image_index_bootstrap_skips_without_remote_embedding(monkeypatch) -> None:
    touched_infra = False

    def fail_if_called(*args, **kwargs):
        nonlocal touched_infra
        touched_infra = True
        raise AssertionError("bootstrap should not touch infra when remote image embedding is not configured")

    monkeypatch.setattr(
        index_image_vectors,
        "get_settings",
        lambda: Settings(
            dashscope_api_key=None,
            image_embedding_api_key=None,
            image_embedding_backend="dashscope",
            image_embedding_model="tongyi-embedding-vision-flash-2026-03-06",
        ),
    )
    monkeypatch.setattr(index_image_vectors, "get_engine", fail_if_called)
    monkeypatch.setattr(index_image_vectors, "get_sessionmaker", fail_if_called)

    result = index_image_vectors.bootstrap_image_index(skip_without_remote=True)

    assert result is True
    assert touched_infra is False
