from __future__ import annotations

import pytest

from scripts.index_text_vectors import TextIndexCheckpoint, _validate_checkpoint


def _checkpoint() -> TextIndexCheckpoint:
    return TextIndexCheckpoint(
        version=1,
        collection="product_chunks",
        build_fingerprint="build-1",
        dataset_fingerprint="dataset-1",
        model="text-embedding-v4",
        embedding_dim=1024,
        total_products=12101,
        next_index=400,
        last_success_product_id="p_beauty_00400",
    )


def test_checkpoint_can_resume_after_only_api_key_changes() -> None:
    checkpoint = _checkpoint()

    _validate_checkpoint(
        checkpoint,
        build_fingerprint="build-1",
        dataset_fingerprint="dataset-1",
        model="text-embedding-v4",
        embedding_dim=1024,
        total_products=12101,
    )


def test_checkpoint_rejects_model_change() -> None:
    checkpoint = _checkpoint()

    with pytest.raises(RuntimeError, match="model changed"):
        _validate_checkpoint(
            checkpoint,
            build_fingerprint="build-2",
            dataset_fingerprint="dataset-1",
            model="different-model",
            embedding_dim=1024,
            total_products=12101,
        )
