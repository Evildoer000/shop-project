from __future__ import annotations

import json
from types import SimpleNamespace

from scripts import seed_products


def test_incomplete_text_checkpoint_is_detected(monkeypatch) -> None:
    monkeypatch.setattr(
        seed_products,
        "get_settings",
        lambda: SimpleNamespace(
            text_milvus_collection="product_chunks",
            milvus_collection="fallback_collection",
        ),
    )
    monkeypatch.setattr(
        seed_products,
        "_read_bootstrap_metadata",
        lambda engine, key: json.dumps(
            {
                "status": "failed",
                "next_index": 400,
                "total_products": 12101,
                "failed_product_id": "p_beauty_00401",
            }
        ),
    )

    checkpoint = seed_products._read_incomplete_text_index_checkpoint(object())

    assert checkpoint is not None
    assert checkpoint["next_index"] == 400


def test_completed_text_checkpoint_is_not_incomplete(monkeypatch) -> None:
    monkeypatch.setattr(
        seed_products,
        "get_settings",
        lambda: SimpleNamespace(
            text_milvus_collection="product_chunks",
            milvus_collection="fallback_collection",
        ),
    )
    monkeypatch.setattr(
        seed_products,
        "_read_bootstrap_metadata",
        lambda engine, key: json.dumps(
            {
                "status": "completed",
                "next_index": 12101,
                "total_products": 12101,
            }
        ),
    )

    assert seed_products._read_incomplete_text_index_checkpoint(object()) is None
