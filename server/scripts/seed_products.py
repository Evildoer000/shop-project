from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
SERVER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER_ROOT))

from app.core.config import get_settings
from app.db.models import Base, Product
from app.db.session import get_engine, get_sessionmaker
from app.rag.llamaindex_milvus import LlamaIndexMilvusRetriever
from app.services.organizer_dataset import load_organizer_products, resolve_dataset_dir


BOOTSTRAP_METADATA_TABLE = "bootstrap_metadata"
TEXT_INDEX_FINGERPRINT_KEY = "text_index_fingerprint"
TEXT_INDEX_PRODUCT_COUNT_KEY = "text_index_product_count"


def main() -> None:
    engine = get_engine()
    SessionLocal = get_sessionmaker()
    Base.metadata.create_all(bind=engine)
    _ensure_bootstrap_metadata_table(engine)
    dataset_dir = resolve_dataset_dir()
    rows = load_organizer_products(dataset_dir)
    fingerprint = _text_index_fingerprint(rows)
    force_reindex = _truthy(os.getenv("BOOTSTRAP_FORCE_REINDEX"))
    with SessionLocal() as db:
        product_count = db.query(Product).count()
        existing_product_ids = {product_id for (product_id,) in db.query(Product.product_id).all()}
        expected_product_ids = {str(row.get("product_id") or "") for row in rows}
        if (
            not force_reindex
            and existing_product_ids == expected_product_ids
            and _can_skip_text_index(engine, product_count, len(rows), fingerprint)
        ):
            print(
                "Text product seed and Milvus collection are already current; "
                "skip rebuild. Set BOOTSTRAP_FORCE_REINDEX=true to force refresh."
            )
            return
        db.query(Product).delete()
        for row in rows:
            db.merge(Product(**row))
        db.commit()
        products = list(db.query(Product).all())

    LlamaIndexMilvusRetriever().index_products(products, overwrite=True)
    _write_bootstrap_metadata(engine, TEXT_INDEX_FINGERPRINT_KEY, fingerprint)
    _write_bootstrap_metadata(engine, TEXT_INDEX_PRODUCT_COUNT_KEY, str(len(products)))
    print(f"Seeded {len(products)} organizer products from {dataset_dir}.")
    print("Refreshed Milvus collection.")


def _ensure_bootstrap_metadata_table(engine) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                f"""
                create table if not exists {BOOTSTRAP_METADATA_TABLE} (
                    key text primary key,
                    value text not null,
                    updated_at timestamptz not null default now()
                )
                """
            )
        )


def _can_skip_text_index(engine, product_count: int, dataset_count: int, fingerprint: str) -> bool:
    if product_count != dataset_count or dataset_count <= 0:
        return False
    collection_count = _text_collection_row_count()
    if collection_count < dataset_count:
        return False
    stored_fingerprint = _read_bootstrap_metadata(engine, TEXT_INDEX_FINGERPRINT_KEY)
    if stored_fingerprint:
        return stored_fingerprint == fingerprint
    # First run after introducing bootstrap metadata: avoid burning embedding
    # API calls when both DB and Milvus are already populated.
    _write_bootstrap_metadata(engine, TEXT_INDEX_FINGERPRINT_KEY, fingerprint)
    _write_bootstrap_metadata(engine, TEXT_INDEX_PRODUCT_COUNT_KEY, str(dataset_count))
    return True


def _text_collection_row_count() -> int:
    from pymilvus import MilvusClient

    settings = get_settings()
    collection_name = settings.text_milvus_collection or settings.milvus_collection
    try:
        client = MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token)
        if not client.has_collection(collection_name):
            return 0
        stats = client.get_collection_stats(collection_name)
        return int(stats.get("row_count") or 0)
    except Exception as exc:
        print(f"Milvus collection readiness check failed; will rebuild text index: {exc}")
        return 0


def _read_bootstrap_metadata(engine, key: str) -> str:
    with engine.begin() as connection:
        result = connection.execute(
            text(f"select value from {BOOTSTRAP_METADATA_TABLE} where key = :key"),
            {"key": key},
        ).scalar()
    return str(result or "")


def _write_bootstrap_metadata(engine, key: str, value: str) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                f"""
                insert into {BOOTSTRAP_METADATA_TABLE} (key, value, updated_at)
                values (:key, :value, now())
                on conflict (key) do update
                set value = excluded.value,
                    updated_at = now()
                """
            ),
            {"key": key, "value": value},
        )


def _text_index_fingerprint(rows: list[dict]) -> str:
    settings = get_settings()
    payload = {
        "dataset": rows,
        "embedding": {
            "base_url": settings.embedding_base_url or "",
            "model": settings.embedding_model or "",
            "dim": int(settings.embedding_dim),
            "fallback": "hash" if not (settings.embedding_api_key and settings.embedding_base_url and settings.embedding_model) else "remote",
        },
        "collection": settings.text_milvus_collection or settings.milvus_collection,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


if __name__ == "__main__":
    main()
