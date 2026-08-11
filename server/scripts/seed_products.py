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
from app.services.elasticsearch_product_index import ElasticsearchProductIndex
from app.rag.llamaindex_milvus import LlamaIndexMilvusRetriever
from app.services.organizer_dataset import load_organizer_products, resolve_dataset_dir


BOOTSTRAP_METADATA_TABLE = "bootstrap_metadata"
TEXT_INDEX_FINGERPRINT_KEY = "text_index_fingerprint"
TEXT_INDEX_PRODUCT_COUNT_KEY = "text_index_product_count"
TEXT_INDEX_CHECKPOINT_KEY_PREFIX = "text_index_build_checkpoint"
KEYWORD_INDEX_FINGERPRINT_KEY = "keyword_index_fingerprint"
KEYWORD_INDEX_PRODUCT_COUNT_KEY = "keyword_index_product_count"
KEYWORD_INDEX_SCHEMA_VERSION = "ik_max_word_ik_smart_fields_v3"


def main() -> None:
    engine = get_engine()
    SessionLocal = get_sessionmaker()
    Base.metadata.create_all(bind=engine)
    _ensure_bootstrap_metadata_table(engine)
    dataset_dir = resolve_dataset_dir()
    rows = load_organizer_products(dataset_dir)
    fingerprint = _text_index_fingerprint(rows)
    keyword_fingerprint = _keyword_index_fingerprint(rows)
    force_reindex = _truthy(os.getenv("BOOTSTRAP_FORCE_REINDEX"))
    force_keyword_reindex = force_reindex or _truthy(os.getenv("BOOTSTRAP_FORCE_ES_REINDEX"))
    with SessionLocal() as db:
        product_count = db.query(Product).count()
        existing_product_ids = {product_id for (product_id,) in db.query(Product.product_id).all()}
        expected_product_ids = {str(row.get("product_id") or "") for row in rows}
        db_is_current = existing_product_ids == expected_product_ids and product_count == len(rows)
        if not db_is_current or force_reindex:
            db.query(Product).delete()
            for row in rows:
                db.merge(Product(**row))
            db.commit()
            products = list(db.query(Product).all())
        else:
            products = list(db.query(Product).all())

    refreshed_text_index = False
    incomplete_checkpoint = _read_incomplete_text_index_checkpoint(engine)
    if incomplete_checkpoint and not force_reindex:
        print(
            "A resumable Milvus text-index build is incomplete "
            f"({incomplete_checkpoint.get('next_index', 0)}/"
            f"{incomplete_checkpoint.get('total_products', len(products))}, "
            f"status={incomplete_checkpoint.get('status', 'unknown')}). "
            "Preserving it; run scripts/index_text_vectors.py without --reset to continue."
        )
    elif force_reindex or not _can_skip_text_index(engine, len(products), len(rows), fingerprint):
        LlamaIndexMilvusRetriever().index_products(products, overwrite=True)
        _write_bootstrap_metadata(engine, TEXT_INDEX_FINGERPRINT_KEY, fingerprint)
        _write_bootstrap_metadata(engine, TEXT_INDEX_PRODUCT_COUNT_KEY, str(len(products)))
        refreshed_text_index = True
    else:
        print("Text product seed and Milvus collection are already current; skip rebuild.")

    _refresh_keyword_index(
        engine,
        products,
        keyword_fingerprint=keyword_fingerprint,
        force_reindex=force_keyword_reindex,
    )
    print(f"Seeded {len(products)} organizer products from {dataset_dir}.")
    if refreshed_text_index:
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


def _refresh_keyword_index(
    engine,
    products: list[Product],
    *,
    keyword_fingerprint: str,
    force_reindex: bool,
) -> None:
    index = ElasticsearchProductIndex()
    if not index.enabled:
        print("Elasticsearch is disabled; skip keyword index refresh.")
        return
    try:
        if not force_reindex and _can_skip_keyword_index(engine, len(products), keyword_fingerprint):
            print("Elasticsearch keyword index is already current; skip rebuild.")
            return
        indexed = index.index_products(products, overwrite=True)
        _write_bootstrap_metadata(engine, KEYWORD_INDEX_FINGERPRINT_KEY, keyword_fingerprint)
        _write_bootstrap_metadata(engine, KEYWORD_INDEX_PRODUCT_COUNT_KEY, str(indexed))
        print(f"Refreshed Elasticsearch keyword index with {indexed} products.")
    except Exception as exc:
        print(f"Elasticsearch keyword index refresh failed; fallback to BM25 only: {exc}")


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


def _keyword_collection_doc_count() -> int:
    index = ElasticsearchProductIndex()
    try:
        return index.count_documents()
    except Exception as exc:
        print(f"Elasticsearch readiness check failed; will rebuild keyword index: {exc}")
        return 0


def _read_bootstrap_metadata(engine, key: str) -> str:
    with engine.begin() as connection:
        result = connection.execute(
            text(f"select value from {BOOTSTRAP_METADATA_TABLE} where key = :key"),
            {"key": key},
        ).scalar()
    return str(result or "")


def _read_incomplete_text_index_checkpoint(engine) -> dict | None:
    settings = get_settings()
    collection_name = settings.text_milvus_collection or settings.milvus_collection
    key = f"{TEXT_INDEX_CHECKPOINT_KEY_PREFIX}:{collection_name}"
    raw = _read_bootstrap_metadata(engine, key)
    if not raw:
        return None
    try:
        checkpoint = json.loads(raw)
    except (TypeError, ValueError):
        return None
    status = str(checkpoint.get("status") or "")
    next_index = int(checkpoint.get("next_index") or 0)
    total_products = int(checkpoint.get("total_products") or 0)
    if status != "completed" or next_index < total_products:
        return checkpoint
    return None


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


def _keyword_index_fingerprint(rows: list[dict]) -> str:
    settings = get_settings()
    payload = {
        "dataset": rows,
        "schema": KEYWORD_INDEX_SCHEMA_VERSION,
        "index": settings.elasticsearch_index or "",
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _can_skip_keyword_index(engine, dataset_count: int, fingerprint: str) -> bool:
    if dataset_count <= 0:
        return False
    collection_count = _keyword_collection_doc_count()
    if collection_count < dataset_count:
        return False
    stored_fingerprint = _read_bootstrap_metadata(engine, KEYWORD_INDEX_FINGERPRINT_KEY)
    if stored_fingerprint:
        return stored_fingerprint == fingerprint
    _write_bootstrap_metadata(engine, KEYWORD_INDEX_FINGERPRINT_KEY, fingerprint)
    _write_bootstrap_metadata(engine, KEYWORD_INDEX_PRODUCT_COUNT_KEY, str(dataset_count))
    return True


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


if __name__ == "__main__":
    main()
