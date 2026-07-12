from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


SERVER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(SERVER_ROOT))

from app.core.config import get_settings
from app.db.models import Product
from app.db.session import get_engine
from app.db.session import get_sessionmaker
from app.services.embedding_client import EmbeddingClient
from app.services.organizer_dataset import resolve_dataset_dir
from seed_products import (
    _ensure_bootstrap_metadata_table,
    _read_bootstrap_metadata,
    _truthy,
    _write_bootstrap_metadata,
)


IMAGE_INDEX_FINGERPRINT_KEY = "image_index_fingerprint"
IMAGE_INDEX_PRODUCT_COUNT_KEY = "image_index_product_count"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Milvus image-vector collection for product images.")
    parser.add_argument("--overwrite", action="store_true", help="Drop and recreate the image collection before indexing.")
    parser.add_argument("--limit", type=int, default=0, help="Only index the first N products with images.")
    parser.add_argument("--collection", default="", help="Override IMAGE_MILVUS_COLLECTION for this run.")
    parser.add_argument(
        "--allow-hash-fallback",
        action="store_true",
        help="Allow running without local Chinese-CLIP dependencies. This is only useful for dev smoke checks.",
    )
    args = parser.parse_args()

    result = bootstrap_image_index(
        overwrite=bool(args.overwrite),
        limit=max(0, int(args.limit or 0)),
        collection_override=args.collection,
        allow_hash_fallback=bool(args.allow_hash_fallback),
    )
    return 0 if result else 1


def bootstrap_image_index(
    *,
    overwrite: bool = False,
    limit: int = 0,
    collection_override: str = "",
    allow_hash_fallback: bool = False,
    force_reindex: bool | None = None,
    skip_without_remote: bool = False,
) -> bool:
    settings = get_settings()
    if not allow_hash_fallback:
        if _remote_image_embedding_configured(settings):
            _assert_dashscope_available()
        elif skip_without_remote:
            print("Remote image embedding is not configured; skip Docker image vector index bootstrap.")
            return True
        else:
            _assert_clip_dependencies_available()

    collection_name = collection_override or settings.image_milvus_collection
    dataset_dir = resolve_dataset_dir()
    engine = get_engine()
    _ensure_bootstrap_metadata_table(engine)
    SessionLocal = get_sessionmaker()
    with SessionLocal() as db:
        products = list(db.query(Product).order_by(Product.product_id).all())

    image_items = _image_items(products, dataset_dir, limit=limit)
    if not image_items:
        print("No product images found; image collection was not updated.")
        return True

    fingerprint = _image_index_fingerprint(image_items, collection_name)
    should_force = bool(overwrite) or (
        _truthy(os.getenv("BOOTSTRAP_FORCE_REINDEX"))
        or _truthy(os.getenv("BOOTSTRAP_FORCE_IMAGE_REINDEX"))
        if force_reindex is None
        else bool(force_reindex)
    )
    if not should_force and _can_skip_image_index(engine, collection_name, len(image_items), fingerprint):
        print(
            "Image Milvus collection is already current; skip rebuild. "
            "Set BOOTSTRAP_FORCE_IMAGE_REINDEX=true to force refresh."
        )
        return True

    strict_remote = _remote_image_embedding_configured(settings) and not allow_hash_fallback
    embedder = EmbeddingClient()
    client = _ensure_milvus_collection(
        collection_name=collection_name,
        dim=int(settings.image_embedding_dim),
        overwrite=should_force,
    )
    total = len(image_items)
    indexed_count = 0
    skipped_count = 0
    pending_rows: list[dict[str, Any]] = []
    for start in range(0, total, 50):
        chunk = image_items[start : start + 50]
        existing_ids = set() if should_force else _existing_ids(
            client,
            collection_name,
            [product.product_id for product, _ in chunk],
        )
        for product, image_path in chunk:
            if product.product_id in existing_ids:
                skipped_count += 1
                continue
            try:
                vector = _embed_image_for_index(embedder, image_path, strict_remote=strict_remote)
            except Exception:
                if pending_rows:
                    _upsert_milvus_rows(client, collection_name, pending_rows)
                    client.flush(collection_name)
                raise
            pending_rows.append(
                {
                    "id": product.product_id,
                    "product_id": product.product_id,
                    "vector": vector,
                    "category": product.category,
                    "sub_category": product.sub_category or "",
                    "image_path": str(image_path),
                }
            )
            indexed_count += 1
        if pending_rows:
            _upsert_milvus_rows(client, collection_name, pending_rows)
            client.flush(collection_name)
            pending_rows = []
        processed = min(start + len(chunk), total)
        print(
            f"Prepared image vectors {processed}/{total} "
            f"(new={indexed_count}, skipped={skipped_count}).",
            flush=True,
        )

    client.flush(collection_name)
    collection_count = _collection_row_count(collection_name)
    if collection_count < total:
        print(
            f"Image index is incomplete: Milvus has {collection_count}/{total} rows in '{collection_name}'.",
            flush=True,
        )
        return False
    _write_bootstrap_metadata(engine, IMAGE_INDEX_FINGERPRINT_KEY, fingerprint)
    _write_bootstrap_metadata(engine, IMAGE_INDEX_PRODUCT_COUNT_KEY, str(total))
    print(
        f"Indexed {collection_count} product images into Milvus collection '{collection_name}' "
        f"(new={indexed_count}, skipped={skipped_count})."
    )
    return True


def _assert_clip_dependencies_available() -> None:
    try:
        import torch  # noqa: F401
        from PIL import Image  # noqa: F401
        from cn_clip.clip import load_from_name  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Chinese-CLIP image indexing needs torch, pillow and cn-clip. "
            "Install server requirements first, or pass --allow-hash-fallback for dev-only smoke checks."
        ) from exc


def _assert_dashscope_available() -> None:
    try:
        import dashscope  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Remote DashScope image indexing needs the dashscope package. "
            "Install server requirements first, or pass --allow-hash-fallback for dev-only smoke checks."
        ) from exc


def _remote_image_embedding_configured(settings) -> bool:
    backend = str(settings.image_embedding_backend or "auto").lower()
    if backend in {"local", "clip", "cn_clip", "hash", "none", "off"}:
        return False
    api_key = settings.image_embedding_api_key or settings.dashscope_api_key
    return bool(api_key and settings.image_embedding_model)


def _image_items(products: list[Product], dataset_dir: Path, *, limit: int = 0) -> list[tuple[Product, Path]]:
    result: list[tuple[Product, Path]] = []
    for product in products:
        image_path = _resolve_product_image_path(product, dataset_dir)
        if image_path is None:
            continue
        result.append((product, image_path))
        if limit and len(result) >= limit:
            break
    return result


def _resolve_product_image_path(product: Product, dataset_dir: Path) -> Path | None:
    raw_path = (
        (product.structured_attributes or {}).get("image_path")
        or (product.specs or {}).get("source_image_path")
        or ""
    )
    if not raw_path:
        return None
    relative = str(raw_path).replace("\\", "/").lstrip("/")
    candidates = [
        dataset_dir / relative,
        PROJECT_ROOT / relative,
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def _can_skip_image_index(engine, collection_name: str, expected_count: int, fingerprint: str) -> bool:
    if expected_count <= 0:
        return True
    collection_count = _collection_row_count(collection_name)
    if collection_count < expected_count:
        return False
    stored_fingerprint = _read_bootstrap_metadata(engine, IMAGE_INDEX_FINGERPRINT_KEY)
    if stored_fingerprint:
        return stored_fingerprint == fingerprint
    _write_bootstrap_metadata(engine, IMAGE_INDEX_FINGERPRINT_KEY, fingerprint)
    _write_bootstrap_metadata(engine, IMAGE_INDEX_PRODUCT_COUNT_KEY, str(expected_count))
    return True


def _collection_row_count(collection_name: str) -> int:
    from pymilvus import MilvusClient

    settings = get_settings()
    try:
        client = MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token)
        if not client.has_collection(collection_name):
            return 0
        stats = client.get_collection_stats(collection_name)
        return int(stats.get("row_count") or 0)
    except Exception as exc:
        print(f"Milvus image collection readiness check failed; will rebuild image index: {exc}")
        return 0


def _image_index_fingerprint(image_items: list[tuple[Product, Path]], collection_name: str) -> str:
    settings = get_settings()
    payload = {
        "images": [
            {
                "product_id": product.product_id,
                "image_path": str(image_path),
                "mtime_ns": image_path.stat().st_mtime_ns,
                "size": image_path.stat().st_size,
            }
            for product, image_path in image_items
        ],
        "embedding": {
            "backend": settings.image_embedding_backend or "",
            "model": settings.image_embedding_model or "",
            "dim": int(settings.image_embedding_dim),
            "fallback": "remote" if _remote_image_embedding_configured(settings) else "local_or_hash",
        },
        "collection": collection_name,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_milvus_collection(
    *,
    collection_name: str,
    rows: list[dict[str, Any]],
    dim: int,
    overwrite: bool,
) -> None:
    client = _ensure_milvus_collection(collection_name=collection_name, dim=dim, overwrite=overwrite)
    _upsert_milvus_rows(client, collection_name, rows)
    client.flush(collection_name)


def _ensure_milvus_collection(*, collection_name: str, dim: int, overwrite: bool):
    from pymilvus import DataType, MilvusClient

    settings = get_settings()
    client = MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token)
    if overwrite and client.has_collection(collection_name):
        client.drop_collection(collection_name)
    if not client.has_collection(collection_name):
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field(field_name="product_id", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="sub_category", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
        index_params = client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
    return client


def _upsert_milvus_rows(client, collection_name: str, rows: list[dict[str, Any]]) -> None:
    for start in range(0, len(rows), 100):
        batch = rows[start : start + 100]
        if hasattr(client, "upsert"):
            client.upsert(collection_name=collection_name, data=batch)
        else:
            client.insert(collection_name=collection_name, data=batch)


def _existing_ids(client, collection_name: str, product_ids: list[str]) -> set[str]:
    if not product_ids or not client.has_collection(collection_name):
        return set()
    expression = "id in " + json.dumps(product_ids, ensure_ascii=False)
    rows = client.query(
        collection_name=collection_name,
        filter=expression,
        output_fields=["id"],
        limit=len(product_ids),
    )
    return {str(row.get("id")) for row in rows if row.get("id")}


def _embed_image_for_index(
    embedder: EmbeddingClient,
    image_path: Path,
    *,
    strict_remote: bool,
) -> list[float]:
    if strict_remote:
        return embedder._remote_image_embedding(image_path)
    return embedder.embed_image(image_path)


if __name__ == "__main__":
    raise SystemExit(main())
