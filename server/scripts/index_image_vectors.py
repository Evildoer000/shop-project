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

    rows = []
    embedder = EmbeddingClient()
    for product, image_path in image_items:
        vector = embedder.embed_image(image_path)
        rows.append(
            {
                "id": product.product_id,
                "product_id": product.product_id,
                "vector": vector,
                "category": product.category,
                "sub_category": product.sub_category or "",
                "image_path": str(image_path),
            }
        )

    _write_milvus_collection(
        collection_name=collection_name,
        rows=rows,
        dim=int(settings.image_embedding_dim),
        overwrite=True,
    )
    _write_bootstrap_metadata(engine, IMAGE_INDEX_FINGERPRINT_KEY, fingerprint)
    _write_bootstrap_metadata(engine, IMAGE_INDEX_PRODUCT_COUNT_KEY, str(len(rows)))
    print(f"Indexed {len(rows)} product images into Milvus collection '{collection_name}'.")
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

    for start in range(0, len(rows), 100):
        batch = rows[start : start + 100]
        if hasattr(client, "upsert"):
            client.upsert(collection_name=collection_name, data=batch)
        else:
            client.insert(collection_name=collection_name, data=batch)
    client.flush(collection_name)


if __name__ == "__main__":
    raise SystemExit(main())
