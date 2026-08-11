from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_ROOT = Path(__file__).resolve().parent
SERVER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(SERVER_ROOT))

from app.core.config import get_settings
from app.db.models import Product
from app.db.session import get_engine, get_sessionmaker
from app.rag.llamaindex_milvus import LlamaIndexMilvusRetriever
from app.services.embedding_client import EmbeddingClient
from app.services.organizer_dataset import load_organizer_products, resolve_dataset_dir
from seed_products import (
    TEXT_INDEX_FINGERPRINT_KEY,
    TEXT_INDEX_CHECKPOINT_KEY_PREFIX,
    TEXT_INDEX_PRODUCT_COUNT_KEY,
    _ensure_bootstrap_metadata_table,
    _read_bootstrap_metadata,
    _text_index_fingerprint,
    _write_bootstrap_metadata,
)


CHECKPOINT_KEY_PREFIX = TEXT_INDEX_CHECKPOINT_KEY_PREFIX
DEFAULT_BATCH_SIZE = 20


@dataclass
class TextIndexCheckpoint:
    version: int
    collection: str
    build_fingerprint: str
    dataset_fingerprint: str
    model: str
    embedding_dim: int
    total_products: int
    next_index: int = 0
    last_success_product_id: str = ""
    failed_product_id: str = ""
    failed_position: int | None = None
    status: str = "running"
    error: str = ""
    updated_at: str = ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build only the Milvus text-vector collection with a PostgreSQL checkpoint. "
            "Remote embedding failures stop the run; hash fallback is never used."
        )
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop the target text collection and start a new checkpoint from the first product.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print the persisted checkpoint and Milvus row count without generating embeddings.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Flush vectors and persist progress every N products (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--collection",
        default="",
        help="Override TEXT_MILVUS_COLLECTION for this run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N products in this invocation, then leave a resumable checkpoint.",
    )
    args = parser.parse_args()

    if args.status:
        return show_status(collection_override=args.collection)
    return 0 if build_text_index(
        reset=bool(args.reset),
        batch_size=max(1, int(args.batch_size)),
        collection_override=args.collection,
        run_limit=max(0, int(args.limit)),
    ) else 1


def build_text_index(
    *,
    reset: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    collection_override: str = "",
    run_limit: int = 0,
) -> bool:
    from pymilvus import MilvusClient

    settings = get_settings()
    collection_name = (
        collection_override
        or settings.text_milvus_collection
        or settings.milvus_collection
    )
    embedder = EmbeddingClient()
    _assert_remote_text_embedding_configured(embedder)

    engine = get_engine()
    _ensure_bootstrap_metadata_table(engine)
    products = _load_products()
    if not products:
        raise RuntimeError("PostgreSQL contains no products; text index was not changed.")

    dataset_rows = load_organizer_products(resolve_dataset_dir())
    dataset_fingerprint = _text_index_fingerprint(dataset_rows)
    build_fingerprint = _build_fingerprint(products, collection_name)
    checkpoint_key = _checkpoint_key(collection_name)
    client = MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token)
    precomputed_vectors: dict[int, list[float]] = {}

    if reset:
        print(
            "Checking the remote embedding key and model before replacing the old text collection...",
            flush=True,
        )
        precomputed_vectors[0] = embedder.embed_remote_required(products[0].search_text())
        _ensure_collection(client, collection_name, int(settings.embedding_dim), overwrite=True)
        checkpoint = TextIndexCheckpoint(
            version=1,
            collection=collection_name,
            build_fingerprint=build_fingerprint,
            dataset_fingerprint=dataset_fingerprint,
            model=str(settings.embedding_model),
            embedding_dim=int(settings.embedding_dim),
            total_products=len(products),
        )
        _save_checkpoint(engine, checkpoint_key, checkpoint)
        print(
            f"Started a new text-index build for '{collection_name}' with {len(products)} products.",
            flush=True,
        )
    else:
        checkpoint = _load_checkpoint(engine, checkpoint_key)
        if checkpoint is None:
            row_count = _collection_row_count(client, collection_name)
            if row_count:
                raise RuntimeError(
                    f"Milvus collection '{collection_name}' has {row_count} rows but no resumable checkpoint. "
                    "Run once with --reset to explicitly replace the old text index."
                )
            _ensure_collection(client, collection_name, int(settings.embedding_dim), overwrite=False)
            checkpoint = TextIndexCheckpoint(
                version=1,
                collection=collection_name,
                build_fingerprint=build_fingerprint,
                dataset_fingerprint=dataset_fingerprint,
                model=str(settings.embedding_model),
                embedding_dim=int(settings.embedding_dim),
                total_products=len(products),
            )
            _save_checkpoint(engine, checkpoint_key, checkpoint)
        else:
            _validate_checkpoint(
                checkpoint,
                build_fingerprint=build_fingerprint,
                dataset_fingerprint=dataset_fingerprint,
                model=str(settings.embedding_model),
                embedding_dim=int(settings.embedding_dim),
                total_products=len(products),
            )
            _ensure_collection(client, collection_name, int(settings.embedding_dim), overwrite=False)

    if checkpoint.status == "completed" and checkpoint.next_index >= len(products):
        print(
            f"Text index is already complete: {checkpoint.next_index}/{len(products)} products.",
            flush=True,
        )
        return True

    start_index = checkpoint.next_index
    stop_index = len(products)
    if run_limit:
        stop_index = min(stop_index, start_index + run_limit)
    checkpoint.status = "running"
    checkpoint.failed_product_id = ""
    checkpoint.failed_position = None
    checkpoint.error = ""
    _save_checkpoint(engine, checkpoint_key, checkpoint)

    pending_rows: list[dict[str, Any]] = []
    pending_last_index = start_index
    for product_index in range(start_index, stop_index):
        product = products[product_index]
        try:
            pending_rows.append(
                _build_row(
                    product,
                    embedder,
                    embedding=precomputed_vectors.pop(product_index, None),
                )
            )
            pending_last_index = product_index + 1
        except Exception as exc:
            if pending_rows:
                _commit_batch(
                    client,
                    collection_name,
                    pending_rows,
                    engine,
                    checkpoint_key,
                    checkpoint,
                    next_index=pending_last_index,
                    last_success_product_id=products[pending_last_index - 1].product_id,
                )
                pending_rows = []
            checkpoint.status = "failed"
            checkpoint.failed_product_id = product.product_id
            checkpoint.failed_position = product_index + 1
            checkpoint.error = _error_text(exc)
            _save_checkpoint(engine, checkpoint_key, checkpoint)
            print(
                f"Text embedding stopped at product {product_index + 1}/{len(products)} "
                f"(product_id={product.product_id}).",
                flush=True,
            )
            print(f"Reason: {checkpoint.error}", flush=True)
            print(
                "Progress has been saved. Replace EMBEDDING_API_KEY and rerun this command "
                "without --reset to retry this product.",
                flush=True,
            )
            return False

        if len(pending_rows) >= batch_size:
            _commit_batch(
                client,
                collection_name,
                pending_rows,
                engine,
                checkpoint_key,
                checkpoint,
                next_index=pending_last_index,
                last_success_product_id=product.product_id,
            )
            pending_rows = []
            print(
                f"Indexed text vectors {checkpoint.next_index}/{len(products)} "
                f"(last_product_id={checkpoint.last_success_product_id}).",
                flush=True,
            )

    if pending_rows:
        last_product = products[pending_last_index - 1]
        _commit_batch(
            client,
            collection_name,
            pending_rows,
            engine,
            checkpoint_key,
            checkpoint,
            next_index=pending_last_index,
            last_success_product_id=last_product.product_id,
        )

    if stop_index < len(products):
        print(
            f"Stopped after this invocation's limit at {checkpoint.next_index}/{len(products)}. "
            "Rerun without --reset to continue.",
            flush=True,
        )
        return True

    client.flush(collection_name)
    row_count = _collection_row_count(client, collection_name)
    if row_count < len(products):
        checkpoint.status = "failed"
        checkpoint.error = (
            f"Milvus row count validation failed: expected at least {len(products)}, got {row_count}."
        )
        _save_checkpoint(engine, checkpoint_key, checkpoint)
        print(checkpoint.error, flush=True)
        return False

    checkpoint.status = "completed"
    checkpoint.next_index = len(products)
    checkpoint.failed_product_id = ""
    checkpoint.failed_position = None
    checkpoint.error = ""
    _save_checkpoint(engine, checkpoint_key, checkpoint)
    _write_bootstrap_metadata(engine, TEXT_INDEX_FINGERPRINT_KEY, dataset_fingerprint)
    _write_bootstrap_metadata(engine, TEXT_INDEX_PRODUCT_COUNT_KEY, str(len(products)))
    LlamaIndexMilvusRetriever._index_cache.clear()
    print(
        f"Completed text index '{collection_name}': {row_count} Milvus rows, "
        f"{len(products)} products.",
        flush=True,
    )
    return True


def show_status(*, collection_override: str = "") -> int:
    from pymilvus import MilvusClient

    settings = get_settings()
    collection_name = (
        collection_override
        or settings.text_milvus_collection
        or settings.milvus_collection
    )
    engine = get_engine()
    _ensure_bootstrap_metadata_table(engine)
    checkpoint = _load_checkpoint(engine, _checkpoint_key(collection_name))
    client = MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token)
    row_count = _collection_row_count(client, collection_name)
    print(f"collection={collection_name}")
    print(f"milvus_row_count={row_count}")
    if checkpoint is None:
        print("checkpoint=missing")
        return 0
    print(json.dumps(asdict(checkpoint), ensure_ascii=False, indent=2))
    return 0


def _load_products() -> list[Product]:
    SessionLocal = get_sessionmaker()
    with SessionLocal() as db:
        return list(db.query(Product).order_by(Product.product_id.asc()).all())


def _assert_remote_text_embedding_configured(embedder: EmbeddingClient) -> None:
    if not embedder._is_configured():
        raise RuntimeError(
            "Remote text embedding is required. Set EMBEDDING_API_KEY, "
            "EMBEDDING_BASE_URL and EMBEDDING_MODEL before rebuilding."
        )


def _ensure_collection(client: Any, collection_name: str, dim: int, *, overwrite: bool) -> None:
    from pymilvus import DataType, MilvusClient

    if overwrite and client.has_collection(collection_name):
        client.drop_collection(collection_name)
    if client.has_collection(collection_name):
        return
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=65535)
    schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dim)
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
        consistency_level="Bounded",
    )


def _build_row(
    product: Product,
    embedder: EmbeddingClient,
    *,
    embedding: list[float] | None = None,
) -> dict[str, Any]:
    from llama_index.core.schema import TextNode

    text = product.search_text()
    metadata = {
        "product_id": product.product_id,
        "category": product.category,
        "brand": product.brand,
        "price": float(product.price),
        "tags": product.tags,
    }
    node = TextNode(text="", id_=product.product_id, metadata=metadata)
    return {
        "id": product.product_id,
        "doc_id": "None",
        "text": text,
        "embedding": embedding if embedding is not None else embedder.embed_remote_required(text),
        **metadata,
        "_node_content": node.to_json(),
        "_node_type": "TextNode",
        "document_id": "None",
        "ref_doc_id": "None",
    }


def _commit_batch(
    client: Any,
    collection_name: str,
    rows: list[dict[str, Any]],
    engine: Any,
    checkpoint_key: str,
    checkpoint: TextIndexCheckpoint,
    *,
    next_index: int,
    last_success_product_id: str,
) -> None:
    if hasattr(client, "upsert"):
        client.upsert(collection_name=collection_name, data=rows)
    else:
        client.insert(collection_name=collection_name, data=rows)
    client.flush(collection_name)
    checkpoint.next_index = next_index
    checkpoint.last_success_product_id = last_success_product_id
    checkpoint.status = "running"
    checkpoint.failed_product_id = ""
    checkpoint.failed_position = None
    checkpoint.error = ""
    _save_checkpoint(engine, checkpoint_key, checkpoint)


def _checkpoint_key(collection_name: str) -> str:
    return f"{CHECKPOINT_KEY_PREFIX}:{collection_name}"


def _build_fingerprint(products: list[Product], collection_name: str) -> str:
    settings = get_settings()
    digest = hashlib.sha256()
    header = {
        "version": 1,
        "collection": collection_name,
        "base_url": settings.embedding_base_url or "",
        "model": settings.embedding_model or "",
        "dim": int(settings.embedding_dim),
    }
    digest.update(json.dumps(header, sort_keys=True).encode("utf-8"))
    for product in products:
        digest.update(product.product_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(product.search_text().encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_checkpoint(
    checkpoint: TextIndexCheckpoint,
    *,
    build_fingerprint: str,
    dataset_fingerprint: str,
    model: str,
    embedding_dim: int,
    total_products: int,
) -> None:
    mismatches: list[str] = []
    if checkpoint.build_fingerprint != build_fingerprint:
        mismatches.append("product text/order or embedding configuration changed")
    if checkpoint.dataset_fingerprint != dataset_fingerprint:
        mismatches.append("dataset fingerprint changed")
    if checkpoint.model != model:
        mismatches.append(f"model changed ({checkpoint.model} -> {model})")
    if checkpoint.embedding_dim != embedding_dim:
        mismatches.append(
            f"embedding dimension changed ({checkpoint.embedding_dim} -> {embedding_dim})"
        )
    if checkpoint.total_products != total_products:
        mismatches.append(
            f"product count changed ({checkpoint.total_products} -> {total_products})"
        )
    if checkpoint.next_index < 0 or checkpoint.next_index > total_products:
        mismatches.append(f"invalid next_index {checkpoint.next_index}")
    if mismatches:
        raise RuntimeError(
            "The saved checkpoint cannot be resumed because "
            + "; ".join(mismatches)
            + ". Run with --reset to start a clean text index."
        )


def _load_checkpoint(engine: Any, key: str) -> TextIndexCheckpoint | None:
    raw = _read_bootstrap_metadata(engine, key)
    if not raw:
        return None
    try:
        return TextIndexCheckpoint(**json.loads(raw))
    except Exception as exc:
        raise RuntimeError(f"Invalid text-index checkpoint '{key}': {exc}") from exc


def _save_checkpoint(engine: Any, key: str, checkpoint: TextIndexCheckpoint) -> None:
    checkpoint.updated_at = datetime.now(timezone.utc).isoformat()
    _write_bootstrap_metadata(
        engine,
        key,
        json.dumps(asdict(checkpoint), ensure_ascii=False, sort_keys=True),
    )


def _collection_row_count(client: Any, collection_name: str) -> int:
    if not client.has_collection(collection_name):
        return 0
    stats = client.get_collection_stats(collection_name)
    return int(stats.get("row_count") or 0)


def _error_text(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}".strip()
    return text[:2000]


if __name__ == "__main__":
    raise SystemExit(main())
