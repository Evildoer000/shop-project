from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from sqlalchemy import text


SCRIPT_ROOT = Path(__file__).resolve().parent
SERVER_ROOT = SCRIPT_ROOT.parents[0]
sys.path.insert(0, str(SERVER_ROOT))
sys.path.insert(0, str(SCRIPT_ROOT))

from app.core.config import get_settings
from app.db.session import get_engine
from index_image_vectors import bootstrap_image_index
from seed_products import main as seed_products_main


def main() -> None:
    timeout_seconds = int(os.getenv("BOOTSTRAP_WAIT_SECONDS", "180"))
    _wait_for_postgres(timeout_seconds)
    _wait_for_milvus(timeout_seconds)
    seed_products_main()
    bootstrap_image_index(skip_without_remote=True)


def _wait_for_postgres(timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with get_engine().connect() as connection:
                connection.execute(text("select 1"))
            print("PostgreSQL is ready.")
            return
        except Exception as exc:
            last_error = exc
            print(f"Waiting for PostgreSQL: {exc}")
            time.sleep(2)
    raise RuntimeError("Timed out waiting for PostgreSQL.") from last_error


def _wait_for_milvus(timeout_seconds: int) -> None:
    from pymilvus import MilvusClient

    settings = get_settings()
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            client = MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token)
            client.list_collections()
            print("Milvus is ready.")
            return
        except Exception as exc:
            last_error = exc
            print(f"Waiting for Milvus: {exc}")
            time.sleep(2)
    raise RuntimeError("Timed out waiting for Milvus.") from last_error


if __name__ == "__main__":
    main()
