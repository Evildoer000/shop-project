from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.db.session import get_sessionmaker
from app.db.schema import ensure_database_schema
from app.db.session import get_engine
from app.domain.memory_distiller import LongTermDistiller


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    ensure_database_schema(get_engine())
    SessionLocal = get_sessionmaker()
    with SessionLocal() as db:
        result = await LongTermDistiller(db).run_daily()
    print(f"distill done: {result}")


if __name__ == "__main__":
    asyncio.run(main())
