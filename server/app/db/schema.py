from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from app.db.models import Base


def ensure_database_schema(engine: Engine) -> None:
    Base.metadata.create_all(bind=engine)
    _ensure_session_memory_distilled_at(engine)


def _ensure_session_memory_distilled_at(engine: Engine) -> None:
    inspector = inspect(engine)
    if not inspector.has_table("session_memory_states"):
        return
    columns = {column["name"] for column in inspector.get_columns("session_memory_states")}
    if "distilled_at" in columns:
        return
    column_type = "DATETIME" if engine.dialect.name == "sqlite" else "TIMESTAMP WITH TIME ZONE"
    with engine.begin() as connection:
        connection.execute(
            text(f"ALTER TABLE session_memory_states ADD COLUMN distilled_at {column_type}")
        )
