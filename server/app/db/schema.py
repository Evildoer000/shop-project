from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from app.db.models import Base


def ensure_database_schema(engine: Engine) -> None:
    Base.metadata.create_all(bind=engine)
    _ensure_session_memory_distilled_at(engine)
    _ensure_product_stock_defaults(engine)
    _ensure_agent_run_span_tree_columns(engine)


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


def _ensure_product_stock_defaults(engine: Engine) -> None:
    inspector = inspect(engine)
    if not inspector.has_table("products"):
        return
    with engine.begin() as connection:
        connection.execute(text("UPDATE products SET stock = 1 WHERE stock IS NULL"))


def _ensure_agent_run_span_tree_columns(engine: Engine) -> None:
    """Add execution-tree fields without deleting or rebuilding existing spans."""
    inspector = inspect(engine)
    if not inspector.has_table("agent_run_spans"):
        return

    existing_columns = {column["name"] for column in inspector.get_columns("agent_run_spans")}
    is_sqlite = engine.dialect.name == "sqlite"
    string_type = "TEXT" if is_sqlite else "VARCHAR(96)"
    task_type = "TEXT" if is_sqlite else "VARCHAR(128)"
    agent_type = "TEXT" if is_sqlite else "VARCHAR(96)"
    version_type = "TEXT" if is_sqlite else "VARCHAR(32)"
    reason_type = "TEXT" if is_sqlite else "VARCHAR(128)"
    additions = {
        "span_key": f"{string_type}",
        "parent_span_key": f"{string_type}",
        "task_id": f"{task_type} NOT NULL DEFAULT ''",
        "agent_id": f"{agent_type} NOT NULL DEFAULT ''",
        "span_type": f"{version_type} NOT NULL DEFAULT 'stage'",
        "attempt": "INTEGER NOT NULL DEFAULT 1",
        "sequence": "INTEGER NOT NULL DEFAULT 0",
        "trace_schema_version": f"{version_type} NOT NULL DEFAULT 'v1'",
        "termination_reason": f"{reason_type} NOT NULL DEFAULT ''",
    }
    with engine.begin() as connection:
        for column_name, column_definition in additions.items():
            if column_name not in existing_columns:
                connection.execute(
                    text(
                        "ALTER TABLE agent_run_spans "
                        f"ADD COLUMN {column_name} {column_definition}"
                    )
                )

        existing_indexes = {index["name"] for index in inspect(connection).get_indexes("agent_run_spans")}
        indexes = {
            "ix_agent_run_spans_parent": "parent_span_key",
            "ix_agent_run_spans_task": "task_id",
            "ix_agent_run_spans_agent": "agent_id",
            "ix_agent_run_spans_sequence": "run_id, sequence",
        }
        for index_name, index_columns in indexes.items():
            if index_name not in existing_indexes:
                connection.execute(
                    text(
                        f"CREATE INDEX {index_name} "
                        f"ON agent_run_spans ({index_columns})"
                    )
                )
