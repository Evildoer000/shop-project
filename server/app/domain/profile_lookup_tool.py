from __future__ import annotations

from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import UserMemory


SOURCE_PRIORITY = {
    "explicit": 0,
    "distilled": 1,
}


class ProfileLookupTool:
    def __init__(self, db: Session) -> None:
        self.db = db

    def lookup(self, user_id: str, query: str = "", limit: int = 8) -> list[dict[str, Any]]:
        rows = self.db.scalars(
            select(UserMemory).where(UserMemory.user_id == user_id)
        ).all()
        if not rows:
            return []
        terms = self._terms(query)

        def sort_key(row: UserMemory) -> tuple[int, int, float, int]:
            text = f"{row.key} {row.value}"
            matched = 1 if terms and any(term in text for term in terms) else 0
            confidence = self._confidence(row.confidence)
            return (
                SOURCE_PRIORITY.get(row.source, 2),
                -matched,
                -confidence,
                row.memory_id,
            )

        sorted_rows = sorted(rows, key=sort_key)
        result = []
        for row in sorted_rows[: max(1, limit)]:
            result.append(
                {
                    "memory_id": row.memory_id,
                    "memory_type": row.memory_type,
                    "key": row.key,
                    "value": row.value,
                    "confidence": self._confidence(row.confidence),
                    "source": row.source,
                    "usage_rule": (
                        "只能作为软偏好；不能覆盖 current_query，不能自动改 product_type、need_slots、budget_scope 或 negative_terms。"
                    ),
                }
            )
        return result

    @staticmethod
    def _terms(query: str) -> list[str]:
        return [term for term in str(query or "").replace("，", " ").replace(",", " ").split() if len(term) >= 2]

    @staticmethod
    def _confidence(value: Decimal | float | int | None) -> float:
        if value is None:
            return 0.0
        return float(value)
