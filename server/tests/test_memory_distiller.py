from decimal import Decimal

from app.db.models import UserMemory
from app.domain.memory_distiller import LongTermDistiller


class FakeDb:
    def __init__(self, existing=None) -> None:
        self.existing = existing
        self.added = []

    def scalar(self, statement):
        return self.existing

    def add(self, item) -> None:
        self.added.append(item)


def test_distiller_does_not_downgrade_explicit_memory_source() -> None:
    existing = UserMemory(
        user_id="u1",
        memory_type="preference",
        key="肤质",
        value="油皮",
        confidence=Decimal("1.00"),
        source="explicit",
    )
    distiller = LongTermDistiller.__new__(LongTermDistiller)
    distiller.db = FakeDb(existing)

    distiller._upsert_distilled_memory("u1", {"key": "肤质", "value": "油皮", "confidence": 0.7})

    assert existing.source == "explicit"
    assert existing.confidence == Decimal("1.00")


def test_distiller_updates_existing_distilled_memory() -> None:
    existing = UserMemory(
        user_id="u1",
        memory_type="preference",
        key="关注品牌",
        value="Nike",
        confidence=Decimal("0.50"),
        source="distilled",
    )
    distiller = LongTermDistiller.__new__(LongTermDistiller)
    distiller.db = FakeDb(existing)

    distiller._upsert_distilled_memory("u1", {"key": "关注品牌", "value": "Nike", "confidence": 0.8})

    assert existing.source == "distilled"
    assert existing.confidence == Decimal("0.80")


def test_distiller_adds_new_distilled_memory() -> None:
    db = FakeDb(existing=None)
    distiller = LongTermDistiller.__new__(LongTermDistiller)
    distiller.db = db

    distiller._upsert_distilled_memory("u1", {"key": "价位偏好", "value": "500-1000", "confidence": 0.75})

    assert len(db.added) == 1
    assert db.added[0].source == "distilled"
    assert db.added[0].key == "价位偏好"
