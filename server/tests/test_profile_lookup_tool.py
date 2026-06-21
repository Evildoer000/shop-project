from decimal import Decimal

from app.db.models import UserMemory
from app.domain.profile_lookup_tool import ProfileLookupTool


class FakeScalarResult:
    def __init__(self, rows) -> None:
        self.rows = rows

    def all(self):
        return self.rows


class FakeDb:
    def __init__(self, rows) -> None:
        self.rows = rows

    def scalars(self, statement):
        return FakeScalarResult(self.rows)


def make_memory(memory_id: int, key: str, value: str, source: str, confidence: str) -> UserMemory:
    return UserMemory(
        memory_id=memory_id,
        user_id="u1",
        memory_type="preference",
        key=key,
        value=value,
        source=source,
        confidence=Decimal(confidence),
    )


def test_profile_lookup_prioritizes_explicit_and_marks_soft_usage() -> None:
    rows = [
        make_memory(1, "关注品牌", "Nike", "distilled", "0.95"),
        make_memory(2, "肤质", "油皮", "explicit", "0.80"),
    ]

    result = ProfileLookupTool(FakeDb(rows)).lookup("u1", "肤质 偏好")

    assert [item["memory_id"] for item in result] == [2, 1]
    assert result[0]["source"] == "explicit"
    assert "只能作为软偏好" in result[0]["usage_rule"]
