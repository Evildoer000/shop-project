from datetime import datetime, timedelta, timezone

from app.harness.evidence_cache import EvidenceBundle, EvidenceCandidate, InMemoryEvidenceCache


def _bundle(turn_id: str, created_at: datetime, candidate_count: int = 3) -> EvidenceBundle:
    product_ids = [f"p{i}" for i in range(candidate_count)]
    return EvidenceBundle(
        user_id="u1",
        session_id="s1",
        turn_id=turn_id,
        query=f"query {turn_id}",
        execution_path="single_retrieval",
        final_route="recommend",
        displayed_product_ids=product_ids[:2],
        selected_product_ids=product_ids[:2],
        candidate_product_ids=product_ids,
        candidates=[
            EvidenceCandidate(product_id=product_id, score=1.0 - index / 100, stage="rerank")
            for index, product_id in enumerate(product_ids)
        ],
        created_at=created_at,
    )


def test_evidence_cache_expires_by_ttl() -> None:
    now = [datetime(2026, 1, 1, tzinfo=timezone.utc)]
    cache = InMemoryEvidenceCache(ttl_seconds=60, now=lambda: now[0])

    cache.put_turn_evidence(_bundle("t1", now[0]))
    now[0] = now[0] + timedelta(seconds=61)

    assert cache.get_latest_evidence("s1") is None


def test_evidence_cache_keeps_recent_20_and_truncates_candidates() -> None:
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    cache = InMemoryEvidenceCache(ttl_seconds=3600, recent_turns=20, max_candidates_per_turn=20, now=lambda: base_time)

    for index in range(25):
        cache.put_turn_evidence(_bundle(f"t{index}", base_time + timedelta(seconds=index), candidate_count=25))

    recent = cache.get_recent_evidence("s1", limit=50)

    assert [bundle.turn_id for bundle in recent] == [f"t{index}" for index in range(5, 25)]
    assert all(len(bundle.candidates) == 20 for bundle in recent)
    assert all(len(bundle.candidate_product_ids) == 20 for bundle in recent)


def test_evidence_cache_returns_deep_copies() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    cache = InMemoryEvidenceCache(now=lambda: now)
    cache.put_turn_evidence(_bundle("t1", now))

    latest = cache.get_latest_evidence("s1")
    assert latest is not None
    latest.selected_product_ids.append("mutated")
    latest.candidates[0].compact_product["name"] = "mutated"

    fresh = cache.get_latest_evidence("s1")
    assert fresh is not None
    assert fresh.selected_product_ids == ["p0", "p1"]
    assert fresh.candidates[0].compact_product == {}


def test_evidence_cache_get_turn_and_compact_recent() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    cache = InMemoryEvidenceCache(now=lambda: now)
    cache.put_turn_evidence(_bundle("t1", now))

    assert cache.get_turn_evidence("s1", "t1") is not None
    compact = cache.compact_recent("s1")
    assert compact[0]["turn_id"] == "t1"
    assert compact[0]["displayed_product_ids"] == ["p0", "p1"]
