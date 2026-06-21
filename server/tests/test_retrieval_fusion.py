from app.domain.retrieval_fusion import rrf_fuse


def test_rrf_fuse_promotes_items_found_by_multiple_retrievers() -> None:
    fused = rrf_fuse(
        [
            ["p1", "p2", "p3"],
            ["p4", "p2", "p5"],
        ],
        top_k=5,
    )

    assert fused[0] == "p2"
    assert set(fused) == {"p1", "p2", "p3", "p4", "p5"}


def test_rrf_fuse_respects_top_k() -> None:
    fused = rrf_fuse(
        [
            ["p1", "p2", "p3"],
            ["p2", "p4", "p5"],
        ],
        top_k=3,
    )

    assert len(fused) == 3
    assert "p2" in fused
