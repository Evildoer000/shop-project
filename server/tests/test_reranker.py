from decimal import Decimal
from types import SimpleNamespace

import pytest

from app.db.models import Product
from app.domain.reranker import HybridReranker, RemoteCrossEncoderReranker, build_reranker


def make_product(product_id: str, name: str) -> Product:
    return Product(
        product_id=product_id,
        name=name,
        category="美妆护肤",
        sub_category="防晒",
        brand="Demo",
        price=Decimal("99"),
        stock=10,
        image_url="",
        description=name,
        specs={},
        ingredients_or_material="",
        suitable_for="",
        avoid_for="",
        tags=["防晒"],
        rating=Decimal("4.8"),
        sales=1200,
        review_summary="",
        image_caption="",
        structured_attributes={},
    )


def test_reranker_parses_results_format() -> None:
    reranker = RemoteCrossEncoderReranker()
    products = [make_product("p1", "A 防晒"), make_product("p2", "B 防晒")]

    scores = reranker._parse_scores(
        {"results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.2}]},
        len(products),
    )

    assert scores == [(1, 0.9), (0, 0.2)]


def test_hybrid_reranker_preserves_fused_order_without_remote_call() -> None:
    reranker = HybridReranker()
    products = [
        make_product("p1", "第一名"),
        make_product("p2", "第二名"),
        make_product("p3", "第三名"),
    ]

    ranked = reranker.rerank("防晒", products, top_k=2)

    assert [product.product_id for product, _score in ranked] == ["p1", "p2"]
    assert ranked[0][1] > ranked[1][1]


def test_build_reranker_defaults_to_hybrid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.domain.reranker.get_settings",
        lambda: SimpleNamespace(rerank_backend="hybrid"),
    )

    assert isinstance(build_reranker(), HybridReranker)


def test_reranker_requires_remote_config(monkeypatch: pytest.MonkeyPatch) -> None:
    reranker = RemoteCrossEncoderReranker()
    monkeypatch.setattr(reranker, "settings", type("Settings", (), {
        "rerank_api_key": None,
        "rerank_base_url": None,
        "rerank_model": None,
    })())

    with pytest.raises(RuntimeError, match="Rerank API 未配置"):
        reranker.rerank("防晒", [make_product("p1", "A 防晒")])
