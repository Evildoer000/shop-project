from decimal import Decimal
from pathlib import Path

import pytest

from app.db.models import Product
from app.domain.image_search_tool import ImageSearchTool
from app.domain.need_slot_schemas import NeedSlot
from app.rag.image_retriever import ImageIndexUnavailable, ImageRetriever
from app.schemas import QueryPlan


def make_product(product_id: str = "p1") -> Product:
    return Product(
        product_id=product_id,
        name="相似商品",
        category="家居日用",
        sub_category="纸品",
        brand="Demo",
        price=Decimal("29.9"),
        stock=10,
        image_url="",
        description="相似商品",
        specs={},
        ingredients_or_material="",
        suitable_for="",
        avoid_for="",
        tags=[],
        rating=Decimal("4.5"),
        sales=100,
        review_summary="",
        image_caption="",
        structured_attributes={},
    )


class MissingCollectionClient:
    def has_collection(self, collection_name: str) -> bool:
        return False

    def search(self, **kwargs):
        raise AssertionError("search should not run when the image collection is missing")


class CountingEmbeddingClient:
    def __init__(self) -> None:
        self.image_calls = 0

    def embed_image(self, image_path):
        self.image_calls += 1
        raise AssertionError("embedding should not run when the image collection is missing")


class FakeProductRepository:
    def __init__(self, products: list[Product]) -> None:
        self.products = products

    def list_for_plan(self, plan: QueryPlan, limit: int) -> list[Product]:
        return self.products[:limit]

    def count_available(self) -> int:
        return len(self.products)


class UnavailableImageRetriever:
    def retrieve(self, image_path, candidates, top_k):
        raise ImageIndexUnavailable("Milvus image collection 'product_image_chunks' is not available.")


def test_image_retriever_does_not_embed_when_collection_missing() -> None:
    embedding_client = CountingEmbeddingClient()
    retriever = ImageRetriever(embedding_client=embedding_client)  # type: ignore[arg-type]
    retriever._client = MissingCollectionClient()

    with pytest.raises(ImageIndexUnavailable):
        retriever.retrieve("uploaded.jpg", [make_product()])

    assert embedding_client.image_calls == 0


def test_image_search_tool_returns_empty_evidence_when_image_index_unavailable() -> None:
    product = make_product()
    tool = ImageSearchTool(
        product_repository=FakeProductRepository([product]),  # type: ignore[arg-type]
        image_retriever=UnavailableImageRetriever(),  # type: ignore[arg-type]
    )

    result = tool.search(
        slot=NeedSlot(slot_id="image", goal="找相似商品", query="找相似商品"),
        base_plan=QueryPlan(),
        image_path=Path("uploaded.jpg"),
        top_k=5,
    )

    assert result.category_resolution == "image_retrieval_unavailable"
    assert result.candidates == []
    assert result.counts["image_hits"] == 0
    assert result.attempts[0]["status"] == "unavailable"
    assert result.structured_products == [product]
