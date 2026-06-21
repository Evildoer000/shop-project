from decimal import Decimal

from app.db.models import Product
from app.domain.need_slot_schemas import NeedSlot
from app.domain.product_search_tool import ProductSearchTool
from app.schemas import IntentPlan, QueryPlan


def make_product(product_id: str, name: str, category: str, sub_category: str) -> Product:
    return Product(
        product_id=product_id,
        name=name,
        category=category,
        sub_category=sub_category,
        brand="Demo",
        price=Decimal("99"),
        stock=10,
        image_url="",
        description=name,
        specs={},
        ingredients_or_material="",
        suitable_for="",
        avoid_for="",
        tags=[sub_category],
        rating=Decimal("4.8"),
        sales=100,
        review_summary="",
        image_caption="",
        structured_attributes={},
    )


class DummyRepository:
    def __init__(self, products: list[Product]) -> None:
        self.products = products
        self.last_categories: list[str] = []

    def count_available(self) -> int:
        return len(self.products)

    def list_for_plan(self, plan: QueryPlan, limit: int = 200) -> list[Product]:
        self.last_categories = plan.categories
        return self.products[:limit]

    def keyword_scores(self, query: str, products: list[Product], top_k: int | None = None) -> dict[str, float]:
        result = {
            product.product_id: 0.5
            for product in products
            if any(token and token in product.search_text() for token in query.split())
        }
        return dict(list(result.items())[:top_k]) if top_k else result


class DummyRetriever:
    def retrieve(self, query: str, products: list[Product], top_k: int = 12) -> dict[str, float]:
        return {
            product.product_id: 0.5
            for product in products
            if any(token and token in product.search_text() for token in query.split())
        }


class DummyReranker:
    def rerank(self, query: str, products: list[Product], top_k: int = 5) -> list[tuple[Product, float]]:
        return [(product, 0.4) for product in products[:top_k]]


def make_tool(products: list[Product]) -> ProductSearchTool:
    return ProductSearchTool(DummyRepository(products), DummyRetriever(), DummyReranker())


def test_resolves_exact_sub_category_before_top_category() -> None:
    tool = make_tool([])
    slot = NeedSlot(slot_id="s1", goal="露营收纳", product_type="背包", query="背包 徒步鞋 方便食品")

    categories, resolution = tool._resolve_categories(slot)

    assert categories == ["背包"]
    assert resolution == "exact_sub_category"


def test_specific_unknown_shoe_term_searches_top_category() -> None:
    tool = make_tool([])
    slot = NeedSlot(slot_id="s1", goal="旅行步行轻便鞋履", product_type="轻便鞋", query="轻便鞋")

    categories, resolution = tool._resolve_categories(slot)

    assert categories == ["服饰运动"]
    assert resolution == "top_category_from_product_type"


def test_out_of_catalog_desktop_host_searches_digital_top_category() -> None:
    tool = make_tool([])
    slot = NeedSlot(slot_id="s1", goal="主机", product_type="主机", query="主机")

    categories, resolution = tool._resolve_categories(slot)

    assert categories == ["数码电子"]
    assert resolution == "top_category_from_product_type"


def test_slot_query_variants_do_not_use_polluted_original_query() -> None:
    tool = make_tool([])
    slot = NeedSlot(
        slot_id="s1",
        goal="露营收纳",
        product_type="背包",
        query="背包 徒步鞋 方便食品 帽子",
        soft_constraints=["露营"],
    )

    variants = tool._query_variants(slot)

    assert variants
    assert all("方便食品" not in variant for variant in variants)
    assert all("徒步鞋" not in variant for variant in variants)


def test_search_records_repair_attempts_and_slot_pool() -> None:
    product = make_product("p1", "Osprey 户外双肩背包", "服饰运动", "背包")
    tool = make_tool([product])
    slot = NeedSlot(
        slot_id="s1",
        goal="露营收纳",
        product_type="背包",
        query="背包 徒步鞋 方便食品 帽子",
        soft_constraints=["露营"],
    )

    result = tool.search(slot, QueryPlan(), IntentPlan(original_query="露营装备"))

    assert result.categories == ["背包"]
    assert result.attempts
    assert result.attempts[0]["categories"] == ["背包"]
    assert result.candidates[0].product_id == "p1"


def test_search_uses_up_to_three_attempts_when_signal_is_empty() -> None:
    product = make_product("p1", "雅诗兰黛持妆粉底液", "美妆护肤", "粉底液")
    tool = make_tool([product])
    slot = NeedSlot(
        slot_id="s1",
        goal="露营收纳",
        product_type="背包",
        query="背包 徒步鞋 方便食品 帽子",
        soft_constraints=["露营"],
    )

    result = tool.search(slot, QueryPlan(), IntentPlan(original_query="露营装备"))

    assert len(result.attempts) == 3
    assert result.candidates == []
