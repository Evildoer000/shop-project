from decimal import Decimal

from app.db.models import Product
from app.services.product_repository import ProductRepository


def make_product(product_id: str, name: str, sub_category: str, brand: str, tags: list[str]) -> Product:
    return Product(
        product_id=product_id,
        name=name,
        category="食品饮料" if sub_category == "咖啡" else "数码电子",
        sub_category=sub_category,
        brand=brand,
        price=Decimal("100"),
        stock=None,
        image_url="",
        description=name,
        specs={},
        ingredients_or_material="",
        suitable_for="",
        avoid_for="",
        tags=tags,
        rating=Decimal("4.8"),
        sales=None,
        review_summary="",
        image_caption="",
        structured_attributes={},
    )


def test_keyword_scores_prefer_specific_coffee_terms() -> None:
    products = [
        make_product("black", "三顿半 冷萃超即溶 黑咖啡 精品速溶咖啡", "咖啡", "三顿半", ["咖啡"]),
        make_product("latte", "雀巢咖啡 1+2原味 三合一速溶咖啡粉 即冲奶香咖啡饮品", "咖啡", "雀巢", ["咖啡"]),
    ]

    scores = ProductRepository(None).keyword_scores("想买奶香一点、即冲方便的三合一速溶咖啡", products)  # type: ignore[arg-type]

    assert scores["latte"] > scores["black"]


def test_keyword_scores_prefer_professional_laptop_terms() -> None:
    products = [
        make_product("air", "Apple MacBook Air 13英寸 M5 芯片 轻薄便携笔记本电脑", "笔记本电脑", "Apple 苹果", ["笔记本电脑"]),
        make_product("pro", "Apple MacBook Pro 16英寸 M5 Pro 芯片 创意设计师专业高性能笔记本电脑", "笔记本电脑", "Apple 苹果", ["笔记本电脑"]),
    ]

    scores = ProductRepository(None).keyword_scores("做视频剪辑和设计，想买性能强的专业笔记本", products)  # type: ignore[arg-type]

    assert scores["pro"] > scores["air"]


def test_repository_exclude_terms_are_not_hard_filters() -> None:
    product = make_product("earbuds", "主动降噪真无线蓝牙耳机", "真无线耳机", "Demo", ["耳机"])
    product.description = "可连接手机、平板和电脑使用。"

    assert ProductRepository(None)._matches_exclude(product, ["手机"]) is False  # type: ignore[attr-defined]


class FakeScalarResult:
    def __init__(self, products: list[Product]) -> None:
        self.products = products

    def all(self) -> list[Product]:
        return self.products


class FakeDb:
    def __init__(self, products: list[Product]) -> None:
        self.products = products

    def scalars(self, stmt) -> FakeScalarResult:
        return FakeScalarResult(self.products)


def test_get_by_ids_preserves_requested_order_and_ignores_missing_ids() -> None:
    products = [
        make_product("a", "商品 A", "咖啡", "Demo", []),
        make_product("b", "商品 B", "咖啡", "Demo", []),
    ]

    result = ProductRepository(FakeDb(products)).get_by_ids(["b", "missing", "a"])  # type: ignore[arg-type]

    assert [product.product_id for product in result] == ["b", "a"]
