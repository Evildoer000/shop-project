from decimal import Decimal

from app.db.models import Product, UserEvent
from app.domain.recommendation_service import RecommendationService
from app.schemas import EventReportRequest, RecommendationResponse
from app.services.event_service import EventService


def make_product(product_id: str, name: str, brand: str = "Demo", rating: str = "4.8") -> Product:
    return Product(
        product_id=product_id,
        name=name,
        category="美妆个护",
        sub_category="防晒",
        brand=brand,
        price=Decimal("99"),
        stock=10,
        image_url=f"/dataset/{product_id}.jpg",
        description=name,
        specs={},
        ingredients_or_material="",
        suitable_for="",
        avoid_for="",
        tags=["防晒"],
        rating=Decimal(rating),
        sales=100,
        review_summary="",
        image_caption="",
        structured_attributes={},
    )


def test_recommendation_service_returns_products_from_db() -> None:
    products = [make_product("p1", "防晒霜"), make_product("p2", "修护霜")]
    response = RecommendationService(FakeRecommendationDb(products)).get_home_recommendations("u1", size=2)

    assert isinstance(response, RecommendationResponse)
    assert {card.product_id for card in response.products}.issubset({"p1", "p2"})
    assert len(response.products) == 2


def test_recommendation_response_schema_is_stable() -> None:
    products = [make_product("p1", "防晒霜")]
    response = RecommendationService(FakeRecommendationDb(products)).get_home_recommendations("u1", size=1)

    assert response.model_dump()["products"][0]["product_id"] == "p1"
    assert response.stage == "cold"


def test_event_service_accepts_mall_behavior_event() -> None:
    req = EventReportRequest(
        user_id="u1",
        session_id="mall",
        event_type="impression",
        product_id="p1",
        context={"from": "mall"},
    )
    db = FakeEventDb()

    event = EventService(db).write_event(req)  # type: ignore[arg-type]

    assert event.event_type == "impression"
    assert db.committed


class FakeRecommendationDb:
    def __init__(self, products: list[Product]) -> None:
        self.products = products
        self.scalar_calls = 0
        self.scalars_calls = 0

    def scalar(self, stmt):
        self.scalar_calls += 1
        return 0

    def scalars(self, stmt):
        self.scalars_calls += 1
        if self.scalars_calls == 1:
            return FakeScalarResult(self.products)
        return FakeScalarResult([])

    def execute(self, stmt):
        return FakeScalarResult([])


class FakeScalarResult:
    def __init__(self, rows) -> None:
        self.rows = rows

    def all(self):
        return self.rows


class FakeEventDb:
    def __init__(self) -> None:
        self.events: list[UserEvent] = []
        self.committed = False

    def add(self, event: UserEvent) -> None:
        event.event_id = len(self.events) + 1
        self.events.append(event)

    def commit(self) -> None:
        self.committed = True

    def refresh(self, event: UserEvent) -> None:
        return None
