from datetime import datetime, timedelta, timezone
from decimal import Decimal

from app.db.models import Product, UserEvent
from app.schemas import EventReportRequest
from app.services.event_service import CHAT_SOURCE_MULTIPLIER, EVENT_WEIGHT, EventService


def test_event_weight_boosts_chat_clicks() -> None:
    req = EventReportRequest(
        user_id="u1",
        session_id="s1",
        event_type="click",
        product_id="p1",
        context={"from": "chat"},
    )

    assert EventService._compute_weight(req) == EVENT_WEIGHT["click"] * CHAT_SOURCE_MULTIPLIER


def test_event_decay_keeps_recent_affinity_higher_than_old_affinity() -> None:
    now = datetime.now(timezone.utc)

    recent = EventService._decay(1.0, now - timedelta(days=1), now)
    old = EventService._decay(1.0, now - timedelta(days=60), now)

    assert recent > old
    assert 0 < old < 1.0


def test_cart_snapshot_aggregates_cart_events_across_sessions() -> None:
    product = Product(
        product_id="p1",
        name="防晒霜",
        category="美妆个护",
        sub_category="防晒",
        brand="Demo",
        price=Decimal("99"),
        stock=10,
        image_url="/dataset/p1.jpg",
        description="",
        specs={},
        ingredients_or_material="",
        suitable_for="",
        avoid_for="",
        tags=[],
        rating=Decimal("4.8"),
        sales=100,
        review_summary="",
        image_caption="",
        structured_attributes={},
    )
    events = [
        UserEvent(event_id=1, user_id="u1", session_id="s1", event_type="cart_add", product_id="p1", context={}),
        UserEvent(event_id=2, user_id="u1", session_id="s2", event_type="cart_add", product_id="p1", context={}),
        UserEvent(event_id=3, user_id="u1", session_id="cart", event_type="cart_remove", product_id="p1", context={}),
    ]

    cart = EventService(FakeCartSession(events, [product])).cart_snapshot("u1", "all")

    assert cart.total_quantity == 1
    assert cart.total_price == Decimal("99")
    assert cart.items[0].product_id == "p1"
    assert cart.items[0].quantity == 1


class FakeCartSession:
    def __init__(self, events, products) -> None:
        self.events = events
        self.products = products
        self.calls = 0

    def scalars(self, stmt):
        self.calls += 1
        return FakeScalarResult(self.events if self.calls == 1 else self.products)


class FakeScalarResult:
    def __init__(self, rows) -> None:
        self.rows = rows

    def all(self):
        return self.rows
