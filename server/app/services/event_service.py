from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Product, UserBrandAffinity, UserEvent, UserProductAffinity
from app.schemas import CartItemResponse, CartResponse, EventReportRequest


LOGGER = logging.getLogger(__name__)

EVENT_WEIGHT = {
    "impression": 0.03,
    "click": 0.30,
    "view": 0.10,
    "detail_view": 0.15,
    "cart_add": 0.60,
    "cart_remove": -0.30,
    "buy": 1.00,
    "favorite": 0.50,
    "dismiss": -0.20,
}
CHAT_SOURCE_MULTIPLIER = 1.5
HALF_LIFE_DAYS = 30


class EventService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def write_event(self, req: EventReportRequest) -> UserEvent:
        event = UserEvent(
            user_id=req.user_id,
            session_id=req.session_id,
            event_type=req.event_type,
            product_id=req.product_id,
            turn_id=req.turn_id,
            position=req.position,
            context=req.context or {},
        )
        self.db.add(event)
        self.db.commit()
        self.db.refresh(event)
        return event

    def cart_snapshot(self, user_id: str, session_id: str) -> CartResponse:
        import json as _json

        stmt = (
            select(UserEvent)
            .where(UserEvent.user_id == user_id)
            .where(UserEvent.event_type.in_(["cart_add", "cart_remove", "buy"]))
            .order_by(UserEvent.created_at.asc(), UserEvent.event_id.asc())
        )
        if session_id != "all":
            stmt = stmt.where(UserEvent.session_id == session_id)
        events = self.db.scalars(stmt).all()

        # 按 (product_id, sku_signature) 聚合, 不同 sku 的同商品分开计数
        # sku_signature: sku dict 排序 JSON 化 → "" 当无 sku
        def _sku_sig(event: UserEvent) -> tuple[str, dict]:
            ctx = event.context or {}
            raw = ctx.get("sku") if isinstance(ctx, dict) else None
            if isinstance(raw, str) and raw.strip():
                try:
                    sku_dict = _json.loads(raw)
                    if isinstance(sku_dict, dict):
                        sig = _json.dumps(sku_dict, sort_keys=True, ensure_ascii=False)
                        return sig, sku_dict
                except Exception:
                    pass
            elif isinstance(raw, dict):
                sig = _json.dumps(raw, sort_keys=True, ensure_ascii=False)
                return sig, raw
            return "", {}

        # quantities[(pid, sig)] = qty;  sku_dicts[(pid, sig)] = sku_dict
        quantities: dict[tuple[str, str], int] = {}
        sku_dicts: dict[tuple[str, str], dict] = {}
        for event in events:
            sig, sku_dict = _sku_sig(event)
            key = (event.product_id, sig)
            sku_dicts[key] = sku_dict
            if event.event_type == "cart_add":
                quantities[key] = quantities.get(key, 0) + 1
            elif event.event_type == "cart_remove":
                quantities[key] = max(0, quantities.get(key, 0) - 1)
            elif event.event_type == "buy":
                # buy 清空该 (pid, sku) 数量
                quantities[key] = 0

        # 过滤数量 > 0
        active = [(key, qty) for key, qty in quantities.items() if qty > 0]
        if not active:
            return CartResponse(user_id=user_id, session_id=session_id)

        product_ids = sorted({pid for (pid, _), _ in active})
        products = self.db.scalars(select(Product).where(Product.product_id.in_(product_ids))).all()
        product_by_id = {product.product_id: product for product in products}

        items: list[CartItemResponse] = []
        total_price = Decimal("0")
        total_quantity = 0
        for (pid, sig), qty in active:
            product = product_by_id.get(pid)
            if product is None:
                continue
            total_quantity += qty
            total_price += product.price * qty
            items.append(
                CartItemResponse(
                    product_id=product.product_id,
                    name=product.name,
                    category=product.category,
                    sub_category=product.sub_category,
                    brand=product.brand,
                    price=product.price,
                    image_url=product.image_url,
                    quantity=qty,
                    rating=product.rating,
                    sku=sku_dicts.get((pid, sig), {}),
                )
            )
        return CartResponse(
            user_id=user_id,
            session_id=session_id,
            items=items,
            total_quantity=total_quantity,
            total_price=total_price,
        )

    def update_affinity(self, req: EventReportRequest) -> None:
        weight = self._compute_weight(req)
        if weight == 0.0:
            return

        product = self.db.scalar(select(Product).where(Product.product_id == req.product_id))
        if product is None:
            LOGGER.info("event product missing, skip affinity update: %s", req.product_id)
            return

        now = datetime.now(timezone.utc)
        self._upsert_product_affinity(req, weight, now)
        if product.brand:
            self._upsert_brand_affinity(req, product.brand, weight, now)
        self.db.commit()

    @staticmethod
    def _compute_weight(req: EventReportRequest) -> float:
        weight = EVENT_WEIGHT.get(req.event_type, 0.0)
        if req.event_type == "click" and req.context.get("from") == "chat":
            weight *= CHAT_SOURCE_MULTIPLIER
        return weight

    def _upsert_product_affinity(
        self,
        req: EventReportRequest,
        weight: float,
        now: datetime,
    ) -> None:
        row = self.db.scalar(
            select(UserProductAffinity).where(
                UserProductAffinity.user_id == req.user_id,
                UserProductAffinity.product_id == req.product_id,
            )
        )
        if row is None:
            row = UserProductAffinity(
                user_id=req.user_id,
                product_id=req.product_id,
                click_count=0,
                cart_count=0,
                buy_count=0,
                affinity=0.0,
                owned=False,
                last_event_at=now,
            )
            self.db.add(row)

        row.affinity = self._decay(row.affinity or 0.0, row.last_event_at, now) + weight
        if req.event_type == "click":
            row.click_count = (row.click_count or 0) + 1
        elif req.event_type == "cart_add":
            row.cart_count = (row.cart_count or 0) + 1
        elif req.event_type == "buy":
            row.buy_count = (row.buy_count or 0) + 1
            row.owned = True
        row.last_event_at = now

    def _upsert_brand_affinity(
        self,
        req: EventReportRequest,
        brand: str,
        weight: float,
        now: datetime,
    ) -> None:
        row = self.db.scalar(
            select(UserBrandAffinity).where(
                UserBrandAffinity.user_id == req.user_id,
                UserBrandAffinity.brand == brand,
            )
        )
        if row is None:
            row = UserBrandAffinity(
                user_id=req.user_id,
                brand=brand,
                click_count=0,
                cart_count=0,
                buy_count=0,
                affinity=0.0,
                last_event_at=now,
            )
            self.db.add(row)

        row.affinity = self._decay(row.affinity or 0.0, row.last_event_at, now) + weight
        if req.event_type == "click":
            row.click_count = (row.click_count or 0) + 1
        elif req.event_type == "cart_add":
            row.cart_count = (row.cart_count or 0) + 1
        elif req.event_type == "buy":
            row.buy_count = (row.buy_count or 0) + 1
        row.last_event_at = now

    @staticmethod
    def _decay(old_affinity: float, last_event_at: datetime | None, now: datetime) -> float:
        if last_event_at is None or old_affinity <= 0:
            return old_affinity
        if last_event_at.tzinfo is None:
            last_event_at = last_event_at.replace(tzinfo=timezone.utc)
        delta_days = (now - last_event_at).total_seconds() / 86400.0
        if delta_days <= 0:
            return old_affinity
        return old_affinity * (0.5 ** (delta_days / HALF_LIFE_DAYS))
