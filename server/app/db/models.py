from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import DateTime, Float, Index, Integer, Numeric, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Product(Base):
    __tablename__ = "products"

    product_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    category: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    sub_category: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    brand: Mapped[str] = mapped_column(String(128), nullable=False)
    price: Mapped[Decimal] = mapped_column(Numeric(10, 2), index=True, nullable=False)
    stock: Mapped[int | None] = mapped_column(Integer, nullable=True)
    image_url: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    specs: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    ingredients_or_material: Mapped[str] = mapped_column(Text, default="", nullable=False)
    suitable_for: Mapped[str] = mapped_column(Text, default="", nullable=False)
    avoid_for: Mapped[str] = mapped_column(Text, default="", nullable=False)
    tags: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    rating: Mapped[Decimal] = mapped_column(Numeric(3, 2), default=0, nullable=False)
    sales: Mapped[int | None] = mapped_column(Integer, nullable=True)
    review_summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    image_caption: Mapped[str] = mapped_column(Text, default="", nullable=False)
    structured_attributes: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    def search_text(self) -> str:
        tags = _json_to_text(self.tags or [])
        specs = _json_to_text(self.specs or {})
        attrs = _json_to_text(self.structured_attributes or {})
        return (
            f"{self.name} {self.brand} {self.category} {self.sub_category or ''} 价格{self.price} "
            f"{self.description} {specs} {self.ingredients_or_material} "
            f"{self.suitable_for} {self.avoid_for} {tags} {self.review_summary} "
            f"{self.image_caption} {attrs}"
        )


class UserMemory(Base):
    __tablename__ = "user_memories"

    memory_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    memory_type: Mapped[str] = mapped_column(String(64), nullable=False)
    key: Mapped[str] = mapped_column(String(128), nullable=False)
    value: Mapped[str] = mapped_column(String(256), nullable=False)
    confidence: Mapped[Decimal] = mapped_column(Numeric(3, 2), default=1, nullable=False)
    source: Mapped[str] = mapped_column(String(64), default="explicit", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class ConversationTurn(Base):
    __tablename__ = "conversation_turns"

    turn_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    session_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    user_message: Mapped[str] = mapped_column(Text, nullable=False)
    assistant_message: Mapped[str] = mapped_column(Text, default="", nullable=False)
    route: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    product_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    rewrite_summary: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    trace_summary: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class SessionMemoryState(Base):
    __tablename__ = "session_memory_states"
    __table_args__ = (UniqueConstraint("user_id", "session_id", name="uq_session_memory_user_session"),)

    state_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    session_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    session_summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    summarized_through_turn_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    distilled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class UserEvent(Base):
    __tablename__ = "user_events"
    __table_args__ = (
        Index("ix_events_user_time", "user_id", "created_at"),
        Index("ix_events_user_type", "user_id", "event_type"),
        Index("ix_events_user_product", "user_id", "product_id"),
    )

    event_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False)
    session_id: Mapped[str] = mapped_column(String(128), nullable=False)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    product_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    turn_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    position: Mapped[int | None] = mapped_column(Integer, nullable=True)
    context: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class UserProductAffinity(Base):
    __tablename__ = "user_product_affinity"
    __table_args__ = (UniqueConstraint("user_id", "product_id", name="uq_upa_user_product"),)

    upa_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    product_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    click_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cart_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    buy_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    affinity: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    owned: Mapped[bool] = mapped_column(default=False, nullable=False)
    last_event_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class UserBrandAffinity(Base):
    __tablename__ = "user_brand_affinity"
    __table_args__ = (UniqueConstraint("user_id", "brand", name="uq_uba_user_brand"),)

    uba_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    brand: Mapped[str] = mapped_column(String(128), nullable=False)
    click_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cart_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    buy_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    affinity: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    last_event_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


def _json_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool, Decimal)):
        return str(value)
    if isinstance(value, list):
        return " ".join(_json_to_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(f"{key} {_json_to_text(item)}" for key, item in value.items())
    return json.dumps(value, ensure_ascii=False, default=str)
