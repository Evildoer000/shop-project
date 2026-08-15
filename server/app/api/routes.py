from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncGenerator
from contextlib import aclosing
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sse_starlette.sse import EventSourceResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import AgentRun, AgentRunSpan, AppUser, ConversationTurn
from app.db.session import get_db, get_sessionmaker
from app.domain.orchestrator import EcommerceOrchestrator
from app.domain.recommendation_service import RecommendationService
from app.schemas import (
    AgentRunConversation,
    AgentRunDetailResponse,
    AgentRunListResponse,
    AgentRunSpanResponse,
    AgentRunSummary,
    CartResponse,
    ChatSessionDetailResponse,
    ChatSessionListResponse,
    ChatSessionSummary,
    ChatSessionTurn,
    ChatStreamRequest,
    EventReportRequest,
    EventReportResponse,
    ImageUploadResponse,
    LoginRequest,
    LoginResponse,
    ProductCatalogResponse,
    ProductCategoriesResponse,
    ProductCategorySummary,
    ProductResponse,
    ProductSubCategorySummary,
    RecommendationCard,
    RecommendationResponse,
)
from app.services.event_service import EventService
from app.services.upload_storage import save_uploaded_image
from app.services.product_repository import ProductRepository

LOGGER = logging.getLogger(__name__)
router = APIRouter()

DEFAULT_LOGIN_PASSWORD = "88888"
PHONE_RE = re.compile(r"^\d{5,20}$")


@router.post("/auth/login", response_model=LoginResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> LoginResponse:
    phone = _normalize_phone(payload.phone)
    if not PHONE_RE.fullmatch(phone):
        raise HTTPException(status_code=400, detail="请输入有效手机号")
    if payload.password.strip() != DEFAULT_LOGIN_PASSWORD:
        raise HTTPException(status_code=401, detail="手机号或密码错误")

    user = db.scalar(select(AppUser).where(AppUser.phone == phone))
    now = datetime.now(timezone.utc)
    if user is None:
        user = AppUser(
            user_id=_user_id_from_phone(phone),
            phone=phone,
            display_name=f"用户{phone[-4:]}",
            last_login_at=now,
        )
        db.add(user)
    else:
        user.last_login_at = now
    db.commit()
    db.refresh(user)
    return LoginResponse(
        ok=True,
        user_id=user.user_id,
        phone=user.phone,
        display_name=user.display_name,
    )


def _normalize_phone(phone: str) -> str:
    return re.sub(r"\D", "", phone or "")


def _user_id_from_phone(phone: str) -> str:
    return f"phone_{phone}"


@router.get("/debug/agent-runs", response_model=AgentRunListResponse)
def list_agent_runs(
    user_id: str | None = None,
    session_id: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
) -> AgentRunListResponse:
    """Read-only historical Agent run list for local debugging."""
    safe_limit = max(1, min(limit, 100))
    safe_offset = max(0, offset)
    filters = []
    if user_id:
        filters.append(AgentRun.user_id == user_id)
    if session_id:
        filters.append(AgentRun.session_id == session_id)
    if status:
        filters.append(AgentRun.status == status)

    total = int(db.scalar(select(func.count()).select_from(AgentRun).where(*filters)) or 0)
    rows = db.scalars(
        select(AgentRun)
        .where(*filters)
        .order_by(AgentRun.created_at.desc(), AgentRun.run_id.desc())
        .offset(safe_offset)
        .limit(safe_limit)
    ).all()
    return AgentRunListResponse(
        runs=[AgentRunSummary.model_validate(row) for row in rows],
        total=total,
        limit=safe_limit,
        offset=safe_offset,
    )


@router.get("/debug/agent-runs/{run_id}", response_model=AgentRunDetailResponse)
def get_agent_run(
    run_id: str,
    user_id: str | None = None,
    db: Session = Depends(get_db),
) -> AgentRunDetailResponse:
    """Read-only historical Agent run detail, including its span chain."""
    filters = [AgentRun.run_id == run_id]
    if user_id:
        filters.append(AgentRun.user_id == user_id)
    run = db.scalar(select(AgentRun).where(*filters))
    if run is None:
        raise HTTPException(status_code=404, detail="Agent run not found")

    spans = db.scalars(
        select(AgentRunSpan)
        .where(AgentRunSpan.run_id == run_id)
        .order_by(AgentRunSpan.sequence.asc(), AgentRunSpan.span_id.asc())
    ).all()

    # New turns carry run_id in trace_summary, which keeps repeated or concurrent
    # identical queries attributable to the correct execution tree. The text
    # fallback preserves access to historical rows written before trace v2.
    conversation_row = db.scalar(
        select(ConversationTurn)
        .where(
            ConversationTurn.user_id == run.user_id,
            ConversationTurn.session_id == run.session_id,
            ConversationTurn.trace_summary["run_id"].as_string() == run.run_id,
        )
        .order_by(ConversationTurn.created_at.desc(), ConversationTurn.turn_id.desc())
    )
    if conversation_row is None:
        conversation_row = db.scalar(
            select(ConversationTurn)
            .where(
                ConversationTurn.user_id == run.user_id,
                ConversationTurn.session_id == run.session_id,
                ConversationTurn.user_message == run.query_summary,
                ConversationTurn.trace_summary["run_id"].as_string().is_(None),
            )
            .order_by(ConversationTurn.created_at.desc(), ConversationTurn.turn_id.desc())
        )
    conversation = (
        AgentRunConversation.model_validate(conversation_row)
        if conversation_row is not None
        else None
    )
    return AgentRunDetailResponse(
        run=AgentRunSummary.model_validate(run),
        spans=[AgentRunSpanResponse.model_validate(span) for span in spans],
        conversation=conversation,
    )


@router.post("/images", response_model=ImageUploadResponse)
async def upload_image(file: UploadFile = File(...)) -> ImageUploadResponse:
    content = await file.read()
    try:
        image_id, path = save_uploaded_image(file.filename or "upload.jpg", content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ImageUploadResponse(
        image_id=image_id,
        image_url=f"/uploads/{path.name}",
        bytes=len(content),
    )


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatStreamRequest,
    db: Session = Depends(get_db),
) -> EventSourceResponse:
    async def events() -> AsyncGenerator[dict[str, str], None]:
        preflight = _chat_stream_preflight_event(payload)
        yield {
            "event": preflight["type"],
            "data": json.dumps(preflight, ensure_ascii=False),
        }
        await asyncio.sleep(0.01)

        orchestrator = EcommerceOrchestrator(db)
        async with aclosing(orchestrator.stream(payload)) as stream:
            async for event in stream:
                event_type = event.get("type", "message")
                yield {
                    "event": event_type,
                    "data": json.dumps(event, ensure_ascii=False),
                }

    return EventSourceResponse(events())


def _chat_stream_preflight_event(payload: ChatStreamRequest) -> dict[str, object]:
    content = "我已经收到图片，正在准备检索。" if payload.image_id else "我已经收到需求，正在准备分析。"
    return {
        "type": "agent_update",
        "stage": "planner",
        "title": "接收请求",
        "content_delta": content,
        "done": False,
    }


@router.get("/chat/sessions", response_model=ChatSessionListResponse)
def list_chat_sessions(user_id: str, db: Session = Depends(get_db)) -> ChatSessionListResponse:
    rows = db.scalars(
        select(ConversationTurn)
        .where(ConversationTurn.user_id == user_id)
        .order_by(ConversationTurn.session_id.asc(), ConversationTurn.turn_id.asc())
    ).all()
    sessions: dict[str, dict[str, object]] = {}
    for row in rows:
        session = sessions.setdefault(
            row.session_id,
            {
                "session_id": row.session_id,
                "title": _compact_text(row.user_message, 28) or "新会话",
                "last_message": "",
                "turn_count": 0,
                "updated_at": row.updated_at or row.created_at,
            },
        )
        session["turn_count"] = int(session["turn_count"]) + 1
        session["last_message"] = _compact_text(row.assistant_message or row.user_message, 42)
        session["updated_at"] = row.updated_at or row.created_at

    result = [
        ChatSessionSummary(
            session_id=str(item["session_id"]),
            title=str(item["title"]),
            last_message=str(item["last_message"]),
            turn_count=int(item["turn_count"]),
            updated_at=item["updated_at"],  # type: ignore[arg-type]
        )
        for item in sessions.values()
    ]
    result.sort(key=lambda item: item.updated_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return ChatSessionListResponse(sessions=result)


@router.get("/chat/sessions/{session_id}", response_model=ChatSessionDetailResponse)
def get_chat_session(
    session_id: str,
    user_id: str,
    db: Session = Depends(get_db),
) -> ChatSessionDetailResponse:
    rows = db.scalars(
        select(ConversationTurn)
        .where(ConversationTurn.user_id == user_id, ConversationTurn.session_id == session_id)
        .order_by(ConversationTurn.turn_id.asc())
    ).all()
    if not rows:
        raise HTTPException(status_code=404, detail="Chat session not found")

    product_repository = ProductRepository(db)
    turns: list[ChatSessionTurn] = []
    for row in rows:
        turns.append(
            ChatSessionTurn(
                turn_id=row.turn_id,
                user_message=row.user_message,
                assistant_message=row.assistant_message,
                route=row.route,
                product_ids=row.product_ids or [],
                products=[_product_to_card(product) for product in product_repository.get_by_ids(row.product_ids or [])],
                rewrite_summary=row.rewrite_summary or {},
                trace_summary=row.trace_summary or {},
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
        )
    return ChatSessionDetailResponse(session_id=session_id, turns=turns)


@router.get("/products", response_model=ProductCatalogResponse)
def list_products(
    category: str | None = None,
    sub_category: str | None = None,
    q: str = "",
    page: int = 1,
    page_size: int = 24,
    sort: str = "popular",
    db: Session = Depends(get_db),
) -> ProductCatalogResponse:
    result = ProductRepository(db).list_catalog_with_scores(
        category=category,
        sub_category=sub_category,
        query=q,
        page=page,
        page_size=page_size,
        sort=sort,
    )
    normalized_page = max(page, 1)
    normalized_page_size = min(max(page_size, 1), 60)
    return ProductCatalogResponse(
        products=[_product_to_card(product, score=result.scores.get(product.product_id, 0.0)) for product in result.products],
        total=result.total,
        page=normalized_page,
        page_size=normalized_page_size,
    )


@router.get("/products/categories", response_model=ProductCategoriesResponse)
def list_product_categories(db: Session = Depends(get_db)) -> ProductCategoriesResponse:
    grouped: dict[str, ProductCategorySummary] = {}
    for category, sub_category, count in ProductRepository(db).list_categories():
        summary = grouped.setdefault(category, ProductCategorySummary(name=category, count=0, sub_categories=[]))
        summary.count += count
        if sub_category:
            summary.sub_categories.append(ProductSubCategorySummary(name=sub_category, count=count))
    return ProductCategoriesResponse(categories=list(grouped.values()))


@router.get("/products/{product_id}", response_model=ProductResponse)
def get_product(product_id: str, db: Session = Depends(get_db)) -> ProductResponse:
    product = ProductRepository(db).get_by_id(product_id)
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return ProductResponse.model_validate(product)


@router.post("/events", response_model=EventReportResponse)
async def report_event(
    payload: EventReportRequest,
    db: Session = Depends(get_db),
) -> EventReportResponse:
    service = EventService(db)
    try:
        event = service.write_event(payload)
    except Exception as exc:
        LOGGER.warning("event write failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="事件写入失败") from exc

    asyncio.create_task(_update_affinity_async(payload))
    return EventReportResponse(ok=True, event_id=event.event_id)


@router.get("/cart", response_model=CartResponse)
def get_cart(
    user_id: str,
    session_id: str = "all",
    db: Session = Depends(get_db),
) -> CartResponse:
    return EventService(db).cart_snapshot(user_id=user_id, session_id=session_id)


@router.get("/recommendations", response_model=RecommendationResponse)
def get_recommendations(
    user_id: str,
    size: int = 24,
    db: Session = Depends(get_db),
) -> RecommendationResponse:
    return RecommendationService(db).get_home_recommendations(user_id=user_id, size=size)


async def _update_affinity_async(payload: EventReportRequest) -> None:
    SessionLocal = get_sessionmaker()
    try:
        with SessionLocal() as db:
            EventService(db).update_affinity(payload)
    except Exception as exc:
        LOGGER.warning("affinity update failed: %s", exc, exc_info=True)


def _product_to_card(product, score: float = 0.0) -> RecommendationCard:
    tags = product.tags or []
    return RecommendationCard(
        product_id=product.product_id,
        name=product.name,
        category=product.category,
        sub_category=product.sub_category,
        brand=product.brand,
        price=float(product.price or 0),
        image_url=product.image_url,
        tags=tags[:6] if isinstance(tags, list) else [],
        rating=float(product.rating or 0),
        reason=product.review_summary or product.description[:80],
        score=round(float(score or 0.0), 4),
    )


def _compact_text(value: str, limit: int) -> str:
    text = " ".join((value or "").split())
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."
