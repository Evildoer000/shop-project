from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sse_starlette.sse import EventSourceResponse
from sqlalchemy.orm import Session

from app.db.session import get_db, get_sessionmaker
from app.domain.orchestrator import EcommerceOrchestrator
from app.domain.recommendation_service import RecommendationService
from app.schemas import (
    CartResponse,
    ChatStreamRequest,
    EventReportRequest,
    EventReportResponse,
    ImageUploadResponse,
    ProductResponse,
    RecommendationResponse,
)
from app.services.event_service import EventService
from app.services.upload_storage import save_uploaded_image
from app.services.product_repository import ProductRepository

LOGGER = logging.getLogger(__name__)
router = APIRouter()


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
        async for event in orchestrator.stream(payload):
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
