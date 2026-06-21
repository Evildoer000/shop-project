from __future__ import annotations

import asyncio
import sys
from decimal import Decimal
from pathlib import Path

SERVER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER_ROOT))

from app.core.config import get_settings
from app.db.models import Product
from app.domain.reranker import build_reranker
from app.services.embedding_client import EmbeddingClient
from app.services.llm_client import LlmClient


async def main() -> None:
    settings = get_settings()
    print(f"LLM model: {settings.llm_model}")
    print(f"Embedding model: {settings.embedding_model}")
    print(f"Rerank backend: {settings.rerank_backend}")
    print(f"Rerank model: {settings.rerank_model or '-'}")

    embedding = EmbeddingClient().embed("测试：适合通勤的降噪蓝牙耳机")
    print(f"Embedding dimension: {len(embedding)}")

    answer = await LlmClient().generate_required(
        "你是接口连通性测试助手，只回答 OK。",
        "请回复 OK",
    )
    print(f"LLM response: {answer[:80]}")

    product = Product(
        product_id="check_rerank",
        name="测试降噪蓝牙耳机",
        category="数码电子",
        sub_category="真无线耳机",
        brand="Check",
        price=Decimal("1"),
        stock=1,
        image_url="",
        description="适合通勤的主动降噪真无线蓝牙耳机",
        specs={},
        ingredients_or_material="",
        suitable_for="通勤 降噪",
        avoid_for="",
        tags=["耳机", "降噪"],
        rating=Decimal("5"),
        sales=1,
        review_summary="通勤降噪效果好",
        image_caption="",
        structured_attributes={},
    )
    reranked = build_reranker().rerank("适合通勤的降噪蓝牙耳机", [product], top_k=1)
    print(f"Rerank score: {reranked[0][1]:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
