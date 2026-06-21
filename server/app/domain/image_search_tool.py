from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from app.core.config import get_settings
from app.domain.need_slot_schemas import NeedSlot, SlotCandidate, SlotSearchResult
from app.rag.image_retriever import ImageIndexUnavailable, ImageRetriever
from app.schemas import QueryPlan
from app.services.product_repository import ProductRepository


class ImageSearchTool:
    """图片搜索原子能力（ImageSearchTool）。

    它只做图片向量召回和候选证据包装，不决定 execution_path / repair / final_route。
    """

    MIN_IMAGE_SCORE = 0.20
    DEFAULT_TOP_K = 12

    def __init__(
        self,
        product_repository: ProductRepository,
        image_retriever: ImageRetriever | None = None,
    ) -> None:
        self.product_repository = product_repository
        self.image_retriever = image_retriever or ImageRetriever()
        self.settings = get_settings()

    def search(
        self,
        slot: NeedSlot,
        base_plan: QueryPlan,
        image_path: Path,
        top_k: int | None = None,
    ) -> SlotSearchResult:
        top_k = top_k or self.DEFAULT_TOP_K
        plan = base_plan.model_copy(deep=True)
        plan.retrieval_strategy.candidate_limit = max(plan.retrieval_strategy.candidate_limit, 200)

        products = self.product_repository.list_for_plan(plan, limit=plan.retrieval_strategy.candidate_limit)
        before_filter = self.product_repository.count_available()
        if not products:
            return SlotSearchResult(
                slot_id=slot.slot_id,
                query=slot.query,
                vector_query="<image>",
                keyword_query="",
                candidates=[],
                counts={"before_structured_filter": before_filter, "after_structured_filter": 0},
                categories=plan.categories,
                category_resolution="image_retrieval_no_candidates",
            )

        try:
            scores = self.image_retriever.retrieve(image_path, products, top_k=top_k * 2)
        except ImageIndexUnavailable as exc:
            return SlotSearchResult(
                slot_id=slot.slot_id,
                query=slot.query,
                vector_query="<image>",
                keyword_query="",
                candidates=[],
                counts={
                    "before_structured_filter": before_filter,
                    "after_structured_filter": len(products),
                    "image_hits": 0,
                    "after_score_filter": 0,
                    "after_rerank": 0,
                },
                attempts=[
                    {
                        "tool": "image_search",
                        "status": "unavailable",
                        "reason": str(exc),
                    }
                ],
                categories=plan.categories,
                category_resolution="image_retrieval_unavailable",
                structured_products=products,
            )
        threshold = float(self.settings.image_relevance_threshold or self.MIN_IMAGE_SCORE)
        passing = [(product_id, score) for product_id, score in scores.items() if score >= threshold]
        passing.sort(key=lambda item: item[1], reverse=True)
        product_by_id = {product.product_id: product for product in products}
        candidates: list[SlotCandidate] = []
        for product_id, score in passing[:top_k]:
            product = product_by_id.get(product_id)
            if product is None:
                continue
            candidates.append(
            SlotCandidate(
                    product=product,
                    product_id=product.product_id,
                    name=product.name,
                    category=product.category,
                    sub_category=product.sub_category,
                    price=float(product.price or Decimal("0")),
                    vector_score=round(score, 4),
                    keyword_score=0.0,
                    rrf_score=0.0,
                    rerank_score=round(score, 4),
                    coverage_reason="image_retrieval",
                )
            )

        return SlotSearchResult(
            slot_id=slot.slot_id,
            query=slot.query,
            vector_query="<image>",
            keyword_query="",
            candidates=candidates,
            counts={
                "before_structured_filter": before_filter,
                "after_structured_filter": len(products),
                "image_hits": len(scores),
                "after_score_filter": len(passing),
                "after_rerank": len(candidates),
            },
            attempts=[
                {
                    "tool": "image_search",
                    "image_score_threshold": threshold,
                    "max_score": max(scores.values()) if scores else 0.0,
                }
            ],
            categories=plan.categories,
            category_resolution="image_retrieval",
            structured_products=products,
            score_filtered_products=[candidate.product for candidate in candidates],
            hybrid_ranked_products=[candidate.product for candidate in candidates],
            vector_scores={candidate.product_id: candidate.vector_score for candidate in candidates},
            keyword_scores={},
        )
