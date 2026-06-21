from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.db.models import Product
from app.domain.image_search_tool import ImageSearchTool
from app.domain.need_slot_schemas import NeedSlot
from app.domain.single_retrieval_worker import SingleRetrievalEvidence
from app.schemas import IntentPlan, QueryPlan


@dataclass
class ImageRetrievalEvidence(SingleRetrievalEvidence):
    image_path: str = ""
    max_image_score: float = 0.0

    def counts(self, after_corrective: int = 0) -> dict[str, int]:
        counts = super().counts(after_corrective=after_corrective)
        counts["image_hits"] = len(self.vector_scores)
        return counts

    def summary(self, intent_plan: IntentPlan, plan: QueryPlan) -> dict[str, Any]:
        summary = super().summary(intent_plan, plan)
        summary.update(
            {
                "worker": "ImageRetrievalWorker",
                "image_path_resolved": bool(self.image_path),
                "image_hits": len(self.vector_scores),
                "max_image_score": self.max_image_score,
            }
        )
        return summary


class ImageRetrievalWorker:
    """图片检索执行节点（ImageRetrievalWorker）。

    只执行图片召回并返回候选证据；是否进入该路径、是否低相关降级、最终 route 都由 Orchestrator 决定。
    """

    def __init__(self, image_search_tool: ImageSearchTool) -> None:
        self.image_search_tool = image_search_tool

    def run(
        self,
        *,
        original_query: str,
        intent_plan: IntentPlan,
        plan: QueryPlan,
        image_path: Path,
    ) -> ImageRetrievalEvidence:
        query = intent_plan.vector_query or intent_plan.keyword_query or original_query or "用户上传图片找相似商品"
        slot = NeedSlot(
            slot_id="image_retrieval",
            goal=query,
            product_type="",
            query=query,
            hard_constraints=list(plan.filters),
            soft_constraints=[*plan.preferences, *plan.scene],
            exclude_terms=list(plan.exclude),
            min_candidates=plan.retrieval_strategy.final_top_k,
        )
        search_result = self.image_search_tool.search(
            slot=slot,
            base_plan=plan,
            image_path=image_path,
            top_k=max(plan.retrieval_strategy.final_top_k, 10),
        )
        ranked: list[tuple[Product, float]] = [
            (candidate.product, candidate.rerank_score)
            for candidate in search_result.candidates
        ]
        evidence = ImageRetrievalEvidence(
            before_structured_filter=search_result.counts.get("before_structured_filter", 0),
            structured_products=search_result.structured_products,
            vector_query="<image>",
            keyword_query=search_result.query,
            vector_scores={candidate.product_id: candidate.vector_score for candidate in search_result.candidates},
            keyword_scores={},
            score_filtered_products=[candidate.product for candidate in search_result.candidates],
            hybrid_ranked_products=[candidate.product for candidate in search_result.candidates],
            ranked=ranked,
            rerank_query=query,
            tool_call_count=1,
            image_path=str(image_path),
            max_image_score=max((score for _, score in ranked), default=0.0),
        )
        if not evidence.ranked:
            if search_result.category_resolution == "image_retrieval_unavailable":
                evidence.failure_trigger = "image_index_unavailable"
            else:
                evidence.failure_trigger = "image_low_relevance"
        return evidence
