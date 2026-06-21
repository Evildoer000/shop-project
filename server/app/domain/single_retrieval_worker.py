from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.db.models import Product
from app.domain.need_slot_schemas import NeedSlot
from app.domain.product_search_tool import ProductSearchTool
from app.schemas import IntentPlan, QueryPlan, SINGLE_RETRIEVAL_REVIEW_LIMIT


@dataclass
class SingleRetrievalEvidence:
    before_structured_filter: int = 0
    structured_products: list[Product] = field(default_factory=list)
    vector_query: str = ""
    keyword_query: str = ""
    vector_scores: dict[str, float] = field(default_factory=dict)
    keyword_scores: dict[str, float] = field(default_factory=dict)
    score_filtered_products: list[Product] = field(default_factory=list)
    hybrid_ranked_products: list[Product] = field(default_factory=list)
    ranked: list[tuple[Product, float]] = field(default_factory=list)
    rerank_query: str = ""
    failure_trigger: str = ""
    tool_call_count: int = 0

    @property
    def after_structured_filter(self) -> int:
        return len(self.structured_products)

    @property
    def after_score_filter(self) -> int:
        return len(self.score_filtered_products)

    @property
    def after_hybrid_rank(self) -> int:
        return len(self.hybrid_ranked_products)

    @property
    def after_rerank(self) -> int:
        return len(self.ranked)

    def counts(self, after_corrective: int = 0) -> dict[str, int]:
        return {
            "before_structured_filter": self.before_structured_filter,
            "after_structured_filter": self.after_structured_filter,
            "vector_hits": len(self.vector_scores),
            "keyword_hits": len(self.keyword_scores),
            "after_score_filter": self.after_score_filter,
            "after_hybrid_rank": self.after_hybrid_rank,
            "after_rerank": self.after_rerank,
            "after_corrective": after_corrective,
        }

    def summary(self, intent_plan: IntentPlan, plan: QueryPlan) -> dict[str, Any]:
        return {
            "worker": "SingleRetrievalWorker",
            "structured_candidates": self.after_structured_filter,
            "vector_hits": len(self.vector_scores),
            "keyword_hits": len(self.keyword_scores),
            "score_filtered_candidates": self.after_score_filter,
            "hybrid_ranked_candidates": self.after_hybrid_rank,
            "hybrid_ranker": "rrf",
            "hybrid_top_k": plan.retrieval_strategy.hybrid_top_k,
            "reranked_candidates": self.after_rerank,
            "rewritten_query": intent_plan.vector_query,
            "vector_query": self.vector_query,
            "keyword_query": self.keyword_query,
            "plan_type": intent_plan.plan_type,
            "retrieval_strategy": plan.retrieval_strategy.model_dump(),
            "failure_trigger": self.failure_trigger,
        }


class SingleRetrievalWorker:
    def __init__(self, product_search_tool: ProductSearchTool) -> None:
        self.product_search_tool = product_search_tool

    def run(
        self,
        original_query: str,
        intent_plan: IntentPlan,
        plan: QueryPlan,
    ) -> SingleRetrievalEvidence:
        self._widen_review_pool(plan)
        query = intent_plan.vector_query or intent_plan.keyword_query or intent_plan.original_query or original_query
        slot = NeedSlot(
            slot_id="single_retrieval",
            goal=query,
            product_type="",
            query=query,
            hard_constraints=list(plan.filters),
            soft_constraints=[*plan.preferences, *plan.scene],
            exclude_terms=list(plan.exclude),
            min_candidates=plan.retrieval_strategy.final_top_k,
        )
        search_result = self.product_search_tool.search_query(
            slot=slot,
            base_plan=plan,
            intent_plan=intent_plan,
            query=query,
            attempt_index=1,
            reason="single_retrieval",
            use_base_plan=True,
        )
        evidence = SingleRetrievalEvidence(
            before_structured_filter=search_result.counts.get("before_structured_filter", 0),
            structured_products=search_result.structured_products,
            vector_query=search_result.vector_query,
            keyword_query=search_result.keyword_query,
            vector_scores=search_result.vector_scores,
            keyword_scores=search_result.keyword_scores,
            score_filtered_products=search_result.score_filtered_products,
            hybrid_ranked_products=search_result.hybrid_ranked_products,
            ranked=[(candidate.product, candidate.rerank_score) for candidate in search_result.candidates],
            rerank_query=self._rerank_query(original_query, intent_plan),
            tool_call_count=self._tool_call_count(search_result.counts),
        )
        if evidence.after_structured_filter == 0:
            evidence.failure_trigger = "no_candidates"
        elif evidence.after_score_filter == 0:
            evidence.failure_trigger = "score_filter_empty"
        return evidence

    def _tool_call_count(self, counts: dict[str, int]) -> int:
        if counts.get("after_structured_filter", 0) <= 0:
            return 1
        if counts.get("after_score_filter", 0) <= 0:
            return 2
        return 3

    def _rerank_query(self, original_query: str, intent_plan: IntentPlan) -> str:
        return " ".join(
            part
            for part in [
                original_query,
                intent_plan.vector_query,
                intent_plan.keyword_query,
            ]
            if part
        )

    def _widen_review_pool(self, plan: QueryPlan) -> None:
        strategy = plan.retrieval_strategy
        strategy.final_top_k = max(strategy.final_top_k, SINGLE_RETRIEVAL_REVIEW_LIMIT)
        strategy.hybrid_top_k = max(strategy.hybrid_top_k, strategy.final_top_k * 3)
