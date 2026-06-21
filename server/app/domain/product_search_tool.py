from __future__ import annotations

from decimal import Decimal

from app.db.models import Product
from app.domain.need_slot_schemas import SlotCandidate, NeedSlot, SlotSearchResult
from app.domain.retrieval_plan_builder import RetrievalPlanBuilder
from app.domain.retrieval_fusion import rrf_fuse
from app.rag.llamaindex_milvus import LlamaIndexMilvusRetriever
from app.schemas import IntentPlan, QueryPlan, QueryRetrievalStrategy
from app.services.product_repository import ProductRepository


class ProductSearchTool:
    MIN_VECTOR_SCORE: float = 0.10
    MIN_KEYWORD_SCORE: float = 0.05
    MAX_ATTEMPTS_PER_SLOT: int = 3
    TOP_CATEGORY_HINTS: dict[str, tuple[str, ...]] = {
        "服饰运动": ("鞋", "衣", "裤", "帽", "包", "穿", "运动", "户外", "露营"),
        "食品饮料": ("食品", "食物", "吃", "零食", "饮料", "咖啡", "茶", "奶", "速食", "泡面", "方便"),
        "数码电子": ("数码", "手机", "电脑", "主机", "台式", "耳机", "平板", "支架", "充电", "蓝牙"),
        "美妆护肤": ("护肤", "美妆", "化妆", "防晒霜", "防晒乳", "洁面", "面膜", "粉底", "唇"),
    }

    def __init__(
        self,
        product_repository: ProductRepository,
        retriever: LlamaIndexMilvusRetriever,
        reranker: object,
    ) -> None:
        self.product_repository = product_repository
        self.retriever = retriever
        self.reranker = reranker

    def search(
        self,
        slot: NeedSlot,
        base_plan: QueryPlan,
        intent_plan: IntentPlan,
    ) -> SlotSearchResult:
        categories, category_resolution = self._resolve_categories(slot)
        plan = self._plan_for_slot(slot, base_plan, categories)
        strategy = plan.retrieval_strategy
        before_structured_filter = self.product_repository.count_available()
        structured_products = self.product_repository.list_for_plan(plan, limit=strategy.candidate_limit)
        variants = self._query_variants(slot)
        attempts: list[dict] = []

        if not structured_products:
            return SlotSearchResult(
                slot_id=slot.slot_id,
                query=slot.query,
                vector_query=variants[0],
                keyword_query=variants[0],
                counts=self._counts(before_structured_filter=before_structured_filter),
                attempts=[
                    {
                        "attempt": 1,
                        "query": variants[0],
                        "categories": categories,
                        "category_resolution": category_resolution,
                        "structured_candidates": 0,
                        "reason": "structured_filter_empty",
                    }
                ],
                categories=categories,
                category_resolution=category_resolution,
            )

        merged: dict[str, SlotCandidate] = {}
        total_counts = self._counts(
            before_structured_filter=before_structured_filter,
            after_structured_filter=len(structured_products),
        )
        searched_queries: set[str] = set()
        last_vector_query = variants[0]
        last_keyword_query = variants[0]
        for attempt_index, query in enumerate(variants[: self.MAX_ATTEMPTS_PER_SLOT], start=1):
            query = query.strip()
            if not query or query in searched_queries:
                continue
            searched_queries.add(query)
            result = self._search_once(
                slot=slot,
                products=structured_products,
                strategy=strategy,
                query=query,
            )
            last_vector_query = result["vector_query"]
            last_keyword_query = result["keyword_query"]
            attempts.append(
                {
                    "attempt": attempt_index,
                    "query": query,
                    "categories": categories,
                    "category_resolution": category_resolution,
                    "structured_candidates": len(structured_products),
                    "vector_hits": len(result["vector_scores"]),
                    "keyword_hits": len(result["keyword_scores"]),
                    "score_filtered": len(result["score_filtered"]),
                    "hybrid_ranked": len(result["fused_products"]),
                    "reranked": len(result["ranked"]),
                    "raw_candidate_ids": [product.product_id for product in result["fused_products"]],
                    "reranked_candidate_ids": [product.product_id for product, _ in result["ranked"]],
                    "reason": "initial_search" if attempt_index == 1 else "repair_search",
                }
            )
            total_counts = self._merge_counts(
                total_counts,
                self._counts(
                    before_structured_filter=before_structured_filter,
                    after_structured_filter=len(structured_products),
                    vector_hits=len(result["vector_scores"]),
                    keyword_hits=len(result["keyword_scores"]),
                    after_score_filter=len(result["score_filtered"]),
                    after_hybrid_rank=len(result["fused_products"]),
                    after_rerank=len(result["ranked"]),
                ),
            )
            for product, rerank_score in result["ranked"]:
                candidate = self._candidate(
                    product,
                    rerank_score=rerank_score,
                    vector_score=result["vector_scores"].get(product.product_id, 0.0),
                    keyword_score=result["keyword_scores"].get(product.product_id, 0.0),
                    rrf_score=self._rrf_score(
                        product.product_id,
                        result["vector_scores"],
                        result["keyword_scores"],
                    ),
                    slot=slot,
                )
                current = merged.get(candidate.product_id)
                if current is None or candidate.rerank_score > current.rerank_score:
                    merged[candidate.product_id] = candidate

            if self._has_enough_signal(result["ranked"], slot):
                break

        candidates = sorted(
            merged.values(),
            key=lambda candidate: (candidate.rerank_score, candidate.rrf_score, candidate.keyword_score),
            reverse=True,
        )[: strategy.final_top_k]
        return SlotSearchResult(
            slot_id=slot.slot_id,
            query=slot.query,
            vector_query=last_vector_query,
            keyword_query=last_keyword_query,
            candidates=candidates,
            counts=total_counts,
            attempts=attempts,
            categories=categories,
            category_resolution=category_resolution,
        )

    def search_query(
        self,
        slot: NeedSlot,
        base_plan: QueryPlan,
        intent_plan: IntentPlan,
        query: str,
        attempt_index: int,
        reason: str,
        *,
        use_base_plan: bool = False,
    ) -> SlotSearchResult:
        if use_base_plan:
            categories = list(base_plan.categories)
            category_resolution = "base_plan"
            plan = base_plan
        else:
            categories, category_resolution = self._resolve_categories(slot)
            plan = self._plan_for_slot(slot, base_plan, categories)
        strategy = plan.retrieval_strategy
        before_structured_filter = self.product_repository.count_available()
        structured_products = self.product_repository.list_for_plan(plan, limit=strategy.candidate_limit)
        query = query.strip() or slot.query

        if not structured_products:
            return SlotSearchResult(
                slot_id=slot.slot_id,
                query=slot.query,
                vector_query=query,
                keyword_query=query,
                counts=self._counts(before_structured_filter=before_structured_filter),
                attempts=[
                    {
                        "attempt": attempt_index,
                        "query": query,
                        "categories": categories,
                        "category_resolution": category_resolution,
                        "structured_candidates": 0,
                        "reason": "structured_filter_empty",
                    }
                ],
                categories=categories,
                category_resolution=category_resolution,
                structured_products=[],
            )

        result = self._search_once(
            slot=slot,
            products=structured_products,
            strategy=strategy,
            query=query,
        )
        candidates = [
            self._candidate(
                product,
                rerank_score=rerank_score,
                vector_score=result["vector_scores"].get(product.product_id, 0.0),
                keyword_score=result["keyword_scores"].get(product.product_id, 0.0),
                rrf_score=self._rrf_score(
                    product.product_id,
                    result["vector_scores"],
                    result["keyword_scores"],
                ),
                slot=slot,
            )
            for product, rerank_score in result["ranked"]
        ]
        return SlotSearchResult(
            slot_id=slot.slot_id,
            query=slot.query,
            vector_query=query,
            keyword_query=query,
            candidates=sorted(
                candidates,
                key=lambda candidate: (candidate.rerank_score, candidate.rrf_score, candidate.keyword_score),
                reverse=True,
            )[: strategy.final_top_k],
            counts=self._counts(
                before_structured_filter=before_structured_filter,
                after_structured_filter=len(structured_products),
                vector_hits=len(result["vector_scores"]),
                keyword_hits=len(result["keyword_scores"]),
                after_score_filter=len(result["score_filtered"]),
                after_hybrid_rank=len(result["fused_products"]),
                after_rerank=len(result["ranked"]),
            ),
            attempts=[
                {
                    "attempt": attempt_index,
                    "query": query,
                    "categories": categories,
                    "category_resolution": category_resolution,
                    "structured_candidates": len(structured_products),
                    "vector_hits": len(result["vector_scores"]),
                    "keyword_hits": len(result["keyword_scores"]),
                    "score_filtered": len(result["score_filtered"]),
                    "hybrid_ranked": len(result["fused_products"]),
                    "reranked": len(result["ranked"]),
                    "raw_candidate_ids": [product.product_id for product in result["fused_products"]],
                    "reranked_candidate_ids": [product.product_id for product, _ in result["ranked"]],
                    "reason": reason,
                }
            ],
            categories=categories,
            category_resolution=category_resolution,
            structured_products=structured_products,
            score_filtered_products=result["score_filtered"],
            hybrid_ranked_products=result["fused_products"],
            vector_scores=result["vector_scores"],
            keyword_scores=result["keyword_scores"],
        )

    def query_variants(self, slot: NeedSlot) -> list[str]:
        return self._query_variants(slot)

    def _search_once(
        self,
        slot: NeedSlot,
        products: list[Product],
        strategy: QueryRetrievalStrategy,
        query: str,
    ) -> dict:
        vector_scores = (
            self.retriever.retrieve(query, products, top_k=strategy.vector_top_k)
            if strategy.use_vector
            else {}
        )
        keyword_scores = (
            self.product_repository.keyword_scores(query, products, top_k=strategy.keyword_top_k)
            if strategy.use_keyword
            else {}
        )
        passing_ids = {
            product_id
            for product_id in {*vector_scores, *keyword_scores}
            if vector_scores.get(product_id, 0.0) >= self.MIN_VECTOR_SCORE
            or keyword_scores.get(product_id, 0.0) >= self.MIN_KEYWORD_SCORE
        }
        score_filtered = [product for product in products if product.product_id in passing_ids]
        fused_products = self._hybrid_rank_products(
            score_filtered,
            vector_scores,
            keyword_scores,
            top_k=strategy.hybrid_top_k,
        )
        ranked = self.reranker.rerank(
            self._rerank_query(slot, query),
            fused_products,
            top_k=strategy.final_top_k,
        )
        return {
            "vector_query": query,
            "keyword_query": query,
            "vector_scores": vector_scores,
            "keyword_scores": keyword_scores,
            "score_filtered": score_filtered,
            "fused_products": fused_products,
            "ranked": ranked,
        }

    def _plan_for_slot(self, slot: NeedSlot, base_plan: QueryPlan, categories: list[str]) -> QueryPlan:
        strategy = base_plan.retrieval_strategy.model_copy()
        strategy.final_top_k = max(strategy.final_top_k, slot.min_candidates)
        strategy.hybrid_top_k = max(strategy.hybrid_top_k, strategy.final_top_k * 3)
        plan = QueryPlan(
            intent=base_plan.intent,
            categories=categories,
            scene=base_plan.scene,
            budget=base_plan.budget.model_copy(),
            preferences=self._unique([*base_plan.preferences, *slot.soft_constraints]),
            exclude=self._unique([*base_plan.exclude, *slot.exclude_terms]),
            retrieval_strategy=strategy,
        )
        plan.filters = RetrievalPlanBuilder()._filters(plan)
        return plan

    def _resolve_categories(self, slot: NeedSlot) -> tuple[list[str], str]:
        product_type = slot.product_type.strip()
        goal = slot.goal.strip()
        for exact in [product_type, goal]:
            if exact in RetrievalPlanBuilder.CATEGORY_KEYWORDS:
                return [exact], "exact_sub_category"
            if exact in RetrievalPlanBuilder.TOP_LEVEL_CATEGORIES:
                return [exact], "exact_top_category"

        text = self._slot_core_text(slot)
        if not text:
            return [], "unknown"

        for top_category, hints in self.TOP_CATEGORY_HINTS.items():
            if any(hint in text for hint in hints):
                if top_category == "服饰运动" and "防晒" in text and not any(hint in text for hint in ("衣", "服", "帽", "鞋", "裤", "包")):
                    continue
                return [top_category], "top_category_from_product_type"

        for category, keywords in RetrievalPlanBuilder.CATEGORY_KEYWORDS.items():
            if any(keyword and keyword in text for keyword in keywords):
                return [category], "keyword_sub_category_from_product_type"
        return [], "unknown"

    def _query_variants(self, slot: NeedSlot) -> list[str]:
        product_type = slot.product_type.strip()
        goal = slot.goal.strip()
        soft_constraints = [value for value in slot.soft_constraints if value not in {product_type, goal}]
        keyword_expansion = self._category_keywords(product_type or goal)
        variants = [
            self._compact(" ".join([product_type, goal, *soft_constraints])),
            self._compact(" ".join([product_type or goal, *keyword_expansion[:4], *soft_constraints[:2]])),
            self._compact(" ".join([product_type or goal, goal, *keyword_expansion])),
        ]
        if not any(variants) and slot.query:
            variants.append(slot.query)
        return self._unique(variants)

    def _slot_core_text(self, slot: NeedSlot) -> str:
        return self._compact(" ".join(part for part in [slot.product_type, slot.goal] if part))

    def _category_keywords(self, product_type: str) -> list[str]:
        if product_type in RetrievalPlanBuilder.CATEGORY_KEYWORDS:
            return RetrievalPlanBuilder.CATEGORY_KEYWORDS[product_type]
        for category, keywords in RetrievalPlanBuilder.CATEGORY_KEYWORDS.items():
            if category in product_type or any(keyword and keyword in product_type for keyword in keywords):
                return [category, *keywords]
        return []

    def _rerank_query(self, slot: NeedSlot, query: str) -> str:
        return self._compact(" ".join(part for part in [slot.product_type, slot.goal, query] if part))

    def _has_enough_signal(self, ranked: list[tuple[Product, float]], slot: NeedSlot) -> bool:
        if len(ranked) >= max(1, slot.min_candidates * 2):
            return True
        if len(ranked) >= slot.min_candidates and any(score >= 0.35 for _, score in ranked):
            return True
        return False

    def _hybrid_rank_products(
        self,
        products: list[Product],
        vector_scores: dict[str, float],
        keyword_scores: dict[str, float],
        top_k: int,
    ) -> list[Product]:
        product_by_id = {product.product_id: product for product in products}
        fused_ids = rrf_fuse(
            [
                [product_id for product_id in vector_scores if product_id in product_by_id],
                [product_id for product_id in keyword_scores if product_id in product_by_id],
            ],
            top_k=top_k,
        )
        return [product_by_id[product_id] for product_id in fused_ids if product_id in product_by_id]

    def _rrf_score(
        self,
        product_id: str,
        vector_scores: dict[str, float],
        keyword_scores: dict[str, float],
        rrf_k: int = 60,
    ) -> float:
        score = 0.0
        for scores in (vector_scores, keyword_scores):
            ranked_ids = list(scores)
            if product_id in scores:
                score += 1.0 / (rrf_k + ranked_ids.index(product_id) + 1)
        return score

    def _candidate(
        self,
        product: Product,
        rerank_score: float,
        vector_score: float,
        keyword_score: float,
        rrf_score: float,
        slot: NeedSlot,
    ) -> SlotCandidate:
        return SlotCandidate(
            product=product,
            product_id=product.product_id,
            name=product.name,
            category=product.category,
            sub_category=product.sub_category,
            price=float(product.price or Decimal("0")),
            vector_score=round(vector_score, 4),
            keyword_score=round(keyword_score, 4),
            rrf_score=round(rrf_score, 6),
            rerank_score=round(rerank_score, 4),
            coverage_reason=f"匹配子需求：{slot.goal or slot.product_type}",
        )

    def _counts(
        self,
        before_structured_filter: int = 0,
        after_structured_filter: int = 0,
        vector_hits: int = 0,
        keyword_hits: int = 0,
        after_score_filter: int = 0,
        after_hybrid_rank: int = 0,
        after_rerank: int = 0,
    ) -> dict[str, int]:
        return {
            "before_structured_filter": before_structured_filter,
            "after_structured_filter": after_structured_filter,
            "vector_hits": vector_hits,
            "keyword_hits": keyword_hits,
            "after_score_filter": after_score_filter,
            "after_hybrid_rank": after_hybrid_rank,
            "after_rerank": after_rerank,
        }

    def _merge_counts(self, left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
        keys = set(left) | set(right)
        merged: dict[str, int] = {}
        for key in keys:
            if key in {"before_structured_filter", "after_structured_filter"}:
                merged[key] = max(left.get(key, 0), right.get(key, 0))
            else:
                merged[key] = max(left.get(key, 0), right.get(key, 0))
        return merged

    def _compact(self, text: str) -> str:
        return " ".join(text.split())

    def _unique(self, values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            text = str(value).strip()
            if text and text not in result:
                result.append(text)
        return result
