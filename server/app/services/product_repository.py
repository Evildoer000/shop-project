from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import String, func, or_, select
from sqlalchemy.orm import Session

from app.db.models import Product
from app.rag.bm25 import ProductBM25Scorer
from app.schemas import QueryPlan
from app.services.elasticsearch_product_index import ElasticsearchProductIndex


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProductCatalogResult:
    products: list[Product]
    total: int
    scores: dict[str, float]


class ProductRepository:
    def __init__(self, db: Session | None) -> None:
        self.db = db
        self.keyword_index = ElasticsearchProductIndex()

    def get_by_id(self, product_id: str) -> Product | None:
        if self.db is None:
            return None
        return self.db.get(Product, product_id)

    def get_by_ids(self, product_ids: list[str]) -> list[Product]:
        if self.db is None or not product_ids:
            return []
        unique_ids = list(dict.fromkeys(product_id for product_id in product_ids if product_id))
        if not unique_ids:
            return []
        products = self.db.scalars(select(Product).where(Product.product_id.in_(unique_ids))).all()
        by_id = {product.product_id: product for product in products}
        return [by_id[product_id] for product_id in unique_ids if product_id in by_id]

    def count_available(self) -> int:
        if self.db is None:
            return 0
        stmt = select(func.count()).select_from(Product).where(or_(Product.stock.is_(None), Product.stock > 0))
        return int(self.db.scalar(stmt) or 0)

    def list_catalog(
        self,
        *,
        category: str | None = None,
        sub_category: str | None = None,
        query: str | None = None,
        page: int = 1,
        page_size: int = 24,
        sort: str = "popular",
    ) -> tuple[list[Product], int]:
        result = self.list_catalog_with_scores(
            category=category,
            sub_category=sub_category,
            query=query,
            page=page,
            page_size=page_size,
            sort=sort,
        )
        return result.products, result.total

    def list_catalog_with_scores(
        self,
        *,
        category: str | None = None,
        sub_category: str | None = None,
        query: str | None = None,
        page: int = 1,
        page_size: int = 24,
        sort: str = "popular",
    ) -> ProductCatalogResult:
        query = (query or "").strip()
        if query:
            es_result = self._list_catalog_elasticsearch(
                category=category,
                sub_category=sub_category,
                query=query,
                page=page,
                page_size=page_size,
                sort=sort,
            )
            if es_result is not None:
                return es_result
        products, total = self._list_catalog_sql(
            category=category,
            sub_category=sub_category,
            query=query,
            page=page,
            page_size=page_size,
            sort=sort,
        )
        return ProductCatalogResult(products=products, total=total, scores={})

    def _list_catalog_sql(
        self,
        *,
        category: str | None = None,
        sub_category: str | None = None,
        query: str | None = None,
        page: int = 1,
        page_size: int = 24,
        sort: str = "popular",
    ) -> tuple[list[Product], int]:
        if self.db is None:
            return [], 0
        page = max(page, 1)
        page_size = min(max(page_size, 1), 60)
        filters = self._catalog_filters(category=category, sub_category=sub_category, query=query)
        total = int(self.db.scalar(select(func.count()).select_from(Product).where(*filters)) or 0)
        stmt = (
            select(Product)
            .where(*filters)
            .order_by(*self._catalog_order(sort))
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        return list(self.db.scalars(stmt).all()), total

    def _list_catalog_elasticsearch(
        self,
        *,
        category: str | None,
        sub_category: str | None,
        query: str,
        page: int,
        page_size: int,
        sort: str,
    ) -> ProductCatalogResult | None:
        if self.db is None or not self.keyword_index.enabled:
            return None
        page = max(page, 1)
        page_size = min(max(page_size, 1), 60)
        try:
            result = self.keyword_index.search(
                query,
                top_k=page_size,
                offset=(page - 1) * page_size,
                category=category,
                sub_category=sub_category,
                sort=sort,
            )
        except Exception as exc:
            LOGGER.warning("Elasticsearch catalog search failed; falling back to SQL: %s", exc)
            return None
        if not result.hits:
            return None
        products = self.get_by_ids([hit.product_id for hit in result.hits])
        scores = self._normalized_scores({hit.product_id: hit.score for hit in result.hits})
        return ProductCatalogResult(products=products, total=result.total, scores=scores)

    def list_categories(self) -> list[tuple[str, str | None, int]]:
        if self.db is None:
            return []
        stmt = (
            select(Product.category, Product.sub_category, func.count())
            .where(or_(Product.stock.is_(None), Product.stock > 0))
            .group_by(Product.category, Product.sub_category)
            .order_by(Product.category.asc(), Product.sub_category.asc())
        )
        return [(str(category or "未分类"), sub_category, int(count or 0)) for category, sub_category, count in self.db.execute(stmt).all()]

    def list_available_products(self) -> list[Product]:
        if self.db is None:
            return []
        stmt = (
            select(Product)
            .where(or_(Product.stock.is_(None), Product.stock > 0))
            .order_by(Product.product_id.asc())
        )
        return list(self.db.scalars(stmt).all())

    def list_for_plan(self, plan: QueryPlan, limit: int = 200) -> list[Product]:
        if self.db is None:
            return []
        stmt = select(Product).where(or_(Product.stock.is_(None), Product.stock > 0))
        # 预算、排除词、类目只作为理解/审核信号，不在召回前硬过滤候选池。
        # 这避免“不要手机”误杀说明里提到手机兼容性的耳机，也避免类目误判直接挡掉可用商品。
        stmt = stmt.limit(max(limit, 1000))
        return list(self.db.scalars(stmt).all())

    def keyword_scores(self, query: str, products: list[Product], top_k: int | None = None) -> dict[str, float]:
        if not query or not products:
            return {}
        if self.keyword_index.enabled:
            try:
                search_top_k = max(1, int(top_k or min(len(products), 1000)))
                result = self.keyword_index.search(query, top_k=search_top_k)
                allowed_ids = {product.product_id for product in products}
                scores = self._normalized_scores(
                    {
                        hit.product_id: hit.score
                        for hit in result.hits
                        if hit.product_id in allowed_ids
                    }
                )
                if scores:
                    return scores
            except Exception as exc:
                LOGGER.warning("Elasticsearch keyword search failed; falling back to local BM25: %s", exc)
        return ProductBM25Scorer(products).score(query, top_k=top_k)

    def _normalized_scores(self, scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        max_score = max(scores.values())
        if max_score <= 0:
            return {product_id: 0.0 for product_id in scores}
        return {product_id: float(score) / float(max_score) for product_id, score in scores.items()}

    def _matches_exclude(self, product: Product, excludes: list[str]) -> bool:
        return False

    def _catalog_filters(
        self,
        *,
        category: str | None,
        sub_category: str | None,
        query: str | None,
    ) -> list:
        filters = [or_(Product.stock.is_(None), Product.stock > 0)]
        category = (category or "").strip()
        sub_category = (sub_category or "").strip()
        query = (query or "").strip()
        if category:
            filters.append(Product.category == category)
        if sub_category:
            filters.append(Product.sub_category == sub_category)
        if query:
            pattern = f"%{query}%"
            filters.append(
                or_(
                    Product.name.ilike(pattern),
                    Product.brand.ilike(pattern),
                    Product.category.ilike(pattern),
                    Product.sub_category.ilike(pattern),
                    Product.description.ilike(pattern),
                    Product.tags.cast(String).ilike(pattern),
                    Product.structured_attributes.cast(String).ilike(pattern),
                )
            )
        return filters

    def _catalog_order(self, sort: str) -> list:
        if sort == "price_asc":
            return [Product.price.asc(), Product.product_id.asc()]
        if sort == "price_desc":
            return [Product.price.desc(), Product.product_id.asc()]
        if sort == "rating":
            return [Product.rating.desc(), Product.sales.desc().nullslast(), Product.product_id.asc()]
        if sort == "newest":
            return [Product.updated_at.desc().nullslast(), Product.product_id.asc()]
        return [Product.sales.desc().nullslast(), Product.rating.desc(), Product.product_id.asc()]
