from __future__ import annotations

from sqlalchemy import String, func, or_, select
from sqlalchemy.orm import Session

from app.db.models import Product
from app.rag.bm25 import ProductBM25Scorer
from app.schemas import QueryPlan


class ProductRepository:
    def __init__(self, db: Session | None) -> None:
        self.db = db

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

    def list_for_plan(self, plan: QueryPlan, limit: int = 200) -> list[Product]:
        if self.db is None:
            return []
        stmt = select(Product).where(or_(Product.stock.is_(None), Product.stock > 0))
        # 预算、排除词、类目只作为理解/审核信号，不在召回前硬过滤候选池。
        # 这避免“不要手机”误杀说明里提到手机兼容性的耳机，也避免类目误判直接挡掉可用商品。
        stmt = stmt.limit(max(limit, 1000))
        return list(self.db.scalars(stmt).all())

    def keyword_scores(self, query: str, products: list[Product], top_k: int | None = None) -> dict[str, float]:
        return ProductBM25Scorer(products).score(query, top_k=top_k)

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
