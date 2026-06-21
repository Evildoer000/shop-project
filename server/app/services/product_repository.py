from __future__ import annotations

from sqlalchemy import func, or_, select
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
