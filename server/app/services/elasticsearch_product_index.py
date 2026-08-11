from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

from app.core.config import get_settings
from app.db.models import Product


LOGGER = logging.getLogger(__name__)


class ElasticsearchProductIndexError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProductSearchHit:
    product_id: str
    score: float
    rank: int


@dataclass(frozen=True)
class ProductSearchResult:
    hits: list[ProductSearchHit]
    total: int


class ElasticsearchProductIndex:
    """Product keyword search backed by Elasticsearch.

    Elasticsearch analyzes original Chinese text with IK. Local BM25 remains
    only as an outage fallback and does not participate in the normal ES path.
    """

    SEARCH_FIELDS = [
        "name^8",
        "sub_category.text^7",
        "category.text^5",
        "brand.text^4",
        "tags^4",
        "description^2",
        "review_summary^1.5",
        "search_text",
    ]

    def __init__(self) -> None:
        self.settings = get_settings()
        self.enabled = bool(self.settings.elasticsearch_enabled)
        self.base_url = self.settings.elasticsearch_url.rstrip("/")
        self.index_name = self.settings.elasticsearch_index
        self.timeout = float(self.settings.elasticsearch_timeout_seconds)

    def ping(self) -> bool:
        if not self.enabled:
            return False
        try:
            with self._client() as client:
                response = client.get("/")
            return response.status_code < 400
        except Exception as exc:
            LOGGER.debug("Elasticsearch ping failed: %s", exc)
            return False

    def count_documents(self) -> int:
        if not self.enabled:
            return 0
        try:
            with self._client() as client:
                response = client.get(f"/{self.index_name}/_count")
            if response.status_code == 404:
                return 0
            response.raise_for_status()
            data = response.json()
            return int(data.get("count") or 0)
        except Exception as exc:
            LOGGER.warning("Elasticsearch count failed: %s", exc)
            return 0

    def index_products(self, products: list[Product], *, overwrite: bool = True) -> int:
        if not self.enabled:
            return 0
        if not products:
            return 0
        if overwrite:
            self.delete_index(ignore_missing=True)
        self.ensure_index()

        total = 0
        with self._client() as client:
            for start in range(0, len(products), 500):
                batch = products[start : start + 500]
                lines: list[str] = []
                for product in batch:
                    lines.append(
                        json.dumps(
                            {"index": {"_index": self.index_name, "_id": product.product_id}},
                            ensure_ascii=False,
                        )
                    )
                    lines.append(json.dumps(self._document(product), ensure_ascii=False, default=str))
                payload = ("\n".join(lines) + "\n").encode("utf-8")
                response = client.post(
                    "/_bulk",
                    content=payload,
                    headers={"Content-Type": "application/x-ndjson"},
                )
                response.raise_for_status()
                data = response.json()
                if data.get("errors"):
                    errors = [
                        item.get("index", {}).get("error")
                        for item in data.get("items", [])
                        if item.get("index", {}).get("error")
                    ]
                    raise ElasticsearchProductIndexError(f"bulk index failed: {errors[:3]}")
                total += len(batch)
            client.post(f"/{self.index_name}/_refresh").raise_for_status()
        return total

    def ensure_index(self) -> None:
        if not self.enabled:
            return
        if self._index_exists():
            return
        with self._client() as client:
            response = client.put(f"/{self.index_name}", json=self._index_body())
        response.raise_for_status()

    def delete_index(self, *, ignore_missing: bool = False) -> None:
        if not self.enabled:
            return
        with self._client() as client:
            response = client.delete(f"/{self.index_name}")
        if response.status_code == 404 and ignore_missing:
            return
        response.raise_for_status()

    def search(
        self,
        query: str,
        *,
        top_k: int = 12,
        offset: int = 0,
        category: str | None = None,
        sub_category: str | None = None,
        boost_categories: list[str] | None = None,
        max_price: float | None = None,
        sort: str = "relevance",
    ) -> ProductSearchResult:
        if not self.enabled:
            raise ElasticsearchProductIndexError("Elasticsearch is disabled.")
        query_text = query.strip()
        if not query_text:
            return ProductSearchResult(hits=[], total=0)

        body = self._search_body(
            query_text=query_text,
            top_k=top_k,
            offset=offset,
            category=category,
            sub_category=sub_category,
            boost_categories=boost_categories or [],
            max_price=max_price,
            sort=sort,
        )
        with self._client() as client:
            response = client.post(f"/{self.index_name}/_search", json=body)
        if response.status_code == 404:
            raise ElasticsearchProductIndexError(f"Elasticsearch index '{self.index_name}' is missing.")
        response.raise_for_status()
        data = response.json()
        hits_payload = data.get("hits", {})
        total_payload = hits_payload.get("total", {})
        if isinstance(total_payload, dict):
            total = int(total_payload.get("value") or 0)
        else:
            total = int(total_payload or 0)
        hits = [
            ProductSearchHit(
                product_id=str(item.get("_source", {}).get("product_id") or item.get("_id") or ""),
                score=float(item.get("_score") or 0.0),
                rank=offset + index + 1,
            )
            for index, item in enumerate(hits_payload.get("hits", []))
        ]
        return ProductSearchResult(hits=[hit for hit in hits if hit.product_id], total=total)

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def _index_exists(self) -> bool:
        with self._client() as client:
            response = client.head(f"/{self.index_name}")
        if response.status_code == 200:
            return True
        if response.status_code == 404:
            return False
        response.raise_for_status()
        return False

    def _index_body(self) -> dict[str, Any]:
        analyzed_text = {
            "type": "text",
            "analyzer": "ik_max_word",
            "search_analyzer": "ik_smart",
        }
        exact_with_text = {
            "type": "keyword",
            "fields": {
                "text": analyzed_text,
            },
        }
        return {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
            },
            "mappings": {
                "properties": {
                    "product_id": {"type": "keyword"},
                    "name": analyzed_text,
                    "brand": exact_with_text,
                    "category": exact_with_text,
                    "sub_category": exact_with_text,
                    "description": analyzed_text,
                    "tags": analyzed_text,
                    "price": {"type": "double"},
                    "stock": {"type": "integer"},
                    "rating": {"type": "double"},
                    "sales": {"type": "integer"},
                    "updated_at": {"type": "date"},
                    "review_summary": analyzed_text,
                    "search_text": analyzed_text,
                }
            },
        }

    def _search_body(
        self,
        *,
        query_text: str,
        top_k: int,
        offset: int,
        category: str | None,
        sub_category: str | None,
        boost_categories: list[str],
        max_price: float | None,
        sort: str,
    ) -> dict[str, Any]:
        filters: list[dict[str, Any]] = [{"range": {"stock": {"gt": 0}}}]
        if category:
            filters.append({"term": {"category": category}})
        if sub_category:
            filters.append({"term": {"sub_category": sub_category}})
        if max_price is not None:
            filters.append({"range": {"price": {"lte": float(max_price)}}})

        must: list[dict[str, Any]] = [
            {
                "multi_match": {
                    "query": query_text,
                    "fields": self.SEARCH_FIELDS,
                    "type": "best_fields",
                    "operator": "or",
                    "minimum_should_match": "30%",
                }
            }
        ]
        should: list[dict[str, Any]] = []
        for value in self._unique(boost_categories):
            should.extend(
                [
                    {"term": {"category": {"value": value, "boost": 2.5}}},
                    {"term": {"sub_category": {"value": value, "boost": 4.0}}},
                ]
            )

        query: dict[str, Any] = {
            "function_score": {
                "query": {
                    "bool": {
                        "must": must,
                        "filter": filters,
                        "should": should,
                    }
                },
                "functions": [
                    {
                        "field_value_factor": {
                            "field": "rating",
                            "factor": 0.03,
                            "modifier": "sqrt",
                            "missing": 0,
                        }
                    },
                    {
                        "field_value_factor": {
                            "field": "sales",
                            "factor": 0.001,
                            "modifier": "log1p",
                            "missing": 0,
                        }
                    },
                ],
                "score_mode": "sum",
                "boost_mode": "sum",
            }
        }

        body: dict[str, Any] = {
            "from": max(offset, 0),
            "size": max(top_k, 1),
            "track_total_hits": True,
            "_source": ["product_id"],
            "query": query,
        }
        sort_clause = self._sort_clause(sort)
        if sort_clause:
            body["sort"] = sort_clause
        return body

    def _sort_clause(self, sort: str) -> list[dict[str, Any]]:
        if sort == "price_asc":
            return [{"price": {"order": "asc"}}, {"_score": {"order": "desc"}}]
        if sort == "price_desc":
            return [{"price": {"order": "desc"}}, {"_score": {"order": "desc"}}]
        if sort == "rating":
            return [{"rating": {"order": "desc"}}, {"_score": {"order": "desc"}}]
        if sort == "newest":
            return [{"updated_at": {"order": "desc", "missing": "_last"}}, {"_score": {"order": "desc"}}]
        return []

    def _document(self, product: Product) -> dict[str, Any]:
        tags = product.tags or []
        sub_category = product.sub_category or ""
        updated_at = self._datetime(product.updated_at)
        return {
            "product_id": product.product_id,
            "name": product.name,
            "brand": product.brand,
            "category": product.category,
            "sub_category": sub_category,
            "description": product.description,
            "tags": tags if isinstance(tags, list) else [],
            "price": float(product.price or 0),
            "stock": int(product.stock or 0),
            "rating": float(product.rating or 0),
            "sales": int(product.sales or 0),
            "updated_at": updated_at,
            "review_summary": product.review_summary,
            "search_text": product.search_text(),
        }

    def _datetime(self, value: datetime | None) -> str | None:
        if value is None:
            return None
        return value.isoformat()

    def _unique(self, values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            text = str(value or "").strip()
            if text and text not in result:
                result.append(text)
        return result
