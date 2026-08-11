from __future__ import annotations

import sys
from pathlib import Path


SERVER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER_ROOT))

from app.db.models import Product
from app.db.session import get_sessionmaker
from app.services.elasticsearch_product_index import ElasticsearchProductIndex


def main() -> None:
    index = ElasticsearchProductIndex()
    if not index.enabled:
        print("Elasticsearch is disabled; set ELASTICSEARCH_ENABLED=true to build keyword index.")
        return

    SessionLocal = get_sessionmaker()
    with SessionLocal() as db:
        products = list(db.query(Product).all())

    indexed = index.index_products(products, overwrite=True)
    print(f"Indexed {indexed} products into Elasticsearch index '{index.index_name}'.")


if __name__ == "__main__":
    main()
