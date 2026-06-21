from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from app.core.config import get_settings


class OrganizerDatasetError(RuntimeError):
    pass


def resolve_dataset_dir() -> Path:
    settings = get_settings()
    candidates = []
    if settings.organizer_dataset_dir:
        candidates.append(Path(settings.organizer_dataset_dir))
    candidates.append(Path(settings.fallback_dataset_dir))
    candidates.append(Path(__file__).resolve().parents[3] / "data" / "raw" / "ecommerce_agent_dataset")
    candidates.extend(_downloads_candidates())

    for candidate in candidates:
        path = candidate.expanduser()
        if path.exists() and path.is_dir():
            return path
    raise OrganizerDatasetError(
        "Organizer dataset directory not found. Set ORGANIZER_DATASET_DIR in .env."
    )


def _downloads_candidates() -> list[Path]:
    home = Path.home()
    downloads = home / "Downloads"
    if not downloads.exists():
        return []
    direct = downloads / "ecommerce_agent_dataset_供参考" / "ecommerce_agent_dataset"
    candidates = [direct]
    candidates.extend(path / "ecommerce_agent_dataset" for path in downloads.glob("ecommerce_agent_dataset*"))
    return candidates


def load_organizer_products(dataset_dir: Path | None = None) -> list[dict[str, Any]]:
    root = dataset_dir or resolve_dataset_dir()
    files = sorted(root.glob("*_/data/*.json"))
    if not files:
        files = sorted(root.glob("*/data/*.json"))
    products = [_normalize_product(path, root) for path in files]
    if not products:
        raise OrganizerDatasetError(f"No product JSON files found under {root}")
    return products


def _normalize_product(path: Path, root: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    knowledge = raw.get("rag_knowledge", {})
    reviews = knowledge.get("user_reviews", [])
    skus = raw.get("skus", [])
    image_path = raw.get("image_path", "")
    image_url = f"/dataset/{image_path.replace(chr(92), '/')}" if image_path else ""

    review_ratings = [review.get("rating") for review in reviews if review.get("rating") is not None]
    review_contents = [
        f"{review.get('nickname', '用户')}({review.get('rating', '-')}/5)：{review.get('content', '')}"
        for review in reviews
        if review.get("content")
    ]
    sku_terms = _sku_terms(skus)

    return {
        "product_id": raw["product_id"],
        "name": raw["title"],
        "category": raw["category"],
        "sub_category": raw.get("sub_category"),
        "brand": raw["brand"],
        "price": raw["base_price"],
        "stock": None,
        "image_url": image_url,
        "description": knowledge.get("marketing_description", ""),
        "specs": {
            "skus": skus,
            "source_image_path": image_path,
            "source_json_path": str(path.relative_to(root)),
        },
        "ingredients_or_material": "",
        "suitable_for": "",
        "avoid_for": "",
        "tags": _unique([raw["category"], raw.get("sub_category"), raw["brand"], *sku_terms]),
        "rating": round(mean(review_ratings), 2) if review_ratings else 0,
        "sales": None,
        "review_summary": "\n".join(review_contents),
        "image_caption": "",
        "structured_attributes": {
            "source": "organizer_dataset",
            "category": raw["category"],
            "sub_category": raw.get("sub_category"),
            "base_price": raw["base_price"],
            "image_path": image_path,
            "official_faq": knowledge.get("official_faq", []),
            "user_reviews": reviews,
        },
    }


def _sku_terms(skus: list[dict[str, Any]]) -> list[str]:
    terms: list[str] = []
    for sku in skus:
        for value in sku.get("properties", {}).values():
            terms.append(str(value))
    return terms


def _unique(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result
