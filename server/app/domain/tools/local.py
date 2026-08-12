from __future__ import annotations

from typing import Any


class ProductDetailTool:
    """Read-only local product facts for comparison and verification."""

    def __init__(self, repository: Any) -> None:
        self.repository = repository

    def get_by_ids(self, product_ids: list[str]) -> list[Any]:
        return self.repository.get_by_ids(product_ids)


class ImageUnderstandingTool:
    """Explicit multimodal capability supplied to recommendation Agents."""

    def __init__(self, extractor: Any) -> None:
        self.extractor = extractor

    async def understand(self, image_path: str, query: str = "") -> Any:
        return await self.extractor.extract(image_path, user_text=query)
