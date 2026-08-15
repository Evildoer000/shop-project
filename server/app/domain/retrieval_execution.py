from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TypeVar

from app.db.session import get_sessionmaker
from app.domain.image_search_tool import ImageSearchTool
from app.domain.product_search_tool import ProductSearchTool
from app.services.product_repository import ProductRepository


ResultT = TypeVar("ResultT")


class RetrievalExecutionBoundary:
    """Run blocking retrieval with thread-local runtime dependencies.

    Request-scoped SQLAlchemy sessions must not cross into worker threads. Some
    retrieval clients also expect an event loop to exist in the worker thread.
    This boundary owns both requirements while reusing the configured retriever,
    reranker, and image retriever.
    """

    def __init__(
        self,
        product_search_tool: ProductSearchTool,
        image_search_tool: ImageSearchTool | None = None,
    ) -> None:
        self.product_search_tool = product_search_tool
        self.image_search_tool = image_search_tool

    async def run_product(
        self,
        operation: Callable[[ProductSearchTool], ResultT],
    ) -> ResultT:
        return await asyncio.to_thread(self.run_product_sync, operation)

    def run_product_sync(
        self,
        operation: Callable[[ProductSearchTool], ResultT],
    ) -> ResultT:
        self._ensure_thread_event_loop()
        repository = getattr(self.product_search_tool, "product_repository", None)
        current_db = getattr(repository, "db", None)
        retriever = getattr(self.product_search_tool, "retriever", None)
        reranker = getattr(self.product_search_tool, "reranker", None)
        if current_db is None or retriever is None or reranker is None:
            return operation(self.product_search_tool)

        db = get_sessionmaker()()
        try:
            isolated_tool = ProductSearchTool(ProductRepository(db), retriever, reranker)
            return operation(isolated_tool)
        finally:
            db.close()

    async def run_image(
        self,
        operation: Callable[[ImageSearchTool], ResultT],
    ) -> ResultT:
        return await asyncio.to_thread(self.run_image_sync, operation)

    def run_image_sync(
        self,
        operation: Callable[[ImageSearchTool], ResultT],
    ) -> ResultT:
        self._ensure_thread_event_loop()
        if self.image_search_tool is None:
            raise RuntimeError("Image retrieval execution boundary is not configured")
        repository = getattr(self.image_search_tool, "product_repository", None)
        current_db = getattr(repository, "db", None)
        image_retriever = getattr(self.image_search_tool, "image_retriever", None)
        if current_db is None or image_retriever is None:
            return operation(self.image_search_tool)

        db = get_sessionmaker()()
        try:
            isolated_tool = ImageSearchTool(ProductRepository(db), image_retriever)
            return operation(isolated_tool)
        finally:
            db.close()

    @staticmethod
    def _ensure_thread_event_loop() -> None:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
            return
        if loop.is_closed():
            asyncio.set_event_loop(asyncio.new_event_loop())
