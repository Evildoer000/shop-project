import asyncio
from types import SimpleNamespace

from app.domain.product_search_tool import ProductSearchTool
from app.domain.retrieval_execution import RetrievalExecutionBoundary


def test_product_retrieval_uses_thread_event_loop_and_isolated_session(monkeypatch) -> None:
    request_session = object()
    worker_session = SimpleNamespace(closed=False)

    def close() -> None:
        worker_session.closed = True

    worker_session.close = close
    monkeypatch.setattr(
        "app.domain.retrieval_execution.get_sessionmaker",
        lambda: lambda: worker_session,
    )
    monkeypatch.setattr(
        "app.domain.retrieval_execution.ProductRepository",
        lambda db: SimpleNamespace(db=db),
    )
    request_tool = SimpleNamespace(
        product_repository=SimpleNamespace(db=request_session),
        retriever=object(),
        reranker=object(),
    )
    boundary = RetrievalExecutionBoundary(request_tool)  # type: ignore[arg-type]

    def operation(tool: ProductSearchTool) -> tuple[bool, bool]:
        loop = asyncio.get_event_loop()
        return loop.is_closed(), tool.product_repository.db is worker_session

    result = asyncio.run(boundary.run_product(operation))

    assert result == (False, True)
    assert worker_session.closed is True
