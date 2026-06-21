from decimal import Decimal

from app.db.models import Product
from app.domain.need_slot_schemas import SlotCandidate, SlotSearchResult
from app.domain.repair_worker import RepairPlan
from app.domain.retrieval_worker import RetrievalWorker
from app.schemas import IntentPlan, QueryPlan


def make_product(product_id: str, name: str) -> Product:
    return Product(
        product_id=product_id,
        name=name,
        category="数码电子",
        sub_category="配件",
        brand="Demo",
        price=Decimal("59"),
        stock=10,
        image_url="",
        description=name,
        specs={},
        ingredients_or_material="",
        suitable_for="",
        avoid_for="",
        tags=["配件"],
        rating=Decimal("4.8"),
        sales=100,
        review_summary="",
        image_caption="",
        structured_attributes={},
    )


class FakeSearchTool:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def search_query(self, **kwargs) -> SlotSearchResult:
        self.calls.append(kwargs)
        product = make_product(f"p{len(self.calls)}", kwargs["query"])
        candidate = SlotCandidate(
            product=product,
            product_id=product.product_id,
            name=product.name,
            category=product.category,
            sub_category=product.sub_category,
            price=float(product.price),
            vector_score=0.4,
            keyword_score=0.3,
            rrf_score=0.03,
            rerank_score=0.9,
        )
        return SlotSearchResult(
            slot_id=kwargs["slot"].slot_id,
            query=kwargs["query"],
            vector_query=kwargs["query"],
            keyword_query=kwargs["query"],
            candidates=[candidate],
            counts={
                "before_structured_filter": 1,
                "after_structured_filter": 1,
                "vector_hits": 1,
                "keyword_hits": 1,
                "after_score_filter": 1,
                "after_hybrid_rank": 1,
                "after_rerank": 1,
            },
            structured_products=[product],
            score_filtered_products=[product],
            hybrid_ranked_products=[product],
            vector_scores={product.product_id: 0.4},
            keyword_scores={product.product_id: 0.3},
        )


class FakeImageRetrievalWorker:
    def __init__(self) -> None:
        self.kwargs: dict | None = None

    def run(self, **kwargs):
        self.kwargs = kwargs
        return {"ok": True}


def test_retrieval_worker_executes_single_repair_plan_via_product_search_tool() -> None:
    search_tool = FakeSearchTool()
    worker = RetrievalWorker(
        product_search_tool=search_tool,  # type: ignore[arg-type]
        single_retrieval_worker=object(),  # type: ignore[arg-type]
        multi_need_coordinator=object(),  # type: ignore[arg-type]
        image_retrieval_worker=object(),  # type: ignore[arg-type]
    )

    evidence = worker.run_single_repair(
        "买根type-c数据线和充电头",
        IntentPlan(original_query="买根type-c数据线和充电头"),
        QueryPlan(),
        RepairPlan(targets=["single"], queries_by_slot={"single": ["type-c 数据线", "充电头"]}),
    )

    assert [call["query"] for call in search_tool.calls] == ["type-c 数据线", "充电头"]
    assert evidence.tool_call_count == 2
    assert [product.product_id for product, _score in evidence.ranked] == ["p1", "p2"]


def test_retrieval_worker_forwards_image_initial_with_keyword_arguments() -> None:
    image_worker = FakeImageRetrievalWorker()
    worker = RetrievalWorker(
        product_search_tool=object(),  # type: ignore[arg-type]
        single_retrieval_worker=object(),  # type: ignore[arg-type]
        multi_need_coordinator=object(),  # type: ignore[arg-type]
        image_retrieval_worker=image_worker,  # type: ignore[arg-type]
    )
    intent_plan = IntentPlan(original_query="找相似商品")
    plan = QueryPlan()

    result = worker.run_image_initial("找相似商品", intent_plan, plan, "upload.jpg")

    assert result == {"ok": True}
    assert image_worker.kwargs == {
        "original_query": "找相似商品",
        "intent_plan": intent_plan,
        "plan": plan,
        "image_path": "upload.jpg",
    }
