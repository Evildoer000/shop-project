from decimal import Decimal

from app.db.models import Product
from app.domain.need_slot_schemas import SlotCandidate, SlotSearchResult
from app.domain.single_retrieval_worker import SingleRetrievalWorker
from app.schemas import IntentPlan, QueryPlan


def make_product(product_id: str, name: str) -> Product:
    return Product(
        product_id=product_id,
        name=name,
        category="beauty",
        sub_category="sunscreen",
        brand="Demo",
        price=Decimal("100"),
        stock=10,
        image_url="",
        description=name,
        specs={},
        ingredients_or_material="",
        suitable_for="",
        avoid_for="",
        tags=["sunscreen"],
        rating=Decimal("4.8"),
        sales=100,
        review_summary="",
        image_caption="",
        structured_attributes={},
    )


class FakeSearchTool:
    def __init__(self, result: SlotSearchResult) -> None:
        self.result = result
        self.calls: list[dict] = []

    def search_query(self, **kwargs) -> SlotSearchResult:
        self.calls.append(kwargs)
        assert kwargs["use_base_plan"] is True
        return self.result


def make_candidate(product: Product, score: float) -> SlotCandidate:
    return SlotCandidate(
        product=product,
        product_id=product.product_id,
        name=product.name,
        category=product.category,
        sub_category=product.sub_category,
        price=float(product.price),
        vector_score=0.3,
        keyword_score=0.2,
        rrf_score=0.03,
        rerank_score=score,
    )


def test_single_retrieval_worker_maps_product_search_tool_result_without_corrective_agent() -> None:
    products = [make_product("p1", "Sunscreen milk"), make_product("p2", "Sunscreen spray")]
    result = SlotSearchResult(
        slot_id="single_retrieval",
        query="sunscreen",
        vector_query="sunscreen",
        keyword_query="sunscreen",
        candidates=[make_candidate(products[0], 0.9), make_candidate(products[1], 0.8)],
        counts={
            "before_structured_filter": 2,
            "after_structured_filter": 2,
            "vector_hits": 1,
            "keyword_hits": 1,
            "after_score_filter": 2,
            "after_hybrid_rank": 2,
            "after_rerank": 2,
        },
        structured_products=products,
        score_filtered_products=products,
        hybrid_ranked_products=products,
        vector_scores={"p1": 0.3},
        keyword_scores={"p2": 0.2},
    )
    search_tool = FakeSearchTool(result)
    worker = SingleRetrievalWorker(search_tool)  # type: ignore[arg-type]

    evidence = worker.run(
        "help me find sunscreen",
        IntentPlan(original_query="help me find sunscreen", vector_query="sunscreen", keyword_query="sunscreen"),
        QueryPlan(),
    )

    assert search_tool.calls
    assert evidence.failure_trigger == ""
    assert evidence.after_structured_filter == 2
    assert evidence.after_score_filter == 2
    assert evidence.after_rerank == 2
    assert evidence.tool_call_count == 3
    assert [product.product_id for product, _ in evidence.ranked] == ["p1", "p2"]


def test_single_retrieval_worker_stops_when_structured_pool_is_empty() -> None:
    result = SlotSearchResult(
        slot_id="single_retrieval",
        query="sunscreen",
        vector_query="sunscreen",
        keyword_query="sunscreen",
        counts={"before_structured_filter": 0},
    )
    worker = SingleRetrievalWorker(FakeSearchTool(result))  # type: ignore[arg-type]

    evidence = worker.run(
        "help me find sunscreen",
        IntentPlan(original_query="help me find sunscreen", vector_query="sunscreen", keyword_query="sunscreen"),
        QueryPlan(),
    )

    assert evidence.failure_trigger == "no_candidates"
    assert evidence.tool_call_count == 1
    assert evidence.ranked == []
