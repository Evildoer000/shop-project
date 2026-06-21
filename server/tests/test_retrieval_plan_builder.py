from app.domain.retrieval_plan_builder import RetrievalPlanBuilder
from app.schemas import IntentPlan, RewriteNeedSlot


def test_builds_query_plan_from_minimal_intent_plan() -> None:
    intent_plan = IntentPlan(
        original_query="预算300到500，推荐降噪耳机，别太重",
        plan_type="single_retrieval",
        vector_query="降噪耳机 预算300到500",
        keyword_query="降噪 耳机 预算300到500",
        budget_min=300,
        budget_max=500,
        budget_scope="per_item",
    )

    plan = RetrievalPlanBuilder().plan(intent_plan)

    assert plan.intent == "recommendation"
    assert "真无线耳机" in plan.categories
    assert plan.budget.min == 300
    assert plan.budget.max == 500
    assert "降噪" in plan.preferences
    assert "price >= 300" in plan.filters
    assert "price <= 500" in plan.filters


def test_clarify_plan_sets_clarification_flag() -> None:
    intent_plan = IntentPlan(original_query="推荐点东西吧", plan_type="clarify")

    plan = RetrievalPlanBuilder().plan(intent_plan)

    assert plan.need_clarification is True
    assert plan.clarification_question


def test_comparison_plan_uses_original_query_signal() -> None:
    intent_plan = IntentPlan(
        original_query="对比笔记本电脑和电脑主机，预算8000",
        plan_type="single_retrieval",
        vector_query="笔记本电脑 电脑主机 预算8000",
        keyword_query="笔记本电脑 电脑主机 预算8000",
        budget_max=8000,
    )

    plan = RetrievalPlanBuilder().plan(intent_plan)

    assert plan.intent == "comparison"
    assert "笔记本电脑" in plan.categories
    assert "电脑主机" in plan.compare_targets
    assert "price <= 8000" in plan.filters


def test_multi_need_slots_contribute_to_category_understanding() -> None:
    intent_plan = IntentPlan(
        original_query="总预算1000，买运动鞋和运动服装",
        plan_type="multi_retrieval",
        budget_max=1000,
        budget_scope="total",
        need_slots=[
            RewriteNeedSlot(slot_id="s1", goal="运动鞋", product_type="运动鞋", query="运动鞋"),
            RewriteNeedSlot(slot_id="s2", goal="运动服装", product_type="运动服装", query="运动服装"),
        ],
    )

    plan = RetrievalPlanBuilder().plan(intent_plan)

    assert "服饰运动" in plan.categories
    assert plan.budget.max == 1000
    assert "price <= 1000" in plan.filters


def test_profile_refined_negative_terms_contribute_to_excludes() -> None:
    intent_plan = IntentPlan(
        original_query="按我的运动习惯推荐双鞋，日常跑步穿。",
        plan_type="single_retrieval",
        vector_query="日常慢跑缓震稳定跑鞋，避免竞速碳板",
        keyword_query="跑鞋 慢跑 缓震 稳定 避开竞速碳板",
    )

    plan = RetrievalPlanBuilder().plan(intent_plan)

    assert "竞速碳板" in plan.exclude
    assert "exclude not matched: 竞速碳板" in plan.filters
