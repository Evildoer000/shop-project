from __future__ import annotations

from deepeval import assert_test

from benchmark.v2.eval_metrics import (
    CorrectiveDecisionMetric,
    EvidenceSetMetric,
    PlanContractMetric,
    ProductRankingMetric,
    build_deepeval_case,
)


def approved_fixture() -> dict:
    return {
        "id": "framework_smoke_sunscreen",
        "review": {"status": "approved"},
        "coverage": {"scenario": "framework_smoke", "tags": ["smoke"]},
        "input": {"message": "油皮预算150以内，推荐清爽通勤防晒。"},
        "expected": {
            "route": {"allowed": ["recommend"]},
            "plan": {
                "allowed_plan_types": ["single_retrieval"],
                "slot_count": {"min": 1, "max": 1},
                "budget": {"max": 150, "scope": "per_item", "numeric_tolerance": 0},
            },
            "retrieval": {
                "judged_products": [
                    {
                        "product_id": "p_good_4",
                        "approved_grade": 4,
                        "type_match": True,
                        "hard_pass": True,
                    },
                    {
                        "product_id": "p_good_3",
                        "approved_grade": 3,
                        "type_match": True,
                        "hard_pass": True,
                    },
                    {
                        "product_id": "p_unknown_2",
                        "approved_grade": 2,
                        "type_match": True,
                        "hard_pass": True,
                    },
                    {
                        "product_id": "p_over_budget_1",
                        "approved_grade": 1,
                        "type_match": True,
                        "hard_pass": False,
                    },
                    {
                        "product_id": "p_wrong_0",
                        "approved_grade": 0,
                        "type_match": False,
                        "hard_pass": True,
                    },
                ],
                "metric_gates": {
                    "wrong_type_rate@10_max": 0.1,
                    "hard_violation_rate@10_max": 0.2,
                    "acceptable_recall@20_min": 0.6,
                    "strong_recall@20_min": 0.5,
                    "ndcg@10_min": 0.65,
                },
            },
        },
    }


def successful_actual_payload() -> dict:
    return {
        "route": "recommend",
        "plan": {
            "plan_type": "single_retrieval",
            "slot_count": 1,
            "budget_max": 150,
            "budget_scope": "per_item",
        },
        "retrieval_stages": {
            "es": ["p_good_4", "p_good_3", "p_unknown_2"],
            "milvus": ["p_good_3", "p_good_4", "p_unknown_2"],
            "rrf": ["p_good_4", "p_good_3", "p_unknown_2"],
            "reranker": ["p_good_4", "p_good_3", "p_unknown_2"],
        },
        "corrective": {
            "passed_product_ids": ["p_good_4", "p_good_3", "p_unknown_2"],
            "rejected_product_ids": ["p_over_budget_1", "p_wrong_0"],
        },
        "evidence_product_ids": ["p_good_4", "p_good_3", "p_unknown_2"],
        "final_product_ids": ["p_good_4", "p_good_3"],
    }


def test_deepeval_runs_project_deterministic_metrics() -> None:
    test_case = build_deepeval_case(approved_fixture(), successful_actual_payload())
    metrics = [
        PlanContractMetric(),
        ProductRankingMetric("es"),
        ProductRankingMetric("milvus"),
        ProductRankingMetric("rrf"),
        ProductRankingMetric("reranker"),
        CorrectiveDecisionMetric(threshold=0.8),
        EvidenceSetMetric(),
    ]

    assert_test(test_case, metrics, run_async=False)

    assert all(metric.is_successful() for metric in metrics)
    assert metrics[1].score_breakdown["ndcg"] == 1.0
    assert metrics[-1].score == 1.0
