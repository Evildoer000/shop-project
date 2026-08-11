from __future__ import annotations

import json
import math
from collections.abc import Iterable
from typing import Any

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


def _payload(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("DeepEval actual_output/expected_output 必须是JSON对象")
    return parsed


def _unique_strings(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


class DeterministicMetric(BaseMetric):
    """Base class for local metrics that never call an LLM judge."""

    async_mode = False
    verbose_mode = False

    def __init__(self, *, threshold: float = 1.0) -> None:
        self.threshold = threshold
        self.score = None
        self.score_breakdown = {}
        self.reason = None
        self.success = None
        self.error = None

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        if self.error is not None:
            return False
        return bool(self.score is not None and self.score >= self.threshold)


class PlanContractMetric(DeterministicMetric):
    @property
    def __name__(self) -> str:
        return "计划契约检查"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        actual = _payload(test_case.actual_output)
        expected = _payload(test_case.expected_output)
        actual_plan = actual.get("plan") if isinstance(actual.get("plan"), dict) else {}
        expected_plan = expected.get("plan") if isinstance(expected.get("plan"), dict) else {}
        expected_route = expected.get("route") if isinstance(expected.get("route"), dict) else {}

        checks: dict[str, bool] = {}
        allowed_plan_types = set(expected_plan.get("allowed_plan_types") or [])
        if allowed_plan_types:
            checks["plan_type"] = str(actual_plan.get("plan_type") or "") in allowed_plan_types

        allowed_routes = set(expected_route.get("allowed") or [])
        if allowed_routes:
            checks["route"] = str(actual.get("route") or "") in allowed_routes

        slot_contract = expected_plan.get("slot_count") if isinstance(expected_plan.get("slot_count"), dict) else {}
        if slot_contract:
            actual_slot_count = int(actual_plan.get("slot_count") or 0)
            checks["slot_count"] = int(slot_contract.get("min") or 0) <= actual_slot_count <= int(
                slot_contract.get("max") or 0
            )

        budget_contract = expected_plan.get("budget") if isinstance(expected_plan.get("budget"), dict) else {}
        expected_budget_max = budget_contract.get("max")
        if expected_budget_max is not None:
            actual_budget_max = actual_plan.get("budget_max")
            tolerance = float(budget_contract.get("numeric_tolerance") or 0)
            checks["budget_max"] = actual_budget_max is not None and abs(
                float(actual_budget_max) - float(expected_budget_max)
            ) <= tolerance
        expected_scope = str(budget_contract.get("scope") or "unknown")
        if expected_scope != "unknown":
            checks["budget_scope"] = str(actual_plan.get("budget_scope") or "unknown") == expected_scope

        self.score = 1.0 if not checks else sum(checks.values()) / len(checks)
        self.score_breakdown = checks
        failed = [name for name, passed in checks.items() if not passed]
        self.reason = "计划契约全部满足。" if not failed else f"未满足：{'、'.join(failed)}。"
        self.success = self.is_successful()
        return self.score


class ProductRankingMetric(DeterministicMetric):
    def __init__(self, stage: str, *, top_k: int = 10, threshold: float = 0.65) -> None:
        super().__init__(threshold=threshold)
        self.stage = stage
        self.top_k = top_k

    @property
    def __name__(self) -> str:
        return f"{self.stage} 商品召回与排序"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        actual = _payload(test_case.actual_output)
        expected = _payload(test_case.expected_output)
        stages = actual.get("retrieval_stages") if isinstance(actual.get("retrieval_stages"), dict) else {}
        ranked_ids = _unique_strings(stages.get(self.stage) or [])[: self.top_k]
        retrieval = expected.get("retrieval") if isinstance(expected.get("retrieval"), dict) else {}
        judgments = retrieval.get("judged_products") or []
        approved = {
            str(item.get("product_id") or ""): {
                "grade": int(item["approved_grade"]),
                "type_match": bool(item.get("type_match")),
                "hard_pass": bool(item.get("hard_pass")),
            }
            for item in judgments
            if item.get("approved_grade") is not None
        }
        if not approved:
            self.error = "没有人工 approved_grade，禁止计算正式检索指标。"
            self.score = 0.0
            self.reason = self.error
            self.success = False
            return self.score

        relevant_ids = {product_id for product_id, item in approved.items() if item["grade"] >= 2}
        strong_ids = {product_id for product_id, item in approved.items() if item["grade"] >= 3}
        wrong_type_count = sum(
            1 for product_id in ranked_ids if product_id in approved and not approved[product_id]["type_match"]
        )
        hard_violation_count = sum(
            1 for product_id in ranked_ids if product_id in approved and not approved[product_id]["hard_pass"]
        )
        wrong_type_rate = wrong_type_count / len(ranked_ids) if ranked_ids else 0.0
        hard_violation_rate = hard_violation_count / len(ranked_ids) if ranked_ids else 0.0
        acceptable_recall = (
            len(set(ranked_ids) & relevant_ids) / len(relevant_ids)
            if relevant_ids
            else 1.0
        )
        strong_recall = (
            len(set(ranked_ids) & strong_ids) / len(strong_ids)
            if strong_ids
            else 1.0
        )
        ndcg = _ndcg(ranked_ids, approved, self.top_k)

        gates = retrieval.get("metric_gates") if isinstance(retrieval.get("metric_gates"), dict) else {}
        checks = {
            "wrong_type_rate": wrong_type_rate <= float(gates.get("wrong_type_rate@10_max", 1.0)),
            "hard_violation_rate": hard_violation_rate <= float(gates.get("hard_violation_rate@10_max", 1.0)),
            "acceptable_recall": acceptable_recall >= float(gates.get("acceptable_recall@20_min", 0.0)),
            "strong_recall": strong_recall >= float(gates.get("strong_recall@20_min", 0.0)),
            "ndcg": ndcg >= float(gates.get("ndcg@10_min", 0.0)),
        }
        self.score = round(
            0.15 * (1 - wrong_type_rate)
            + 0.15 * (1 - hard_violation_rate)
            + 0.25 * acceptable_recall
            + 0.15 * strong_recall
            + 0.30 * ndcg,
            6,
        )
        self.score_breakdown = {
            "ranked_ids": ranked_ids,
            "wrong_type_rate": round(wrong_type_rate, 6),
            "hard_violation_rate": round(hard_violation_rate, 6),
            "acceptable_recall": round(acceptable_recall, 6),
            "strong_recall": round(strong_recall, 6),
            "ndcg": round(ndcg, 6),
            "gate_checks": checks,
        }
        failed = [name for name, passed in checks.items() if not passed]
        self.reason = (
            f"{self.stage} 的分级召回与排序满足门槛。"
            if not failed
            else f"{self.stage} 未满足：{'、'.join(failed)}。"
        )
        self.success = not failed and self.is_successful()
        return self.score

    def is_successful(self) -> bool:
        checks = self.score_breakdown.get("gate_checks") if self.score_breakdown else None
        return bool(
            self.error is None
            and self.score is not None
            and self.score >= self.threshold
            and (not checks or all(checks.values()))
        )


class CorrectiveDecisionMetric(DeterministicMetric):
    @property
    def __name__(self) -> str:
        return "CorrectiveAgent 放行与误拒"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        actual = _payload(test_case.actual_output)
        expected = _payload(test_case.expected_output)
        corrective = actual.get("corrective") if isinstance(actual.get("corrective"), dict) else {}
        passed_ids = set(_unique_strings(corrective.get("passed_product_ids") or []))
        rejected_ids = set(_unique_strings(corrective.get("rejected_product_ids") or []))
        judgments = expected.get("retrieval", {}).get("judged_products") or []
        relevant = {
            str(item.get("product_id") or "")
            for item in judgments
            if item.get("approved_grade") is not None and int(item["approved_grade"]) >= 2
        }
        invalid = {
            str(item.get("product_id") or "")
            for item in judgments
            if item.get("approved_grade") is not None and int(item["approved_grade"]) < 2
        }
        if not relevant and not invalid:
            self.error = "没有人工审核商品等级。"
            self.score = 0.0
            self.reason = self.error
            self.success = False
            return self.score

        pass_precision = len(passed_ids & relevant) / len(passed_ids) if passed_ids else 0.0
        pass_recall = len(passed_ids & relevant) / len(relevant) if relevant else 1.0
        false_reject_rate = len(rejected_ids & relevant) / len(relevant) if relevant else 0.0
        invalid_leak_rate = len(passed_ids & invalid) / len(invalid) if invalid else 0.0
        self.score = round(
            0.35 * pass_precision
            + 0.35 * pass_recall
            + 0.15 * (1 - false_reject_rate)
            + 0.15 * (1 - invalid_leak_rate),
            6,
        )
        self.score_breakdown = {
            "pass_precision": round(pass_precision, 6),
            "pass_recall": round(pass_recall, 6),
            "false_reject_rate": round(false_reject_rate, 6),
            "invalid_leak_rate": round(invalid_leak_rate, 6),
        }
        self.reason = (
            "CorrectiveAgent 正确放行相关商品且未放行无效商品。"
            if self.score >= self.threshold
            else "CorrectiveAgent 存在漏放、误拒或相关商品覆盖不足。"
        )
        self.success = self.is_successful()
        return self.score


class EvidenceSetMetric(DeterministicMetric):
    @property
    def __name__(self) -> str:
        return "最终商品证据一致性"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        actual = _payload(test_case.actual_output)
        final_ids = set(_unique_strings(actual.get("final_product_ids") or []))
        evidence_ids = set(_unique_strings(actual.get("evidence_product_ids") or []))
        if not final_ids:
            self.score = 1.0
            self.reason = "本轮没有输出商品，无越界商品。"
        elif not evidence_ids:
            self.score = 0.0
            self.reason = "输出了商品，但没有可对照的证据集合。"
        else:
            self.score = len(final_ids & evidence_ids) / len(final_ids)
            missing = sorted(final_ids - evidence_ids)
            self.reason = (
                "最终商品全部来自证据集合。"
                if not missing
                else f"以下商品不在证据集合：{', '.join(missing)}。"
            )
        self.score_breakdown = {
            "final_product_ids": sorted(final_ids),
            "evidence_product_ids": sorted(evidence_ids),
        }
        self.success = self.is_successful()
        return self.score


def _ndcg(ranked_ids: list[str], judgments: dict[str, dict[str, Any]], top_k: int) -> float:
    # Grade 0 (wrong type) and grade 1 (hard-constraint violation) are both
    # non-relevant for ranking. Acceptable relevance starts at grade 2.
    actual_gains = [
        max(int(judgments.get(product_id, {}).get("grade", 0)) - 1, 0)
        for product_id in ranked_ids[:top_k]
    ]
    ideal_gains = sorted(
        (max(int(item["grade"]) - 1, 0) for item in judgments.values()),
        reverse=True,
    )[:top_k]
    actual_dcg = _dcg(actual_gains)
    ideal_dcg = _dcg(ideal_gains)
    return actual_dcg / ideal_dcg if ideal_dcg else 1.0


def _dcg(gains: list[int]) -> float:
    return sum((2**gain - 1) / math.log2(index + 2) for index, gain in enumerate(gains))


def build_deepeval_case(
    case: dict[str, Any],
    actual_payload: dict[str, Any],
) -> LLMTestCase:
    return LLMTestCase(
        name=str(case.get("id") or ""),
        input=str(case.get("input", {}).get("message") or ""),
        actual_output=json.dumps(actual_payload, ensure_ascii=False),
        expected_output=json.dumps(case.get("expected") or {}, ensure_ascii=False),
        metadata={
            "case_id": case.get("id"),
            "review_status": case.get("review", {}).get("status"),
            "scenario": case.get("coverage", {}).get("scenario"),
        },
        tags=list(case.get("coverage", {}).get("tags") or []),
    )
