from __future__ import annotations

from pathlib import Path

from benchmark.v2.validate_dataset import load_jsonl, validate_case, validate_coverage


DATASET_DIR = Path("benchmark/v2/datasets")


def test_v2_dataset_contract_and_coverage() -> None:
    all_cases = []
    errors = []
    for path in sorted(DATASET_DIR.glob("*.jsonl")):
        rows = load_jsonl(path)
        all_cases.extend(rows)
        for line_no, case in enumerate(rows, start=1):
            errors.extend(validate_case(case, path, line_no))
    errors.extend(validate_coverage(all_cases))
    assert errors == []


def test_machine_proposals_are_not_marked_as_ground_truth() -> None:
    cases = load_jsonl(DATASET_DIR / "single_turn.jsonl")
    judged_products = [
        product
        for case in cases
        for product in case.get("expected", {}).get("retrieval", {}).get("judged_products") or []
    ]
    assert judged_products
    assert all(product.get("approved_grade") is None for product in judged_products)
    assert all(case.get("review", {}).get("status") == "needs_review" for case in cases)
