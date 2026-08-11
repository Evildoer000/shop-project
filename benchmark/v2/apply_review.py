from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "datasets" / "single_turn.jsonl"
DEFAULT_REVIEW_QUEUE = ROOT / "review_queue.csv"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} JSON格式错误: {exc}") from exc
    return rows


def load_review_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    approved: dict[tuple[str, str], dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for line_no, row in enumerate(reader, start=2):
            case_id = str(row.get("case_id") or "").strip()
            product_id = str(row.get("product_id") or "").strip()
            grade_text = str(row.get("approved_grade") or "").strip()
            decision = str(row.get("reviewer_decision") or "").strip().lower()
            if not case_id or not product_id:
                raise ValueError(f"{path}:{line_no} 缺少case_id或product_id")
            if decision in {"reject", "rejected", "删除", "驳回"}:
                approved[(case_id, product_id)] = {
                    "decision": "reject",
                    "grade": "",
                    "notes": str(row.get("reviewer_notes") or "").strip(),
                }
                continue
            if not grade_text:
                continue
            try:
                grade = int(grade_text)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no} approved_grade必须是0到4的整数") from exc
            if grade not in {0, 1, 2, 3, 4}:
                raise ValueError(f"{path}:{line_no} approved_grade超出0到4范围")
            approved[(case_id, product_id)] = {
                "decision": "approve",
                "grade": str(grade),
                "notes": str(row.get("reviewer_notes") or "").strip(),
            }
    return approved


def apply_reviews(
    cases: list[dict[str, Any]],
    reviews: dict[tuple[str, str], dict[str, str]],
    *,
    reviewer: str,
    approve_complete_cases: bool,
) -> dict[str, int]:
    counts = defaultdict(int)
    now = datetime.now().isoformat(timespec="seconds")
    for case in cases:
        case_id = str(case.get("id") or "")
        judged_products = case.get("expected", {}).get("retrieval", {}).get("judged_products") or []
        kept_products: list[dict[str, Any]] = []
        for product in judged_products:
            key = (case_id, str(product.get("product_id") or ""))
            review = reviews.get(key)
            if not review:
                kept_products.append(product)
                counts["unchanged_products"] += 1
                continue
            if review["decision"] == "reject":
                counts["rejected_products"] += 1
                continue
            product["approved_grade"] = int(review["grade"])
            product["review_status"] = "approved"
            product["reviewer_notes"] = review["notes"]
            kept_products.append(product)
            counts["approved_products"] += 1
        case["expected"]["retrieval"]["judged_products"] = kept_products

        if not approve_complete_cases:
            continue
        if kept_products and all(item.get("review_status") == "approved" for item in kept_products):
            case["review"]["status"] = "approved"
            case["review"]["reviewer"] = reviewer
            case["review"]["reviewed_at"] = now
            case["expected"]["retrieval"]["metric_gates"]["enabled_after_human_review"] = False
            counts["approved_cases"] += 1
    return dict(counts)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把人工审核CSV中的等级回填到V2离线测试集。")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--review-queue", type=Path, default=DEFAULT_REVIEW_QUEUE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--reviewer", default="manual_reviewer")
    parser.add_argument(
        "--approve-complete-cases",
        action="store_true",
        help="某case的全部保留商品都已审核时，将case状态改为approved并启用正式指标。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.dataset
    cases = load_jsonl(args.dataset)
    reviews = load_review_rows(args.review_queue)
    counts = apply_reviews(
        cases,
        reviews,
        reviewer=str(args.reviewer),
        approve_complete_cases=bool(args.approve_complete_cases),
    )
    write_jsonl(output, cases)
    print(json.dumps({"output": str(output), **counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
