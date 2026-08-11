from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets"


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


def validate_case(case: dict[str, Any], path: Path, line_no: int) -> list[str]:
    errors: list[str] = []
    prefix = f"{path.name}:{line_no}:{case.get('id', '<missing-id>')}"
    required_top = ["schema_version", "id", "case_kind", "enabled", "review", "coverage", "annotation"]
    for key in required_top:
        if key not in case:
            errors.append(f"{prefix} 缺少顶层字段 {key}")
    if case.get("schema_version") != "2.0":
        errors.append(f"{prefix} schema_version必须为2.0")
    if case.get("review", {}).get("status") not in {"needs_review", "approved", "rejected"}:
        errors.append(f"{prefix} review.status非法")

    case_kind = case.get("case_kind")
    if case_kind == "multi_turn":
        turns = case.get("turns")
        if not isinstance(turns, list) or len(turns) < 2:
            errors.append(f"{prefix} 多轮case至少需要2个turn")
        for index, turn in enumerate(turns or [], start=1):
            if not str(turn.get("input", {}).get("message") or "").strip():
                errors.append(f"{prefix} 第{index}轮缺少message")
            if not turn.get("expected", {}).get("plan", {}).get("allowed_plan_types"):
                errors.append(f"{prefix} 第{index}轮缺少allowed_plan_types")
    else:
        if "input" not in case or "expected" not in case:
            errors.append(f"{prefix} 单轮/异常case缺少input或expected")

    retrieval = case.get("expected", {}).get("retrieval")
    if isinstance(retrieval, dict) and retrieval.get("required_subcategories"):
        constraints = retrieval.get("soft_constraints") or []
        weight_sum = sum(float(item.get("weight") or 0) for item in constraints)
        if constraints and abs(weight_sum - 1.0) > 0.001:
            errors.append(f"{prefix} soft_constraints权重之和应为1，当前为{weight_sum:g}")
        judged = retrieval.get("judged_products") or []
        if len(judged) < 3:
            errors.append(f"{prefix} judged_products少于3条，难以做严格排序评估")
        product_ids = [str(item.get("product_id") or "") for item in judged]
        if len(product_ids) != len(set(product_ids)):
            errors.append(f"{prefix} judged_products存在重复product_id")
        for item in judged:
            proposed = item.get("proposed_grade")
            approved = item.get("approved_grade")
            if proposed not in {0, 1, 2, 3, 4}:
                errors.append(f"{prefix} {item.get('product_id')} proposed_grade非法")
            if approved is not None and approved not in {0, 1, 2, 3, 4}:
                errors.append(f"{prefix} {item.get('product_id')} approved_grade非法")
            if case.get("review", {}).get("status") == "approved" and approved is None:
                errors.append(f"{prefix} case已approved但{item.get('product_id')}未填写approved_grade")
    return errors


def validate_coverage(all_cases: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    kinds = Counter(str(item.get("case_kind") or "") for item in all_cases)
    domains = Counter(str(item.get("coverage", {}).get("domain") or "") for item in all_cases)
    tags = Counter(
        str(tag)
        for item in all_cases
        for tag in item.get("coverage", {}).get("tags") or []
    )
    thresholds = {
        ("case_kind", "single_turn"): 50,
        ("case_kind", "multi_turn"): 15,
        ("case_kind", "adversarial"): 20,
        ("domain", "beauty_skincare"): 10,
        ("domain", "digital_electronics"): 10,
        ("domain", "clothing_sports"): 10,
        ("domain", "food_lifestyle"): 10,
        ("tag", "multi_need"): 5,
        ("tag", "security"): 5,
        ("tag", "fault"): 8,
    }
    for (group, key), minimum in thresholds.items():
        source = {"case_kind": kinds, "domain": domains, "tag": tags}[group]
        if source[key] < minimum:
            errors.append(f"覆盖不足: {group}={key} 需要至少{minimum}条，当前{source[key]}条")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="校验V2离线测试集格式、审核状态和覆盖率。")
    parser.add_argument("--datasets", type=Path, default=DATASET_DIR)
    parser.add_argument("--require-approved", action="store_true", help="要求所有case均已人工审核通过。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.datasets.glob("*.jsonl"))
    if not paths:
        raise SystemExit(f"未找到JSONL数据集: {args.datasets}")
    all_cases: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in paths:
        rows = load_jsonl(path)
        all_cases.extend(rows)
        for line_no, case in enumerate(rows, start=1):
            errors.extend(validate_case(case, path, line_no))
            if args.require_approved and case.get("review", {}).get("status") != "approved":
                errors.append(f"{path.name}:{line_no}:{case.get('id')} 尚未人工审核通过")

    ids = [str(case.get("id") or "") for case in all_cases]
    duplicates = [case_id for case_id, count in Counter(ids).items() if count > 1]
    if duplicates:
        errors.append(f"存在重复case id: {', '.join(sorted(duplicates))}")
    errors.extend(validate_coverage(all_cases))

    summary = {
        "dataset_files": [path.name for path in paths],
        "total_cases": len(all_cases),
        "case_kinds": dict(Counter(str(item.get("case_kind") or "") for item in all_cases)),
        "review_status": dict(Counter(str(item.get("review", {}).get("status") or "") for item in all_cases)),
        "errors": errors,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
