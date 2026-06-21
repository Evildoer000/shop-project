from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVER_ROOT = PROJECT_ROOT / "server"
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from sqlalchemy import delete  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.db.models import Base, ConversationTurn, SessionMemoryState, UserMemory  # noqa: E402
from app.db.session import get_engine, get_sessionmaker  # noqa: E402
from app.domain.orchestrator import EcommerceOrchestrator  # noqa: E402
from app.schemas import ChatStreamRequest  # noqa: E402


DATASET_DIR = Path(__file__).resolve().parent / "datasets"
DEFAULT_JSON_REPORT = Path(__file__).resolve().parent / "report.json"
DEFAULT_MD_REPORT = Path(__file__).resolve().parent / "report.md"
CASE_DATASETS = [
    "retrieval_core",
    "personalized_coarse",
    "cross_scenario",
    "out_of_catalog",
    "route_boundary",
]
DIALOGUE_DATASETS = ["context_dialogues"]


@dataclass
class TurnEvalResult:
    dataset: str
    case_id: str
    turn_index: int
    query: str
    expected_route: str
    actual_route: str = ""
    route_ok: bool = False
    product_ids: list[str] = field(default_factory=list)
    hit_at_5: bool | None = None
    recall_at_5: float | None = None
    diverse_met_at_5: bool | None = None
    forbidden_clean_at_5: bool | None = None
    profile_used_ok: bool | None = None
    repair_triggered_ok: bool | None = None
    context_reuse_ok: bool | None = None
    final_result_ok: bool = False
    called_tools: list[str] = field(default_factory=list)
    internal_actions: list[str] = field(default_factory=list)
    repository_reads: list[str] = field(default_factory=list)
    cache_reads: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "case_id": self.case_id,
            "turn_index": self.turn_index,
            "query": self.query,
            "expected_route": self.expected_route,
            "actual_route": self.actual_route,
            "route_ok": self.route_ok,
            "product_ids": self.product_ids,
            "hit@5": self.hit_at_5,
            "recall@5": self.recall_at_5,
            "diverse_met@5": self.diverse_met_at_5,
            "forbidden_clean@5": self.forbidden_clean_at_5,
            "profile_used_ok": self.profile_used_ok,
            "repair_triggered_ok": self.repair_triggered_ok,
            "context_reuse_ok": self.context_reuse_ok,
            "final_result_ok": self.final_result_ok,
            "called_tools": self.called_tools,
            "internal_actions": self.internal_actions,
            "repository_reads": self.repository_reads,
            "cache_reads": self.cache_reads,
            "errors": self.errors,
            "duration_ms": self.duration_ms,
        }


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
                raise ValueError(f"{path}:{line_no} invalid JSONL: {exc}") from exc
    return rows


def ensure_memory_tables() -> None:
    Base.metadata.create_all(
        bind=get_engine(),
        tables=[ConversationTurn.__table__, SessionMemoryState.__table__, UserMemory.__table__],
    )


def reset_eval_user(db: Session, user_id: str, session_id: str) -> None:
    db.execute(delete(ConversationTurn).where(ConversationTurn.user_id == user_id, ConversationTurn.session_id == session_id))
    db.execute(delete(SessionMemoryState).where(SessionMemoryState.user_id == user_id, SessionMemoryState.session_id == session_id))
    db.execute(delete(UserMemory).where(UserMemory.user_id == user_id))
    db.commit()


def seed_profile_fixture(db: Session, user_id: str, fixture: dict[str, Any] | None) -> None:
    if not fixture:
        return
    rows = []
    for key, value in fixture.items():
        if value in (None, "", []):
            continue
        if isinstance(value, (list, tuple)):
            text = "，".join(str(item) for item in value if str(item).strip())
        elif isinstance(value, dict):
            text = json.dumps(value, ensure_ascii=False)
        else:
            text = str(value)
        rows.append(
            UserMemory(
                user_id=user_id,
                memory_type="preference",
                key=str(key),
                value=text[:256],
                confidence=Decimal("1.0"),
                source="explicit",
            )
        )
    if rows:
        db.add_all(rows)
        db.commit()


async def run_turn(db: Session, *, user_id: str, session_id: str, query: str, image_id: str | None = None) -> dict[str, Any]:
    orchestrator = EcommerceOrchestrator(db)
    request = ChatStreamRequest(user_id=user_id, session_id=session_id, message=query, image_id=image_id)
    events: list[dict[str, Any]] = []
    started = time.perf_counter()
    async for event in orchestrator.stream(request):
        events.append(event)
    # Memory writes are scheduled as background tasks by the Orchestrator.
    await asyncio.sleep(0.25)
    return {"events": events, "duration_ms": round((time.perf_counter() - started) * 1000, 2)}


def evaluate_turn(dataset: str, case_id: str, turn_index: int, spec: dict[str, Any], run: dict[str, Any]) -> TurnEvalResult:
    trace = latest_decision_trace(run["events"])
    products = latest_product_cards(run["events"])
    product_ids = [str(item.get("product_id") or "") for item in products if item.get("product_id")]
    called_tools = sorted(extract_tool_calls(run["events"], trace))
    internal_actions = sorted(extract_internal_actions(run["events"], trace))
    repository_reads = sorted(extract_repository_reads(trace))
    cache_reads = sorted(extract_cache_reads(run["events"], trace, spec))

    result = TurnEvalResult(
        dataset=dataset,
        case_id=case_id,
        turn_index=turn_index,
        query=str(spec.get("query") or ""),
        expected_route=str(spec.get("expected_route") or ""),
        actual_route=actual_route(trace),
        product_ids=product_ids,
        called_tools=called_tools,
        internal_actions=internal_actions,
        repository_reads=repository_reads,
        cache_reads=cache_reads,
        duration_ms=float(run.get("duration_ms") or 0.0),
    )
    result.route_ok = route_matches(result.expected_route, result.actual_route)
    result.hit_at_5 = hit_at_5(product_ids, spec)
    result.recall_at_5 = recall_at_5(product_ids, spec)
    result.diverse_met_at_5 = diverse_met_at_5(products, spec)
    result.forbidden_clean_at_5 = forbidden_clean_at_5(product_ids, spec)
    result.profile_used_ok = expected_items_ok("profile_lookup", spec.get("expected_tool_calls"), called_tools)
    result.repair_triggered_ok = expected_items_ok(
        "repair_plan_generated",
        spec.get("expected_internal_actions"),
        internal_actions,
    )
    result.context_reuse_ok = context_reuse_ok(spec, called_tools, repository_reads, trace)
    result.errors.extend(expectation_errors(spec, called_tools, internal_actions, repository_reads, cache_reads))
    result.final_result_ok = final_result_ok(result)
    return result


def latest_decision_trace(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in reversed(events):
        if event.get("type") == "decision_trace" and isinstance(event.get("trace"), dict):
            return event["trace"]
    return {}


def latest_product_cards(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for event in reversed(events):
        if event.get("type") == "product_cards" and isinstance(event.get("products"), list):
            return [item for item in event["products"] if isinstance(item, dict)]
    return []


def actual_route(trace: dict[str, Any]) -> str:
    task = trace.get("task") if isinstance(trace.get("task"), dict) else {}
    retrieval_summary = trace.get("retrieval_summary") if isinstance(trace.get("retrieval_summary"), dict) else {}
    return str(task.get("final_route") or trace.get("route") or retrieval_summary.get("route") or "")


def route_matches(expected: str, actual: str) -> bool:
    if not expected:
        return True
    return expected == actual


def hit_at_5(product_ids: list[str], spec: dict[str, Any]) -> bool | None:
    target = set(spec.get("relevant_product_ids") or []) | set(spec.get("acceptable_product_ids") or [])
    if not target:
        return None
    return bool(set(product_ids[:5]) & target)


def recall_at_5(product_ids: list[str], spec: dict[str, Any]) -> float | None:
    relevant = set(spec.get("relevant_product_ids") or [])
    if not relevant:
        return None
    return round(len(set(product_ids[:5]) & relevant) / len(relevant), 4)


def diverse_met_at_5(products: list[dict[str, Any]], spec: dict[str, Any]) -> bool | None:
    required = int(spec.get("min_diverse_subcategories") or 0)
    if required <= 0:
        return None
    sub_categories = {
        str(item.get("sub_category") or "").strip()
        for item in products[:5]
        if str(item.get("sub_category") or "").strip()
    }
    return len(sub_categories) >= required


def forbidden_clean_at_5(product_ids: list[str], spec: dict[str, Any]) -> bool | None:
    forbidden = set(spec.get("forbidden_product_ids") or [])
    if not forbidden:
        return None
    return not bool(set(product_ids[:5]) & forbidden)


def expected_items_ok(item: str, expected_items: Any, actual_items: list[str]) -> bool | None:
    expected = set(expected_items or [])
    if item not in expected:
        return None
    return item in set(actual_items)


def context_reuse_ok(
    spec: dict[str, Any],
    called_tools: list[str],
    repository_reads: list[str],
    trace: dict[str, Any],
) -> bool | None:
    expects_context = "product_search" in set(spec.get("forbidden_tool_calls") or []) and (
        "product_details_by_id" in set(spec.get("expected_repository_reads") or [])
        or "evidence_cache_recent" in set(spec.get("expected_cache_reads") or [])
    )
    if not expects_context:
        return None
    retrieval_summary = trace.get("retrieval_summary") if isinstance(trace.get("retrieval_summary"), dict) else {}
    loaded_ids = retrieval_summary.get("loaded_product_ids") or []
    return "product_search" not in called_tools and bool(loaded_ids) and "product_details_by_id" in repository_reads


def final_result_ok(result: TurnEvalResult) -> bool:
    checks: list[bool] = [result.route_ok, not result.errors]
    for value in [
        result.hit_at_5,
        result.diverse_met_at_5,
        result.forbidden_clean_at_5,
        result.profile_used_ok,
        result.repair_triggered_ok,
        result.context_reuse_ok,
    ]:
        if value is not None:
            checks.append(bool(value))
    return all(checks)


def extract_tool_calls(events: list[dict[str, Any]], trace: dict[str, Any]) -> set[str]:
    tools: set[str] = set()
    for event in events:
        stage = str(event.get("stage") or "")
        if stage == "profile_lookup":
            tools.add("profile_lookup")
        if stage in {"single_retrieval_worker_execution", "multi_need_retrieval", "repair_search_executed"}:
            tools.add("product_search")
        if stage == "image_retrieval_worker_execution":
            tools.add("image_search")

    for decision in trace.get("orchestrator_decisions") or []:
        if not isinstance(decision, dict):
            continue
        if decision.get("decision") == "profile_lookup" and decision.get("approved"):
            tools.add("profile_lookup")

    multi_trace = trace.get("multi_need_trace") if isinstance(trace.get("multi_need_trace"), dict) else {}
    for call in multi_trace.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        action = str(call.get("action") or "")
        if action == "search_products":
            tools.add("product_search")
        elif action == "image_search":
            tools.add("image_search")
        elif action == "profile_lookup":
            tools.add("profile_lookup")
    return tools


def extract_internal_actions(events: list[dict[str, Any]], trace: dict[str, Any]) -> set[str]:
    actions: set[str] = set()
    for event in events:
        stage = str(event.get("stage") or "")
        if stage in {"repair_plan_generated", "repair_search_executed", "evidence_merged"}:
            actions.add(stage)
    retrieval_summary = trace.get("retrieval_summary") if isinstance(trace.get("retrieval_summary"), dict) else {}
    if retrieval_summary.get("repair_plans"):
        actions.add("repair_plan_generated")
    if retrieval_summary.get("repair_retrievals"):
        actions.add("repair_search_executed")
    multi_trace = trace.get("multi_need_trace") if isinstance(trace.get("multi_need_trace"), dict) else {}
    for item in multi_trace.get("internal_actions") or []:
        if isinstance(item, dict) and item.get("action"):
            actions.add(str(item["action"]))
    return actions


def extract_repository_reads(trace: dict[str, Any]) -> set[str]:
    reads: set[str] = set()
    retrieval_summary = trace.get("retrieval_summary") if isinstance(trace.get("retrieval_summary"), dict) else {}
    if retrieval_summary.get("loaded_product_ids"):
        reads.add("product_details_by_id")
    return reads


def extract_cache_reads(events: list[dict[str, Any]], trace: dict[str, Any], spec: dict[str, Any]) -> set[str]:
    reads: set[str] = set()
    # 当前 trace 还没有显式 cache_reads 字段。上下文直答时先用可观测结果标记该读发生过：
    # no product_search + loaded_product_ids 表示 planner/context 找到了上一轮商品引用。
    if "evidence_cache_recent" in set(spec.get("expected_cache_reads") or []):
        if "product_search" not in extract_tool_calls(events, trace) and "product_details_by_id" in extract_repository_reads(trace):
            reads.add("evidence_cache_recent")
    return reads


def expectation_errors(
    spec: dict[str, Any],
    called_tools: list[str],
    internal_actions: list[str],
    repository_reads: list[str],
    cache_reads: list[str],
) -> list[str]:
    errors: list[str] = []
    actual_tools = set(called_tools)
    actual_actions = set(internal_actions)
    actual_repositories = set(repository_reads)
    actual_cache_reads = set(cache_reads)

    for tool in spec.get("expected_tool_calls") or []:
        if tool not in actual_tools:
            errors.append(f"missing expected tool call: {tool}")
    for tool in spec.get("forbidden_tool_calls") or []:
        if tool in actual_tools:
            errors.append(f"forbidden tool call observed: {tool}")
    for action in spec.get("expected_internal_actions") or []:
        if action not in actual_actions:
            errors.append(f"missing expected internal action: {action}")
    for action in spec.get("forbidden_internal_actions") or []:
        if action in actual_actions:
            errors.append(f"forbidden internal action observed: {action}")
    for read in spec.get("expected_repository_reads") or []:
        if read not in actual_repositories:
            errors.append(f"missing expected repository read: {read}")
    for read in spec.get("expected_cache_reads") or []:
        if read not in actual_cache_reads:
            errors.append(f"missing expected cache read: {read}")
    return errors


async def evaluate_single_case(dataset: str, case: dict[str, Any], limit_index: int) -> TurnEvalResult:
    user_id = f"benchmark_{dataset}_{case['id']}"
    session_id = f"{user_id}_session"
    SessionLocal = get_sessionmaker()
    with SessionLocal() as db:
        reset_eval_user(db, user_id, session_id)
        seed_profile_fixture(db, user_id, case.get("profile_fixture"))
        run = await run_turn(
            db,
            user_id=user_id,
            session_id=session_id,
            query=str(case.get("query") or ""),
            image_id=case.get("image_id"),
        )
        return evaluate_turn(dataset, str(case.get("id") or f"{dataset}_{limit_index}"), 1, case, run)


async def evaluate_dialogue_case(dataset: str, case: dict[str, Any], limit_index: int) -> list[TurnEvalResult]:
    case_id = str(case.get("id") or f"{dataset}_{limit_index}")
    user_id = f"benchmark_{dataset}_{case_id}"
    session_id = f"{user_id}_session"
    results: list[TurnEvalResult] = []
    SessionLocal = get_sessionmaker()
    with SessionLocal() as db:
        reset_eval_user(db, user_id, session_id)
        seed_profile_fixture(db, user_id, case.get("profile_fixture"))
        turns = case.get("turns") or []
        for turn_index, turn in enumerate(turns, start=1):
            run = await run_turn(
                db,
                user_id=user_id,
                session_id=session_id,
                query=str(turn.get("query") or ""),
                image_id=turn.get("image_id"),
            )
            # Dialogue setup turns create conversation state; only the final turn is
            # scored for the context-reuse benchmark.
            if turn_index == len(turns):
                results.append(evaluate_turn(dataset, case_id, turn_index, turn, run))
    return results


async def evaluate_datasets(selected: list[str], limit: int | None, fail_fast: bool) -> list[TurnEvalResult]:
    ensure_memory_tables()
    results: list[TurnEvalResult] = []
    for dataset in selected:
        path = DATASET_DIR / f"{dataset}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"dataset not found: {path}")
        rows = load_jsonl(path)
        if limit is not None:
            rows = rows[: max(0, limit)]
        for index, case in enumerate(rows, start=1):
            try:
                if "turns" in case:
                    results.extend(await evaluate_dialogue_case(dataset, case, index))
                else:
                    results.append(await evaluate_single_case(dataset, case, index))
            except Exception as exc:
                result = TurnEvalResult(
                    dataset=dataset,
                    case_id=str(case.get("id") or f"{dataset}_{index}"),
                    turn_index=1,
                    query=str(case.get("query") or ""),
                    expected_route=str(case.get("expected_route") or ""),
                    actual_route="__error__",
                    errors=[str(exc)],
                )
                results.append(result)
                if fail_fast:
                    raise
            print_progress(results[-1])
    return results


def print_progress(result: TurnEvalResult) -> None:
    status = "PASS" if result.final_result_ok else "FAIL"
    print(
        f"[{status}] {result.dataset}/{result.case_id}#t{result.turn_index} "
        f"route={result.actual_route or '-'} expected={result.expected_route or '-'} "
        f"products={','.join(result.product_ids[:5]) or '-'}"
    )


def aggregate(results: list[TurnEvalResult]) -> dict[str, Any]:
    by_dataset: dict[str, list[TurnEvalResult]] = {}
    for result in results:
        by_dataset.setdefault(result.dataset, []).append(result)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_turns": len(results),
        "overall": aggregate_group(results),
        "datasets": {dataset: aggregate_group(items) for dataset, items in sorted(by_dataset.items())},
        "failures": [result.to_dict() for result in results if not result.final_result_ok],
        "results": [result.to_dict() for result in results],
    }


def aggregate_group(items: list[TurnEvalResult]) -> dict[str, Any]:
    return {
        "turns": len(items),
        "pass_rate": ratio([item.final_result_ok for item in items]),
        "route_ok": ratio([item.route_ok for item in items]),
        "hit@5": ratio([item.hit_at_5 for item in items if item.hit_at_5 is not None]),
        "recall@5": mean([item.recall_at_5 for item in items if item.recall_at_5 is not None]),
        "diverse_met@5": ratio([item.diverse_met_at_5 for item in items if item.diverse_met_at_5 is not None]),
        "forbidden_clean@5": ratio([item.forbidden_clean_at_5 for item in items if item.forbidden_clean_at_5 is not None]),
        "profile_used_ok": ratio([item.profile_used_ok for item in items if item.profile_used_ok is not None]),
        "context_reuse_ok": ratio([item.context_reuse_ok for item in items if item.context_reuse_ok is not None]),
    }


def ratio(values: list[bool]) -> float | None:
    if not values:
        return None
    return round(sum(1 for value in values if value) / len(values), 4)


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def write_reports(summary: dict[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(summary), encoding="utf-8")


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Benchmark Report",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Total turns: `{summary['total_turns']}`",
        "",
        "## Overall",
        "",
        metric_table({"overall": summary["overall"]}),
        "",
        "## By Dataset",
        "",
        metric_table(summary["datasets"]),
    ]
    failures = summary.get("failures") or []
    lines.extend(["", "## Failures", ""])
    if not failures:
        lines.append("No failures.")
    else:
        for item in failures[:30]:
            errors = "; ".join(item.get("errors") or [])
            lines.append(
                f"- `{item['dataset']}/{item['case_id']}#t{item['turn_index']}` "
                f"expected `{item.get('expected_route') or '-'}`, got `{item.get('actual_route') or '-'}`, "
                f"products `{', '.join(item.get('product_ids') or []) or '-'}`"
                + (f", errors: {errors}" if errors else "")
            )
        if len(failures) > 30:
            lines.append(f"- ... {len(failures) - 30} more failures in JSON report.")
    lines.append("")
    return "\n".join(lines)


def metric_table(groups: dict[str, dict[str, Any]]) -> str:
    headers = [
        "dataset",
        "turns",
        "pass_rate",
        "route_ok",
        "hit@5",
        "recall@5",
        "diverse_met@5",
        "forbidden_clean@5",
        "profile_used_ok",
        "context_reuse_ok",
    ]
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for name, metrics in groups.items():
        rows.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(metrics.get("turns", "")),
                    fmt(metrics.get("pass_rate")),
                    fmt(metrics.get("route_ok")),
                    fmt(metrics.get("hit@5")),
                    fmt(metrics.get("recall@5")),
                    fmt(metrics.get("diverse_met@5")),
                    fmt(metrics.get("forbidden_clean@5")),
                    fmt(metrics.get("profile_used_ok")),
                    fmt(metrics.get("context_reuse_ok")),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all new Harness benchmark datasets.")
    parser.add_argument(
        "--datasets",
        default=",".join(CASE_DATASETS + DIALOGUE_DATASETS),
        help="Comma-separated dataset names without .jsonl. Default: main benchmark datasets.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit cases per dataset for smoke runs.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD_REPORT)
    parser.add_argument("--no-write-report", action="store_true", help="Only print progress; do not write report files.")
    parser.add_argument("--fail-fast", action="store_true", help="Raise on the first runtime exception.")
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    selected = [item.strip() for item in str(args.datasets).split(",") if item.strip()]
    results = await evaluate_datasets(selected, args.limit, args.fail_fast)
    summary = aggregate(results)
    if not args.no_write_report:
        write_reports(summary, args.output_json, args.output_md)
        print(f"Wrote {args.output_json}")
        print(f"Wrote {args.output_md}")
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
