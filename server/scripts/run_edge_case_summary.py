from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import time
import uuid
from typing import Any, Iterator

import httpx


EDGE_CASE_MESSAGES = [
    "三顿半",
    "查一下三顿半有什么产品",
    "去三亚旅行，要防晒和轻便鞋，预算300以内",
    "露营装备要背包、徒步鞋和方便食品，顺便看看帽子",
    "新手化妆套装帮我配齐，敏感肌，不要酒精味重",
    "帮我找防晒衣，不要防晒霜",
    "想买手机支架，预算100以内",
    "推荐一个咖啡，不要三顿半",
    "数码",
    "咖啡",
    "三顿半的耳机",
    "苹果",
    "胡言乱语",
    "买个东西",
    "不要咖啡",
    "有三顿半的手机吗",
    "我要三顿半，预算100以内",
    "咖啡不要咖啡",
]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Run compact edge-case probes against the chat SSE endpoint.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="FastAPI base URL")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N messages")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of text")
    parser.add_argument("--direct", action="store_true", help="Call EcommerceOrchestrator directly instead of HTTP SSE")
    args = parser.parse_args()

    messages = EDGE_CASE_MESSAGES[: args.limit] if args.limit else EDGE_CASE_MESSAGES
    if args.direct:
        summaries = asyncio.run(run_direct(messages))
        if args.json:
            print(json.dumps(summaries, ensure_ascii=False, indent=2))
        else:
            for summary in summaries:
                print_summary(summary)
                print()
        return 0 if all(not item.get("error") for item in summaries) else 1

    run_id = uuid.uuid4().hex[:8]
    url = f"{args.base_url.rstrip('/')}/api/chat/stream"

    summaries: list[dict[str, Any]] = []
    started = time.monotonic()
    with httpx.Client(timeout=None) as client:
        for index, message in enumerate(messages, start=1):
            summary = run_turn(
                client=client,
                url=url,
                user_id=f"edge_case_{run_id}_{index}",
                session_id=f"edge_case_session_{run_id}_{index}",
                message=message,
            )
            summary["index"] = index
            summary["message"] = message
            summaries.append(summary)
            if not args.json:
                print_summary(summary)
                print()

    if args.json:
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
    else:
        print(f"Completed {len(summaries)} probes in {time.monotonic() - started:.1f}s")

    return 0 if all(not item.get("error") for item in summaries) else 1


async def run_direct(messages: list[str]) -> list[dict[str, Any]]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from app.db.session import get_sessionmaker
    from app.domain.orchestrator import EcommerceOrchestrator
    from app.schemas import ChatStreamRequest

    summaries: list[dict[str, Any]] = []
    for index, message in enumerate(messages, start=1):
        db = get_sessionmaker()()
        answer_parts: list[str] = []
        trace_events: dict[str, Any] = {}
        decision_trace: dict[str, Any] = {}
        product_cards: list[dict[str, Any]] = []
        try:
            orchestrator = EcommerceOrchestrator(db)
            async for event in orchestrator.stream(
                ChatStreamRequest(
                    user_id=f"edge_case_direct_{index}",
                    session_id=f"edge_case_direct_{uuid.uuid4().hex[:8]}_{index}",
                    message=message,
                    image_id=None,
                )
            ):
                event_type = event.get("type")
                if event_type == "trace":
                    stage = event.get("stage")
                    if stage:
                        trace_events[str(stage)] = event
                elif event_type == "decision_trace":
                    decision_trace = event.get("trace") or {}
                elif event_type == "token":
                    answer_parts.append(str(event.get("content", "")))
                elif event_type == "product_cards":
                    product_cards = list(event.get("products") or [])
        except Exception as exc:  # noqa: BLE001 - probe output should report any failure plainly.
            summary = {"error": repr(exc)}
        else:
            summary = build_summary(
                answer="".join(answer_parts).strip(),
                trace_events=trace_events,
                decision_trace=decision_trace,
                product_cards=product_cards,
            )
        finally:
            db.close()
        summary["index"] = index
        summary["message"] = message
        summaries.append(summary)
    return summaries


def run_turn(
    client: httpx.Client,
    url: str,
    user_id: str,
    session_id: str,
    message: str,
) -> dict[str, Any]:
    payload = {
        "user_id": user_id,
        "session_id": session_id,
        "message": message,
        "image_id": None,
    }
    answer_parts: list[str] = []
    trace_events: dict[str, Any] = {}
    decision_trace: dict[str, Any] = {}
    product_cards: list[dict[str, Any]] = []

    try:
        with client.stream("POST", url, headers={"Accept": "text/event-stream"}, json=payload) as response:
            if response.status_code >= 400:
                return {"error": f"HTTP {response.status_code}: {response.text}"}

            for event in iter_sse_events(response):
                event_type = event.get("type")
                if event_type == "trace":
                    stage = event.get("stage")
                    if stage:
                        trace_events[str(stage)] = event
                elif event_type == "decision_trace":
                    decision_trace = event.get("trace") or {}
                elif event_type == "token":
                    answer_parts.append(str(event.get("content", "")))
                elif event_type == "product_cards":
                    product_cards = list(event.get("products") or [])
                elif event_type == "error":
                    return {"error": event.get("message") or "server error"}
                elif event_type == "done":
                    break
    except Exception as exc:  # noqa: BLE001 - probe output should report any failure plainly.
        return {"error": repr(exc)}

    return build_summary(
        answer="".join(answer_parts).strip(),
        trace_events=trace_events,
        decision_trace=decision_trace,
        product_cards=product_cards,
    )


def build_summary(
    answer: str,
    trace_events: dict[str, Any],
    decision_trace: dict[str, Any],
    product_cards: list[dict[str, Any]],
) -> dict[str, Any]:
    retrieval_summary = decision_trace.get("retrieval_summary") or {}
    counts = decision_trace.get("candidate_counts") or {}
    query_understanding = decision_trace.get("query_understanding") or {}
    multi_need_trace = decision_trace.get("multi_need_trace") or {}
    selection = multi_need_trace.get("selection") if isinstance(multi_need_trace, dict) else {}
    selection = selection if isinstance(selection, dict) else {}

    detect_event = trace_events.get("need_slot_detect") or {}
    detect_payload = detect_event.get("multi_need") if isinstance(detect_event, dict) else {}
    detect_payload = detect_payload if isinstance(detect_payload, dict) else {}

    route = (
        decision_trace.get("route")
        or retrieval_summary.get("corrective_route")
        or retrieval_summary.get("route")
        or "-"
    )
    slot_count = retrieval_summary.get("slot_count") or detect_payload.get("slot_count") or 0
    is_multi_need = bool(retrieval_summary.get("multi_need") or multi_need_trace or detect_payload.get("is_multi_need"))

    products = [
        {
            "id": product.get("product_id"),
            "name": product.get("name"),
            "brand": product.get("brand"),
            "price": product.get("price"),
        }
        for product in product_cards
    ]
    selected_by_slot = compact_selected_by_slot(selection.get("selected_by_slot") or {})
    coverage_by_slot = compact_coverage(multi_need_trace.get("coverage_by_slot") if isinstance(multi_need_trace, dict) else {})

    return {
        "route": route,
        "failure_stage": decision_trace.get("failure_stage") or "",
        "failure_reason": decision_trace.get("failure_reason") or "",
        "final_reason": decision_trace.get("final_reason") or "",
        "intent": query_understanding.get("intent") or "",
        "categories": query_understanding.get("categories") or [],
        "product_type": query_understanding.get("detected_product_type") or "",
        "is_multi_need": is_multi_need,
        "slot_count": slot_count,
        "slot_planner": detect_payload.get("planner_source") or "",
        "coverage_by_slot": coverage_by_slot,
        "selected_by_slot": selected_by_slot,
        "counts": counts,
        "products": products,
        "answer_preview": answer[:180].replace("\n", " "),
    }


def compact_selected_by_slot(selected_by_slot: dict[str, Any]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for slot_id, candidates in selected_by_slot.items():
        if not isinstance(candidates, list):
            continue
        result[str(slot_id)] = [
            "{product_id}:{name}".format(
                product_id=str(candidate.get("product_id") or ""),
                name=str(candidate.get("name") or ""),
            )
            for candidate in candidates
            if isinstance(candidate, dict)
        ]
    return result


def compact_coverage(coverage_by_slot: Any) -> dict[str, str]:
    if not isinstance(coverage_by_slot, dict):
        return {}
    result: dict[str, str] = {}
    for slot_id, item in coverage_by_slot.items():
        if isinstance(item, dict):
            status = item.get("status") or "-"
            reason = item.get("reason") or ""
            result[str(slot_id)] = f"{status}: {reason}" if reason else str(status)
    return result


def print_summary(summary: dict[str, Any]) -> None:
    if summary.get("error"):
        print(f"{summary['index']:02d}. {summary['message']} -> ERROR {summary['error']}")
        return

    route = summary.get("route") or "-"
    multi_need = (
        f"multi_need={summary.get('is_multi_need')} "
        f"slots={summary.get('slot_count')} "
        f"planner={summary.get('slot_planner') or '-'}"
    )
    product_names = [
        "{id}:{name}".format(id=product.get("id"), name=product.get("name"))
        for product in summary.get("products") or []
    ]
    products_text = "；".join(product_names) if product_names else "(none)"
    counts = summary.get("counts") or {}
    count_text = (
        f"sql={counts.get('after_structured_filter', 0)} "
        f"vec={counts.get('vector_hits', 0)} "
        f"kw={counts.get('keyword_hits', 0)} "
        f"final={counts.get('after_corrective', 0)}"
    )

    print(f"{summary['index']:02d}. {summary['message']}")
    print(f"    route={route} | {multi_need} | {count_text}")
    if summary.get("failure_stage"):
        print(f"    failure={summary.get('failure_stage')}: {summary.get('failure_reason') or '-'}")
    print(f"    products={products_text}")
    if summary.get("coverage_by_slot"):
        coverage = "；".join(f"{slot}: {value}" for slot, value in summary["coverage_by_slot"].items())
        print(f"    coverage={coverage}")
    if summary.get("selected_by_slot"):
        selected = "；".join(
            f"{slot}: {', '.join(values) or '(none)'}"
            for slot, values in summary["selected_by_slot"].items()
        )
        print(f"    selected={selected}")
    if summary.get("final_reason"):
        print(f"    reason={summary['final_reason']}")
    if summary.get("answer_preview"):
        print(f"    answer={summary['answer_preview']}")


def iter_sse_events(response: httpx.Response) -> Iterator[dict[str, Any]]:
    event_name: str | None = None
    data_lines: list[str] = []

    def flush() -> dict[str, Any] | None:
        nonlocal event_name, data_lines
        if not data_lines:
            return None
        raw_data = "\n".join(data_lines)
        current_event_name = event_name
        data_lines = []
        event_name = None
        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError:
            return {"type": current_event_name or "unknown", "raw": raw_data}
        if "type" not in data and current_event_name:
            data["type"] = current_event_name
        return data

    for line in response.iter_lines():
        if line == "":
            event = flush()
            if event is not None:
                yield event
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").lstrip())

    event = flush()
    if event is not None:
        yield event


if __name__ == "__main__":
    raise SystemExit(main())
