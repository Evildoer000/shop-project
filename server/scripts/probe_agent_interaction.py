from __future__ import annotations

import argparse
import json
import sys
import uuid
from typing import Any, Iterator

import httpx


DEFAULT_MESSAGES = [
    "我是油皮，预算150以内，推荐一款夏天通勤用不闷的防晒，不要酒精味重",
    "对比一下刚才推荐里前两款，哪款更适合通勤",
    "想买黑咖啡，最好冷萃或冻干，不要奶香三合一",
]


def configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def main() -> int:
    configure_console_encoding()
    parser = argparse.ArgumentParser(
        description="Probe the ecommerce RAG agent SSE flow and print decision traces.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="FastAPI base URL")
    parser.add_argument("--user-id", default="probe_user", help="User id sent to the backend")
    parser.add_argument(
        "--session-id",
        default=f"probe_{uuid.uuid4().hex[:8]}",
        help="Session id sent to the backend",
    )
    parser.add_argument(
        "--message",
        action="append",
        dest="messages",
        help="Message to send. Repeat this flag for multi-turn probing.",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Start an interactive terminal chat after any provided messages.",
    )
    parser.add_argument(
        "--no-defaults",
        action="store_true",
        help="Do not run the built-in default messages when no --message is provided.",
    )
    parser.add_argument(
        "--raw-json",
        action="store_true",
        help="Print the raw decision_trace JSON after the human-readable chain report.",
    )
    args = parser.parse_args()

    messages = args.messages or ([] if args.no_defaults or args.interactive else DEFAULT_MESSAGES)
    url = f"{args.base_url.rstrip('/')}/api/chat/stream"
    had_error = False

    with httpx.Client(timeout=None) as client:
        for index, message in enumerate(messages, start=1):
            print(f"\n=== Turn {index} ===")
            print(f"User: {message}")
            result = run_turn(
                client=client,
                url=url,
                user_id=args.user_id,
                session_id=args.session_id,
                message=message,
                raw_json=args.raw_json,
            )
            had_error = had_error or result["had_error"]

        if args.interactive:
            had_error = run_interactive_chat(
                client=client,
                url=url,
                user_id=args.user_id,
                session_id=args.session_id,
                start_turn=len(messages) + 1,
                raw_json=args.raw_json,
            ) or had_error

    return 1 if had_error else 0


def run_interactive_chat(
    client: httpx.Client,
    url: str,
    user_id: str,
    session_id: str,
    start_turn: int,
    raw_json: bool,
) -> bool:
    print("\n=== Interactive Agent Probe ===")
    print(f"session_id={session_id}")
    print("输入你的测试指令；输入 exit / quit / q 结束。")
    print("每轮都会打印 trace、回答、商品卡片和 decision_trace。")

    turn = start_turn
    had_error = False
    while True:
        try:
            message = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if message.lower() in {"exit", "quit", "q"}:
            print("bye")
            break
        if not message:
            continue

        print(f"\n=== Turn {turn} ===")
        result = run_turn(
            client=client,
            url=url,
            user_id=user_id,
            session_id=session_id,
            message=message,
            raw_json=raw_json,
        )
        had_error = had_error or result["had_error"]
        turn += 1

    return had_error


def run_turn(
    client: httpx.Client,
    url: str,
    user_id: str,
    session_id: str,
    message: str,
    raw_json: bool = False,
) -> dict[str, bool]:
    payload = {
        "user_id": user_id,
        "session_id": session_id,
        "message": message,
        "image_id": None,
    }
    answer_parts: list[str] = []
    decision_trace: dict[str, Any] | None = None
    product_cards: list[dict[str, Any]] = []
    had_error = False

    try:
        with client.stream(
            "POST",
            url,
            headers={"Accept": "text/event-stream"},
            json=payload,
        ) as response:
            if response.status_code >= 400:
                print(f"[http_error] {response.status_code}: {response.text}")
                return {"had_error": True}

            for event in iter_sse_events(response):
                event_type = event.get("type")
                if event_type == "trace":
                    print(format_trace_event(event))
                elif event_type == "decision_trace":
                    decision_trace = event.get("trace") or {}
                    print("[decision_trace] received")
                elif event_type == "token":
                    token = str(event.get("content", ""))
                    answer_parts.append(token)
                    print(token, end="", flush=True)
                elif event_type == "product_cards":
                    product_cards = list(event.get("products") or [])
                elif event_type == "error":
                    had_error = True
                    print(f"\n[error:{event.get('stage', 'server')}] {event.get('message', '')}")
                elif event_type == "done":
                    break
    except httpx.RemoteProtocolError as exc:
        had_error = True
        print(
            "\n[stream_error] Backend closed the SSE stream before sending a complete response. "
            "Check the uvicorn server terminal for the original exception."
        )
        print(f"[stream_error_detail] {exc}")
    except httpx.HTTPError as exc:
        had_error = True
        print(f"\n[http_error] {exc}")

    if answer_parts:
        print("\n\nAssistant: (streamed above)")
    else:
        print("\n\nAssistant:")
        print("(no answer tokens)")

    print("\nProduct cards:")
    if product_cards:
        for product in product_cards:
            price = product.get("price", "")
            rating = product.get("rating", "")
            print(
                "- {product_id} | {name} | {brand} | ¥{price} | rating={rating}".format(
                    product_id=product.get("product_id", ""),
                    name=product.get("name", ""),
                    brand=product.get("brand", ""),
                    price=price,
                    rating=rating,
                )
            )
            reason = product.get("reason")
            if reason:
                print(f"  reason: {reason}")
    else:
        print("(none)")

    print("\nDecision chain:")
    if decision_trace:
        print(format_decision_chain(decision_trace))
        if raw_json:
            print("\nRaw decision_trace JSON:")
            print(json.dumps(decision_trace, ensure_ascii=False, indent=2))
    else:
        print("(none)")

    return {"had_error": had_error}


def format_trace_event(event: dict[str, Any]) -> str:
    stage = str(event.get("stage", ""))
    content = str(event.get("content", ""))
    lines = [f"[trace:{stage}] {content}"]
    if stage in {"intent_planning", "query_rewrite"}:
        rewrite = event.get("intent_plan") or event.get("query_rewrite") or {}
        if isinstance(rewrite, dict):
            lines.extend(
                [
                    f"  - plan_type: {execution_path_code_label(rewrite.get('plan_type'))}",
                    f"  - 原始问题: {rewrite.get('original_query') or '-'}",
                    f"  - 向量检索句: {rewrite.get('vector_query') or '-'}",
                    f"  - 关键词检索句: {rewrite.get('keyword_query') or '-'}",
                    f"  - 预算下限: {rewrite.get('budget_min') if rewrite.get('budget_min') is not None else '-'}",
                    f"  - 预算上限: {rewrite.get('budget_max') if rewrite.get('budget_max') is not None else '-'}",
                    f"  - 预算范围: {rewrite.get('budget_scope') or '-'}",
                    f"  - 引用商品: {join_values(rewrite.get('referenced_product_ids'))}",
                    f"  - 原因: {rewrite.get('plan_reason') or '-'}",
                ]
            )
    elif stage == "query_planning":
        plan = event.get("query_plan") or {}
        if isinstance(plan, dict):
            budget = plan.get("budget") or {}
            budget_max = budget.get("max") if isinstance(budget, dict) else None
            lines.extend(
                [
                    f"  - 识别类目: {join_values(plan.get('categories'))}",
                    f"  - 识别商品类型: {plan.get('detected_product_type') or '-'}",
                    f"  - 预算上限: {format_budget(budget_max)}",
                    f"  - 场景: {join_values(plan.get('scene'))}",
                    f"  - 偏好: {join_values(plan.get('preferences'))}",
                    f"  - 排除项: {join_values(plan.get('exclude'))}",
                    f"  - 需要澄清: {yes_no(plan.get('need_clarification'))}",
                    f"  - 不支持品类: {yes_no(plan.get('unsupported_product_type'))}",
                ]
            )
    elif stage in {"corrective_review", "corrective_reflection"}:
        decision = event.get("reflection_result") or event.get("corrective_agent") or {}
        if isinstance(decision, dict):
            lines.extend(
                [
                    f"  - 是否有通过商品: {yes_no(decision.get('has_passed_products'))}",
                    f"  - 评估质量: {quality_label(decision.get('quality'))}",
                    f"  - 是否需要澄清: {yes_no(decision.get('needs_clarification'))}",
                    f"  - 通过商品: {join_values(decision.get('passed_product_ids'))}",
                    f"  - 原因: {decision.get('reason') or '-'}",
                ]
            )
            combo = decision.get("combo_summary") if isinstance(decision.get("combo_summary"), dict) else {}
            if combo:
                lines.append(f"  - 组合状态: {combo.get('status') or '-'}")
    elif stage == "need_slot_detect":
        multi_need = event.get("multi_need") or {}
        if isinstance(multi_need, dict):
            lines.extend(
                [
                    f"  - 是否多需求检索: {yes_no(multi_need.get('is_multi_need'))}",
                    f"  - slot 数: {multi_need.get('slot_count', 0)}",
                    f"  - planner: {multi_need.get('planner_source') or '-'}",
                    f"  - 原因: {multi_need.get('reason') or '-'}",
                ]
            )
    elif stage == "need_slot_plan":
        slots = event.get("need_slot_plan") or []
        if isinstance(slots, list):
            lines.append("  - 子需求:")
            for slot in slots:
                if isinstance(slot, dict):
                    lines.append(
                        "    - {slot_id} [{need_type}] {goal} / {product_type}".format(
                            slot_id=slot.get("slot_id", "-"),
                            need_type=slot.get("need_type", "-"),
                            goal=slot.get("goal", "-"),
                            product_type=slot.get("product_type", "-"),
                        )
                    )
    elif stage == "multi_need_retrieval":
        trace = event.get("multi_need") or event.get("multi_need_trace") or {}
        if isinstance(trace, dict):
            budgets = trace.get("budgets") or {}
            lines.extend(
                [
                    f"  - 停止原因: {trace.get('termination_reason') or '-'}",
                    f"  - 决策步数: {budgets.get('decision_steps', 0)}",
                    f"  - search 调用: {budgets.get('search_calls', 0)}",
                ]
            )
    return "\n".join(lines)


def format_decision_chain(trace: dict[str, Any]) -> str:
    summary = trace.get("retrieval_summary") if isinstance(trace.get("retrieval_summary"), dict) else {}
    task = trace.get("task") if isinstance(trace.get("task"), dict) else {}
    multi_need_trace = trace.get("multi_need_trace") if isinstance(trace.get("multi_need_trace"), dict) else {}
    counts = trace.get("candidate_counts") if isinstance(trace.get("candidate_counts"), dict) else {}

    final_route = (
        task.get("final_route")
        or summary.get("orchestrator_final_route")
        or trace.get("route")
        or summary.get("route")
        or "-"
    )
    execution_path = task.get("execution_path") or selected_decision(trace, "execution_path") or "-"
    task_status = trace.get("task_status") or task.get("status") or "-"
    failure_stage = trace.get("failure_stage") or "none"
    failure_reason = trace.get("failure_reason") or ""

    lines = ["RAG Harness 决策链"]
    lines.append(f"- execution_path: {execution_path_code_label(execution_path)}")
    lines.append(f"- final_route: {route_code_label(final_route)}")
    lines.append(f"- task_status: {status_label(task_status)}")
    lines.append(f"- 停止阶段: {stage_label(failure_stage)}")
    if failure_reason:
        lines.append(f"- 停止原因: {failure_reason}")

    append_agent_path(lines, trace)
    append_planner_proposal(lines, trace.get("planner_proposal") or {})
    append_orchestrator_decisions(lines, trace.get("orchestrator_decisions") or [])
    append_reflection_result(lines, summary)

    is_multi_need = bool(summary.get("multi_need") or multi_need_trace)
    if counts:
        lines.append("\n候选数量流转:")
        lines.append(f"- 商品库可用商品: {counts.get('before_structured_filter', 0)}")
        if is_multi_need:
            lines.append(f"- 各 slot 候选总数/入审候选数: {counts.get('after_structured_filter', 0)}")
        else:
            lines.append(f"- 可售候选池: {counts.get('after_structured_filter', 0)}")
        lines.append(f"- 向量召回命中: {counts.get('vector_hits', 0)}")
        lines.append(f"- BM25 关键词命中: {counts.get('keyword_hits', 0)}")
        lines.append(f"- 分数阈值过滤后: {counts.get('after_score_filter', 0)}")
        lines.append(f"- RRF 粗排后: {counts.get('after_hybrid_rank', 0)}")
        lines.append(f"- Cross-Encoder 重排后: {counts.get('after_rerank', 0)}")
        lines.append(f"- Reflection 通过后: {counts.get('after_corrective', 0)}")

    plan = trace.get("query_understanding") or {}
    if plan:
        lines.append("\n需求理解:")
        lines.append(f"- 意图: {intent_label(plan.get('intent'))}")
        lines.append(f"- 类目: {join_values(plan.get('categories'))}")
        lines.append(f"- 商品类型: {plan.get('detected_product_type') or '-'}")
        budget = plan.get("budget") or {}
        lines.append(f"- 预算: {format_budget(budget.get('max') if isinstance(budget, dict) else None)}")
        lines.append(f"- 场景: {join_values(plan.get('scene'))}")
        lines.append(f"- 偏好: {join_values(plan.get('preferences'))}")
        lines.append(f"- 排除项: {join_values(plan.get('exclude'))}")

    if summary:
        lines.append("\n检索与证据:")
        if summary.get("keyword_query") or summary.get("rewritten_query"):
            lines.append(f"- 改写后的关键词查询: {summary.get('keyword_query') or summary.get('rewritten_query') or '-'}")
        if summary.get("vector_query"):
            lines.append(f"- 向量语义查询: {summary.get('vector_query') or '-'}")
        lines.append(f"- Orchestrator final_route: {route_code_label(summary.get('orchestrator_final_route') or final_route)}")
        if summary.get("reflection_reason"):
            lines.append(f"- Reflection 原因: {summary.get('reflection_reason')}")
        combo = summary.get("combo_summary") if isinstance(summary.get("combo_summary"), dict) else {}
        if combo:
            lines.append("- 预算组合:")
            lines.append(f"  - 状态: {combo.get('status') or '-'}")
            if combo.get("budget_max") is not None:
                lines.append(f"  - 预算上限: {combo.get('budget_max')}")
            if combo.get("total_price") is not None:
                lines.append(f"  - 组合总价: {combo.get('total_price')}")
            if combo.get("over_budget_amount"):
                lines.append(f"  - 超预算: {combo.get('over_budget_amount')}")
            if combo.get("missing_required_slot_ids"):
                lines.append(f"  - 缺失 required slots: {join_values(combo.get('missing_required_slot_ids'))}")
        if summary.get("alternatives_by_slot"):
            lines.append("- 同 slot 备选商品:")
            alternatives = summary.get("alternatives_by_slot") or {}
            if isinstance(alternatives, dict):
                for slot_id, items in alternatives.items():
                    for item in items or []:
                        if isinstance(item, dict):
                            lines.append(
                                "  - {slot_id}: {product_id}（{name}）".format(
                                    slot_id=slot_id,
                                    product_id=item.get("product_id", "-"),
                                    name=item.get("name", "-"),
                                )
                            )
        if summary.get("rejected_products"):
            lines.append("- Reflection 拒绝商品:")
            for item in summary.get("rejected_products") or []:
                if isinstance(item, dict):
                    lines.append(f"  - {item.get('product_id', '-')}: {item.get('reason', '-')}")
        if summary.get("reflection_omitted_products"):
            lines.append("- Reflection 明确拒绝/未通过候选:")
            for item in summary.get("reflection_omitted_products") or []:
                if isinstance(item, dict):
                    lines.append(
                        "  - {product_id}（{name}）: {reason}".format(
                            product_id=item.get("product_id", "-"),
                            name=item.get("name", "-"),
                            reason=item.get("reason", "-"),
                        )
                    )

    if multi_need_trace:
        lines.append("\n多需求 Worker 轨迹:")
        lines.append(f"- 停止原因: {multi_need_trace.get('termination_reason') or '-'}")
        lines.append("- 覆盖状态:")
        coverage = multi_need_trace.get("coverage_by_slot") or {}
        if isinstance(coverage, dict):
            for slot_id, item in coverage.items():
                if isinstance(item, dict):
                    lines.append(f"  - {slot_id}: {status_label(item.get('status'))}，{item.get('reason') or '-'}")
        selection = multi_need_trace.get("final_selection") or multi_need_trace.get("selection") or {}
        if isinstance(selection, dict):
            lines.append(
                f"- Orchestrator 最终组合: {route_code_label(selection.get('route'))}，{selection.get('reason') or '-'}"
            )

    rerank_scores = parse_rerank_factors(trace.get("rerank_factors") or [])
    entered_rerank = bool(counts and counts.get("after_score_filter", 0) and counts.get("after_rerank", 0))
    lines.append("\n重排状态:")
    if entered_rerank and rerank_scores:
        lines.append("- 本轮进入远程 Cross-Encoder 精排，以下是精排分:")
        for key, value in rerank_scores.items():
            lines.append(f"  - {weight_label(key)}: {value}")
    elif entered_rerank:
        lines.append("- 本轮进入远程 Cross-Encoder 精排；多需求链路按 slot 汇总，不展示单一 query 权重分。")
    else:
        lines.append("- 本轮没有可排序候选，或未进入重排阶段。")

    stages = trace.get("stages") or []
    if stages:
        lines.append("\n阶段明细:")
        for index, stage in enumerate(stages, start=1):
            if not isinstance(stage, dict):
                continue
            name = str(stage.get("name", ""))
            status = str(stage.get("status", ""))
            reason = str(stage.get("reason", ""))
            details = stage.get("details") if isinstance(stage.get("details"), dict) else {}
            lines.append(f"{index}. {stage_label(name)}: {status_label(status)}")
            if reason:
                lines.append(f"   - 说明: {reason}")
            for detail_line in format_stage_details(name, details):
                lines.append(f"   - {detail_line}")

    final_reason = trace.get("final_reason")
    if final_reason:
        lines.append(f"\n最终判断: {final_reason}")
    return "\n".join(lines)


def append_agent_path(lines: list[str], trace: dict[str, Any]) -> None:
    agent_path = trace.get("agent_path") if isinstance(trace.get("agent_path"), list) else []
    if not agent_path:
        agent_path = build_agent_path_from_trace(trace)
    if not agent_path:
        return
    lines.append("\nAgent 决策路径:")
    for item in agent_path:
        if not isinstance(item, dict):
            continue
        node = item.get("node") or "-"
        role = item.get("role") or "-"
        decision = item.get("decision") or ""
        selected = item.get("selected") or item.get("output") or item.get("input_final_route") or ""
        approved = item.get("approved")
        reason = item.get("reason") or ""
        label = f"- {node}（{role}）"
        if decision:
            label += f": {decision_label(decision)} -> {decision_value_label(str(decision), selected, approved)}"
        elif selected:
            label += f": {selected}"
        lines.append(label)
        decision_path = item.get("decision_path") if isinstance(item.get("decision_path"), list) else []
        for step in decision_path[:8]:
            lines.append(f"  path: {short_text(step, 140)}")
        if len(decision_path) > 8:
            lines.append(f"  path: ... 还有 {len(decision_path) - 8} 步")
        if item.get("execution_path"):
            lines.append(f"  execution_path: {execution_path_code_label(item.get('execution_path'))}")
        if item.get("fallback_plan"):
            lines.append(f"  fallback_plan: {item.get('fallback_plan')}")
        if item.get("passed_product_ids"):
            lines.append(f"  passed_product_ids: {join_values(item.get('passed_product_ids'))}")
        if reason:
            lines.append(f"  reason: {short_text(reason, 220)}")


def build_agent_path_from_trace(trace: dict[str, Any]) -> list[dict[str, Any]]:
    summary = trace.get("retrieval_summary") if isinstance(trace.get("retrieval_summary"), dict) else {}
    task = trace.get("task") if isinstance(trace.get("task"), dict) else {}
    planner = trace.get("planner_proposal") if isinstance(trace.get("planner_proposal"), dict) else {}
    plan = trace.get("query_understanding") if isinstance(trace.get("query_understanding"), dict) else {}
    multi_need_trace = trace.get("multi_need_trace") if isinstance(trace.get("multi_need_trace"), dict) else {}
    agent_path: list[dict[str, Any]] = []

    if planner:
        intent_plan = planner.get("intent_plan") if isinstance(planner.get("intent_plan"), dict) else {}
        need_slots = intent_plan.get("need_slots") or planner.get("need_slots") or []
        agent_path.append(
            {
                "node": "IntentPlanner",
                "role": "声明式意图提案",
                "route_field": "plan_type",
                "route": intent_plan.get("plan_type") or planner.get("plan_type") or fallback_planner_route(planner, need_slots),
                "status": "proposed",
                "decision_path": [
                    f"plan_type={intent_plan.get('plan_type') or planner.get('plan_type') or '-'}",
                    f"budget_min={intent_plan.get('budget_min', planner.get('budget_min'))}",
                    f"budget_max={intent_plan.get('budget_max', planner.get('budget_max'))}",
                    f"budget_scope={intent_plan.get('budget_scope') or planner.get('budget_scope') or '-'}",
                    f"need_slots={len(need_slots) if isinstance(need_slots, list) else 0}",
                ],
                "reason": intent_plan.get("plan_reason") or planner.get("plan_reason") or "",
            }
        )

    if plan:
        budget = plan.get("budget") if isinstance(plan.get("budget"), dict) else {}
        agent_path.append(
            {
                "node": "QueryPlanner",
                "role": "结构化查询规划",
                "route_field": "planning_signal",
                "route": "clarify" if plan.get("need_clarification") else "continue",
                "status": "planned",
                "decision_path": [
                    f"categories={plan.get('categories') or []}",
                    f"product_type={plan.get('detected_product_type') or '-'}",
                    f"budget_max={budget.get('max') if budget else None}",
                    f"need_clarification={plan.get('need_clarification', False)}",
                ],
                "reason": plan.get("clarification_question") or "",
            }
        )

    decisions = trace.get("orchestrator_decisions") if isinstance(trace.get("orchestrator_decisions"), list) else []
    agent_path.append(
        {
            "node": "Orchestrator",
            "role": "流程控制与最终路线裁决",
            "route_field": "final_route",
            "route": task.get("final_route") or summary.get("orchestrator_final_route") or trace.get("route") or "",
            "status": trace.get("task_status") or task.get("status") or "",
            "execution_path": task.get("execution_path") or selected_decision(trace, "execution_path"),
            "decision_path": [fallback_decision_path_item(item) for item in decisions if isinstance(item, dict)],
            "reason": trace.get("final_reason") or trace.get("failure_reason") or "",
        }
    )

    if multi_need_trace:
        final_signal = multi_need_trace.get("final_signal") if isinstance(multi_need_trace.get("final_signal"), dict) else {}
        budgets = multi_need_trace.get("budgets") if isinstance(multi_need_trace.get("budgets"), dict) else {}
        agent_path.append(
            {
                "node": "MultiNeedRetrievalCoordinator",
                "role": "多需求 Worker 调度",
                "route_field": "worker_route_signal",
                "route": final_signal.get("route") or "slot_agents_completed",
                "status": "completed",
                "decision_path": [
                    f"slot_task_mode={budgets.get('slot_task_mode') or '-'}",
                    f"parallel_slot_agents={budgets.get('parallel_slot_agents') or 0}",
                    f"search_calls={budgets.get('search_calls') or 0}",
                ],
                "reason": final_signal.get("reason") or multi_need_trace.get("termination_reason") or "",
            }
        )
        agent_path.extend(build_slot_agent_path(multi_need_trace))

    reflection = summary.get("reflection_result") if isinstance(summary.get("reflection_result"), dict) else {}
    if reflection or summary.get("reflection_reason"):
        combo = reflection.get("combo_summary") if isinstance(reflection.get("combo_summary"), dict) else {}
        agent_path.append(
            {
                "node": "CorrectiveAgent",
                "role": "证据反射 Worker",
                "route_field": "reflection_signal",
                "route": reflection_signal(reflection, combo, summary),
                "status": "reflected",
                "decision_path": [
                    f"has_passed_products={reflection.get('has_passed_products', bool(summary.get('passed_product_ids')))}",
                    f"fallback_plan={reflection.get('fallback_plan') or '-'}",
                    f"combo_status={combo.get('status') or '-'}",
                    f"passed_product_ids={reflection.get('passed_product_ids') or summary.get('passed_product_ids') or []}",
                ],
                "reason": reflection.get("reason") or summary.get("reflection_reason") or "",
            }
        )

    final_route = task.get("final_route") or summary.get("orchestrator_final_route") or trace.get("route")
    if final_route:
        agent_path.append(
            {
                "node": "AnswerGenerator",
                "role": "回答生成 Worker",
                "route_field": "input_final_route",
                "route": final_route,
                "status": "streamed",
                "decision_path": [
                    f"answer_mode={answer_mode_signal(task.get('execution_path') or selected_decision(trace, 'execution_path'), final_route)}"
                ],
                "reason": trace.get("final_reason") or "",
            }
        )
    return agent_path


def fallback_planner_route(planner: dict[str, Any], need_slots: Any) -> str:
    if planner.get("action") == "direct_answer":
        return "direct_answer"
    if planner.get("need_clarification"):
        return "clarify"
    if isinstance(need_slots, list) and need_slots:
        return "multi_retrieval"
    if planner.get("retrieval_requested") is False:
        return "direct_answer"
    return "single_retrieval"


def build_slot_agent_path(multi_need_trace: dict[str, Any]) -> list[dict[str, Any]]:
    budgets = multi_need_trace.get("budgets") if isinstance(multi_need_trace.get("budgets"), dict) else {}
    results_by_slot = budgets.get("slot_agent_results_by_slot") if isinstance(budgets.get("slot_agent_results_by_slot"), dict) else {}
    coverage_by_slot = multi_need_trace.get("coverage_by_slot") if isinstance(multi_need_trace.get("coverage_by_slot"), dict) else {}
    tool_calls = multi_need_trace.get("tool_calls") if isinstance(multi_need_trace.get("tool_calls"), list) else []
    nodes: list[dict[str, Any]] = []
    for slot in multi_need_trace.get("slots") or []:
        if not isinstance(slot, dict):
            continue
        slot_id = str(slot.get("slot_id") or "")
        result = results_by_slot.get(slot_id) if isinstance(results_by_slot.get(slot_id), dict) else {}
        coverage = coverage_by_slot.get(slot_id) if isinstance(coverage_by_slot.get(slot_id), dict) else {}
        calls = [
            tool_call_path_item(call)
            for call in tool_calls
            if isinstance(call, dict) and call.get("slot_id") == slot_id
        ]
        nodes.append(
            {
                "node": f"SlotRetrievalAgent:{slot_id}",
                "role": "单 slot 检索 Worker",
                "route_field": "slot_status",
                "route": coverage.get("status") or slot.get("status") or "",
                "status": result.get("termination_reason") or "",
                "decision_path": [
                    f"goal={slot.get('goal') or '-'}",
                    f"product_type={slot.get('product_type') or '-'}",
                    f"decision_steps={result.get('decision_steps', 0)}",
                    f"search_calls={result.get('search_calls', 0)}",
                    f"repair_calls={result.get('repair_calls', 0)}",
                    *calls,
                ],
                "reason": coverage.get("reason") or result.get("termination_reason") or "",
            }
        )
    return nodes


def fallback_decision_path_item(decision: dict[str, Any]) -> str:
    name = decision.get("decision") or "-"
    selected = decision.get("selected") or decision.get("internal_decision") or ""
    approved = decision.get("approved")
    if selected:
        return f"{name}={selected}"
    if approved is not None:
        return f"{name}={'approved' if approved else 'rejected'}"
    return str(name)


def tool_call_path_item(call: dict[str, Any]) -> str:
    action = call.get("action") or "-"
    status = call.get("status") or "-"
    inputs = call.get("input_summary") if isinstance(call.get("input_summary"), dict) else {}
    query = inputs.get("query") or inputs.get("previous_query") or ""
    if query:
        return f"{action}:{status}:{query}"
    return f"{action}:{status}"


def reflection_signal(reflection: dict[str, Any], combo: dict[str, Any], summary: dict[str, Any]) -> str:
    if combo.get("status"):
        return str(combo.get("status"))
    if reflection.get("fallback_plan") and reflection.get("fallback_plan") != "none":
        return str(reflection.get("fallback_plan"))
    if reflection.get("has_passed_products") or summary.get("passed_product_ids"):
        return "passed_products"
    return "no_passed_products"


def answer_mode_signal(execution_path: Any, final_route: Any) -> str:
    if execution_path == "multi_retrieval":
        return "stream_multi_need_text"
    if execution_path == "direct_answer":
        return "stream_direct_text"
    if final_route in {"no_product", "clarify", "clarification"}:
        return "stream_direct_text"
    return "stream_text"


def append_planner_proposal(lines: list[str], planner: dict[str, Any]) -> None:
    if not isinstance(planner, dict) or not planner:
        return
    intent_plan = planner.get("intent_plan") if isinstance(planner.get("intent_plan"), dict) else {}
    plan_type = intent_plan.get("plan_type") or planner.get("plan_type")
    budget_min = intent_plan.get("budget_min", planner.get("budget_min"))
    budget_max = intent_plan.get("budget_max", planner.get("budget_max"))
    budget_scope = intent_plan.get("budget_scope") or planner.get("budget_scope")
    referenced_product_ids = intent_plan.get("referenced_product_ids") or planner.get("referenced_product_ids")
    need_slots = intent_plan.get("need_slots") or planner.get("need_slots") or []

    lines.append("\nPlanner 提案:")
    lines.append(f"- 来源: {planner.get('source') or '-'}")
    if plan_type:
        lines.append(f"- plan_type: {plan_type}")
    if budget_min is not None:
        lines.append(f"- budget_min: {budget_min}")
    if budget_max is not None:
        lines.append(f"- budget_max: {budget_max}")
    if budget_scope:
        lines.append(f"- budget_scope: {budget_scope}")
    if referenced_product_ids:
        lines.append(f"- referenced_product_ids: {join_values(referenced_product_ids)}")
    if need_slots:
        lines.append(f"- need_slots: {len(need_slots)} 个")
        for slot in need_slots:
            if isinstance(slot, dict):
                lines.append(
                    "  - {slot_id} [{need_type}] {goal} / {product_type}".format(
                        slot_id=slot.get("slot_id", "-"),
                        need_type=slot.get("need_type", "-"),
                        goal=slot.get("goal", "-"),
                        product_type=slot.get("product_type", "-"),
                    )
                )


def append_orchestrator_decisions(lines: list[str], decisions: list[Any]) -> None:
    if not isinstance(decisions, list) or not decisions:
        return
    lines.append("\nOrchestrator 裁决:")
    for item in decisions:
        if not isinstance(item, dict):
            continue
        decision = str(item.get("decision") or "-")
        selected = item.get("selected") or item.get("internal_decision")
        approved = item.get("approved")
        reason = item.get("reason") or "-"
        lines.append(f"- {decision_label(decision)}: {decision_value_label(decision, selected, approved)}")
        lines.append(f"  reason: {reason}")


def append_reflection_result(lines: list[str], summary: dict[str, Any]) -> None:
    reflection = summary.get("reflection_result") if isinstance(summary.get("reflection_result"), dict) else {}
    if not reflection and not any(
        key in summary for key in ("reflection_reason", "reflection_quality", "passed_product_ids", "rejected_products")
    ):
        return
    reason = reflection.get("reason") or summary.get("reflection_reason") or "-"
    passed_ids = reflection.get("passed_product_ids") or summary.get("passed_product_ids") or []
    rejected = reflection.get("rejected_products") or summary.get("rejected_products") or []
    has_passed = reflection.get("has_passed_products")
    if has_passed is None:
        has_passed = bool(passed_ids)
    combo = reflection.get("combo_summary") if isinstance(reflection.get("combo_summary"), dict) else {}

    lines.append("\nWorker 反射结果:")
    lines.append("- 节点: CorrectiveAgent（只做 evidence reflection，不裁决 final_route）")
    lines.append(f"- has_passed_products: {yes_no(has_passed)}")
    lines.append(f"- fallback_plan: {reflection.get('fallback_plan') or '-'}")
    lines.append(f"- passed_product_ids: {join_values(passed_ids)}")
    lines.append(f"- rejected_count: {len(rejected) if isinstance(rejected, list) else 0}")
    if combo:
        lines.append(f"- combo_status: {combo.get('status') or '-'}")
    lines.append(f"- reason: {reason}")


def selected_decision(trace: dict[str, Any], decision_name: str) -> str:
    decisions = trace.get("orchestrator_decisions") or []
    if not isinstance(decisions, list):
        return ""
    for item in reversed(decisions):
        if isinstance(item, dict) and item.get("decision") == decision_name:
            return str(item.get("selected") or item.get("internal_decision") or "")
    return ""


def route_code_label(value: Any) -> str:
    text = str(value or "-")
    return "-" if text == "-" else f"{text}（{route_label(text)}）"


def execution_path_code_label(value: Any) -> str:
    text = str(value or "-")
    return "-" if text == "-" else f"{text}（{execution_path_label(text)}）"


def decision_value_label(decision: str, selected: Any, approved: Any) -> str:
    if selected:
        if decision == "execution_path":
            return execution_path_code_label(selected)
        if decision == "final_route":
            return route_code_label(selected)
        return str(selected)
    if approved is None:
        return "-"
    return "approved" if approved else "rejected"


def decision_label(value: str) -> str:
    return {
        "intent_plan": "意图计划裁决（intent_plan）",
        "image_input": "图片输入可用性裁决（image_input）",
        "retrieval_needed": "是否需要新检索（retrieval_needed）",
        "previous_evidence_answer": "上下文证据直答裁决（previous_evidence_answer）",
        "image_retrieval_path": "图片检索路径裁决（image_retrieval_path）",
        "image_relevance": "图片相关性裁决（image_relevance）",
        "execution_path": "执行路径裁决（execution_path）",
        "repair": "内部修复裁决（repair）",
        "final_route": "最终业务路线裁决（final_route）",
    }.get(value, value)


def agent_route_value(route_field: Any, route: Any) -> str:
    text = str(route or "-")
    if text == "-":
        return "-"
    route_field_text = str(route_field or "")
    if route_field_text in {"final_route", "input_final_route"}:
        return route_code_label(text)
    if route_field_text == "execution_path_proposal":
        return execution_path_code_label(text)
    if text in {"direct_answer", "clarify", "single_retrieval", "multi_retrieval", "image_retrieval"}:
        return execution_path_code_label(text)
    if text in {"recommend", "partial_recommend", "over_budget_combo", "no_product", "clarification"}:
        return route_code_label(text)
    return text


def short_text(value: Any, max_length: int = 120) -> str:
    text = str(value)
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def format_stage_details(name: str, details: dict[str, Any]) -> list[str]:
    labels = {
        "before": "进入前候选数",
        "after": "阶段后候选数",
        "vector_hits": "向量命中数",
        "keyword_hits": "关键词命中数",
        "route": "路由",
        "orchestrator_final_route": "Orchestrator final_route",
    }
    result = []
    for key, value in details.items():
        if key in {"route", "orchestrator_final_route"}:
            value = route_code_label(value)
        elif isinstance(value, list):
            value = join_values(value) if all(not isinstance(item, (dict, list)) for item in value) else f"{len(value)} 项"
        elif isinstance(value, dict):
            value = json.dumps(value, ensure_ascii=False)
            if len(value) > 180:
                value = value[:177] + "..."
        result.append(f"{labels.get(key, key)}: {value}")
    return result


def parse_rerank_factors(values: list[Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in values:
        text = str(item)
        if "=" not in text:
            continue
        key, value = text.split("=", 1)
        result[key] = value
    return result


def action_label(value: Any) -> str:
    return {"retrieve": "检索商品", "direct_answer": "直接回答"}.get(str(value), str(value or "-"))


def intent_label(value: Any) -> str:
    return {
        "recommendation": "商品推荐",
        "comparison": "商品对比",
        "cart_action": "购物车操作",
    }.get(str(value), str(value or "-"))


def route_label(value: Any) -> str:
    return {
        "retrieve": "执行 RAG 推荐",
        "recommend": "通过评估并推荐",
        "partial_recommend": "部分覆盖并推荐",
        "over_budget_combo": "完整组合超预算，等待用户确认",
        "direct_answer": "直接回答",
        "clarify": "需要用户澄清",
        "clarification": "需要用户澄清",
        "unsupported_product_type": "商品库不支持该品类",
        "no_relevant_products": "没有相关候选商品",
        "low_relevance": "相关性过低",
        "llm_relevance_rejected": "LLM 相关性校验未通过",
        "reject_candidates": "候选商品被 Corrective Agent 拒绝",
        "no_product": "没有可推荐商品",
    }.get(str(value), str(value or "-"))


def execution_path_label(value: Any) -> str:
    return {
        "direct_answer": "直接回答",
        "clarify": "澄清问题",
        "single_retrieval": "单需求检索",
        "multi_retrieval": "多需求检索",
        "image_retrieval": "图片检索",
    }.get(str(value), str(value or "-"))


def stage_label(value: Any) -> str:
    return {
        "none": "无，链路完成",
        "input": "输入处理",
        "query_rewrite": "意图决策与检索改写",
        "query_planning": "查询规划",
        "direct_answer": "直接回答",
        "context_evidence_answer": "上下文证据直答",
        "category_support": "类目支持检查",
        "structured_filter": "可售候选池读取",
        "candidate_pool": "可售候选池读取",
        "retrieval": "向量/BM25 召回",
        "score_filter": "分数阈值过滤",
        "hybrid_rank": "RRF 混合粗排",
        "need_slot_detect": "需求槽识别",
        "multi_need_retrieval": "并发 Slot Agent 检索",
        "image_retrieval_worker_execution": "图片检索 Worker",
        "corrective_review": "CorrectiveAgent 审核（旧）",
        "corrective_reflection": "CorrectiveAgent 证据反射",
        "single_repair": "RepairWorkerAgent 单检索修复",
        "rerank": "重排",
        "corrective_agent": "CorrectiveAgent 判断",
        "llm_relevance_check": "LLM 最终相关性校验",
    }.get(str(value), str(value or "-"))


def status_label(value: Any) -> str:
    return {
        "passed": "通过",
        "succeeded": "成功",
        "running": "运行中",
        "proposed": "已提案",
        "planned": "已规划",
        "completed": "已完成",
        "attempted": "已尝试",
        "reflected": "已反射",
        "streamed": "已生成回答",
        "slot_agent_completed": "slot 检索完成",
        "stopped": "停止",
        "skipped": "跳过",
        "covered": "已覆盖",
        "weak": "弱覆盖",
        "failed": "失败",
        "pending": "待处理",
    }.get(str(value), str(value or "-"))


def weight_label(value: str) -> str:
    if value.startswith("cross_encoder_score"):
        suffix = value.removeprefix("cross_encoder_score").strip("_")
        return f"Cross-Encoder 相关性分 {suffix}".rstrip()
    return value


def quality_label(value: Any) -> str:
    return {
        "correct": "匹配",
        "incorrect": "不匹配",
        "ambiguous": "不确定",
    }.get(str(value), str(value or "-"))


def yes_no(value: Any) -> str:
    return "是" if bool(value) else "否"


def join_values(value: Any) -> str:
    if not value:
        return "-"
    if isinstance(value, list):
        return "、".join(str(item) for item in value) or "-"
    return str(value)


def format_budget(value: Any) -> str:
    if value is None:
        return "未限制"
    return f"{value} 元以内"


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
    sys.exit(main())
