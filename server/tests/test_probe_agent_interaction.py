from scripts.probe_agent_interaction import format_decision_chain


def test_format_decision_chain_shows_harness_layers_without_corrective_route() -> None:
    text = format_decision_chain(
        {
            "route": "over_budget_combo",
            "failure_stage": "none",
            "planner_proposal": {
                "plan_type": "multi_retrieval",
                "budget_max": 1000,
                "budget_scope": "total",
                "need_slots": [
                    {"slot_id": "s1", "need_type": "required", "goal": "运动鞋", "product_type": "运动鞋"},
                    {"slot_id": "s2", "need_type": "required", "goal": "运动服装", "product_type": "运动服装"},
                ],
            },
            "orchestrator_decisions": [
                {"decision": "execution_path", "selected": "multi_retrieval", "reason": "approved plan_type"},
                {"decision": "final_route", "selected": "over_budget_combo", "reason": "完整组合超预算"},
            ],
            "task": {
                "status": "succeeded",
                "execution_path": "multi_retrieval",
                "final_route": "over_budget_combo",
            },
            "task_status": "succeeded",
            "retrieval_summary": {
                "route": "over_budget_combo",
                "reflection_result": {
                    "has_passed_products": True,
                    "fallback_plan": "none",
                    "passed_product_ids": ["shoe", "shorts"],
                    "reason": "完整组合超预算",
                    "combo_summary": {"status": "over_budget"},
                },
            },
            "multi_need_trace": {
                "stop_reason": "slot agents completed",
                "search_calls": 2,
                "tool_calls": [
                    {
                        "action": "search_products",
                        "slot_id": "s1",
                        "status": "ok",
                        "input_summary": {"query": "运动鞋", "attempt": 1},
                    }
                ],
            },
            "candidate_counts": {"after_corrective": 2},
            "stages": [
                {
                    "name": "corrective_reflection",
                    "status": "passed",
                    "reason": "完整组合超预算",
                    "details": {},
                }
            ],
        }
    )

    assert "IntentPlanner" in text
    assert "Orchestrator" in text
    assert "CorrectiveAgent" in text
    assert "AnswerGenerator" in text
    assert "Corrective Agent 路由" not in text
