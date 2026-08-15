from benchmark.eval_all import extract_internal_actions, extract_tool_calls


def test_benchmark_reads_structured_supervisor_tool_trace() -> None:
    trace = {
        "tool_calls": [
            {"tool": "profile_lookup", "operation": "lookup", "status": "succeeded"},
            {"tool": "product_search", "operation": "single_initial", "status": "succeeded"},
        ],
        "agent_path": [
            {"capability": "repair", "status": "succeeded"},
        ],
    }

    assert extract_tool_calls([], trace) == {"profile_lookup", "product_search"}
    assert extract_internal_actions([], trace) == {"repair_plan_generated"}
