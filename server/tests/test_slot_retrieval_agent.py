from app.domain.need_slot_schemas import NeedSlot, SlotSearchResult
from app.domain.slot_retrieval_agent import SlotRetrievalAgent
from app.schemas import IntentPlan, QueryPlan


class EmptySearchTool:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def query_variants(self, slot: NeedSlot) -> list[str]:
        return [slot.query, f"{slot.product_type} repair 1", f"{slot.product_type} repair 2"]

    def search_query(
        self,
        slot: NeedSlot,
        base_plan: QueryPlan,
        intent_plan: IntentPlan,
        query: str,
        attempt_index: int,
        reason: str,
    ) -> SlotSearchResult:
        self.queries.append(query)
        return SlotSearchResult(
            slot_id=slot.slot_id,
            query=slot.query,
            vector_query=query,
            keyword_query=query,
            attempts=[{"attempt": attempt_index, "query": query, "reason": reason}],
        )


def test_slot_agent_runs_initial_search_only_without_repair_loop() -> None:
    search_tool = EmptySearchTool()
    agent = SlotRetrievalAgent(search_tool)
    slot = NeedSlot(slot_id="s1", goal="防晒", product_type="防晒", query="防晒 initial")

    result = agent.run(slot, QueryPlan(), IntentPlan(original_query="旅行要防晒"))

    assert result.decision_steps == 1
    assert result.search_calls == 1
    assert result.repair_calls == 0
    assert result.termination_reason == "slot_agent_no_candidates"
    assert search_tool.queries == ["防晒 initial"]
    assert [call.action for call in result.tool_calls] == [
        "search_products",
    ]
    assert len(result.slot_result["attempts"]) == 1
