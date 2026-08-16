import asyncio

import pytest

from app.domain.agents import AgentExecutionContext, AgentExecutor, BusinessAgentHandlers, BusinessAgentServices
from app.domain.supervisor.agent_registry import build_foundation_agent_registry
from app.domain.supervisor.task_graph import TaskGraph, TaskGraphNode
from app.domain.tools.commerce_mcp import (
    CommerceMcpTransport,
    CommerceProductDetailTool,
    CommerceResearchTool,
    CommerceReviewsTool,
    CommerceToolError,
    UnavailableCommerceTransport,
    _normalize_response,
)
from app.harness.tool_registry import ToolRegistry
from app.schemas import IntentItem, IntentPlanV3, IntentRouteBasis


class FakeCommerceTransport(CommerceMcpTransport):
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def call(self, operation: str, payload: dict) -> dict:
        self.calls.append((operation, payload))
        if operation == "search":
            return {
                "available": True,
                "operation": operation,
                "platform": payload["platform"],
                "endpoint_id": "fake.search",
                "collected_at": "2026-08-12T00:00:00Z",
                "data": {
                    "result": {
                        "payload": {
                            "products": [
                                {"item_id": f"item_{index}", "title": f"商品 {index}"}
                                for index in range(4)
                            ]
                        }
                    }
                },
            }
        return {
            "available": True,
            "operation": operation,
            "platform": payload["platform"],
            "endpoint_id": f"fake.{operation}",
            "collected_at": "2026-08-12T00:00:00Z",
            "items": [
                {
                    "item_id": payload["product_id"],
                    "title": f"{operation}:{payload['product_id']}",
                }
            ],
        }


def test_nested_mcp_response_is_normalized_without_exposing_raw_endpoint_calls() -> None:
    normalized = _normalize_response(
        {
            "available": True,
            "operation": "search",
            "result": {
                "success": True,
                "data": {"payload": {"note_list": [{"note_id": "n1", "title": "笔记"}]}},
            },
        },
        operation="search",
    )

    assert normalized["available"] is True
    assert normalized["items"] == [{"note_id": "n1", "title": "笔记"}]


def test_unavailable_transport_and_platform_rejection_are_controlled() -> None:
    unavailable = asyncio.run(
        CommerceResearchTool(UnavailableCommerceTransport()).search(
            platform="taobao",
            query="防晒霜",
        )
    )

    assert unavailable["available"] is False
    assert unavailable["items"] == []
    assert unavailable["error"]["code"] == "commerce_mcp_not_configured"
    with pytest.raises(CommerceToolError, match="unsupported_commerce_platform"):
        asyncio.run(CommerceResearchTool().search(platform="jd", query="防晒霜"))


def test_commerce_agent_deeply_enriches_at_most_two_search_results() -> None:
    transport = FakeCommerceTransport()
    tools = ToolRegistry()
    tools.register("commerce_search", CommerceResearchTool(transport), networked=True)
    tools.register("commerce_product_detail", CommerceProductDetailTool(transport), networked=True)
    tools.register("commerce_reviews", CommerceReviewsTool(transport), networked=True)
    handlers = BusinessAgentHandlers(BusinessAgentServices()).as_mapping()
    executor = AgentExecutor(
        agent_registry=build_foundation_agent_registry(),
        tool_registry=tools,
        handlers={"commerce_research": handlers["commerce_research"]},
    )
    graph = TaskGraph(
        graph_id="g_commerce",
        turn_id="t_commerce",
        nodes=[
            TaskGraphNode(
                node_id="commerce",
                task_id="t_commerce:commerce",
                agent_id="commerce_research_agent",
                capability="commerce_research",
                intent_ids=["i1"],
                metadata={
                    "planner_task_id": "t_xhs",
                    "parameters": {
                        "platforms": ["xiaohongshu"],
                        "query": "油皮防晒评价",
                        "trigger_type": "explicit_platform_request",
                        "trigger_text": "小红书",
                    },
                },
            )
        ],
    )
    context = AgentExecutionContext(
        node=graph.nodes[0],
        graph=graph,
        query="看看小红书评价",
        user_id="u1",
        session_id="s1",
        turn_id=graph.turn_id,
        intent_plan=IntentPlanV3(
            original_query="看看小红书评价",
            normalized_query="看看小红书评价",
            intents=[
                IntentItem(
                    intent_id="i1",
                    intent_type="product_comparison",
                    goal="查看口碑",
                    resolved_query="查看小红书油皮防晒口碑",
                    route_basis=IntentRouteBasis(
                        target_clarity="context_product",
                        external_information_need="explicit_platform",
                        trigger_text="小红书",
                        product_family="防晒霜",
                    ),
                )
            ],
        ),
    )

    report = asyncio.run(executor.execute(graph, base_context=context))

    assert report.succeeded is True
    operations = [operation for operation, _ in transport.calls]
    assert operations.count("search") == 1
    assert operations.count("detail") == 2
    assert operations.count("reviews") == 2
    assert all(
        payload.get("product_id") in {"item_0", "item_1"}
        for operation, payload in transport.calls
        if operation in {"detail", "reviews"}
    )
