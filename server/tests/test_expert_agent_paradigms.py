import asyncio
import json
from typing import Any

from app.domain.agents.business import BusinessAgentHandlers, BusinessAgentServices
from app.domain.agents.contracts import AgentExecutionContext
from app.domain.supervisor.task_graph import TaskGraph, TaskGraphNode
from app.domain.supervisor.tool_access import AgentToolAccess
from app.harness.tool_registry import ToolRegistry


class FakeStructuredLlm:
    def __init__(self, outputs: list[dict[str, Any]]) -> None:
        self.outputs = outputs
        self.calls = 0
        self.operations: list[str] = []

    def is_configured(self) -> bool:
        return True

    async def generate_required(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, Any] | None = None,
        operation: str = "",
    ) -> str:
        self.operations.append(operation)
        output = self.outputs[min(self.calls, len(self.outputs) - 1)]
        self.calls += 1
        return json.dumps(output, ensure_ascii=False)


class FakeWebSearch:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.queries: list[str] = []

    async def search(
        self,
        *,
        query: str,
        freshness: str = "recent",
        limit: int = 8,
    ) -> dict[str, Any]:
        self.queries.append(query)
        return self.responses[min(len(self.queries) - 1, len(self.responses) - 1)]


def make_context(
    *,
    agent_id: str,
    capability: str,
    metadata: dict[str, Any] | None = None,
) -> AgentExecutionContext:
    node = TaskGraphNode(
        node_id="node",
        task_id=f"turn:{capability}",
        agent_id=agent_id,
        capability=capability,
        metadata=metadata or {},
    )
    graph = TaskGraph(
        graph_id="graph",
        turn_id="turn",
        nodes=[node],
        entry_node_ids=[node.node_id],
        terminal_node_ids=[node.node_id],
    )
    return AgentExecutionContext(
        node=node,
        graph=graph,
        query="敏感肌应该关注哪些护肤成分",
        user_id="user",
        session_id="session",
        turn_id="turn",
        intent_plan=None,
        metadata=dict(metadata or {}),
    )


def knowledge_output(
    *,
    sufficient: bool,
    followup_query: str = "",
    evidence_indexes: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "concepts": [
            {
                "concept": "神经酰胺面霜",
                "concept_type": "product_type",
                "evidence_indexes": evidence_indexes or [0],
            }
        ],
        "risks": [],
        "coverage": {
            "sufficient": sufficient,
            "missing_aspects": [] if sufficient else ["适用边界"],
            "followup_query": followup_query,
            "reason": "证据已覆盖目标" if sufficient else "还缺少适用边界资料",
        },
    }


def comparison_draft(winner_id: str) -> dict[str, Any]:
    return {
        "dimensions": [
            {
                "name": "价格",
                "values": [
                    {"product_id": "p1", "value": "199", "evidence": "local"},
                    {"product_id": "p2", "value": "159", "evidence": "local"},
                ],
            }
        ],
        "winner_by_goal": [
            {"goal": "预算优先", "product_id": winner_id, "reason": "价格更低"}
        ],
        "unknowns": [],
    }


def reflection_output(final: dict[str, Any], *, changed: bool = True) -> dict[str, Any]:
    return {
        "issues": [
            {
                "type": "unsupported_winner",
                "location": "winner_by_goal[0]",
                "reason": "初稿胜出商品与本地价格证据不一致",
            }
        ],
        "changed": changed,
        "reflection_reason": "已按本地商品事实修正",
        "final_comparison": final,
    }


def test_knowledge_research_stops_when_first_round_is_sufficient() -> None:
    web_search = FakeWebSearch(
        [
            {
                "available": True,
                "items": [
                    {
                        "title": "敏感肌屏障护理",
                        "snippet": "神经酰胺可用于屏障护理产品。",
                        "url": "https://example.test/first",
                    }
                ],
            }
        ]
    )
    llm = FakeStructuredLlm([knowledge_output(sufficient=True)])
    handlers = BusinessAgentHandlers(BusinessAgentServices(knowledge_llm=llm))
    metadata = {
        "allowed_tools": ["web_search"],
        "parameters": {"query": "敏感肌屏障护理商品类型"},
    }
    context = make_context(
        agent_id="product_knowledge_agent",
        capability="knowledge_research",
        metadata=metadata,
    )
    tools = ToolRegistry()
    tools.register("web_search", web_search, networked=True)
    context.tool_access = AgentToolAccess(tools)

    result = asyncio.run(handlers.knowledge_research(context))

    assert web_search.queries == ["敏感肌屏障护理商品类型"]
    assert llm.operations == ["product_knowledge_agent.reflect_round_1"]
    assert result.output["research_round_count"] == 1
    assert result.output["coverage"]["sufficient"] is True
    assert result.output["research_rounds"][0]["stop_reason"] == "coverage_sufficient"


def test_knowledge_research_performs_one_targeted_followup() -> None:
    web_search = FakeWebSearch(
        [
            {
                "available": True,
                "items": [
                    {
                        "title": "敏感肌基础护理",
                        "snippet": "神经酰胺常见于屏障护理产品。",
                        "url": "https://example.test/basic",
                    }
                ],
            },
            {
                "available": True,
                "items": [
                    {
                        "title": "神经酰胺适用边界",
                        "snippet": "敏感肌仍需结合配方刺激物和个体反应判断。",
                        "url": "https://example.test/boundary",
                    }
                ],
            },
        ]
    )
    llm = FakeStructuredLlm(
        [
            knowledge_output(
                sufficient=False,
                followup_query="神经酰胺敏感肌适用边界",
            ),
            knowledge_output(sufficient=True, evidence_indexes=[0, 1]),
        ]
    )
    handlers = BusinessAgentHandlers(BusinessAgentServices(knowledge_llm=llm))
    metadata = {
        "allowed_tools": ["web_search"],
        "parameters": {"query": "敏感肌屏障护理商品类型"},
    }
    context = make_context(
        agent_id="product_knowledge_agent",
        capability="knowledge_research",
        metadata=metadata,
    )
    tools = ToolRegistry()
    tools.register("web_search", web_search, networked=True)
    context.tool_access = AgentToolAccess(tools)

    result = asyncio.run(handlers.knowledge_research(context))

    assert web_search.queries == [
        "敏感肌屏障护理商品类型",
        "神经酰胺敏感肌适用边界",
    ]
    assert llm.operations == [
        "product_knowledge_agent.reflect_round_1",
        "product_knowledge_agent.reflect_round_2",
    ]
    assert result.output["research_round_count"] == 2
    assert result.output["research_rounds"][0]["stop_reason"] == "followup_scheduled"
    assert result.output["research_rounds"][1]["stop_reason"] == "coverage_sufficient"


def test_knowledge_research_can_reformulate_an_empty_successful_search() -> None:
    web_search = FakeWebSearch(
        [
            {"available": True, "items": []},
            {
                "available": True,
                "items": [
                    {
                        "title": "敏感肌屏障护理",
                        "snippet": "神经酰胺常见于屏障护理产品。",
                        "url": "https://example.test/reformulated",
                    }
                ],
            },
        ]
    )
    first_reflection = {
        "concepts": [],
        "risks": [],
        "coverage": {
            "sufficient": False,
            "missing_aspects": ["基础证据"],
            "followup_query": "敏感肌屏障护理 神经酰胺",
            "reason": "首轮没有检索结果，需要改写查询",
        },
    }
    llm = FakeStructuredLlm(
        [
            first_reflection,
            knowledge_output(sufficient=True),
        ]
    )
    handlers = BusinessAgentHandlers(BusinessAgentServices(knowledge_llm=llm))
    metadata = {
        "allowed_tools": ["web_search"],
        "parameters": {"query": "敏感肌该买什么"},
    }
    context = make_context(
        agent_id="product_knowledge_agent",
        capability="knowledge_research",
        metadata=metadata,
    )
    tools = ToolRegistry()
    tools.register("web_search", web_search, networked=True)
    context.tool_access = AgentToolAccess(tools)

    result = asyncio.run(handlers.knowledge_research(context))

    assert web_search.queries == ["敏感肌该买什么", "敏感肌屏障护理 神经酰胺"]
    assert result.output["research_round_count"] == 2
    assert result.output["claim_count"] == 1
    assert result.output["coverage"]["sufficient"] is True


def test_comparison_reflection_can_correct_the_validated_draft() -> None:
    corrected = comparison_draft("p2")
    llm = FakeStructuredLlm(
        [
            comparison_draft("p1"),
            reflection_output(corrected),
        ]
    )
    handlers = BusinessAgentHandlers(BusinessAgentServices(comparison_llm=llm))
    context = make_context(
        agent_id="comparison_agent",
        capability="comparison",
    )
    products = [
        {"product_id": "p1", "name": "商品一", "price": 199},
        {"product_id": "p2", "name": "商品二", "price": 159},
    ]

    result = asyncio.run(
        handlers._reason_over_comparison(
            context,
            products=products,
            external=[],
            knowledge={},
            upstream_evidence=[],
            fallback_dimensions=[],
        )
    )

    assert llm.operations == ["comparison_agent.compare", "comparison_agent.reflect"]
    assert result["winner_by_goal"][0]["product_id"] == "p2"
    assert result["reflection"]["status"] == "completed"
    assert result["reflection"]["changed"] is True
    assert result["reflection"]["issue_count"] == 1


def test_comparison_keeps_draft_when_reflection_introduces_unknown_product() -> None:
    invalid_final = comparison_draft("p999")
    llm = FakeStructuredLlm(
        [
            comparison_draft("p1"),
            reflection_output(invalid_final),
        ]
    )
    handlers = BusinessAgentHandlers(BusinessAgentServices(comparison_llm=llm))
    context = make_context(
        agent_id="comparison_agent",
        capability="comparison",
    )
    products = [
        {"product_id": "p1", "name": "商品一", "price": 199},
        {"product_id": "p2", "name": "商品二", "price": 159},
    ]

    result = asyncio.run(
        handlers._reason_over_comparison(
            context,
            products=products,
            external=[],
            knowledge={},
            upstream_evidence=[],
            fallback_dimensions=[],
        )
    )

    assert result["winner_by_goal"][0]["product_id"] == "p1"
    assert result["reflection"]["status"] == "failed"
    assert result["reflection"]["changed"] is False
    assert "p999" not in json.dumps(result, ensure_ascii=False)
