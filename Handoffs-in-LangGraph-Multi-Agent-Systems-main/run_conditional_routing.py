"""Example 2: Conditional Edge Routing with Agent Subgraphs.

This example showcases:
1. Using add_conditional_edges for routing between agent subgraphs
2. All three entities (supervisor, transaction_history, property_profile) are agents created with create_agent
3. Each agent has domain-specific tools:
   - Supervisor: calculate_mortgage_affordability
   - Transaction History: calculate_price_per_sqft
   - Property Profile: calculate_remaining_lease
4. Named routing function (should_continue) that extracts keywords from messages
5. Pre-defined routing paths with explicit edges
6. Graph visualization with Mermaid diagram
7. Pure routing WITHOUT state updates - routing decision is in message content

Key Point: Even with agent subgraphs, conditional edges work for routing. The supervisor
agent makes a routing decision. The should_continue function extracts keywords from
the supervisor's message to determine which agent subgraph to invoke next.

Use conditional edges when you only need routing decisions based on message content.

Usage:
    python run_conditional_routing.py                          # Uses default query
    python run_conditional_routing.py "Your custom query"      # Uses custom query
"""

import argparse
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from src.nodes import (
    SupervisorState,
    property_profile_agent,
    supervisor_conditional_node,
    transaction_history_agent,
)
from src.prompts import PROPERTY_PROFILE_AGENT_PROMPT, TRANSACTION_HISTORY_AGENT_PROMPT
from src.utils import save_mermaid_diagram


# ============================================================
# Helper Functions
# ============================================================
def create_specialist_node(agent, prompt_template: str, agent_name: str):
    """Generic wrapper for specialist agents in conditional routing.

    Returns a function that:
    1. Adds property name context to prompt
    2. Invokes the agent
    3. Returns dict (not Command) since conditional edges handle routing
    """

    def node_func(state: SupervisorState) -> dict:
        # Add property name as context if available
        property_name = state.get("property_name", "")
        context = f"Property: {property_name}" if property_name else ""
        prompt = prompt_template.format(context=context)

        # Invoke agent and extract response
        agent_input = {"messages": [SystemMessage(content=prompt)] + state["messages"]}
        agent_result = agent.invoke(agent_input)
        response_message = agent_result["messages"][-1]
        response_message.name = agent_name

        # Print response with property context
        context_str = f" (Property: {property_name})" if property_name else ""
        print(
            f"{agent_name.replace('_', ' ').title()}{context_str}: {response_message.content}"
        )

        return {"messages": [response_message]}

    return node_func


# Create specialist node wrappers using factory function
transaction_history_agent_conditional = create_specialist_node(
    transaction_history_agent,
    TRANSACTION_HISTORY_AGENT_PROMPT,
    "transaction_history_agent",
)
property_profile_agent_conditional = create_specialist_node(
    property_profile_agent, PROPERTY_PROFILE_AGENT_PROMPT, "property_profile_agent"
)


def should_continue(
    state: SupervisorState,
) -> Literal["transaction_history_agent", "property_profile_agent", "end"]:
    """Extract routing decision from supervisor's message.

    Looks for agent names in the supervisor's latest message to determine routing.
    Returns "end" if supervisor provides a direct response (no routing needed).
    """
    # Find latest supervisor message
    for msg in reversed(state["messages"]):
        if (
            isinstance(msg, AIMessage)
            and hasattr(msg, "name")
            and msg.name == "supervisor"
        ):
            content = msg.content.strip().lower()
            # Check for agent names in content
            if "transaction_history_agent" in content:
                return "transaction_history_agent"
            elif "property_profile_agent" in content:
                return "property_profile_agent"
            return "end"  # No agent name found - supervisor handled directly
    return "end"


def build_graph() -> StateGraph:
    """Build graph using conditional edges for routing.

    Architecture:
    - Supervisor makes routing decision (outputs agent name)
    - should_continue() extracts decision from message
    - Conditional edges route based on decision
    - Explicit edges route specialists to END
    """
    graph = StateGraph(SupervisorState)

    # Add agent nodes
    graph.add_node("supervisor", supervisor_conditional_node)
    graph.add_node("transaction_history_agent", transaction_history_agent_conditional)
    graph.add_node("property_profile_agent", property_profile_agent_conditional)

    # Add edges
    graph.add_edge(START, "supervisor")

    # Conditional routing based on supervisor's message
    graph.add_conditional_edges(
        "supervisor",
        should_continue,
        {
            "transaction_history_agent": "transaction_history_agent",
            "property_profile_agent": "property_profile_agent",
            "end": END,
        },
    )

    # Specialists always route to END after processing
    graph.add_edge("transaction_history_agent", END)
    graph.add_edge("property_profile_agent", END)

    return graph.compile()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Conditional edge routing example")
    parser.add_argument(
        "query", nargs="?", default="Describe the 38 Oxley Road property in Singapore"
    )
    args = parser.parse_args()

    print("EXAMPLE: Conditional Edge Routing")
    print("=" * 70)

    # Build graph and save visualization
    graph = build_graph()
    save_mermaid_diagram(graph, "artifacts/graph_setup.png")

    # Invoke graph with query
    print(f"\nQuery: {args.query}\n")
    graph.invoke({"messages": [HumanMessage(content=args.query)]})
