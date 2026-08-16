"""Command-based Routing with State Updates and Agent Integration.

This example showcases Command-based routing in a supervisor-agent architecture.

The system uses:
1. create_agent to create three agents (supervisor, transaction_history, property_profile)
2. Each agent has domain-specific tools for demo:
   - Supervisor: calculate_mortgage_affordability
   - Transaction History: calculate_price_per_sqft
   - Property Profile: calculate_remaining_lease
3. All node functions use Command for dynamic routing (no Command.PARENT needed)
4. Structured output for routing decisions and property_name extraction
5. Agents are invoked within node functions in the parent graph

Key Benefit: Clean supervisor architecture using Command for routing where:
- Supervisor orchestrates and routes to specialist agents
- Specialist agents complete their work and return
- All routing uses Command without Command.PARENT (flat graph structure)

Note: Command.PARENT is NOT needed here because all node functions are in the parent graph.

Usage:
    python run_command_routing.py                                    # Default query
    python run_command_routing.py "Tell me about 38 Oxley Road"      # Extracts property_name
"""

import argparse

from langchain_core.messages import HumanMessage
from langgraph.graph import START, StateGraph

from src.nodes import (
    SupervisorState,
    property_profile_agent_node,
    supervisor_command_node,
    transaction_history_agent_node,
)


def build_graph() -> StateGraph:
    """Build graph using Command for dynamic routing.

    Architecture:
    - Flat graph with three agent nodes
    - Command handles all routing dynamically (no explicit edges needed beyond START)
    - Supervisor extracts property name and routes to specialists
    """
    graph = StateGraph(SupervisorState)

    # Add agent nodes
    graph.add_node("supervisor", supervisor_command_node)
    graph.add_node("transaction_history_agent", transaction_history_agent_node)
    graph.add_node("property_profile_agent", property_profile_agent_node)

    # Set entry point (Command handles all other routing)
    graph.add_edge(START, "supervisor")

    return graph.compile()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Command-based routing example")
    parser.add_argument(
        "query", nargs="?", default="Tell me about the property at Sunset Boulevard"
    )
    args = parser.parse_args()

    print("EXAMPLE: Command-based Routing")
    print("=" * 70)
    print(f"Query: {args.query}\n")

    # Build and invoke graph
    graph = build_graph()
    final_state = graph.invoke({"messages": [HumanMessage(content=args.query)]})

    # Show extracted property name
    print("\n" + "=" * 70)
    print("STATE UPDATE:")
    print(f"  property_name: '{final_state.get('property_name', '(not extracted)')}'")
    print("=" * 70)
