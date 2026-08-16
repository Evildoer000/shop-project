"""Node functions for supervisor and specialist agents."""

from __future__ import annotations

from typing import Annotated, Literal

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.types import Command
from pydantic import BaseModel
from typing_extensions import TypedDict

from src.prompts import (
    PROPERTY_PROFILE_AGENT_PROMPT,
    SUPERVISOR_PROMPT,
    TRANSACTION_HISTORY_AGENT_PROMPT,
)
from src.tools import (
    calculate_mortgage_affordability,
    calculate_price_per_sqft,
    calculate_remaining_lease,
)

# ============================================================
# Configuration
# ============================================================
MODEL_NAME = "gpt-4.1"
TEMPERATURE = 0.0

# ============================================================
# Agent Creation
# ============================================================
# Create agents with domain-specific tools using create_agent
# Each agent is a compiled subgraph that can invoke tools during reasoning

supervisor_agent = create_agent(
    model=init_chat_model(MODEL_NAME, temperature=TEMPERATURE),
    tools=[calculate_mortgage_affordability],  # Mortgage calculator
)

transaction_history_agent = create_agent(
    model=init_chat_model(MODEL_NAME, temperature=TEMPERATURE),
    tools=[calculate_price_per_sqft],  # Property valuation calculator
)

property_profile_agent = create_agent(
    model=init_chat_model(MODEL_NAME, temperature=TEMPERATURE),
    tools=[calculate_remaining_lease],  # Lease years calculator
)


# ============================================================
# State and Models
# ============================================================
class SupervisorState(TypedDict, total=False):
    """State shared across all agents in the graph."""

    messages: Annotated[
        list, add_messages
    ]  # Conversation history (uses add_messages reducer)
    next: str | None  # Next node to route to (for conditional routing)
    property_name: str  # Property name extracted from query (e.g., "38 Oxley Road")


class SupervisorDecision(BaseModel):
    """Structured output from supervisor for routing decisions."""

    next_agent: Literal["transaction_history_agent", "property_profile_agent", "none"]
    property_name: str = ""  # Extracted property name from query
    response: str = ""  # Direct response if no routing needed


# ============================================================
# Helper Functions
# ============================================================
def _invoke_agent(agent, prompt: str, messages: list, agent_name: str):
    """Helper to invoke an agent and return formatted response.

    This consolidates the common pattern of:
    1. Adding system prompt to messages
    2. Invoking the agent subgraph
    3. Extracting and naming the response message
    """
    agent_input = {"messages": [SystemMessage(content=prompt)] + messages}
    agent_result = agent.invoke(agent_input)
    response_message = agent_result["messages"][-1]
    response_message.name = agent_name
    return response_message


# ============================================================
# Node Functions
# ============================================================
def supervisor_conditional_node(state: SupervisorState) -> dict:
    """Supervisor for conditional routing - returns dict for conditional edges."""
    response = _invoke_agent(
        supervisor_agent, SUPERVISOR_PROMPT, state["messages"], "supervisor"
    )
    print(f"Supervisor: {response.content}")
    return {"messages": [response]}


def supervisor_command_node(state: SupervisorState) -> Command:
    """Supervisor for Command routing - uses structured output and extracts property name."""
    # Use structured output to get routing decision and extract property name
    llm = init_chat_model(MODEL_NAME, temperature=TEMPERATURE)
    decision: SupervisorDecision = llm.with_structured_output(
        SupervisorDecision
    ).invoke([SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"])

    # Handle direct response (no routing needed - e.g., greetings)
    if decision.next_agent == "none":
        response = _invoke_agent(
            supervisor_agent, SUPERVISOR_PROMPT, state["messages"], "supervisor"
        )
        return Command(goto="__end__", update={"messages": [response]})

    # Route to specialist agent and update state with extracted property name
    update = {
        "messages": [
            AIMessage(content=f"Routing to {decision.next_agent}", name="supervisor")
        ]
    }
    if decision.property_name:
        update["property_name"] = decision.property_name
        print(f"Supervisor: Extracted property_name='{decision.property_name}'")

    print(f"Supervisor: Routing to {decision.next_agent}")
    return Command[str](goto=decision.next_agent, update=update)


def transaction_history_agent_node(state: SupervisorState) -> Command:
    """Transaction history specialist - provides market data and sales history."""
    # Add property name as context if available (set by supervisor)
    property_name = state.get("property_name", "")
    context = f"Property: {property_name}" if property_name else ""
    prompt = TRANSACTION_HISTORY_AGENT_PROMPT.format(context=context)

    # Invoke agent and return Command to end
    response = _invoke_agent(
        transaction_history_agent,
        prompt,
        state["messages"],
        "transaction_history_agent",
    )
    print(f"Transaction History Agent: {response.content}")

    return Command[str](goto="__end__", update={"messages": [response]})


def property_profile_agent_node(state: SupervisorState) -> Command:
    """Property profile specialist - provides property details and features."""
    # Add property name as context if available (set by supervisor)
    property_name = state.get("property_name", "")
    context = f"Property: {property_name}" if property_name else ""
    prompt = PROPERTY_PROFILE_AGENT_PROMPT.format(context=context)

    # Invoke agent and return Command to end
    response = _invoke_agent(
        property_profile_agent, prompt, state["messages"], "property_profile_agent"
    )

    context_str = f" (Property: {property_name})" if property_name else ""
    print(f"Property Profile Agent{context_str}: {response.content}")

    return Command[str](goto="__end__", update={"messages": [response]})
