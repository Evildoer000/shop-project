from __future__ import annotations

from typing import Any

from app.harness.tool_registry import ToolRegistry, ToolRegistration


class ToolAccessDenied(PermissionError):
    pass


class AgentToolAccess:
    """Per-agent capability boundary over the shared Harness ToolRegistry."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def registration_for(self, agent_id: str, tool_name: str, allowed_tools: tuple[str, ...]) -> ToolRegistration:
        if tool_name not in allowed_tools:
            raise ToolAccessDenied(f"Agent {agent_id} is not allowlisted for tool {tool_name}")
        registration = self.registry.registration(tool_name)
        if registration is None:
            raise ToolAccessDenied(f"Tool {tool_name} is not registered")
        return registration

    def get(self, agent_id: str, tool_name: str, allowed_tools: tuple[str, ...]) -> Any:
        return self.registration_for(agent_id, tool_name, allowed_tools).instance

    def describe_for_agent(self, agent_id: str, allowed_tools: tuple[str, ...]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for tool_name in allowed_tools:
            registration = self.registry.registration(tool_name)
            if registration is None:
                continue
            result.append(
                {
                    "name": registration.name,
                    "kind": registration.kind,
                    "description": registration.description,
                    "networked": registration.networked,
                    "sensitive": registration.sensitive,
                    "platforms": list(registration.platforms),
                    "timeout_ms": registration.timeout_ms,
                }
            )
        return result
