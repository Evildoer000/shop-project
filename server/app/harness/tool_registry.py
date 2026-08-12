from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class ToolRegistration:
    name: str
    kind: str
    owner: str
    description: str
    instance: Any
    networked: bool = False
    sensitive: bool = False
    platforms: tuple[str, ...] = ()
    timeout_ms: int = 30_000


class ToolRegistry:
    """业务原子能力注册表（ToolRegistry）。

    只登记 Orchestrator 可批准调用的 Tool，例如 ProductSearchTool、ImageSearchTool、ProfileLookupTool。
    """

    def __init__(self) -> None:
        self._items: dict[str, ToolRegistration] = {}

    def register(
        self,
        name: str,
        instance: Any,
        *,
        kind: str = "tool",
        owner: str = "Orchestrator",
        description: str = "",
        networked: bool = False,
        sensitive: bool = False,
        platforms: tuple[str, ...] = (),
        timeout_ms: int = 30_000,
    ) -> None:
        if name in self._items:
            raise ValueError(f"Tool already registered: {name}")
        self._items[name] = ToolRegistration(
            name=name,
            kind=kind,
            owner=owner,
            description=description,
            instance=instance,
            networked=networked,
            sensitive=sensitive,
            platforms=platforms,
            timeout_ms=timeout_ms,
        )

    def get(self, name: str) -> Any:
        registration = self._items.get(name)
        if registration is None:
            raise KeyError(f"Tool not registered: {name}")
        return registration.instance

    def registration(self, name: str) -> ToolRegistration | None:
        return self._items.get(name)

    def require(self, name: str, expected_type: type[T]) -> T:
        instance = self.get(name)
        if not isinstance(instance, expected_type):
            raise TypeError(f"Tool {name} is {type(instance).__name__}, expected {expected_type.__name__}")
        return instance

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "name": item.name,
                "kind": item.kind,
                "owner": item.owner,
                "description": item.description,
                "networked": item.networked,
                "sensitive": item.sensitive,
                "platforms": list(item.platforms),
                "timeout_ms": item.timeout_ms,
            }
            for item in self._items.values()
        ]
