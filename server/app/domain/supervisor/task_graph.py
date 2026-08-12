from __future__ import annotations

from collections import deque
from typing import Literal

from pydantic import BaseModel, Field


TaskNodeStatus = Literal[
    "pending",
    "running",
    "succeeded",
    "skipped",
    "failed",
    "cancelled",
    "timeout",
]
DependencyPolicy = Literal["all_succeeded", "all_terminal"]

TERMINAL_STATUSES = frozenset({"succeeded", "skipped", "failed", "cancelled", "timeout"})
SUCCESS_STATUSES = frozenset({"succeeded", "skipped"})
FAILURE_STATUSES = frozenset({"failed", "cancelled", "timeout"})


class TaskGraphNode(BaseModel):
    node_id: str
    task_id: str
    agent_id: str
    capability: str
    intent_ids: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    input_refs: list[str] = Field(default_factory=list)
    required: bool = True
    asynchronous: bool = False
    dependency_policy: DependencyPolicy = "all_succeeded"
    status: TaskNodeStatus = "pending"
    attempt: int = 1
    max_attempts: int = 1
    metadata: dict = Field(default_factory=dict)


class TaskGraph(BaseModel):
    schema_version: str = "1.0"
    graph_id: str
    turn_id: str
    nodes: list[TaskGraphNode] = Field(default_factory=list)
    entry_node_ids: list[str] = Field(default_factory=list)
    terminal_node_ids: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    def add_node(self, node: TaskGraphNode) -> None:
        if self.get_node(node.node_id) is not None:
            raise ValueError(f"Task graph node already exists: {node.node_id}")
        self.nodes.append(node)

    def get_node(self, node_id: str) -> TaskGraphNode | None:
        return next((node for node in self.nodes if node.node_id == node_id), None)

    def require_node(self, node_id: str) -> TaskGraphNode:
        node = self.get_node(node_id)
        if node is None:
            raise KeyError(f"Task graph node not found: {node_id}")
        return node

    def replace_dependency(self, old_node_id: str, new_node_ids: list[str]) -> None:
        for node in self.nodes:
            if old_node_id not in node.depends_on:
                continue
            updated: list[str] = []
            for dependency_id in node.depends_on:
                replacements = new_node_ids if dependency_id == old_node_id else [dependency_id]
                for replacement in replacements:
                    if replacement not in updated:
                        updated.append(replacement)
            node.depends_on = updated

    def validate_graph(self) -> None:
        errors: list[str] = []
        node_ids = [node.node_id for node in self.nodes]
        task_ids = [node.task_id for node in self.nodes]
        duplicates = sorted({node_id for node_id in node_ids if node_ids.count(node_id) > 1})
        if duplicates:
            errors.append(f"duplicate node_ids: {duplicates}")
        duplicate_tasks = sorted({task_id for task_id in task_ids if task_ids.count(task_id) > 1})
        if duplicate_tasks:
            errors.append(f"duplicate task_ids: {duplicate_tasks}")
        known_ids = set(node_ids)
        for node in self.nodes:
            if not node.node_id:
                errors.append("node_id cannot be empty")
            if not node.task_id:
                errors.append(f"node {node.node_id} task_id cannot be empty")
            if not node.agent_id:
                errors.append(f"node {node.node_id} agent_id cannot be empty")
            if not node.capability:
                errors.append(f"node {node.node_id} capability cannot be empty")
            if node.node_id in node.depends_on:
                errors.append(f"node {node.node_id} cannot depend on itself")
            if len(node.depends_on) != len(set(node.depends_on)):
                errors.append(f"node {node.node_id} has duplicate dependencies")
            unknown = set(node.depends_on) - known_ids
            if unknown:
                errors.append(f"node {node.node_id} has unknown dependencies {sorted(unknown)}")
            if node.attempt < 1 or node.max_attempts < node.attempt:
                errors.append(f"node {node.node_id} has invalid attempt bounds")
        cycle = self._find_cycle()
        if cycle:
            errors.append(f"task graph cycle detected: {' -> '.join(cycle)}")
        if set(self.entry_node_ids) - known_ids:
            errors.append("entry_node_ids contains unknown nodes")
        if set(self.terminal_node_ids) - known_ids:
            errors.append("terminal_node_ids contains unknown nodes")
        if errors:
            raise ValueError("Invalid task graph: " + "; ".join(errors))

    def topological_order(self) -> list[str]:
        self.validate_graph()
        indegree = {node.node_id: len(node.depends_on) for node in self.nodes}
        dependents: dict[str, list[str]] = {node.node_id: [] for node in self.nodes}
        for node in self.nodes:
            for dependency_id in node.depends_on:
                dependents[dependency_id].append(node.node_id)
        queue = deque(node_id for node_id in indegree if indegree[node_id] == 0)
        result: list[str] = []
        while queue:
            node_id = queue.popleft()
            result.append(node_id)
            for dependent_id in dependents[node_id]:
                indegree[dependent_id] -= 1
                if indegree[dependent_id] == 0:
                    queue.append(dependent_id)
        if len(result) != len(self.nodes):
            raise ValueError("Task graph is not acyclic")
        return result

    def ready_nodes(self) -> list[TaskGraphNode]:
        result: list[TaskGraphNode] = []
        for node in self.nodes:
            if node.status != "pending":
                continue
            dependencies = [self.require_node(node_id) for node_id in node.depends_on]
            if node.dependency_policy == "all_terminal":
                ready = all(dependency.status in TERMINAL_STATUSES for dependency in dependencies)
            else:
                ready = all(self._dependency_succeeded(dependency) for dependency in dependencies)
            if ready:
                result.append(node)
        return result

    def blocked_nodes(self) -> list[TaskGraphNode]:
        """Return pending nodes that can never satisfy their dependency policy."""
        result: list[TaskGraphNode] = []
        for node in self.nodes:
            if node.status != "pending" or node.dependency_policy == "all_terminal":
                continue
            dependencies = [self.require_node(node_id) for node_id in node.depends_on]
            if any(
                dependency.status in FAILURE_STATUSES
                or dependency.status == "skipped" and dependency.metadata.get("skipped_reason") == "dependency_failed"
                for dependency in dependencies
            ):
                result.append(node)
        return result

    def mark_blocked_nodes(self) -> list[str]:
        """Mark failed-dependent pending nodes as skipped and preserve the reason."""
        blocked = self.blocked_nodes()
        for node in blocked:
            node.status = "skipped"
            node.metadata = {**node.metadata, "skipped_reason": "dependency_failed"}
        return [node.node_id for node in blocked]

    def all_terminal(self, *, include_asynchronous: bool = True) -> bool:
        nodes = [
            node
            for node in self.nodes
            if include_asynchronous or not node.asynchronous
        ]
        return bool(nodes) and all(node.status in TERMINAL_STATUSES for node in nodes)

    def all_required_succeeded(self, *, include_asynchronous: bool = True) -> bool:
        nodes = [
            node
            for node in self.nodes
            if node.required and (include_asynchronous or not node.asynchronous)
        ]
        return all(self._dependency_succeeded(node) for node in nodes)

    def refresh_terminal_nodes(self) -> None:
        dependency_ids = {dependency_id for node in self.nodes for dependency_id in node.depends_on}
        self.terminal_node_ids = [node.node_id for node in self.nodes if node.node_id not in dependency_ids]

    def _dependency_succeeded(self, node: TaskGraphNode) -> bool:
        if node.status == "succeeded":
            return True
        return node.status == "skipped" and node.metadata.get("skipped_reason") != "dependency_failed"

    def _find_cycle(self) -> list[str]:
        dependencies = {node.node_id: node.depends_on for node in self.nodes}
        visiting: list[str] = []
        visited: set[str] = set()

        def visit(node_id: str) -> list[str]:
            if node_id in visiting:
                start = visiting.index(node_id)
                return visiting[start:] + [node_id]
            if node_id in visited:
                return []
            visiting.append(node_id)
            for dependency_id in dependencies.get(node_id, []):
                cycle = visit(dependency_id)
                if cycle:
                    return cycle
            visiting.pop()
            visited.add(node_id)
            return []

        for node_id in dependencies:
            cycle = visit(node_id)
            if cycle:
                return cycle
        return []
