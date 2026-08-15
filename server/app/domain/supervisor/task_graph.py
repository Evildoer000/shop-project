from __future__ import annotations

from collections import deque
from typing import Any, Literal

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

HandoffStatus = Literal[
    "planned",
    "ready",
    "accepted",
    "consumed",
    "unavailable",
    "blocked",
    "superseded",
]


class AgentHandoff(BaseModel):
    """Supervisor-owned, reference-only transfer between two task nodes."""

    handoff_id: str
    handoff_type: str
    from_node_id: str
    to_node_id: str
    from_agent_id: str
    to_agent_id: str
    intent_ids: list[str] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    required: bool = True
    reason: str = ""
    status: HandoffStatus = "planned"
    sequence: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


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
    schema_version: str = "2.0"
    graph_id: str
    turn_id: str
    nodes: list[TaskGraphNode] = Field(default_factory=list)
    entry_node_ids: list[str] = Field(default_factory=list)
    terminal_node_ids: list[str] = Field(default_factory=list)
    handoffs: list[AgentHandoff] = Field(default_factory=list)
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
            for field_name in ("depends_on", "input_refs"):
                values = getattr(node, field_name)
                if old_node_id not in values:
                    continue
                updated: list[str] = []
                for dependency_id in values:
                    replacements = new_node_ids if dependency_id == old_node_id else [dependency_id]
                    for replacement in replacements:
                        if replacement not in updated:
                            updated.append(replacement)
                setattr(node, field_name, updated)
        self.sync_handoffs()

    def validate_graph(self) -> None:
        self.sync_handoffs()
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
            if len(node.input_refs) != len(set(node.input_refs)):
                errors.append(f"node {node.node_id} has duplicate optional input refs")
            unknown = set(node.depends_on) - known_ids
            if unknown:
                errors.append(f"node {node.node_id} has unknown dependencies {sorted(unknown)}")
            unknown_inputs = set(node.input_refs) - known_ids
            if unknown_inputs:
                errors.append(f"node {node.node_id} has unknown optional input refs {sorted(unknown_inputs)}")
            overlap = set(node.depends_on).intersection(node.input_refs)
            if overlap:
                errors.append(
                    f"node {node.node_id} has hard/optional dependency overlap {sorted(overlap)}"
                )
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
        indegree = {node.node_id: len(self._scheduling_dependencies(node)) for node in self.nodes}
        dependents: dict[str, list[str]] = {node.node_id: [] for node in self.nodes}
        for node in self.nodes:
            for dependency_id in self._scheduling_dependencies(node):
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
            optional_inputs = [self.require_node(node_id) for node_id in node.input_refs]
            if node.dependency_policy == "all_terminal":
                ready = all(dependency.status in TERMINAL_STATUSES for dependency in dependencies)
            else:
                ready = all(self._dependency_succeeded(dependency) for dependency in dependencies)
            if ready and all(item.status in TERMINAL_STATUSES for item in optional_inputs):
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
                or dependency.status == "skipped"
                and dependency.metadata.get("skipped_reason")
                in {"dependency_failed", "optional_branch_failed"}
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
        if blocked:
            self.sync_handoffs()
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
        dependency_ids = {
            dependency_id
            for node in self.nodes
            for dependency_id in self._scheduling_dependencies(node)
        }
        self.terminal_node_ids = [node.node_id for node in self.nodes if node.node_id not in dependency_ids]

    def sync_handoffs(self) -> None:
        """Materialize graph edges as auditable handoffs without copying payloads."""
        active_keys: set[tuple[str, str, bool]] = set()
        by_key = {
            (item.from_node_id, item.to_node_id, item.required): item
            for item in self.handoffs
            if item.status != "superseded"
        }
        next_sequence = max((item.sequence for item in self.handoffs), default=0) + 1
        for target in self.nodes:
            for source_id, required in [
                *((item, True) for item in target.depends_on),
                *((item, False) for item in target.input_refs),
            ]:
                source = self.get_node(source_id)
                if source is None:
                    continue
                key = (source_id, target.node_id, required)
                active_keys.add(key)
                handoff = by_key.get(key)
                if handoff is None:
                    handoff = AgentHandoff(
                        handoff_id=self._handoff_id(source_id, target.node_id, required),
                        handoff_type=self._handoff_type(source, target),
                        from_node_id=source_id,
                        to_node_id=target.node_id,
                        from_agent_id=source.agent_id,
                        to_agent_id=target.agent_id,
                        intent_ids=list(dict.fromkeys([*source.intent_ids, *target.intent_ids])),
                        artifact_refs=[
                            source.node_id,
                            f"evidence:{source.node_id}",
                            f"tool_calls:{source.node_id}",
                        ],
                        required=required,
                        reason=(
                            "Downstream task requires the upstream result."
                            if required
                            else "Downstream task may use this context when available."
                        ),
                        sequence=next_sequence,
                        metadata={
                            "source_capability": source.capability,
                            "target_capability": target.capability,
                        },
                    )
                    next_sequence += 1
                    self.handoffs.append(handoff)
                    by_key[key] = handoff
                self._refresh_handoff_status(handoff, source, target)

        for handoff in self.handoffs:
            key = (handoff.from_node_id, handoff.to_node_id, handoff.required)
            if handoff.status != "superseded" and key not in active_keys:
                handoff.status = "superseded"

    def handoffs_to(self, node_id: str) -> list[AgentHandoff]:
        self.sync_handoffs()
        return sorted(
            [item for item in self.handoffs if item.to_node_id == node_id and item.status != "superseded"],
            key=lambda item: item.sequence,
        )

    def complete_source_handoffs(
        self,
        node_id: str,
        *,
        evidence_refs: list[str] | None = None,
    ) -> list[AgentHandoff]:
        self.sync_handoffs()
        changed: list[AgentHandoff] = []
        source = self.require_node(node_id)
        for handoff in self.handoffs:
            if handoff.from_node_id != node_id or handoff.status == "superseded":
                continue
            before = handoff.status
            if evidence_refs:
                handoff.evidence_refs = list(
                    dict.fromkeys([*handoff.evidence_refs, *evidence_refs])
                )
            target = self.require_node(handoff.to_node_id)
            self._refresh_handoff_status(handoff, source, target)
            if handoff.status != before:
                changed.append(handoff)
        return changed

    def accept_handoffs(self, node_id: str) -> list[AgentHandoff]:
        changed: list[AgentHandoff] = []
        for handoff in self.handoffs_to(node_id):
            if handoff.status == "ready":
                handoff.status = "accepted"
                changed.append(handoff)
        return changed

    def consume_handoffs(self, node_id: str) -> list[AgentHandoff]:
        changed: list[AgentHandoff] = []
        for handoff in self.handoffs_to(node_id):
            if handoff.status == "accepted":
                handoff.status = "consumed"
                changed.append(handoff)
        return changed

    def _dependency_succeeded(self, node: TaskGraphNode) -> bool:
        if node.status == "succeeded":
            return True
        return node.status == "skipped" and node.metadata.get("skipped_reason") not in {
            "dependency_failed",
            "optional_branch_failed",
        }

    def _find_cycle(self) -> list[str]:
        dependencies = {
            node.node_id: self._scheduling_dependencies(node)
            for node in self.nodes
        }
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

    def _scheduling_dependencies(self, node: TaskGraphNode) -> list[str]:
        return list(dict.fromkeys([*node.depends_on, *node.input_refs]))

    def _handoff_id(self, source_id: str, target_id: str, required: bool) -> str:
        kind = "hard" if required else "optional"
        return f"handoff:{source_id}->{target_id}:{kind}"

    def _refresh_handoff_status(
        self,
        handoff: AgentHandoff,
        source: TaskGraphNode,
        target: TaskGraphNode,
    ) -> None:
        if handoff.status in {"accepted", "consumed", "superseded"}:
            return
        source_failed = source.status in FAILURE_STATUSES or (
            source.status == "skipped"
            and source.metadata.get("skipped_reason")
            in {"dependency_failed", "optional_branch_failed"}
        )
        if target.dependency_policy == "all_terminal" and source.status in TERMINAL_STATUSES:
            handoff.status = "ready"
            return
        if source_failed:
            handoff.status = "blocked" if handoff.required else "unavailable"
        elif target.status == "skipped":
            handoff.status = "blocked" if handoff.required else "unavailable"
        elif target.status in {"running", "succeeded", "failed", "timeout", "cancelled"}:
            handoff.status = "consumed"
        elif source.status in SUCCESS_STATUSES:
            handoff.status = "ready"
        else:
            handoff.status = "planned"

    def _handoff_type(self, source: TaskGraphNode, target: TaskGraphNode) -> str:
        if source.capability == "intent_understanding" and target.capability == "policy_gate":
            return "intent_plan_proposal"
        if source.capability == "policy_gate":
            if target.metadata.get("phase") == "knowledge_follow_up_policy_approval":
                return "policy_decision"
            if target.metadata.get("supervisor_approved"):
                return "approved_retrieval_query"
            return "approved_task"
        if source.capability == "profile_preference":
            return "profile_context"
        if target.capability == "evidence_verification":
            return "evidence_submission"
        if source.capability == "evidence_verification":
            return "verified_evidence"
        if target.capability == "repair":
            return "failure_diagnostic"
        if source.capability == "repair":
            return "repair_instruction"
        if target.capability == "answer_generation":
            return "answer_context"
        if target.capability == "memory_distillation":
            return "completed_turn"
        return "task_result"
