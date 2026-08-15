from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.domain.supervisor.capability_catalog import CapabilityCatalog, build_default_capability_catalog


@dataclass(frozen=True)
class AgentManifest:
    agent_id: str
    capability: str
    description: str
    allowed_tools: tuple[str, ...] = ()
    execution_modes: tuple[str, ...] = ()
    timeout_ms: int = 30_000
    max_attempts: int = 1
    priority: int = 100
    enabled: bool = True
    supervisor_managed: bool = False


@dataclass(frozen=True)
class AgentRegistration:
    manifest: AgentManifest
    instance: Any = None


class AgentRegistry:
    """Business-agent registry; atomic tools remain in Harness ToolRegistry."""

    def __init__(self, catalog: CapabilityCatalog | None = None) -> None:
        self.catalog = catalog or build_default_capability_catalog()
        self._items: dict[str, AgentRegistration] = {}

    def register(self, manifest: AgentManifest, instance: Any = None) -> None:
        if manifest.agent_id in self._items:
            raise ValueError(f"Agent already registered: {manifest.agent_id}")
        definition = self.catalog.get(manifest.capability)
        if definition is None:
            raise ValueError(f"Agent {manifest.agent_id} uses unknown capability: {manifest.capability}")
        if manifest.max_attempts < 1:
            raise ValueError("Agent max_attempts must be at least 1")
        self._items[manifest.agent_id] = AgentRegistration(manifest=manifest, instance=instance)

    def get(self, agent_id: str) -> AgentRegistration | None:
        return self._items.get(agent_id)

    def require(self, agent_id: str) -> AgentRegistration:
        registration = self.get(agent_id)
        if registration is None:
            raise KeyError(f"Agent not registered: {agent_id}")
        return registration

    def enabled_for_capability(self, capability: str) -> list[AgentRegistration]:
        return sorted(
            (
                item
                for item in self._items.values()
                if item.manifest.enabled and item.manifest.capability == capability
            ),
            key=lambda item: item.manifest.priority,
        )

    def select_for_capability(self, capability: str, *, execution_mode: str = "") -> AgentRegistration | None:
        candidates = self.enabled_for_capability(capability)
        if execution_mode:
            candidates = [
                item
                for item in candidates
                if not item.manifest.execution_modes or execution_mode in item.manifest.execution_modes
            ]
        return candidates[0] if candidates else None

    def describe(self, *, include_supervisor_managed: bool = True) -> list[dict[str, Any]]:
        registrations = sorted(self._items.values(), key=lambda item: item.manifest.agent_id)
        return [
            {
                "agent_id": item.manifest.agent_id,
                "capability": item.manifest.capability,
                "description": item.manifest.description,
                "allowed_tools": list(item.manifest.allowed_tools),
                "execution_modes": list(item.manifest.execution_modes),
                "enabled": item.manifest.enabled,
                "supervisor_managed": item.manifest.supervisor_managed,
            }
            for item in registrations
            if include_supervisor_managed or not item.manifest.supervisor_managed
        ]


def build_foundation_agent_registry(
    catalog: CapabilityCatalog | None = None,
) -> AgentRegistry:
    resolved_catalog = catalog or build_default_capability_catalog()
    registry = AgentRegistry(resolved_catalog)
    manifests = [
        AgentManifest(
            agent_id="intent_understanding_agent",
            capability="intent_understanding",
            description="强制解析用户意图、约束、上下文引用和候选 Agent 提案。",
            execution_modes=("direct", "clarify", "context_evidence", "single_product", "multi_product"),
            supervisor_managed=True,
        ),
        AgentManifest(
            agent_id="supervisor_policy_gate",
            capability="policy_gate",
            description="确定性代码策略门：校验计划契约、Agent 路由和 Tool 授权，不调用 LLM。",
            execution_modes=("direct", "clarify", "context_evidence", "single_product", "multi_product"),
            supervisor_managed=True,
        ),
        AgentManifest(
            agent_id="profile_preference_agent",
            capability="profile_preference",
            description="按 Supervisor 批准读取长期画像和行为偏好。",
            allowed_tools=("profile_lookup",),
            execution_modes=("single_product", "multi_product", "context_evidence"),
        ),
        AgentManifest(
            agent_id="clarification_agent",
            capability="clarification",
            description="生成阻塞执行的最小澄清问题。",
            execution_modes=("clarify",),
        ),
        AgentManifest(
            agent_id="single_product_recommendation_agent",
            capability="single_product_recommendation",
            description="执行单商品目标的文本、关键词和图片辅助检索。",
            allowed_tools=("product_search", "image_search", "image_understanding", "product_detail"),
            execution_modes=("single_product",),
            max_attempts=3,
        ),
        AgentManifest(
            agent_id="multi_product_bundle_agent",
            capability="multi_product_bundle",
            description="并行执行多个商品 Slot，并形成组合证据。",
            allowed_tools=("product_search", "image_search", "image_understanding", "product_detail"),
            execution_modes=("multi_product",),
            max_attempts=3,
        ),
        AgentManifest(
            agent_id="slot_retrieval_agent",
            capability="slot_product_retrieval",
            description="只处理一个商品 Slot 的独立检索，供多商品路线并行调度。",
            allowed_tools=("product_search", "image_search", "image_understanding", "product_detail"),
            execution_modes=("multi_product",),
            max_attempts=3,
            supervisor_managed=True,
        ),
        AgentManifest(
            agent_id="commerce_research_agent",
            capability="commerce_research",
            description="仅查询淘宝、抖音电商和小红书，输出带来源的外部商品与口碑证据。",
            allowed_tools=("commerce_search", "commerce_product_detail", "commerce_reviews"),
            execution_modes=("context_evidence", "single_product", "multi_product"),
            max_attempts=2,
        ),
        AgentManifest(
            agent_id="comparison_agent",
            capability="comparison",
            description="基于商品证据和获批外部研究执行多维对比。",
            allowed_tools=("web_search", "product_detail"),
            execution_modes=("context_evidence", "single_product", "multi_product"),
            max_attempts=3,
        ),
        AgentManifest(
            agent_id="knowledge_research_agent",
            capability="knowledge_research",
            description="解释商品原理、成分、规格和选购知识。",
            allowed_tools=("web_search", "product_detail"),
            execution_modes=("context_evidence", "single_product", "multi_product"),
            max_attempts=3,
        ),
        AgentManifest(
            agent_id="evidence_verifier_agent",
            capability="evidence_verification",
            description="校验商品和研究证据是否支撑用户需求。",
            max_attempts=3,
            supervisor_managed=True,
        ),
        AgentManifest(
            agent_id="bundle_optimizer",
            capability="bundle_optimization",
            description="多商品证据通过后进行预算和搭配优化。",
            execution_modes=("multi_product",),
            supervisor_managed=True,
        ),
        AgentManifest(
            agent_id="repair_agent",
            capability="repair",
            description="按失败节点生成局部修复计划。",
            supervisor_managed=True,
        ),
        AgentManifest(
            agent_id="answer_generator",
            capability="answer_generation",
            description="生成最终客户端回答。",
            supervisor_managed=True,
        ),
        AgentManifest(
            agent_id="memory_distillation_agent",
            capability="memory_distillation",
            description="回答结束后异步蒸馏会话摘要和记忆候选。",
            supervisor_managed=True,
        ),
    ]
    for manifest in manifests:
        registry.register(manifest)
    return registry
