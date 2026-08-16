from __future__ import annotations

from dataclasses import dataclass

from app.schemas import AgentCapability


@dataclass(frozen=True)
class CapabilityDefinition:
    name: AgentCapability
    description: str
    planner_proposable: bool
    evidence_producing: bool = False
    supervisor_managed: bool = False


class CapabilityCatalog:
    """Allowlist exposed to the planner and enforced by the Supervisor."""

    def __init__(self) -> None:
        self._items: dict[str, CapabilityDefinition] = {}

    def register(self, definition: CapabilityDefinition) -> None:
        if definition.name in self._items:
            raise ValueError(f"Capability already registered: {definition.name}")
        self._items[definition.name] = definition

    def get(self, name: str) -> CapabilityDefinition | None:
        return self._items.get(name)

    def require(self, name: str) -> CapabilityDefinition:
        definition = self.get(name)
        if definition is None:
            raise KeyError(f"Capability not registered: {name}")
        return definition

    def names(self) -> set[str]:
        return set(self._items)

    def proposable_names(self) -> set[str]:
        return {item.name for item in self._items.values() if item.planner_proposable}

    def evidence_capabilities(self) -> set[str]:
        return {item.name for item in self._items.values() if item.evidence_producing}

    def describe_for_prompt(self) -> list[dict[str, str]]:
        return [
            {"capability": item.name, "description": item.description}
            for item in self._items.values()
            if item.planner_proposable
        ]


def build_default_capability_catalog() -> CapabilityCatalog:
    catalog = CapabilityCatalog()
    definitions = [
        CapabilityDefinition(
            name="intent_understanding",
            description="理解当前请求并输出结构化意图提案。",
            planner_proposable=False,
            supervisor_managed=True,
        ),
        CapabilityDefinition(
            name="policy_gate",
            description="用确定性代码规则审批 Agent 提案、研究请求和动态调度。",
            planner_proposable=False,
            supervisor_managed=True,
        ),
        CapabilityDefinition(
            name="profile_preference",
            description="按需读取长期画像，并输出只能作为软偏好的个性化上下文。",
            planner_proposable=True,
        ),
        CapabilityDefinition(
            name="clarification",
            description="围绕阻塞执行的缺失条件生成一个最小澄清问题。",
            planner_proposable=True,
        ),
        CapabilityDefinition(
            name="single_product_recommendation",
            description="处理一个商品目标及其约束，可调用文本或多模态检索工具。",
            planner_proposable=True,
            evidence_producing=True,
        ),
        CapabilityDefinition(
            name="multi_product_bundle",
            description="拆分并协调多个商品槽位，产出组合候选证据。",
            planner_proposable=True,
            evidence_producing=True,
        ),
        CapabilityDefinition(
            name="slot_product_retrieval",
            description="为一个已批准的商品槽位执行独立文本或图片辅助检索。",
            planner_proposable=False,
            evidence_producing=True,
            supervisor_managed=True,
        ),
        CapabilityDefinition(
            name="commerce_research",
            description="从获批的淘宝、抖音电商和小红书接口获取外部商品与口碑证据。",
            planner_proposable=True,
            evidence_producing=True,
        ),
        CapabilityDefinition(
            name="comparison",
            description="基于已有商品和获批研究证据执行多维商品对比。",
            planner_proposable=True,
            evidence_producing=True,
        ),
        CapabilityDefinition(
            name="knowledge_research",
            description="补充本地商品详情、商品原理、成分、规格或选购知识，并保留本地与网页来源。",
            planner_proposable=True,
            evidence_producing=True,
        ),
        CapabilityDefinition(
            name="evidence_verification",
            description="审核商品和研究证据是否支撑用户需求。",
            planner_proposable=False,
            evidence_producing=True,
            supervisor_managed=True,
        ),
        CapabilityDefinition(
            name="bundle_optimization",
            description="在多商品证据通过后执行组合预算与搭配优化。",
            planner_proposable=False,
            supervisor_managed=True,
        ),
        CapabilityDefinition(
            name="repair",
            description="根据运行时失败诊断生成局部修复计划。",
            planner_proposable=False,
            supervisor_managed=True,
        ),
        CapabilityDefinition(
            name="answer_generation",
            description="将获批证据组织为最终用户回答。",
            planner_proposable=False,
            supervisor_managed=True,
        ),
        CapabilityDefinition(
            name="memory_distillation",
            description="回答完成后异步蒸馏会话摘要和长期记忆候选。",
            planner_proposable=False,
            supervisor_managed=True,
        ),
    ]
    for definition in definitions:
        catalog.register(definition)
    return catalog
