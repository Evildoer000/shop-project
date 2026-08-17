from __future__ import annotations

import inspect
import json
import re
from copy import deepcopy
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from app.domain.context_reference_resolver import ContextReferenceResolver
from app.domain.supervisor.capability_catalog import (
    CapabilityCatalog,
    build_default_capability_catalog,
)
from app.domain.supervisor.prompts import PromptRegistry, build_default_prompt_registry
from app.domain.supervisor.validators import (
    IntentPlanContractError,
    validate_intent_plan_contract,
)
from app.schemas import (
    AgentTaskParameters,
    AgentTaskProposal,
    IntentConstraint,
    IntentItem,
    IntentPlanV3,
    IntentProductNeed,
    IntentRouteBasis,
    IntentUncertainty,
    RecommendationPolicy,
)
from app.services.llm_client import LlmClient
from app.services.structured_llm import (
    StructuredLlmValidationError,
    parse_json_object,
)


@dataclass(frozen=True)
class PlannerStreamEvent:
    kind: str
    content: str = ""
    intent_plan: IntentPlanV3 | None = None


class IntentPlanner:
    """Turn one user message into independent intent-scoped Agent tasks."""

    JSON_RESPONSE_FORMAT = {"type": "json_object"}
    INTENT_TYPES = {
        "social_chat",
        "product_recommendation",
        "product_comparison",
        "product_qa",
        "shopping_knowledge",
    }
    PRIORITIES = {"required", "optional"}
    TRIGGER_TYPES = {
        "none",
        "explicit_web_request",
        "explicit_platform_request",
        "freshness_required",
        "knowledge_bridge",
    }
    FORBIDDEN_GLOBAL_FIELDS = {
        "execution_mode",
        "entry_mode",
        "primary_intent",
        "plan_type",
        "vector_query",
        "keyword_query",
        "budget_min",
        "budget_max",
        "budget_scope",
        "need_slots",
        "referenced_product_ids",
        "profile_lookup",
        "context_requests",
        "research_requests",
        "clarification",
        "uncertainties",
        "agent_proposals",
        "local_catalog_status",
    }

    def __init__(
        self,
        llm_client: LlmClient | None = None,
        capability_catalog: CapabilityCatalog | None = None,
        prompt_registry: PromptRegistry | None = None,
    ) -> None:
        self.llm_client = llm_client or LlmClient(component="IntentPlanner")
        self.capability_catalog = capability_catalog or build_default_capability_catalog()
        self.prompt_registry = prompt_registry or build_default_prompt_registry()
        self.context_reference_resolver = ContextReferenceResolver()
        self.last_validation_attempts: list[dict[str, Any]] = []
        self.last_normalizations: list[dict[str, Any]] = []

    async def stream_plan_with_summary(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> AsyncGenerator[PlannerStreamEvent, None]:
        self.last_validation_attempts = []
        self.last_normalizations = []
        yield PlannerStreamEvent(
            kind="summary_delta",
            content="正在识别意图、上下文引用和任务依赖。",
        )
        system_prompt = self._system_prompt(context, tagged=True)
        user_prompt = self._user_prompt(query, context)
        parser = _TaggedPlannerStreamParser()
        content_parts: list[str] = []
        async for delta in self._generate_stream_required(
            system_prompt,
            user_prompt,
            operation="intent_planner.v3.stream_plan_with_summary",
        ):
            content_parts.append(delta)
            parser.feed(delta)
        parser.finish()
        content = "".join(content_parts)
        data = parse_json_object(self._json_text_from_tagged_content(content))
        errors = (
            ["输出不是可解析的 <json> JSON object。"]
            if data is None
            else self._validate_plan_data(data, query=query, context=context)
        )
        self._record_validation_attempt(1, content, errors)
        if errors:
            yield PlannerStreamEvent(
                kind="summary_delta",
                content="初次计划未通过结构校验，正在执行一次定向修复。",
            )
            repair_prompt = self._repair_user_prompt(
                user_prompt,
                content,
                errors,
            )
            repaired_content = await self._generate_required(
                self._system_prompt(context, tagged=False),
                repair_prompt,
                operation="intent_planner.v3.stream_plan_with_summary.structure_repair",
            )
            data = parse_json_object(repaired_content)
            errors = (
                ["修复输出不是可解析的 JSON object。"]
                if data is None
                else self._validate_plan_data(data, query=query, context=context)
            )
            self._record_validation_attempt(2, repaired_content, errors)
            content = repaired_content
        if data is None or errors:
            raise StructuredLlmValidationError(
                "IntentPlanner returned invalid V3 JSON after one repair.",
                errors=errors,
                data=data,
                content=content,
            )
        summary = parser.summary.strip()
        if summary and not str(data.get("summary") or "").strip():
            data["summary"] = summary
        yield PlannerStreamEvent(
            kind="plan",
            intent_plan=self._parse_plan(query, data, context=context),
        )

    async def plan(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> IntentPlanV3:
        self.last_validation_attempts = []
        self.last_normalizations = []
        system_prompt = self._system_prompt(context, tagged=False)
        original_user_prompt = self._user_prompt(query, context)
        user_prompt = original_user_prompt
        last_data: dict[str, Any] | None = None
        last_content = ""
        last_errors: list[str] = []
        for attempt in range(1, 3):
            last_content = await self._generate_required(
                system_prompt,
                user_prompt,
                operation=(
                    "intent_planner.v3.plan"
                    if attempt == 1
                    else "intent_planner.v3.plan.structure_repair"
                ),
            )
            last_data = parse_json_object(last_content)
            last_errors = (
                ["输出不是可解析的 JSON object。"]
                if last_data is None
                else self._validate_plan_data(
                    last_data,
                    query=query,
                    context=context,
                )
            )
            self._record_validation_attempt(attempt, last_content, last_errors)
            if last_data is not None and not last_errors:
                return self._parse_plan(query, last_data, context=context)
            user_prompt = self._repair_user_prompt(
                original_user_prompt,
                last_content,
                last_errors,
            )
        raise StructuredLlmValidationError(
            "IntentPlanner returned invalid V3 JSON after one repair.",
            errors=last_errors,
            data=last_data,
            content=last_content,
        )

    def _system_prompt(self, context: dict[str, Any] | None, *, tagged: bool) -> str:
        output_rule = (
            "你必须只输出两个标签块，顺序固定：\n"
            "<summary>不超过45字的中文计划摘要</summary>\n"
            "<json>{完整 JSON object}</json>\n"
            "标签外不要输出任何文字或 Markdown。\n"
            if tagged
            else "只输出一个 JSON object，不要输出 Markdown 或解释文字。\n"
        )
        return self._versioned_system_prompt() + (
            "\n\n## IntentPlan V3 职责\n"
            "你只负责识别业务意图、整理每个意图自己的信息，并提出声明式 Agent 任务。"
            "不要回答用户，不要查商品，不要调用 Tool，不要决定最终答案。\n"
            f"{output_rule}\n"
            "## 顶层契约\n"
            "顶层只允许 schema_version、summary、intents、task_proposals。schema_version 必须是 3.0。\n"
            "严禁输出 execution_mode、entry_mode、primary_intent、plan_type、全局检索词、全局预算、"
            "need_slots、context_requests、research_requests 或全局 clarification。\n\n"
            "## 多意图拆分\n"
            "- 用户消息中每个可以独立完成、独立失败、独立回答的目标，都建立一个 intent。\n"
            "- intents 必须严格按用户表达顺序排列；最终答案会沿用此顺序。\n"
            "- 同一句话可以有任意多个独立意图。不要因为它们使用同一种 capability 就合并。\n"
            "- 每个 intent 都必须包含 intent_id、intent_type、goal、resolved_query；priority 省略时默认为 required，单一用户目标不得标成 optional。\n"
            "- 约束、上下文商品引用、商品需求和不确定性必须放在所属 intent 内，不得放到全局。\n"
            "- 金额范围只能写入 intent.budget={minimum, maximum, scope, currency}；禁止把 [100,220] 这类数字数组塞进 constraint.value。\n"
            "- 单个商品目标可有一个 product_need 或不写；真正的组合、清单、从头到脚搭配才拆多个 product_needs。\n"
            "- product_needs 只表达业务商品目标，不得生成向量查询、BM25 查询、过滤器、top_k 或其它检索实现字段。\n\n"
            "## 推荐候选策略\n"
            "- 每个 product_recommendation intent 都要填写 recommendation_policy；它只描述候选来源、历史商品角色和用户要求的最终数量，不包含检索实现。\n"
            "- candidate_source=context_only：用户明确要求只在本会话已推荐/已提到商品中筛选；必须选择可信 context_reference_keys，且不要创建商品检索任务。\n"
            "- candidate_source=context_plus_new：既要考虑上下文商品，也要补充新商品；必须创建对应商品推荐任务。\n"
            "- candidate_source=new_only：只需要新商品，或本轮没有可引用的上下文商品；创建对应商品推荐任务。\n"
            "- reference_policy=must_include：用户明确说‘算上/保留前面的商品’；eligible：历史商品与新商品一起参与筛选但不保证入选；comparison_only：历史商品只作比较背景，不进入正式推荐；没有引用时用 none。\n"
            "- requested_count 只记录用户对该 intent 的商品数量要求；‘给我两个/总共五个’用 exact，‘最多三个’用 at_most，‘至少两个’用 at_least，未说明数量则为 null + unknown。\n"
            "- ‘就在刚才推荐的商品里选两个有芦荟的’：context_only + eligible + 2 + exact，不创建 single_product_recommendation。\n"
            "- ‘算上前面两个，总共给我五个护手霜’：context_plus_new + must_include + 5 + exact，创建 single_product_recommendation。\n"
            "- ‘前面两个还可以，再给我一些符合新条件的备选’：通常 context_plus_new + eligible；只有明确说不要旧商品/只看新的时才用 new_only。\n"
            "- 只能从 context.trusted_product_references.groups 复制 context_reference_keys。严禁输出 referenced_product_ids 或按商品名编造 ID；后端会把可信 key 映射为 ID。\n\n"
            "## 任务提案\n"
            "- 每个 task_proposal 必须包含 task_id、capability、单数 intent_id、objective、depends_on、"
            "optional_upstream_task_ids、parameters、reason。\n"
            "- task_id 必须唯一。相同 capability 可以出现多次，例如两个独立知识任务必须是两个 task。\n"
            "- depends_on 只表示没有上游任务结果就不能执行的硬依赖；optional_upstream_task_ids 只表示可选上游任务结果。\n"
            "- depends_on 与 optional_upstream_task_ids 只能填写本次 task_proposals 中其他任务的 task_id；禁止填写当前任务自身。\n"
            "- context_reference_keys 只填写可信历史商品引用，例如 ['latest:item:2', 'turn:403']。"
            "optional_upstream_task_ids 的正确示例是 ['task_knowledge_001']。\n"
            "- 禁止把 latest:item:2、turn:403、product:p_xxx 或任何历史商品引用放入 optional_upstream_task_ids/depends_on；"
            "商品引用只能进入所属 intent.context_reference_keys，商品 ID 由后端可信解析，LLM 不直接生成。\n"
            "- 每个业务任务只能绑定一个 intent_id。跨意图依赖必须确实需要另一个任务的结果。严禁输出 intent_ids 数组。\n"
            "- 不得提案 EvidenceVerifier、BundleOptimizer、Repair、AnswerGenerator 或 MemoryDistillation；这些由 Supervisor 管理。\n"
            "- 不得输出 required 或 expected_output_schema；必需性由 intent.priority 和依赖关系推导。\n\n"
            "## capability 边界\n"
            "- social_chat：不提业务 Agent，Supervisor 最终直接交给 AnswerGenerator。\n"
            "- profile_preference：仅当用户明确依赖历史画像或长期偏好时提案；用户本轮说出的肤质、预算、偏好直接写入当前 intent，不需要查画像。\n"
            "- clarification：只处理自己 intent 的阻塞歧义；一个意图需要澄清不能删除其它可执行意图。\n"
            "- single_product_recommendation：一个商品目标，由 Agent 内部生成 RetrievalPlan。\n"
            "- multi_product_bundle：一个组合目标，product_needs 至少两个；各 need 后续由独立 Slot 任务检索。\n"
            "- comparison：只对已有商品和上游资料做精准对比；它可读取本地 product_detail，但不能联网或搜索平台。\n"
            "- knowledge_research：交给商品信息与知识补充 Agent；parameters.knowledge_mode 必填。knowledge_answer 直接回答知识问题，product_evidence 为推荐/对比准备资料，concept_bridge 才负责把模糊效果收敛为商品概念。\n"
            "- commerce_research：只在用户明确要求淘宝、抖音或小红书商品/口碑时提案。\n\n"
            "## 联网边界\n"
            "以下情况可以提案 knowledge_research：用户明确要求联网/查资料；明确要求最新信息；"
            "明确询问原理、成分、规格、适用范围或选购知识；要求介绍已有商品；或者只描述效果、症状、用途，无法确定可检索商品类型，需要 knowledge_bridge。\n"
            "product_qa 或商品介绍任务如果已经选择 context_reference_keys，应优先交给商品信息与知识补充 Agent 使用 product_detail；"
            "除非用户同时明确要求新商品推荐，否则不要额外创建 single_product_recommendation。\n"
            "商品类型已经明确且用户只要本地推荐时，不要为了补充常识而联网。\n"
            "只有 concept_bridge 才填写 parameters.knowledge_mode=concept_bridge 和 trigger_type=knowledge_bridge。"
            "它的知识结果返回后，同一个 IntentUnderstandingAgent 会只针对该 intent 修订路线；knowledge_answer/product_evidence 直接进入原有下游。\n"
            "用户明确指定平台时，commerce_research.parameters.platforms 只填写用户点名的平台。\n\n"
            "## 上下文与图片\n"
            "当前 query > 最近对话 > 会话摘要。只有省略式追问、明确指代或继续上一轮时才继承上下文。\n"
            "遇到‘这些、刚才的、前面两个、上一轮推荐的’或明确商品名时，只能选择 trusted_product_references 中对应的 key，再决定 recommendation_policy。\n"
            "图片只是输入模态；不要创建纯图片 Agent。商品推荐 Agent 会按请求元数据自行调用图片工具。\n\n"
            "## 典型拆分\n"
            "用户说『敏感肌买什么；喉咙不舒服买什么；下周去海边帮我从头到脚搭一套』："
            "输出三个 intents；前两个分别创建独立 knowledge_research 任务，第三个创建 multi_product_bundle 任务。"
            "三个根任务互不依赖，可并行执行。\n"
            "用户说『推荐油皮防晒，再解释为什么清爽，并和上一款比较』：可以拆推荐、知识解释、对比三个 intent；"
            "知识与对比任务只在确实需要推荐结果时依赖推荐任务。\n"
        ) + self._revision_prompt(context)

    def _revision_prompt(self, context: dict[str, Any] | None) -> str:
        if not context:
            return ""
        if context.get("intent_revision"):
            return (
                "\n\n## 单意图证据返回后的路线修订\n"
                "context.intent_revision 是 Supervisor 提供的可信内部上下文。"
                "本次只能输出 target_intent 对应的一个 intent，必须保留它的 intent_id、原始目标和硬约束。\n"
                "已完成的 knowledge_research 不得再次提案。根据知识证据，决定后续是单商品推荐、多商品组合或"
                "澄清，并至少提出一个后续任务。不得重建或修改其它 intent。\n"
            )
        if context.get("replan_attempt"):
            return (
                "\n\n## 策略拒绝后的受限重规划\n"
                "context.previous_intent_plan 和 policy_evaluation 是可信内部反馈。"
                "保留所有用户目标、顺序和硬约束，只修正被拒绝的任务、参数或依赖。"
                "独立且已合法的分支必须保留；输出完整 V3 计划，不要输出增量补丁。\n"
            )
        return ""

    def _user_prompt(self, query: str, context: dict[str, Any] | None) -> str:
        return json.dumps(
            {
                "query": query,
                "context": context or {},
                "required_output": self._output_contract(),
                "available_agent_capabilities": self.capability_catalog.describe_for_prompt(),
            },
            ensure_ascii=False,
        )

    def _output_contract(self) -> dict[str, Any]:
        schema = deepcopy(IntentPlanV3.model_json_schema(mode="validation"))
        internal_fields = {
            "original_query",
            "normalized_query",
            "trusted_context_product_ids",
            "references_resolved",
            "referenced_product_ids",
            "intent_ids",
        }

        def scrub(value: Any) -> None:
            if isinstance(value, dict):
                properties = value.get("properties")
                if isinstance(properties, dict):
                    for field in internal_fields:
                        properties.pop(field, None)
                    required = value.get("required")
                    if isinstance(required, list):
                        value["required"] = [
                            field for field in required if field not in internal_fields
                        ]
                for item in value.values():
                    scrub(item)
            elif isinstance(value, list):
                for item in value:
                    scrub(item)

        scrub(schema)
        task_schema = (schema.get("$defs") or {}).get("AgentTaskProposal")
        if isinstance(task_schema, dict):
            required = list(task_schema.get("required") or [])
            if "intent_id" not in required:
                required.append("intent_id")
            task_schema["required"] = required
        return schema

    def _legacy_output_contract(self) -> dict[str, Any]:
        return {
            "schema_version": "3.0",
            "summary": "short Chinese planning summary",
            "intents": [
                {
                    "intent_id": "i1",
                    "intent_type": "social_chat | product_recommendation | product_comparison | product_qa | shopping_knowledge",
                    "priority": "required | optional",
                    "goal": "one independently answerable business goal",
                    "resolved_query": "context-resolved natural-language goal",
                    "constraints": [
                        {
                            "name": "canonical constraint name",
                            "value": "structured value",
                            "strength": "hard | soft",
                            "source": "current_query | recent_turn | session_summary | image_inference | long_term_profile",
                            "reason": "source explanation",
                        }
                    ],
                    "referenced_product_ids": [],
                    "recommendation_policy": {
                        "candidate_source": "context_only | context_plus_new | new_only",
                        "reference_policy": "none | must_include | eligible | comparison_only",
                        "requested_count": "positive integer or null",
                        "count_mode": "exact | at_most | at_least | unknown",
                        "reason": "why these candidate and quantity rules match the user's wording",
                    },
                    "product_needs": [
                        {
                            "need_id": "n1",
                            "priority": "required | optional",
                            "goal": "business product goal",
                            "product_type": "natural product family when known",
                            "constraints": [],
                            "exclusions": [],
                        }
                    ],
                    "uncertainties": [
                        {
                            "field": "missing field",
                            "description": "what is uncertain",
                            "blocking": False,
                            "confidence": 0.8,
                        }
                    ],
                    "route_basis": {
                        "target_clarity": "not_applicable | explicit_product | context_product | vague_effect_or_use",
                        "external_information_need": "none | explicit_web | explicit_platform | freshness_required | knowledge_bridge",
                        "trigger_text": "verbatim query fragment or empty",
                        "product_family": "known product family or empty",
                        "reason": "short route basis",
                    },
                }
            ],
            "task_proposals": [
                {
                    "task_id": "t1",
                    "capability": "one available planner-proposable capability",
                    "intent_ids": ["i1"],
                    "objective": "task-specific objective",
                    "depends_on": [],
                    "optional_upstream_task_ids": [],
                    "parameters": {
                        "query": "task-specific research or lookup query",
                        "freshness": "any | recent | realtime",
                        "platforms": [],
                        "trigger_type": "none | explicit_web_request | explicit_platform_request | freshness_required | knowledge_bridge",
                        "trigger_text": "verbatim query fragment or empty",
                        "profile_usage": "intent_refinement | ranking_only | answer_personalization",
                        "missing_fields": [],
                        "question_goal": "",
                        "product_need_ids": [],
                        "referenced_product_ids": [],
                        "comparison_dimensions": [],
                    },
                    "reason": "why this task is needed",
                }
            ],
        }

    def _parse_plan(
        self,
        query: str,
        data: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> IntentPlanV3:
        payload = deepcopy(data)
        payload["original_query"] = query
        payload["normalized_query"] = query.strip()
        plan = IntentPlanV3.model_validate(payload)
        reference_context = (
            (context or {}).get("trusted_product_references")
            if isinstance(context, dict)
            else None
        )
        trusted_ids = list(
            reference_context.get("trusted_product_ids") or []
        ) if isinstance(reference_context, dict) else []
        resolved_intents: list[IntentItem] = []
        for intent in plan.intents:
            keys = list(dict.fromkeys(intent.context_reference_keys))
            if not keys:
                keys = self.context_reference_resolver.suggest_keys(
                    f"{intent.goal} {intent.resolved_query}",
                    reference_context,
                )
            referenced_ids, _ = self.context_reference_resolver.resolve_keys(
                reference_context,
                keys,
            )
            resolved_intents.append(
                intent.model_copy(
                    update={
                        "context_reference_keys": keys,
                        "referenced_product_ids": referenced_ids,
                    }
                )
            )
        return plan.model_copy(
            update={
                "intents": resolved_intents,
                "trusted_context_product_ids": trusted_ids,
                "references_resolved": True,
            }
        )

    def _parse_plan_legacy(self, query: str, data: dict[str, Any]) -> IntentPlanV3:
        tasks = [
            self._task_proposal(item, index)
            for index, item in enumerate(data.get("task_proposals") or [], start=1)
            if isinstance(item, dict)
        ]
        retrieval_intent_ids = {
            intent_id
            for task in tasks
            if task.capability
            in {"single_product_recommendation", "multi_product_bundle"}
            for intent_id in task.intent_ids
        }
        intents = [
            self._intent_item(
                item,
                index,
                has_retrieval_task=str(item.get("intent_id") or f"i{index}").strip()
                in retrieval_intent_ids,
            )
            for index, item in enumerate(data.get("intents") or [], start=1)
            if isinstance(item, dict)
        ]
        return IntentPlanV3(
            original_query=query,
            normalized_query=query.strip(),
            summary=str(data.get("summary") or "").strip(),
            intents=intents,
            task_proposals=tasks,
        )

    def _intent_item(
        self,
        value: dict[str, Any],
        index: int,
        *,
        has_retrieval_task: bool,
    ) -> IntentItem:
        constraints = self._constraints(value.get("constraints"))
        needs = [
            self._product_need(item, need_index)
            for need_index, item in enumerate(value.get("product_needs") or [], start=1)
            if isinstance(item, dict)
        ]
        uncertainties = [
            self._uncertainty(item)
            for item in value.get("uncertainties") or []
            if isinstance(item, dict)
        ]
        route = value.get("route_basis") if isinstance(value.get("route_basis"), dict) else {}
        referenced_product_ids = self._strings(value.get("referenced_product_ids"))
        return IntentItem(
            intent_id=str(value.get("intent_id") or f"i{index}").strip(),
            intent_type=str(value.get("intent_type") or "product_recommendation"),  # type: ignore[arg-type]
            priority=value.get("priority") if value.get("priority") in self.PRIORITIES else "required",
            goal=str(value.get("goal") or value.get("resolved_query") or "").strip(),
            resolved_query=str(value.get("resolved_query") or value.get("goal") or "").strip(),
            constraints=constraints,
            referenced_product_ids=referenced_product_ids,
            recommendation_policy=self._recommendation_policy(
                value.get("recommendation_policy"),
                referenced_product_ids=referenced_product_ids,
                has_retrieval_task=has_retrieval_task,
            ),
            product_needs=needs,
            uncertainties=uncertainties,
            route_basis=IntentRouteBasis(
                target_clarity=(
                    route.get("target_clarity")
                    if route.get("target_clarity")
                    in {"not_applicable", "explicit_product", "context_product", "vague_effect_or_use"}
                    else "not_applicable"
                ),
                external_information_need=(
                    route.get("external_information_need")
                    if route.get("external_information_need")
                    in {"none", "explicit_web", "explicit_platform", "freshness_required", "knowledge_bridge"}
                    else "none"
                ),
                trigger_text=str(route.get("trigger_text") or "").strip(),
                product_family=str(route.get("product_family") or "").strip(),
                reason=str(route.get("reason") or "").strip(),
            ),
        )

    def _recommendation_policy(
        self,
        value: Any,
        *,
        referenced_product_ids: list[str],
        has_retrieval_task: bool,
    ) -> RecommendationPolicy:
        raw = value if isinstance(value, dict) else {}
        candidate_source = raw.get("candidate_source")
        if candidate_source not in {"context_only", "context_plus_new", "new_only"}:
            if referenced_product_ids:
                candidate_source = (
                    "context_plus_new" if has_retrieval_task else "context_only"
                )
            else:
                candidate_source = "new_only"
        reference_policy = raw.get("reference_policy")
        if reference_policy not in {
            "none",
            "must_include",
            "eligible",
            "comparison_only",
        }:
            reference_policy = "eligible" if referenced_product_ids else "none"
        requested_count = self._positive_int_or_none(raw.get("requested_count"))
        count_mode = raw.get("count_mode")
        if count_mode not in {"exact", "at_most", "at_least", "unknown"}:
            count_mode = "exact" if requested_count is not None else "unknown"
        if requested_count is None:
            count_mode = "unknown"
        return RecommendationPolicy(
            candidate_source=candidate_source,
            reference_policy=reference_policy,
            requested_count=requested_count,
            count_mode=count_mode,
            reason=str(raw.get("reason") or "").strip(),
        )

    def _product_need(self, value: dict[str, Any], index: int) -> IntentProductNeed:
        return IntentProductNeed(
            need_id=str(value.get("need_id") or f"n{index}").strip(),
            priority=value.get("priority") if value.get("priority") in self.PRIORITIES else "required",
            goal=str(value.get("goal") or value.get("product_type") or "").strip(),
            product_type=str(value.get("product_type") or "").strip(),
            constraints=self._constraints(value.get("constraints")),
            exclusions=self._strings(value.get("exclusions")),
        )

    def _task_proposal(self, value: dict[str, Any], index: int) -> AgentTaskProposal:
        raw = value.get("parameters") if isinstance(value.get("parameters"), dict) else {}
        trigger_type = raw.get("trigger_type") if raw.get("trigger_type") in self.TRIGGER_TYPES else "none"
        return AgentTaskProposal(
            task_id=str(value.get("task_id") or f"t{index}").strip(),
            capability=str(value.get("capability") or "clarification"),  # type: ignore[arg-type]
            intent_ids=self._strings(value.get("intent_ids")),
            objective=str(value.get("objective") or value.get("reason") or "").strip(),
            reason=str(value.get("reason") or value.get("objective") or "").strip(),
            depends_on=self._strings(value.get("depends_on")),
            optional_upstream_task_ids=self._strings(
                value.get("optional_upstream_task_ids")
            ),
            parameters=AgentTaskParameters(
                query=str(raw.get("query") or "").strip(),
                freshness=(
                    raw.get("freshness")
                    if raw.get("freshness") in {"any", "recent", "realtime"}
                    else "recent"
                ),
                platforms=[
                    item
                    for item in self._strings(raw.get("platforms"))
                    if item in {"taobao", "douyin_ec", "xiaohongshu"}
                ],
                trigger_type=trigger_type,
                trigger_text=str(raw.get("trigger_text") or "").strip(),
                profile_usage=(
                    raw.get("profile_usage")
                    if raw.get("profile_usage")
                    in {"intent_refinement", "ranking_only", "answer_personalization"}
                    else "ranking_only"
                ),
                missing_fields=self._strings(raw.get("missing_fields")),
                question_goal=str(raw.get("question_goal") or "").strip(),
                product_need_ids=self._strings(raw.get("product_need_ids")),
                referenced_product_ids=self._strings(raw.get("referenced_product_ids")),
                comparison_dimensions=self._strings(raw.get("comparison_dimensions")),
            ),
        )

    def _constraints(self, value: Any) -> list[IntentConstraint]:
        if not isinstance(value, list):
            return []
        result: list[IntentConstraint] = []
        valid_sources = {
            "current_query",
            "recent_turn",
            "session_summary",
            "image_inference",
            "long_term_profile",
        }
        for item in value:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            raw_value = item.get("value")
            if not name or raw_value is None or isinstance(raw_value, dict):
                continue
            source = item.get("source") if item.get("source") in valid_sources else "current_query"
            strength = item.get("strength") if item.get("strength") in {"hard", "soft"} else "hard"
            if source == "long_term_profile":
                strength = "soft"
            result.append(
                IntentConstraint(
                    name=name,
                    value=raw_value,
                    strength=strength,
                    source=source,
                    reason=str(item.get("reason") or "").strip(),
                )
            )
        return result

    def _uncertainty(self, value: dict[str, Any]) -> IntentUncertainty:
        confidence = self._float_or_none(value.get("confidence"))
        if confidence is not None:
            confidence = min(1.0, max(0.0, confidence))
        return IntentUncertainty(
            field=str(value.get("field") or "unknown").strip(),
            description=str(value.get("description") or "信息不确定").strip(),
            blocking=self._bool(value.get("blocking")),
            confidence=confidence,
        )

    def _validate_plan_data(
        self,
        data: dict[str, Any],
        *,
        query: str = "",
        context: dict[str, Any] | None = None,
    ) -> list[str]:
        self.last_normalizations = self._normalize_trusted_dependency_references(
            data,
            context,
        )
        errors: list[str] = []
        forbidden = sorted(self.FORBIDDEN_GLOBAL_FIELDS.intersection(data))
        if forbidden:
            errors.append(f"V3 top level contains removed fields: {forbidden}")
        for index, raw_intent in enumerate(data.get("intents") or []):
            if not isinstance(raw_intent, dict):
                continue
            if raw_intent.get("referenced_product_ids"):
                errors.append(
                    f"intents[{index}].referenced_product_ids is backend-derived; "
                    "use context_reference_keys"
                )
        for index, raw_task in enumerate(data.get("task_proposals") or []):
            if not isinstance(raw_task, dict):
                continue
            if "intent_ids" in raw_task:
                errors.append(
                    f"task_proposals[{index}].intent_ids was removed; use scalar intent_id"
                )
            if "optional_context_from" in raw_task:
                errors.append(
                    f"task_proposals[{index}].optional_context_from was renamed; "
                    "use optional_upstream_task_ids"
                )
            for dependency_field in ("depends_on", "optional_upstream_task_ids"):
                dependency_values = raw_task.get(dependency_field)
                if not isinstance(dependency_values, list):
                    continue
                untrusted_references = [
                    str(value).strip()
                    for value in dependency_values
                    if self._looks_like_context_reference(value)
                ]
                if untrusted_references:
                    errors.append(
                        f"task_proposals[{index}].{dependency_field} contains values that "
                        "look like historical product references but are not trusted "
                        f"context keys: {sorted(set(untrusted_references))}; "
                        "put trusted keys in the owning intent.context_reference_keys"
                    )
            if raw_task.get("capability") == "knowledge_research":
                parameters = raw_task.get("parameters")
                if not isinstance(parameters, dict) or not parameters.get(
                    "knowledge_mode"
                ):
                    errors.append(
                        f"task_proposals[{index}].parameters.knowledge_mode is required"
                    )
        try:
            plan = self._parse_plan(query or "validation_query", data, context=context)
        except ValidationError as exc:
            errors.extend(self._pydantic_errors(exc))
            return list(dict.fromkeys(errors))

        reference_context = (
            (context or {}).get("trusted_product_references")
            if isinstance(context, dict)
            else None
        )
        for intent in plan.intents:
            _, unknown_keys = self.context_reference_resolver.resolve_keys(
                reference_context,
                intent.context_reference_keys,
            )
            if unknown_keys:
                errors.append(
                    f"intent {intent.intent_id} uses unknown context_reference_keys: "
                    f"{unknown_keys}"
                )
        try:
            validate_intent_plan_contract(plan, self.capability_catalog)
        except IntentPlanContractError as exc:
            errors.extend(exc.errors)
        return list(dict.fromkeys(errors))

    def _normalize_trusted_dependency_references(
        self,
        data: dict[str, Any],
        context: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Move only exact trusted history keys out of dependency fields.

        A historical product reference is input context for the owning business
        intent, never an edge in the Agent-step graph. Unknown values remain in
        place so normal contract validation reports them instead of guessing.
        """
        reference_context = (
            context.get("trusted_product_references")
            if isinstance(context, dict)
            else None
        )
        groups = (
            reference_context.get("groups")
            if isinstance(reference_context, dict)
            else None
        )
        trusted_keys = {
            str(group.get("key") or "").strip()
            for group in (groups or [])
            if isinstance(group, dict) and str(group.get("key") or "").strip()
        }
        if not trusted_keys:
            return []

        intents_by_id = {
            str(item.get("intent_id") or "").strip(): item
            for item in (data.get("intents") or [])
            if isinstance(item, dict) and str(item.get("intent_id") or "").strip()
        }
        corrections: list[dict[str, Any]] = []
        for task_index, raw_task in enumerate(data.get("task_proposals") or []):
            if not isinstance(raw_task, dict):
                continue
            intent_id = str(raw_task.get("intent_id") or "").strip()
            owning_intent = intents_by_id.get(intent_id)
            if owning_intent is None:
                continue
            for field_name in ("depends_on", "optional_upstream_task_ids"):
                values = raw_task.get(field_name)
                if not isinstance(values, list):
                    continue
                retained: list[Any] = []
                moved: list[str] = []
                for value in values:
                    normalized = str(value or "").strip()
                    if normalized in trusted_keys:
                        moved.append(normalized)
                    else:
                        retained.append(value)
                if not moved:
                    continue
                existing = owning_intent.get("context_reference_keys")
                if existing is not None and not isinstance(existing, list):
                    # Preserve the malformed value so Pydantic reports it; do
                    # not silently discard user/model data while normalizing.
                    continue
                raw_task[field_name] = retained
                if existing is None:
                    existing = []
                    owning_intent["context_reference_keys"] = existing
                added: list[str] = []
                for reference_key in moved:
                    if reference_key not in existing:
                        existing.append(reference_key)
                        added.append(reference_key)
                corrections.append(
                    {
                        "code": "trusted_product_reference_moved",
                        "task_index": task_index,
                        "task_id": str(raw_task.get("task_id") or ""),
                        "intent_id": intent_id,
                        "source_field": field_name,
                        "moved_reference_keys": moved,
                        "added_to_context_reference_keys": added,
                    }
                )
        return corrections

    @staticmethod
    def _looks_like_context_reference(value: Any) -> bool:
        normalized = str(value or "").strip()
        return normalized.startswith(
            (
                "latest:",
                "turn:",
                "product:",
                "category:",
                "session:",
                "previous_turn",
                "query_named_products",
            )
        )

    def _validate_plan_data_legacy(self, data: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        forbidden = sorted(self.FORBIDDEN_GLOBAL_FIELDS.intersection(data))
        if forbidden:
            errors.append(f"V3 顶层包含已删除字段: {forbidden}")
        if str(data.get("schema_version") or "") != "3.0":
            errors.append("schema_version must be 3.0")
        if not isinstance(data.get("summary"), str):
            errors.append("summary must be a string")
        intents = data.get("intents")
        tasks = data.get("task_proposals")
        if not isinstance(intents, list) or not intents:
            errors.append("intents must be a non-empty list")
            intents = []
        if not isinstance(tasks, list):
            errors.append("task_proposals must be a list")
            tasks = []

        intent_ids: list[str] = []
        product_need_ids: set[str] = set()
        intent_items_by_id: dict[str, dict[str, Any]] = {}
        for index, item in enumerate(intents):
            if not isinstance(item, dict):
                errors.append(f"intents[{index}] must be an object")
                continue
            intent_id = str(item.get("intent_id") or "").strip()
            intent_ids.append(intent_id)
            if intent_id:
                intent_items_by_id[intent_id] = item
            if not intent_id:
                errors.append(f"intents[{index}].intent_id is required")
            if item.get("intent_type") not in self.INTENT_TYPES:
                errors.append(f"intents[{index}].intent_type is invalid")
            if item.get("priority") not in self.PRIORITIES:
                errors.append(f"intents[{index}].priority is invalid")
            for field in ("goal", "resolved_query"):
                if not str(item.get(field) or "").strip():
                    errors.append(f"intents[{index}].{field} is required")
            for field in ("constraints", "referenced_product_ids", "product_needs", "uncertainties"):
                if field in item and not isinstance(item.get(field), list):
                    errors.append(f"intents[{index}].{field} must be a list")
            route = item.get("route_basis")
            if not isinstance(route, dict):
                errors.append(f"intents[{index}].route_basis must be an object")
            elif "local_catalog_status" in route:
                errors.append(f"intents[{index}].route_basis.local_catalog_status was removed in V3")
            recommendation_policy = item.get("recommendation_policy")
            if recommendation_policy is not None and not isinstance(
                recommendation_policy, dict
            ):
                errors.append(
                    f"intents[{index}].recommendation_policy must be an object"
                )
            elif isinstance(recommendation_policy, dict):
                if recommendation_policy.get("candidate_source") not in {
                    "context_only",
                    "context_plus_new",
                    "new_only",
                }:
                    errors.append(
                        f"intents[{index}].recommendation_policy.candidate_source is invalid"
                    )
                if recommendation_policy.get("reference_policy") not in {
                    "none",
                    "must_include",
                    "eligible",
                    "comparison_only",
                }:
                    errors.append(
                        f"intents[{index}].recommendation_policy.reference_policy is invalid"
                    )
                count_mode = recommendation_policy.get("count_mode")
                if count_mode not in {"exact", "at_most", "at_least", "unknown"}:
                    errors.append(
                        f"intents[{index}].recommendation_policy.count_mode is invalid"
                    )
                requested_count = recommendation_policy.get("requested_count")
                if requested_count is not None and self._positive_int_or_none(
                    requested_count
                ) is None:
                    errors.append(
                        f"intents[{index}].recommendation_policy.requested_count must be a positive integer or null"
                    )
                if count_mode != "unknown" and requested_count is None:
                    errors.append(
                        f"intents[{index}].recommendation_policy.requested_count is required for {count_mode}"
                    )
            for need_index, need in enumerate(item.get("product_needs") or []):
                if not isinstance(need, dict):
                    errors.append(f"intents[{index}].product_needs[{need_index}] must be an object")
                    continue
                need_id = str(need.get("need_id") or "").strip()
                if not need_id:
                    errors.append(f"intents[{index}].product_needs[{need_index}].need_id is required")
                elif need_id in product_need_ids:
                    errors.append(f"duplicate product need_id: {need_id}")
                product_need_ids.add(need_id)
                if not str(need.get("goal") or "").strip():
                    errors.append(f"intents[{index}].product_needs[{need_index}].goal is required")

        self._duplicates("intent_id", intent_ids, errors)
        known_intents = set(intent_ids)
        task_ids: list[str] = []
        proposable = self.capability_catalog.proposable_names()
        dependencies: dict[str, list[str]] = {}
        covered_intents: set[str] = set()
        retrieval_intents: set[str] = set()
        for index, item in enumerate(tasks):
            if not isinstance(item, dict):
                errors.append(f"task_proposals[{index}] must be an object")
                continue
            task_id = str(item.get("task_id") or "").strip()
            task_ids.append(task_id)
            if not task_id:
                errors.append(f"task_proposals[{index}].task_id is required")
            if item.get("capability") not in proposable:
                errors.append(f"task_proposals[{index}].capability is not planner-proposable")
            if not str(item.get("objective") or "").strip():
                errors.append(f"task_proposals[{index}].objective is required")
            if not str(item.get("reason") or "").strip():
                errors.append(f"task_proposals[{index}].reason is required")
            refs = item.get("intent_ids")
            if not isinstance(refs, list) or not refs:
                errors.append(f"task_proposals[{index}].intent_ids must be non-empty")
                refs = []
            unknown = {str(value) for value in refs} - known_intents
            if unknown:
                errors.append(f"task {task_id or index} references unknown intents {sorted(unknown)}")
            covered_intents.update(str(value) for value in refs)
            if item.get("capability") in {
                "single_product_recommendation",
                "multi_product_bundle",
            }:
                retrieval_intents.update(str(value) for value in refs)
            hard = item.get("depends_on") or []
            optional = item.get("optional_upstream_task_ids") or []
            if not isinstance(hard, list) or not isinstance(optional, list):
                errors.append(f"task {task_id or index} dependencies must be lists")
                hard = []
                optional = []
            if set(hard).intersection(optional):
                errors.append(f"task {task_id or index} has hard/optional dependency overlap")
            dependencies[task_id] = [str(value) for value in hard]
            parameters = item.get("parameters")
            if not isinstance(parameters, dict):
                errors.append(f"task_proposals[{index}].parameters must be an object")
            for removed in ("required", "expected_output_schema", "proposal_id"):
                if removed in item:
                    errors.append(f"task_proposals[{index}].{removed} was removed in V3")

        self._duplicates("task_id", task_ids, errors)
        known_tasks = set(task_ids)
        for task_id, dependency_ids in dependencies.items():
            unknown = set(dependency_ids) - known_tasks
            if unknown:
                errors.append(f"task {task_id} has unknown dependencies {sorted(unknown)}")
            if task_id in dependency_ids:
                errors.append(f"task {task_id} cannot depend on itself")
        cycle = self._find_cycle(dependencies)
        if cycle:
            errors.append(f"task dependency cycle detected: {' -> '.join(cycle)}")

        for intent_id, item in intent_items_by_id.items():
            if item.get("intent_type") != "product_recommendation":
                continue
            references = self._strings(item.get("referenced_product_ids"))
            raw_policy = (
                item.get("recommendation_policy")
                if isinstance(item.get("recommendation_policy"), dict)
                else {}
            )
            candidate_source = raw_policy.get("candidate_source")
            if candidate_source not in {
                "context_only",
                "context_plus_new",
                "new_only",
            }:
                candidate_source = (
                    "context_plus_new"
                    if references and intent_id in retrieval_intents
                    else "context_only"
                    if references
                    else "new_only"
                )
            reference_policy = raw_policy.get("reference_policy")
            if reference_policy not in {
                "none",
                "must_include",
                "eligible",
                "comparison_only",
            }:
                reference_policy = "eligible" if references else "none"
            if candidate_source in {"context_only", "context_plus_new"} and not references:
                errors.append(
                    f"recommendation intent {intent_id} uses {candidate_source} without referenced_product_ids"
                )
            if reference_policy != "none" and not references:
                errors.append(
                    f"recommendation intent {intent_id} uses {reference_policy} without referenced_product_ids"
                )
            if candidate_source == "context_only":
                if intent_id in retrieval_intents:
                    errors.append(
                        f"context-only recommendation intent {intent_id} must not create a product retrieval task"
                    )
                if references:
                    covered_intents.add(intent_id)

        for item in intents:
            if not isinstance(item, dict) or item.get("priority") != "required":
                continue
            if item.get("intent_type") == "social_chat":
                continue
            intent_id = str(item.get("intent_id") or "")
            if intent_id and intent_id not in covered_intents:
                errors.append(f"required intent {intent_id} has no task proposal")
        return errors

    def _pydantic_errors(self, exc: ValidationError) -> list[str]:
        result: list[str] = []
        for item in exc.errors(include_url=False):
            location = ".".join(str(value) for value in item.get("loc") or [])
            message = str(item.get("msg") or "validation failed")
            result.append(f"{location}: {message}" if location else message)
        return result

    def _record_validation_attempt(
        self,
        attempt: int,
        content: str,
        errors: list[str],
    ) -> None:
        self.last_validation_attempts.append(
            {
                "attempt": attempt,
                "valid": not errors,
                "errors": list(errors)[:20],
                "normalizations": list(self.last_normalizations),
                "output_preview": str(content or "")[:1_200],
            }
        )

    def _repair_user_prompt(
        self,
        original_user_prompt: str,
        previous_output: str,
        errors: list[str],
    ) -> str:
        previous_excerpt = self._repair_output_excerpt(previous_output)
        return (
            f"{original_user_prompt}\n\n"
            "上一次输出未通过强类型契约。请只修正结构和字段值，保留用户的全部目标、"
            "顺序、约束与合法依赖；只输出一个 JSON object，不要输出 Markdown。\n"
            "字段边界再次强调：context_reference_keys 只能放可信历史商品引用；"
            "depends_on 和 optional_upstream_task_ids 只能放本次计划中其他 task_proposals 的 task_id。"
            "不要把商品 ID 或历史引用键当作上游任务。\n"
            "校验错误：\n"
            + "\n".join(f"- {error}" for error in errors[:30])
            + "\n\n上一次输出：\n"
            + previous_excerpt
        )

    def _repair_output_excerpt(self, previous_output: str) -> str:
        """Keep all task edges visible to the repair call when possible."""
        parsed = parse_json_object(previous_output)
        if isinstance(parsed, dict):
            focused = {
                "schema_version": parsed.get("schema_version"),
                "intents": parsed.get("intents") or [],
                "task_proposals": parsed.get("task_proposals") or [],
            }
            return json.dumps(focused, ensure_ascii=False)[:12_000]
        return previous_output[:12_000]

    async def _generate_required(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        operation: str,
    ) -> str:
        call = self.llm_client.generate_required
        kwargs: dict[str, Any] = {}
        if self._supports_parameter(call, "response_format"):
            kwargs["response_format"] = self.JSON_RESPONSE_FORMAT
        if self._supports_parameter(call, "operation"):
            kwargs["operation"] = operation
        return await call(system_prompt, user_prompt, **kwargs)

    async def _generate_stream_required(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        operation: str,
    ) -> AsyncGenerator[str, None]:
        call = self.llm_client.generate_stream_required
        kwargs = {"operation": operation} if self._supports_parameter(call, "operation") else {}
        async for delta in call(system_prompt, user_prompt, **kwargs):
            yield delta

    def _versioned_system_prompt(self) -> str:
        return self.prompt_registry.require("intent_understanding_agent").render_system(
            allowed_tools=[]
        )

    def _json_text_from_tagged_content(self, content: str) -> str:
        match = re.search(r"<json>\s*(.*?)\s*</json>", content, flags=re.DOTALL | re.IGNORECASE)
        return match.group(1) if match else content

    def _duplicates(self, label: str, values: list[str], errors: list[str]) -> None:
        duplicates = sorted({value for value in values if value and values.count(value) > 1})
        if duplicates:
            errors.append(f"duplicate {label}: {duplicates}")

    def _find_cycle(self, dependencies: dict[str, list[str]]) -> list[str]:
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

    def _strings(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))

    def _float_or_none(self, value: Any) -> float | None:
        if value in {None, ""}:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _positive_int_or_none(self, value: Any) -> int | None:
        if value is None or value == "" or isinstance(value, bool):
            return None
        try:
            number = int(value)
        except (TypeError, ValueError):
            return None
        if number <= 0 or number > 50:
            return None
        if isinstance(value, float) and not value.is_integer():
            return None
        if isinstance(value, str) and not value.strip().isdigit():
            return None
        return number

    def _bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() == "true"
        return value == 1

    @staticmethod
    def _supports_parameter(callable_obj: Any, name: str) -> bool:
        try:
            signature = inspect.signature(callable_obj)
        except (TypeError, ValueError):
            return False
        return name in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )


class _TaggedPlannerStreamParser:
    SUMMARY_START = "<summary>"
    SUMMARY_END = "</summary>"
    JSON_START = "<json>"

    def __init__(self) -> None:
        self._phase = "before_summary"
        self._buffer = ""
        self._summary_parts: list[str] = []

    @property
    def summary(self) -> str:
        return "".join(self._summary_parts)

    def feed(self, chunk: str) -> list[str]:
        self._buffer += chunk
        emitted: list[str] = []
        while True:
            if self._phase == "before_summary":
                index = self._buffer.lower().find(self.SUMMARY_START)
                if index < 0:
                    self._buffer = self._buffer[-(len(self.SUMMARY_START) - 1) :]
                    break
                self._buffer = self._buffer[index + len(self.SUMMARY_START) :]
                self._phase = "in_summary"
                continue
            if self._phase == "in_summary":
                index = self._buffer.lower().find(self.SUMMARY_END)
                if index >= 0:
                    text = self._buffer[:index]
                    if text:
                        self._summary_parts.append(text)
                        emitted.append(text)
                    self._buffer = self._buffer[index + len(self.SUMMARY_END) :]
                    self._phase = "after_summary"
                    continue
                safe_len = max(0, len(self._buffer) - (len(self.SUMMARY_END) - 1))
                if safe_len:
                    text = self._buffer[:safe_len]
                    self._summary_parts.append(text)
                    emitted.append(text)
                    self._buffer = self._buffer[safe_len:]
                break
            if self._phase == "after_summary":
                index = self._buffer.lower().find(self.JSON_START)
                if index < 0:
                    self._buffer = self._buffer[-(len(self.JSON_START) - 1) :]
                    break
                self._buffer = self._buffer[index + len(self.JSON_START) :]
                self._phase = "in_json"
                break
            break
        return emitted

    def finish(self) -> None:
        if self._phase == "in_summary" and self._buffer:
            end_index = self._buffer.lower().find(self.SUMMARY_END)
            text = self._buffer[:end_index] if end_index >= 0 else self._buffer
            if text:
                self._summary_parts.append(text)
        self._buffer = ""


def extract_budget_range(text: str) -> tuple[float | None, float | None]:
    range_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|到|至|~)\s*(\d+(?:\.\d+)?)", text)
    if range_match:
        return float(range_match.group(1)), float(range_match.group(2))
    max_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:以内|以下|内|不超过)", text)
    if max_match:
        return None, float(max_match.group(1))
    min_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:以上|起)", text)
    if min_match:
        return float(min_match.group(1)), None
    return None, None
