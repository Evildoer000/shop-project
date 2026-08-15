from __future__ import annotations

import json
import inspect
import re
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from app.domain.supervisor.capability_catalog import CapabilityCatalog, build_default_capability_catalog
from app.domain.supervisor.prompts import PromptRegistry, build_default_prompt_registry
from app.schemas import (
    AgentTaskProposal,
    ClarificationProposal,
    ContextRequest,
    IntentConstraint,
    IntentConstraintSet,
    IntentItem,
    IntentPlan,
    IntentQueryRewrite,
    IntentRouteBasis,
    IntentUncertainty,
    ProfileLookupProposal,
    ResearchRequest,
    RewriteNeedSlot,
)
from app.services.llm_client import LlmClient
from app.services.structured_llm import StructuredLlmValidationError, generate_validated_json, parse_json_object


@dataclass(frozen=True)
class PlannerStreamEvent:
    kind: str
    content: str = ""
    intent_plan: IntentPlan | None = None


class IntentPlanner:
    JSON_RESPONSE_FORMAT = {"type": "json_object"}
    PLAN_TYPES = {"direct_answer", "clarify", "single_retrieval", "multi_retrieval"}
    INTENT_TYPES = {
        "social_chat",
        "product_recommendation",
        "product_comparison",
        "product_qa",
        "shopping_knowledge",
        "cart_action",
    }
    EXECUTION_MODES = {"direct", "clarify", "context_evidence", "single_product", "multi_product"}
    INPUT_MODALITIES = {"text", "image", "audio"}
    RESEARCH_CONSUMERS = {"knowledge_research", "comparison", "commerce_research"}
    RESEARCH_TRIGGER_TYPES = {
        "explicit_web_request",
        "explicit_platform_request",
        "freshness_required",
        "knowledge_bridge",
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

    async def stream_plan_with_summary(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> AsyncGenerator[PlannerStreamEvent, None]:
        system_prompt = self._versioned_system_prompt() + (
            "你是电商 RAG Harness 的 IntentPlanner（意图规划智能体）。你的职责只是给 Orchestrator 提出"
            "最小声明式计划；不要回答用户，不要查商品，不要读取数据库，不要决定 final_route，不要编造商品事实。\n\n"
            "## 流式输出契约\n"
            "你必须只输出两个标签块，且顺序固定：\n"
            "<summary>一句中文用户可见计划意图解释，不复述用户原话，不包含 original_query/vector_query/keyword_query/product_id，不超过 45 字。</summary>\n"
            "<json>{完整 IntentPlan JSON object}</json>\n"
            "JSON object 必须包含 summary 字段，且含义与 summary 标签一致。除这两个标签块外不要输出任何文字或 Markdown。\n\n"
            "## plan_type 取值\n"
            "- direct_answer：当前问题不需要商品库证据，例如询问助手身份、系统能力、问候、感谢、简单常识或闲聊。\n"
            "- clarify：用户有推荐/购物意图，但当前没有可执行的商品目标，例如「推荐点东西吧」。\n"
            "- single_retrieval：当前只有一个商品/检索目标。\n"
            "- multi_retrieval：当前有多个商品需求、套装、清单、搭配、配齐、组合或总预算组合。\n\n"
            "## JSON 字段契约\n"
            "- 必填字段只有 summary 和 plan_type。\n"
            "- 检索计划可输出 vector_query / keyword_query；多需求计划应输出 need_slots；用户明确预算时才输出 budget_min / budget_max / budget_scope；"
            "只有需要画像时才输出 profile_lookup；只有当前问题明确引用上下文商品时才输出 referenced_product_ids。\n"
            "- direct_answer 和 clarify 不要输出 vector_query、keyword_query、need_slots。\n\n"
            "## 画像读取规则\n"
            "- 当前 query 出现「按我的」「根据我」「我的肤质」「我的偏好」「我平时」「适合我」等明确依赖个人历史或画像的信息时，应输出 profile_lookup.requested=true。\n"
            "- 当前 query 自己已经说清约束，例如「敏感肌」「油皮」「预算300」「不要酒精」，这些约束仍写入检索计划；profile_memory 只能补充软偏好，不能覆盖当前 query。\n\n"
            "## 最重要的规则：当前 query 永远优先\n"
            "- 上下文优先级：当前 query > recent_turns(n_turns_ago=0 即上一轮) > recent_turns(更早) > pending_summary_turns > session_summary > profile_memory。\n"
            "- recent_turns 每条带 n_turns_ago 字段：0=上一轮，1=两轮前，依次类推。\n"
            "- 当用户用「上次/刚才/上一轮/上面那个」等**明确锚定最近一轮**的词时，优先绑 n_turns_ago=0。\n"
            "- 当用户说「最早/最开始/第一次问的/一开始」时，应绑 n_turns_ago 最大的同类轮次或读 session_summary 里的 [t#: pid...] 锚点；不要默认绑最近一轮的同类商品。\n"
            "- 当用户说「那个/它/这个」等指代词时：(a) 当前 query 同时给出修饰词(颜色/尺寸/价格/型号/商品类型)，可以按修饰词在 recent_turns 找匹配商品并填 referenced_product_ids；(b) 当前轮上传了图片(context.image_attributes 存在)，`这个`通常指当前图，`那个`通常指 n_turns_ago=0，可以填 referenced_product_ids；(c) 既无修饰词又无图片，且 recent_turns 里有 ≥2 类商品时，必须输出 clarify，不要强行绑。\n"
            "- session_summary 末尾可能有 [t#: pid1, pid2; t#(img): pid3] 这样的结构化锚点，记录早期已被压缩轮次的 product_ids。当当前 query 指代这些早期轮次时，referenced_product_ids 应来自这些锚点。\n"
            "- 当前 query 能独立表达意图时，必须按当前 query 规划，不能机械继承上一轮商品目标、预算、need_slots 或 product_ids。\n"
            "- 当前 query 是身份/能力/问候/感谢/闲聊/知识问答时，必须输出 plan_type=direct_answer；即使 recent_turns 有商品推荐结果，也不能继承上一轮商品计划。\n"
            "- 只有当前 query 是省略式追问、指代、收窄筛选、续问或明确补充上一轮需求时，才允许继承 recent_turns / pending_summary_turns。\n\n"
            "## 逐词验证规则\n"
            "- vector_query 和 keyword_query 中的每个实义词，都必须能在当前 query 中找到依据。\n"
            "- 只有省略式追问、指代、收窄或续问时，才允许从 recent_turns / pending_summary_turns 继承商品范围、品牌、预算、排除项。\n"
            "- profile_memory 只能作为软偏好，不得替代当前检索目标，不得强行塞进召回 query。\n\n"
            "## 图片属性上下文规则\n"
            "- context.image_attributes 是图片属性理解服务（ImageAttributeExtractor）给出的视觉语义推测，不是商品事实源。\n"
            "- 当前 query 仍然优先；如果 query 和 image_attributes 冲突，以当前 query 为准，并把图片属性当软补充。\n"
            "- image_attributes 可用于补充商品目标、颜色、风格、材质和场景；不确定内容不要变成硬约束。\n"
            "- 当用户上传图片并有商品推荐/找相似意图时，vector_query / keyword_query 可以合并当前 query 与 image_attributes.retrieval_query 的视觉词。\n\n"
            "## 上下文商品复用规则\n"
            "- 如果当前 query 是针对 recent_turns 中上一轮商品的追问，并且 recent_turns 里有 product_ids，可以输出 plan_type=direct_answer，并填写 referenced_product_ids。\n"
            "- referenced_product_ids 只是 Planner proposal；你不能查商品详情，不能决定最终跳过检索，也不能直接回答商品事实。\n"
            "- 如果上下文商品不足、用户提出新商品目标、需要找新候选，或当前 query 不是针对上一轮商品的追问，不要填写 referenced_product_ids。\n\n"
            "## 检索和多需求规则\n"
            "- 只要当前 query 有粗品类、上位商品词、使用场景或用途线索，就应该先提案检索，而不是直接 clarify。\n"
            "- 粗品类检索不要强行收窄到单一子类：vector_query / keyword_query 应保留粗品类词和场景词，让 Retrieval Worker 召回多个相关子类。\n"
            "- 新手、入门、适合春天、通勤、旅行、预算、轻薄、好看等词本身只是约束，不是多需求触发词；只有目标天然需要多个可购买商品共同完成时，才拆 need_slots。\n"
            "- 单一商品目标 + 多个约束仍是 single_retrieval，例如「推荐适合新手的电脑」「推荐春天穿的外套」「预算200的油皮洗面奶」。\n"
            "- 组合任务/生活任务/搭配任务/装备清单应输出 multi_retrieval，例如「春天穿搭套装」「新手化妆从0开始需要哪些」「开学数码装备清单」「露营装备配齐」「办公桌搭配齐」。\n"
            "- 「几件衣服」「几款护肤品」「推荐一些装备」这类泛数量表达不等于多槽；如果没有明确搭配/配齐/清单/需要哪些，优先 single_retrieval，并让 Retrieval Worker 召回多个子类。\n"
            "- 套装边界：如果用户是在找现成售卖的单一套装 SKU，例如「某品牌礼盒套装」「旅行装套盒」「XX三件套同款套装商品」，且没有帮我配/安排/从0开始/需要哪些/总预算等搭配信号，可以按 single_retrieval。\n"
            "- 如果「套装」修饰的是上位品类、生活任务或搭配目标，而不是明确品牌/SKU，例如「化妆品套装」「彩妆套装」「春天穿搭套装」「露营装备套装」，应按多商品组合任务处理。\n"
            "- 只有「装备」或场景词并不等于多槽；「户外露营装备推荐下」「通勤装备推荐」这类泛场景推荐，优先 single_retrieval，并保留粗场景词让 Retrieval Worker 召回库内多个相关子类。不要擅自补帐篷、睡袋、防潮垫、露营灯等用户没说且商品库未必覆盖的硬 slot。\n"
            "- 场景词不能直接变成 required slot：拍照、露营、徒步、通勤、健身等通常写入 soft_constraints；除非用户明确说要相机、帐篷、睡袋、鞋、裤、包等商品，否则不要把这些推断品类设为 required。\n"
            "- multi_retrieval 的每个 slot 必须是可独立检索的原子商品或商品子类；不要把「新手化妆品套装」「春天穿搭套装」「开学装备」这类组合目标本身当成一个 slot。\n"
            "- need_type 边界：用户明确点名的商品或组合任务最小核心商品可以 required；Planner 自己推断的补充件、配饰、拍照/露营/通勤衍生件应 optional，缺失 optional 不能影响推荐主路线。\n"
            "- 组合任务拆 slot 时，把人群、季节、场景、预算、风格写入各 slot 的 soft_constraints 或 query；不要因为这些约束额外创造一个 slot。\n"
            "- 判定示例：\n"
            "  - 「推荐适合新手的电脑」=> single_retrieval，电脑是单一商品，新手是约束。\n"
            "  - 「推荐一件适合春天穿的外套」=> single_retrieval，外套是单一商品，春天是约束。\n"
            "  - 「根据我平时偏好，选几件日常通勤穿的衣服」=> single_retrieval，衣服是粗品类，通勤和偏好是约束。\n"
            "  - 「适合春天的穿搭套装」=> multi_retrieval，拆上装/下装/鞋或配饰等原子商品。\n"
            "  - 「新手化妆品套装」=> multi_retrieval，拆底妆/定妆/眉妆/唇妆等基础彩妆 slot，而不是一个“化妆品套装” slot。\n"
            "  - 「户外露营装备推荐下」=> single_retrieval，保留户外露营装备粗场景，不擅自拆成帐篷/睡袋等硬 slot。\n"
            "  - 「露营拍照和徒步都要用，推荐一套轻量户外装备」=> 不要拆帐篷/相机；可以 single_retrieval 保留轻量户外装备场景，或拆徒步鞋/户外裤/背包/帽子等可穿戴随身 slot。\n"
            "  - 「健身房训练用的整套装备」=> multi_retrieval 时拆训练上衣/训练裤/训练鞋，配件只能 optional。\n"
            "  - 「开学数码装备清单，帮我配齐」=> multi_retrieval，拆电脑/耳机/充电配件等原子商品。\n"
            "- 多个并列商品、套装、一套、组合、装备清单、配齐、帮我配、总预算组合等，通常是 multi_retrieval。\n"
            "- 单一商品加多个约束不是多需求，例如「预算3500的安卓平板，轻薄」仍是 single_retrieval。\n"
            "- multi_retrieval 的 need_slots 必须是当前轮生效后的完整 slot plan，不是增量片段。\n"
            "- 不要输出 product_type/categories/preferences/exclusions；这些属于下游 RetrievalPlanBuilder / Tool 的内部解析。\n"
        )
        system_prompt += self._expanded_contract_prompt()
        user_prompt = json.dumps(
            {
                "query": query,
                "context": context or {},
                "required_output": {
                    "summary": "short Chinese client-visible planning summary",
                    "plan_type": "direct_answer | clarify | single_retrieval | multi_retrieval",
                },
                "optional_output": {
                    "plan_reason": "short Chinese reason",
                    "vector_query": "semantic retrieval query; omit when direct_answer/clarify",
                    "keyword_query": "keyword retrieval query; omit when direct_answer/clarify",
                    "budget_min": "number; omit when unknown",
                    "budget_max": "number; omit when unknown",
                    "budget_scope": "per_item | total | unknown; omit when unknown",
                    "need_slots": [
                        {
                            "slot_id": "s1",
                            "need_type": "required | optional",
                            "goal": "slot goal",
                            "product_type": "natural product name for this slot",
                            "query": "independent slot query",
                            "soft_constraints": ["slot-level soft constraints"],
                            "exclude_terms": ["slot-level exclusions"],
                            "min_candidates": 1,
                        }
                    ],
                    "referenced_product_ids": ["product IDs from recent turns; omit when empty"],
                    "profile_lookup": {"requested": True, "query": "lookup query", "reason": "why profile lookup is useful"},
                },
                "expanded_output_contract": self._expanded_output_contract(),
                "available_agent_capabilities": self.capability_catalog.describe_for_prompt(),
            },
            ensure_ascii=False,
        )
        parser = _TaggedPlannerStreamParser()
        content_parts: list[str] = []
        async for delta in self._generate_stream_required(
            system_prompt,
            user_prompt,
            operation="intent_planner.stream_plan_with_summary",
        ):
            content_parts.append(delta)
            for summary_delta in parser.feed(delta):
                if summary_delta:
                    yield PlannerStreamEvent(kind="summary_delta", content=summary_delta)
        parser.finish()
        content = "".join(content_parts)
        data = parse_json_object(self._json_text_from_tagged_content(content))
        if data is None:
            raise StructuredLlmValidationError(
                "IntentPlanner returned invalid tagged JSON.",
                errors=["输出不是可解析的 <json> JSON object。"],
                data=None,
                content=content,
            )
        summary = parser.summary.strip()
        if summary and not str(data.get("summary") or "").strip():
            data["summary"] = summary
        errors = self._validate_plan_data(data)
        if errors:
            raise StructuredLlmValidationError(
                "IntentPlanner returned invalid tagged JSON.",
                errors=errors,
                data=data,
                content=content,
            )
        yield PlannerStreamEvent(kind="plan", intent_plan=self._parse_plan(query, data, context))

    async def plan(self, query: str, context: dict[str, Any] | None = None) -> IntentPlan:
        system_prompt = self._versioned_system_prompt() + (
            "你是电商 RAG Harness 的 IntentPlanner（意图规划智能体）。只输出 JSON object，不要输出 Markdown 或解释文字。\n"
            "你的职责只是给 Orchestrator 提出最小声明式计划；不要回答用户，不要查商品，不要读取数据库，"
            "不要决定 final_route，不要编造商品事实。\n\n"
            "## plan_type 取值\n"
            "- direct_answer：当前问题不需要商品库证据，例如询问助手身份、系统能力、问候、感谢、简单常识或闲聊。\n"
            "- clarify：用户有推荐/购物意图，但当前没有可执行的商品目标，例如「推荐点东西吧」。\n"
            "- single_retrieval：当前只有一个商品/检索目标。\n"
            "- multi_retrieval：当前有多个商品需求、套装、清单、搭配、配齐、组合或总预算组合。\n\n"
            "## 输出契约\n"
            "- 必填字段只有 plan_type。\n"
            "- 其它字段为空或未知时可以省略；Orchestrator 会用默认值补齐。\n"
            "- 检索计划可输出 vector_query / keyword_query；多需求计划应输出 need_slots；用户明确预算时才输出 budget_min / budget_max / budget_scope；"
            "只有需要画像时才输出 profile_lookup；只有当前问题明确引用上下文商品时才输出 referenced_product_ids。\n\n"
            "## 画像读取规则\n"
            "- 当前 query 出现「按我的」「根据我」「我的肤质」「我的偏好」「我平时」「适合我」等明确依赖个人历史或画像的信息时，应输出 profile_lookup.requested=true。\n"
            "- 当前 query 自己已经说清约束，例如「敏感肌」「油皮」「预算300」「不要酒精」，这些约束仍写入检索计划；profile_memory 只能补充软偏好，不能覆盖当前 query。\n\n"

            "## 字段含义\n"
            "- plan_type：给 Orchestrator 的执行路径提案，不是最终业务路线 final_route。\n"
            "- vector_query：给向量检索用的语义检索文本，用自然语言表达当前用户的商品目标和明确约束。\n"
            "- keyword_query：给 BM25/关键词检索用的词面检索文本，使用简洁商品词、数字、品牌、材质、场景等。\n"
            "- budget_min / budget_max：用户明确说出的价格下限/上限；不要推测没说出的预算。\n"
            "- budget_scope：per_item 表示每件/单品预算，total 表示多个商品共享总预算，unknown 表示没说清或未提预算。\n"
            "- need_slots：只用于 multi_retrieval。每个 slot 是 Retrieval Worker 要独立检索的一个商品需求。\n"
            "- referenced_product_ids：当前问题明确引用 recent_turns 中已有商品时，填对应 product_ids；不要自己创造 ID。\n"
            "- profile_lookup：读取长期画像的提案，只在用户明确依赖个人偏好/肤质/口味/历史偏好时使用。\n"
            "- plan_reason：用中文简短说明为什么提出这个 plan_type。\n\n"
            "## 最重要的规则：当前 query 永远优先\n"
            "- 上下文优先级：当前 query > recent_turns(n_turns_ago=0 即上一轮) > recent_turns(更早) > pending_summary_turns > session_summary > profile_memory。\n"
            "- recent_turns 每条带 n_turns_ago 字段：0=上一轮，1=两轮前，依次类推。\n"
            "- 当用户用「上次/刚才/上一轮/上面那个」等**明确锚定最近一轮**的词时，优先绑 n_turns_ago=0。\n"
            "- 当用户说「最早/最开始/第一次问的/一开始」时，应绑 n_turns_ago 最大的同类轮次或读 session_summary 里的 [t#: pid...] 锚点；不要默认绑最近一轮的同类商品。\n"
            "- 当用户说「那个/它/这个」等指代词时：(a) 当前 query 同时给出修饰词(颜色/尺寸/价格/型号/商品类型)，可以按修饰词在 recent_turns 找匹配商品并填 referenced_product_ids；(b) 当前轮上传了图片(context.image_attributes 存在)，`这个`通常指当前图，`那个`通常指 n_turns_ago=0，可以填 referenced_product_ids；(c) 既无修饰词又无图片，且 recent_turns 里有 ≥2 类商品时，必须输出 clarify，不要强行绑。\n"
            "- session_summary 末尾可能有 [t#: pid1, pid2; t#(img): pid3] 这样的结构化锚点，记录早期已被压缩轮次的 product_ids。当当前 query 指代这些早期轮次时，referenced_product_ids 应来自这些锚点。\n"
            "- 当前 query 能独立表达意图时，必须按当前 query 规划，不能机械继承上一轮商品目标、预算、need_slots 或 product_ids。\n"
            "- 当前 query 是身份/能力/问候/感谢/闲聊/知识问答时，必须输出 plan_type=direct_answer；即使 recent_turns 有商品推荐结果，也不能继承上一轮商品计划。\n"
            "- 当前 query 出现新品牌、新品类、新目标或独立问题时，不继承上一轮目标。\n"
            "- 只有当前 query 是省略式追问、指代、收窄筛选、续问或明确补充上一轮需求时，才允许继承 recent_turns / pending_summary_turns。\n"
            "- 不要把 recent_turns.rewrite.need_slots 直接复制为当前 need_slots；只有当前 query 明确继续上一轮组合任务时，才继承并按本轮新增约束改写完整 slots。\n\n"
            "## 逐词验证规则\n"
            "- vector_query 和 keyword_query 中的每个实义词，都必须能在当前 query 中找到依据。\n"
            "- 只有省略式追问、指代、收窄或续问时，才允许从 recent_turns / pending_summary_turns 继承商品范围、品牌、预算、排除项。\n"
            "- profile_memory 只能作为软偏好，不得替代当前检索目标，不得强行塞进召回 query。\n"
            "- 输出前检查：这个词来自当前 query，还是来自被允许继承的上下文？两者都不是就不要写。\n\n"
            "## 图片属性上下文规则\n"
            "- context.image_attributes 是图片属性理解服务（ImageAttributeExtractor）给出的视觉语义推测，不是商品事实源。\n"
            "- 当前 query 仍然优先；如果 query 和 image_attributes 冲突，以当前 query 为准，并把图片属性当软补充。\n"
            "- image_attributes 可用于补充商品目标、颜色、风格、材质和场景；不确定内容不要变成硬约束。\n"
            "- 当用户上传图片并有商品推荐/找相似意图时，vector_query / keyword_query 可以合并当前 query 与 image_attributes.retrieval_query 的视觉词。\n\n"
            "## 上下文商品复用规则\n"
            "- 如果当前 query 是针对 recent_turns 中上一轮商品的追问，例如「这两个哪个更适合跑步」「哪个更便宜」「把刚才那两个做个表格」，"
            "并且 recent_turns 里有 product_ids，可以输出 plan_type=direct_answer，并填写 referenced_product_ids。\n"
            "- referenced_product_ids 只是 Planner proposal；你不能查商品详情，不能决定最终跳过检索，也不能直接回答商品事实。\n"
            "- 如果上下文商品不足、用户提出新商品目标、需要找新候选，或当前 query 不是针对上一轮商品的追问，不要填写 referenced_product_ids。\n\n"
            "## direct_answer / clarify 规则\n"
            "- direct_answer 和 clarify 不要输出 vector_query、keyword_query、need_slots。\n"
            "- 用户问「你是谁」「你能做什么」「你好」「谢谢」时，输出 direct_answer。\n"
            "- 用户只说「推荐点东西吧」「帮我买点东西」但没有商品目标时，输出 clarify。\n\n"
            "## 检索和多需求规则\n"
            "- single_retrieval 的 vector_query 或 keyword_query 应包含当前用户的商品/检索目标；如果省略，Orchestrator 会回退用原始 query。\n"
            "- 只要当前 query 有粗品类、上位商品词、使用场景或用途线索，就应该先提案检索，而不是直接 clarify；"
            "例如「护肤品」「衣服」「鞋子」「裤子」「饮料」「零食」「电脑」「手机」「户外装备」「化妆套装」都属于可检索目标。\n"
            "- 粗品类检索不要强行收窄到单一子类：vector_query / keyword_query 应保留粗品类词和场景词，让 Retrieval Worker 召回多个相关子类，再由 CorrectiveAgent 审核证据。\n"
            "- 只有完全没有商品目标、品类线索、使用场景、用途或约束时，才输出 clarify。\n"
            "- 新手、入门、适合春天、通勤、旅行、预算、轻薄、好看等词本身只是约束，不是多需求触发词；只有目标天然需要多个可购买商品共同完成时，才拆 need_slots。\n"
            "- 单一商品目标 + 多个约束仍是 single_retrieval，例如「推荐适合新手的电脑」「推荐春天穿的外套」「预算200的油皮洗面奶」。\n"
            "- 组合任务/生活任务/搭配任务/装备清单应输出 multi_retrieval，例如「春天穿搭套装」「新手化妆从0开始需要哪些」「开学数码装备清单」「露营装备配齐」「办公桌搭配齐」。\n"
            "- 「几件衣服」「几款护肤品」「推荐一些装备」这类泛数量表达不等于多槽；如果没有明确搭配/配齐/清单/需要哪些，优先 single_retrieval，并让 Retrieval Worker 召回多个子类。\n"
            "- 套装边界：如果用户是在找现成售卖的单一套装 SKU，例如「某品牌礼盒套装」「旅行装套盒」「XX三件套同款套装商品」，且没有帮我配/安排/从0开始/需要哪些/总预算等搭配信号，可以按 single_retrieval。\n"
            "- 如果「套装」修饰的是上位品类、生活任务或搭配目标，而不是明确品牌/SKU，例如「化妆品套装」「彩妆套装」「春天穿搭套装」「露营装备套装」，应按多商品组合任务处理。\n"
            "- 只有「装备」或场景词并不等于多槽；「户外露营装备推荐下」「通勤装备推荐」这类泛场景推荐，优先 single_retrieval，并保留粗场景词让 Retrieval Worker 召回库内多个相关子类。不要擅自补帐篷、睡袋、防潮垫、露营灯等用户没说且商品库未必覆盖的硬 slot。\n"
            "- 场景词不能直接变成 required slot：拍照、露营、徒步、通勤、健身等通常写入 soft_constraints；除非用户明确说要相机、帐篷、睡袋、鞋、裤、包等商品，否则不要把这些推断品类设为 required。\n"
            "- multi_retrieval 的每个 slot 必须是可独立检索的原子商品或商品子类；不要把「新手化妆品套装」「春天穿搭套装」「开学装备」这类组合目标本身当成一个 slot。\n"
            "- need_type 边界：用户明确点名的商品或组合任务最小核心商品可以 required；Planner 自己推断的补充件、配饰、拍照/露营/通勤衍生件应 optional，缺失 optional 不能影响推荐主路线。\n"
            "- 组合任务拆 slot 时，把人群、季节、场景、预算、风格写入各 slot 的 soft_constraints 或 query；不要因为这些约束额外创造一个 slot。\n"
            "- 判定示例：\n"
            "  - 「推荐适合新手的电脑」=> single_retrieval，电脑是单一商品，新手是约束。\n"
            "  - 「推荐一件适合春天穿的外套」=> single_retrieval，外套是单一商品，春天是约束。\n"
            "  - 「根据我平时偏好，选几件日常通勤穿的衣服」=> single_retrieval，衣服是粗品类，通勤和偏好是约束。\n"
            "  - 「适合春天的穿搭套装」=> multi_retrieval，拆上装/下装/鞋或配饰等原子商品。\n"
            "  - 「新手化妆品套装」=> multi_retrieval，拆底妆/定妆/眉妆/唇妆等基础彩妆 slot，而不是一个“化妆品套装” slot。\n"
            "  - 「户外露营装备推荐下」=> single_retrieval，保留户外露营装备粗场景，不擅自拆成帐篷/睡袋等硬 slot。\n"
            "  - 「露营拍照和徒步都要用，推荐一套轻量户外装备」=> 不要拆帐篷/相机；可以 single_retrieval 保留轻量户外装备场景，或拆徒步鞋/户外裤/背包/帽子等可穿戴随身 slot。\n"
            "  - 「健身房训练用的整套装备」=> multi_retrieval 时拆训练上衣/训练裤/训练鞋，配件只能 optional。\n"
            "  - 「开学数码装备清单，帮我配齐」=> multi_retrieval，拆电脑/耳机/充电配件等原子商品。\n"
            "- multi_retrieval 的 need_slots 必须是当前轮生效后的完整 slot plan，不是增量片段。每个 slot 只写自己的商品目标和相关约束，不要拼入其它 slot 的商品词。\n"
            "- 多个并列商品、套装、一套、组合、装备清单、配齐、帮我配、总预算组合等，通常是 multi_retrieval。\n"
            "- 单一商品加多个约束不是多需求，例如「预算3500的安卓平板，轻薄」仍是 single_retrieval。\n"
            "- 不要输出 product_type/categories/preferences/exclusions；这些属于下游 RetrievalPlanBuilder / Tool 的内部解析。\n"
        )
        system_prompt += self._expanded_contract_prompt()
        user_prompt = json.dumps(
            {
                "query": query,
                "context": context or {},
                "required_output": {
                    "plan_type": "direct_answer | clarify | single_retrieval | multi_retrieval",
                },
                "optional_output": {
                    "plan_reason": "short Chinese reason",
                    "vector_query": "semantic retrieval query; omit when direct_answer/clarify",
                    "keyword_query": "keyword retrieval query; omit when direct_answer/clarify",
                    "budget_min": "number; omit when unknown",
                    "budget_max": "number; omit when unknown",
                    "budget_scope": "per_item | total | unknown; omit when unknown",
                    "need_slots": [
                        {
                            "slot_id": "s1",
                            "need_type": "required | optional",
                            "goal": "slot goal",
                            "product_type": "natural product name for this slot",
                            "query": "independent slot query",
                            "soft_constraints": ["slot-level soft constraints"],
                            "exclude_terms": ["slot-level exclusions"],
                            "min_candidates": 1,
                        }
                    ],
                    "referenced_product_ids": ["product IDs from recent turns; omit when empty"],
                    "profile_lookup": {"requested": True, "query": "lookup query", "reason": "why profile lookup is useful"},
                },
                "expanded_output_contract": self._expanded_output_contract(),
                "available_agent_capabilities": self.capability_catalog.describe_for_prompt(),
                "examples": [
                    {
                        "query": "你是谁？",
                        "output": {
                            "plan_type": "direct_answer",
                            "plan_reason": "用户在询问助手身份，不需要商品库证据。",
                        },
                    },
                    {
                        "query": "你是谁",
                        "context": {
                            "recent_turns": [
                                {
                                    "user": "帮我买防晒和面霜，预算 500，不要酒精、不油腻",
                                    "route": "recommend",
                                    "product_ids": ["p_beauty_006", "p_beauty_007"],
                                    "rewrite": {
                                        "need_slots": [
                                            {"slot_id": "s1", "goal": "防晒", "query": "防晒"},
                                            {"slot_id": "s2", "goal": "面霜", "query": "面霜"},
                                        ],
                                        "budget_max": 500,
                                        "budget_scope": "total",
                                    },
                                }
                            ]
                        },
                        "output": {
                            "plan_type": "direct_answer",
                            "plan_reason": "当前问题是询问助手身份，不能继承上一轮防晒和面霜的商品计划。",
                        },
                    },
                    {
                        "query": "推荐点东西吧",
                        "output": {
                            "plan_type": "clarify",
                            "plan_reason": "用户有推荐意图，但没有商品目标，需要澄清。",
                        },
                    },
                    {
                        "query": "想买点护肤品，最近皮肤状态不太好",
                        "output": {
                            "plan_type": "single_retrieval",
                            "vector_query": "皮肤状态不太好的护肤品",
                            "keyword_query": "护肤品 皮肤状态 不太好",
                            "plan_reason": "用户给出了护肤品这个粗品类和皮肤状态线索，应先检索多种护肤子类，再由证据反射审核。",
                        },
                    },
                    {
                        "query": "想买件衣服，日常通勤穿",
                        "output": {
                            "plan_type": "single_retrieval",
                            "vector_query": "日常通勤穿的衣服",
                            "keyword_query": "衣服 日常 通勤",
                            "plan_reason": "用户给出了衣服这个粗品类和通勤场景，应先检索上衣相关子类，不应直接澄清。",
                        },
                    },
                    {
                        "query": "预算300到500，推荐耳机",
                        "output": {
                            "plan_type": "single_retrieval",
                            "vector_query": "预算300到500的耳机",
                            "keyword_query": "耳机 预算300到500",
                            "budget_min": 300,
                            "budget_max": 500,
                            "budget_scope": "per_item",
                            "plan_reason": "用户明确要推荐耳机，并给出单品预算区间。",
                        },
                    },
                    {
                        "query": "这两个哪个更适合油皮？",
                        "context": {
                            "recent_turns": [
                                {
                                    "user": "推荐两款防晒",
                                    "route": "recommend",
                                    "product_ids": ["p1", "p2"],
                                    "selected_products": [
                                        {"product_id": "p1", "name": "清爽防晒乳"},
                                        {"product_id": "p2", "name": "滋润防晒霜"},
                                    ],
                                }
                            ]
                        },
                        "output": {
                            "plan_type": "direct_answer",
                            "referenced_product_ids": ["p1", "p2"],
                            "plan_reason": "当前问题是在追问上一轮两个商品的适合度，可提案复用上下文商品证据。",
                        },
                    },
                ],
            },
            ensure_ascii=False,
        )
        data = await generate_validated_json(
            self.llm_client,
            system_prompt,
            user_prompt,
            validate=lambda value: self._validate_plan_data(value),
            error_message="IntentPlanner returned invalid JSON.",
            response_format=self.JSON_RESPONSE_FORMAT,
            operation="intent_planner.plan",
        )
        return self._parse_plan(query, data, context)

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

    def _parse_plan(
        self,
        query: str,
        data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> IntentPlan:
        plan_type = str(data.get("plan_type") or "single_retrieval").strip()
        if plan_type not in self.PLAN_TYPES:
            plan_type = "single_retrieval"
        need_slots = self._sanitize_need_slots(data.get("need_slots"))
        if plan_type != "multi_retrieval":
            need_slots = []
        if plan_type == "multi_retrieval" and not need_slots:
            plan_type = "single_retrieval"
        if plan_type in {"direct_answer", "clarify"}:
            vector_query = ""
            keyword_query = ""
        else:
            vector_query = str(data.get("vector_query") or "").strip()
            keyword_query = str(data.get("keyword_query") or "").strip()
            if not vector_query and not keyword_query:
                vector_query = query
                keyword_query = query
        profile_lookup = data.get("profile_lookup") if isinstance(data.get("profile_lookup"), dict) else {}
        profile_usage = self._profile_usage(profile_lookup.get("usage"))
        referenced_product_ids = self._string_list(data.get("referenced_product_ids"))
        primary_intent = self._normalize_primary_intent(data.get("primary_intent"), plan_type, referenced_product_ids)
        execution_mode = self._normalize_execution_mode(data.get("execution_mode"), plan_type, referenced_product_ids)
        intents = self._sanitize_intents(
            data.get("intents"),
            query=query,
            primary_intent=primary_intent,
            vector_query=vector_query,
            keyword_query=keyword_query,
            referenced_product_ids=referenced_product_ids,
        )
        execution_mode = self._resolve_execution_mode(
            execution_mode,
            plan_type=plan_type,
            intents=intents,
            referenced_product_ids=referenced_product_ids,
        )
        intents = self._ensure_knowledge_bridge_intent(
            intents,
            execution_mode=execution_mode,
            query=query,
        )
        if primary_intent not in {intent.intent_type for intent in intents}:
            primary_intent = intents[0].intent_type
        constraints = self._sanitize_constraints(
            data.get("constraints"),
            budget_min=self._float_or_none(data.get("budget_min")),
            budget_max=self._float_or_none(data.get("budget_max")),
            budget_scope=self._normalize_budget_scope(data.get("budget_scope"), plan_type),
        )
        context_requests = self._sanitize_context_requests(
            data.get("context_requests"),
            profile_lookup=profile_lookup,
            profile_usage=profile_usage,
        )
        clarification = self._sanitize_clarification(data.get("clarification"), execution_mode)
        research_requests = self._sanitize_research_requests(
            data.get("research_requests"),
            intents,
            query=query,
            referenced_product_ids=referenced_product_ids,
        )
        agent_proposals = self._sanitize_agent_proposals(
            data.get("agent_proposals"),
            execution_mode=execution_mode,
            intents=intents,
            context_requests=context_requests,
            research_requests=research_requests,
        )
        profile_context = next(
            (request for request in context_requests if request.context_type == "long_term_profile"),
            None,
        )
        profile_requested = self._bool_or_false(profile_lookup.get("requested")) or profile_context is not None
        budget_min = constraints.budget_min
        budget_max = constraints.budget_max
        budget_scope = constraints.budget_scope
        return IntentPlan(
            schema_version="2.0",
            original_query=query,
            normalized_query=str(data.get("normalized_query") or query).strip(),
            summary=str(data.get("summary") or "").strip(),
            primary_intent=primary_intent,  # type: ignore[arg-type]
            intents=intents,
            execution_mode=execution_mode,  # type: ignore[arg-type]
            input_modalities=self._normalize_modalities(data.get("input_modalities"), query, context),
            constraints=constraints,
            context_requests=context_requests,
            clarification=clarification,
            research_requests=research_requests,
            agent_proposals=agent_proposals,
            uncertainties=self._sanitize_uncertainties(data.get("uncertainties")),
            plan_type=plan_type,  # type: ignore[arg-type]
            vector_query=vector_query,
            keyword_query=keyword_query,
            budget_min=budget_min,
            budget_max=budget_max,
            budget_scope=budget_scope,
            need_slots=need_slots,
            referenced_product_ids=referenced_product_ids,
            profile_lookup=ProfileLookupProposal(
                requested=profile_requested,
                query=str(profile_lookup.get("query") or (profile_context.query if profile_context else "")).strip(),
                usage=(profile_context.usage if profile_context else profile_usage),  # type: ignore[arg-type]
                reason=str(
                    profile_lookup.get("reason") or (profile_context.reason if profile_context else "")
                ).strip(),
            ),
            plan_reason=str(data.get("plan_reason") or "").strip(),
        )

    def _versioned_system_prompt(self) -> str:
        spec = self.prompt_registry.require("intent_understanding_agent")
        return spec.render_system(allowed_tools=[])

    def _expanded_contract_prompt(self) -> str:
        return (
            "\n\n## 扩展意图契约\n"
            "- primary_intent/intents 描述用户要完成的业务目标；一个请求可以同时包含推荐、对比和知识解释。\n"
            "- execution_mode 只描述工作流拓扑：direct、clarify、context_evidence、single_product、multi_product。\n"
            "- input_modalities 只描述 text/image/audio；图片不是独立业务路线，也不要提案纯图片 Agent。\n"
            "- 每个 intent 必须有稳定 intent_id、intent_type、goal；有依赖时通过 depends_on 引用其它 intent_id。\n"
            "- query_rewrite 应属于具体 intent；multi_product 的检索词应继续下沉到对应 need_slots。\n"
            "- context_requests 只提出上下文读取需求。长期画像用途必须是 intent_refinement、ranking_only 或 answer_personalization。\n"
            "- clarification 说明是否阻塞、缺失字段和澄清目标；不要在这里直接生成最终回答。\n"
            "- 每个 intent 的 route_basis 只记录可核验路由事实：商品目标是否明确、本地目录是否足够、为什么需要外部信息，以及当前 query 中的原文触发片段。\n"
            "- research_requests 只描述 web_general/marketplace/social_content 等抽象研究需求和平台，不得填写 MCP 名、URL、Cookie 或具体 Tool。\n"
            "- 只有当前 query 明确要求联网/指定平台、明确需要最新信息，或确实需要 knowledge_bridge 才能确定商品族时，才能输出 research_requests；普通商品推荐和可由本地商品详情回答的问题不得联网。\n"
            "- 每个 research_request 必须指定唯一 consumer_capability。淘宝、抖音电商、小红书请求统一交给 commerce_research；通用网页资料交给 knowledge_research 或 comparison，不能广播给多个 Agent。\n"
            "- trigger_text 必须逐字来自当前 query。knowledge_bridge 还必须说明 local_catalog_gap，且当前 query 不能已经含有明确商品族。\n"
            "- agent_proposals 只能从 available_agent_capabilities 中选择，并说明 intent_ids、硬依赖 depends_on、软上下文 optional_context_from 和理由。capability 是能力提案，不是具体 Agent 实例；最终 Agent 由 Supervisor 绑定。\n"
            "- depends_on / optional_context_from 只能引用 proposal_id，不能引用 context request_id。ranking_only 画像只能放在 optional_context_from，不能成为商品任务的硬依赖；intent_refinement 的执行顺序由 Supervisor 固定节点控制。\n"
            "- EvidenceVerifier、BundleOptimizer、Repair、AnswerGenerator、MemoryDistillation 由 Supervisor 固定或按运行时失败插入，绝不能写入 agent_proposals。\n"
            "- 长期画像产生的约束只能是 soft，不能覆盖当前 query 中的 hard constraints。\n"
            "- 用户只有效果/症状/模糊用途而没有可确认商品族时，优先 execution_mode=context_evidence 并提案 knowledge_research；"
            "不得猜测商品类型。Supervisor 会在知识证据给出可追溯商品概念后决定是否追加推荐。\n"
        )

    def _expanded_output_contract(self) -> dict[str, Any]:
        return {
            "schema_version": "2.0",
            "normalized_query": "normalized current query",
            "primary_intent": "social_chat | product_recommendation | product_comparison | product_qa | shopping_knowledge | cart_action",
            "intents": [
                {
                    "intent_id": "i1",
                    "intent_type": "one primary_intent enum value",
                    "goal": "business goal",
                    "depends_on": [],
                    "query_rewrite": {"semantic_query": "", "keyword_query": ""},
                    "referenced_product_ids": [],
                    "route_basis": {
                        "target_clarity": "not_applicable | explicit_product | context_product | vague_effect_or_use",
                        "local_catalog_status": "sufficient | insufficient | unknown",
                        "external_information_need": "none | explicit_web | explicit_platform | freshness_required | knowledge_bridge",
                        "trigger_text": "verbatim current-query fragment or empty",
                        "product_family": "concrete product family or empty",
                        "reason": "short routing basis",
                    },
                }
            ],
            "execution_mode": "direct | clarify | context_evidence | single_product | multi_product",
            "input_modalities": ["text | image | audio"],
            "constraints": {
                "budget_min": "number or null",
                "budget_max": "number or null",
                "budget_scope": "per_item | total | unknown",
                "items": [
                    {
                        "name": "constraint name",
                        "value": "constraint value",
                        "strength": "hard | soft",
                        "source": "current_query | recent_turn | session_summary | image_inference",
                        "reason": "source reason",
                    }
                ],
            },
            "context_requests": [
                {
                    "request_id": "ctx1",
                    "context_type": "long_term_profile",
                    "usage": "intent_refinement | ranking_only | answer_personalization",
                    "query": "lookup query",
                    "reason": "why context is needed",
                    "required": False,
                }
            ],
            "clarification": {
                "required": False,
                "blocking": False,
                "missing_fields": [],
                "ambiguity": "",
                "question_goal": "",
                "reason": "",
            },
            "research_requests": [
                {
                    "request_id": "r1",
                    "intent_id": "i1",
                    "mode": "web_general | marketplace | social_content",
                    "consumer_capability": "knowledge_research | comparison | commerce_research",
                    "trigger_type": "explicit_web_request | explicit_platform_request | freshness_required | knowledge_bridge",
                    "trigger_text": "verbatim fragment from current query",
                    "platforms": [],
                    "query": "research goal",
                    "freshness": "any | recent | realtime",
                    "local_catalog_gap": "required for knowledge_bridge, otherwise empty",
                    "reason": "why research is needed",
                    "required": False,
                }
            ],
            "agent_proposals": [
                {
                    "proposal_id": "p1",
                    "capability": "one available_agent_capabilities value",
                    "intent_ids": ["i1"],
                    "reason": "why this capability is needed",
                    "required": False,
                    "depends_on": [],
                    "optional_context_from": [],
                    "expected_output_schema": "short schema name",
                }
            ],
            "uncertainties": [
                {
                    "field": "field name",
                    "description": "uncertainty",
                    "blocking": False,
                    "confidence": "0..1 or null",
                }
            ],
        }

    def _normalize_primary_intent(
        self,
        value: Any,
        plan_type: str,
        referenced_product_ids: list[str],
    ) -> str:
        normalized = str(value or "").strip()
        if normalized in self.INTENT_TYPES:
            return normalized
        if plan_type == "direct_answer":
            return "product_qa" if referenced_product_ids else "social_chat"
        return "product_recommendation"

    def _normalize_execution_mode(
        self,
        value: Any,
        plan_type: str,
        referenced_product_ids: list[str],
    ) -> str:
        normalized = str(value or "").strip()
        if normalized in self.EXECUTION_MODES:
            return normalized
        if plan_type == "clarify":
            return "clarify"
        if plan_type == "multi_retrieval":
            return "multi_product"
        if plan_type == "single_retrieval":
            return "single_product"
        return "context_evidence" if referenced_product_ids else "direct"

    def _resolve_execution_mode(
        self,
        proposed: str,
        *,
        plan_type: str,
        intents: list[IntentItem],
        referenced_product_ids: list[str],
    ) -> str:
        if plan_type == "clarify":
            return "clarify"
        if plan_type == "multi_retrieval":
            return "multi_product"
        if plan_type == "single_retrieval":
            return "single_product"
        intent_types = {intent.intent_type for intent in intents}
        has_context_evidence_goal = bool(
            referenced_product_ids
            or intent_types.intersection({"product_comparison", "product_qa", "shopping_knowledge"})
        )
        if has_context_evidence_goal or proposed == "context_evidence":
            return "context_evidence"
        return "direct"

    def _ensure_knowledge_bridge_intent(
        self,
        intents: list[IntentItem],
        *,
        execution_mode: str,
        query: str,
    ) -> list[IntentItem]:
        intent_types = {intent.intent_type for intent in intents}
        if (
            execution_mode != "context_evidence"
            or "product_recommendation" not in intent_types
            or intent_types.intersection({"product_qa", "shopping_knowledge"})
        ):
            return intents
        known_ids = {intent.intent_id for intent in intents}
        intent_id = "i_knowledge_bridge"
        suffix = 2
        while intent_id in known_ids:
            intent_id = f"i_knowledge_bridge_{suffix}"
            suffix += 1
        return [
            *intents,
            IntentItem(
                intent_id=intent_id,
                intent_type="shopping_knowledge",
                goal=f"研究用户目标并映射到可检索商品概念：{query}",
            ),
        ]

    def _normalize_modalities(
        self,
        value: Any,
        query: str,
        context: dict[str, Any] | None,
    ) -> list[str]:
        modalities = [item for item in self._string_list(value) if item in self.INPUT_MODALITIES]
        if query.strip() and "text" not in modalities:
            modalities.insert(0, "text")
        context = context or {}
        modality_context = context.get("input_modalities")
        has_image_context = (
            bool(context.get("image_attributes") or context.get("image_id"))
            or isinstance(modality_context, dict) and bool(modality_context.get("has_image"))
            or isinstance(modality_context, list) and "image" in modality_context
        )
        if has_image_context and "image" not in modalities:
            modalities.append("image")
        return modalities or ["text"]

    def _sanitize_intents(
        self,
        value: Any,
        *,
        query: str,
        primary_intent: str,
        vector_query: str,
        keyword_query: str,
        referenced_product_ids: list[str],
    ) -> list[IntentItem]:
        result: list[IntentItem] = []
        seen_ids: set[str] = set()
        if isinstance(value, list):
            for index, item in enumerate(value, start=1):
                if not isinstance(item, dict):
                    continue
                intent_type = str(item.get("intent_type") or item.get("type") or "").strip()
                if intent_type not in self.INTENT_TYPES:
                    continue
                intent_id = str(item.get("intent_id") or f"i{index}").strip() or f"i{index}"
                if intent_id in seen_ids:
                    continue
                rewrite = item.get("query_rewrite") if isinstance(item.get("query_rewrite"), dict) else {}
                route_basis = item.get("route_basis") if isinstance(item.get("route_basis"), dict) else {}
                target_clarity = str(route_basis.get("target_clarity") or "not_applicable").strip()
                if target_clarity not in {
                    "not_applicable",
                    "explicit_product",
                    "context_product",
                    "vague_effect_or_use",
                }:
                    target_clarity = "not_applicable"
                local_catalog_status = str(route_basis.get("local_catalog_status") or "unknown").strip()
                if local_catalog_status not in {"sufficient", "insufficient", "unknown"}:
                    local_catalog_status = "unknown"
                external_information_need = str(
                    route_basis.get("external_information_need") or "none"
                ).strip()
                if external_information_need not in {
                    "none",
                    "explicit_web",
                    "explicit_platform",
                    "freshness_required",
                    "knowledge_bridge",
                }:
                    external_information_need = "none"
                result.append(
                    IntentItem(
                        intent_id=intent_id,
                        intent_type=intent_type,  # type: ignore[arg-type]
                        goal=str(item.get("goal") or query).strip(),
                        depends_on=self._string_list(item.get("depends_on")),
                        query_rewrite=IntentQueryRewrite(
                            semantic_query=str(rewrite.get("semantic_query") or "").strip(),
                            keyword_query=str(rewrite.get("keyword_query") or "").strip(),
                        ),
                        referenced_product_ids=self._string_list(item.get("referenced_product_ids")),
                        route_basis=IntentRouteBasis(
                            target_clarity=target_clarity,  # type: ignore[arg-type]
                            local_catalog_status=local_catalog_status,  # type: ignore[arg-type]
                            external_information_need=external_information_need,  # type: ignore[arg-type]
                            trigger_text=str(route_basis.get("trigger_text") or "").strip(),
                            product_family=str(route_basis.get("product_family") or "").strip(),
                            reason=str(route_basis.get("reason") or "").strip(),
                        ),
                    )
                )
                seen_ids.add(intent_id)
        if result:
            return result
        return [
            IntentItem(
                intent_id="i1",
                intent_type=primary_intent,  # type: ignore[arg-type]
                goal=query,
                query_rewrite=IntentQueryRewrite(
                    semantic_query=vector_query,
                    keyword_query=keyword_query,
                ),
                referenced_product_ids=referenced_product_ids,
                route_basis=IntentRouteBasis(
                    target_clarity=(
                        "context_product"
                        if referenced_product_ids
                        else "explicit_product"
                        if primary_intent == "product_recommendation"
                        else "not_applicable"
                    ),
                    local_catalog_status="unknown",
                    external_information_need="none",
                ),
            )
        ]

    def _sanitize_constraints(
        self,
        value: Any,
        *,
        budget_min: float | None,
        budget_max: float | None,
        budget_scope: str,
    ) -> IntentConstraintSet:
        payload = value if isinstance(value, dict) else {}
        scope = str(payload.get("budget_scope") or budget_scope).strip()
        if scope not in {"per_item", "total", "unknown"}:
            scope = budget_scope
        items: list[IntentConstraint] = []
        raw_items = payload.get("items") if isinstance(payload.get("items"), list) else []
        for item in raw_items:
            if not isinstance(item, dict) or not str(item.get("name") or "").strip():
                continue
            raw_value = item.get("value")
            if not isinstance(raw_value, (str, float, int, bool, list)):
                continue
            strength = item.get("strength") if item.get("strength") in {"hard", "soft"} else "hard"
            source = item.get("source")
            if source not in {
                "current_query",
                "recent_turn",
                "session_summary",
                "image_inference",
                "long_term_profile",
            }:
                source = "current_query"
            if source == "long_term_profile":
                strength = "soft"
            items.append(
                IntentConstraint(
                    name=str(item.get("name") or "").strip(),
                    value=raw_value,
                    strength=strength,
                    source=source,
                    reason=str(item.get("reason") or "").strip(),
                )
            )
        return IntentConstraintSet(
            budget_min=self._float_or_none(payload.get("budget_min")) if "budget_min" in payload else budget_min,
            budget_max=self._float_or_none(payload.get("budget_max")) if "budget_max" in payload else budget_max,
            budget_scope=scope,  # type: ignore[arg-type]
            items=items,
        )

    def _sanitize_context_requests(
        self,
        value: Any,
        *,
        profile_lookup: dict[str, Any],
        profile_usage: str,
    ) -> list[ContextRequest]:
        result: list[ContextRequest] = []
        if isinstance(value, list):
            for index, item in enumerate(value, start=1):
                if not isinstance(item, dict) or item.get("context_type") != "long_term_profile":
                    continue
                usage = self._profile_usage(item.get("usage"))
                reason = str(item.get("reason") or "").strip()
                if not reason:
                    continue
                result.append(
                    ContextRequest(
                        request_id=str(item.get("request_id") or f"ctx{index}").strip() or f"ctx{index}",
                        usage=usage,  # type: ignore[arg-type]
                        query=str(item.get("query") or "").strip(),
                        reason=reason,
                        required=self._bool_or_false(item.get("required")),
                    )
                )
        if result or not self._bool_or_false(profile_lookup.get("requested")):
            return result
        return [
            ContextRequest(
                request_id="ctx_profile",
                usage=profile_usage,  # type: ignore[arg-type]
                query=str(profile_lookup.get("query") or "").strip(),
                reason=str(profile_lookup.get("reason") or "需要长期画像作为软偏好。").strip(),
            )
        ]

    def _sanitize_clarification(self, value: Any, execution_mode: str) -> ClarificationProposal:
        payload = value if isinstance(value, dict) else {}
        required = self._bool_or_false(payload.get("required")) or execution_mode == "clarify"
        return ClarificationProposal(
            required=required,
            blocking=self._bool_or_false(payload.get("blocking")) or execution_mode == "clarify",
            missing_fields=self._string_list(payload.get("missing_fields")),
            ambiguity=str(payload.get("ambiguity") or "").strip(),
            question_goal=str(payload.get("question_goal") or "补齐可执行的商品目标").strip() if required else "",
            reason=str(payload.get("reason") or "用户需求不足以形成可执行任务。").strip() if required else "",
        )

    def _sanitize_research_requests(
        self,
        value: Any,
        intents: list[IntentItem],
        *,
        query: str,
        referenced_product_ids: list[str],
    ) -> list[ResearchRequest]:
        intent_ids = {intent.intent_id for intent in intents}
        result: list[ResearchRequest] = []
        seen_ids: set[str] = set()
        raw_requests = value if isinstance(value, list) else []
        for index, item in enumerate(raw_requests, start=1):
            if not isinstance(item, dict):
                continue
            mode = str(item.get("mode") or "").strip()
            intent_id = str(item.get("intent_id") or "").strip()
            if mode not in {"web_general", "marketplace", "social_content"} or intent_id not in intent_ids:
                continue
            request_id = str(item.get("request_id") or f"r{index}").strip() or f"r{index}"
            if request_id in seen_ids:
                continue
            consumer_capability = str(item.get("consumer_capability") or "").strip()
            trigger_type = str(item.get("trigger_type") or "").strip()
            trigger_text = str(item.get("trigger_text") or "").strip()
            if (
                consumer_capability not in self.RESEARCH_CONSUMERS
                or trigger_type not in self.RESEARCH_TRIGGER_TYPES
                or not trigger_text
            ):
                continue
            freshness = item.get("freshness") if item.get("freshness") in {"any", "recent", "realtime"} else "recent"
            result.append(
                ResearchRequest(
                    request_id=request_id,
                    intent_id=intent_id,
                    mode=mode,  # type: ignore[arg-type]
                    consumer_capability=consumer_capability,  # type: ignore[arg-type]
                    trigger_type=trigger_type,  # type: ignore[arg-type]
                    trigger_text=trigger_text,
                    platforms=self._string_list(item.get("platforms")),
                    query=str(item.get("query") or "").strip(),
                    freshness=freshness,
                    local_catalog_gap=str(item.get("local_catalog_gap") or "").strip(),
                    reason=str(item.get("reason") or "需要补充外部证据。").strip(),
                    required=self._bool_or_false(item.get("required")),
                )
            )
            seen_ids.add(request_id)

        # Knowledge routes must never silently fall back to model memory when
        # the planner omitted a research request. Product QA about an already
        # referenced catalog item can be grounded locally; open-ended shopping
        # knowledge still receives the controlled web-search template.
        return result

    def _sanitize_agent_proposals(
        self,
        value: Any,
        *,
        execution_mode: str,
        intents: list[IntentItem],
        context_requests: list[ContextRequest],
        research_requests: list[ResearchRequest],
    ) -> list[AgentTaskProposal]:
        result: list[AgentTaskProposal] = []
        seen_ids: set[str] = set()
        canonical_by_capability: dict[str, AgentTaskProposal] = {}
        proposal_aliases: dict[str, str] = {}
        proposable = self.capability_catalog.proposable_names()
        known_intent_ids = {intent.intent_id for intent in intents}
        if isinstance(value, list):
            for index, item in enumerate(value, start=1):
                if not isinstance(item, dict):
                    continue
                capability = str(item.get("capability") or "").strip()
                proposal_id = str(item.get("proposal_id") or f"p{index}").strip() or f"p{index}"
                if capability not in proposable or proposal_id in seen_ids:
                    continue
                intent_ids_for_proposal = [
                    intent_id
                    for intent_id in self._string_list(item.get("intent_ids"))
                    if intent_id in known_intent_ids
                ]
                existing = canonical_by_capability.get(capability)
                if existing is not None:
                    proposal_aliases[proposal_id] = existing.proposal_id
                    existing.intent_ids = list(dict.fromkeys([*existing.intent_ids, *intent_ids_for_proposal]))
                    existing.required = existing.required or self._bool_or_false(item.get("required"))
                    existing.depends_on = list(
                        dict.fromkeys([*existing.depends_on, *self._string_list(item.get("depends_on"))])
                    )
                    existing.optional_context_from = list(
                        dict.fromkeys(
                            [
                                *existing.optional_context_from,
                                *self._string_list(item.get("optional_context_from")),
                            ]
                        )
                    )
                    if not existing.expected_output_schema:
                        existing.expected_output_schema = str(
                            item.get("expected_output_schema") or ""
                        ).strip()
                    seen_ids.add(proposal_id)
                    continue
                proposal = AgentTaskProposal(
                    proposal_id=proposal_id,
                    capability=capability,  # type: ignore[arg-type]
                    intent_ids=intent_ids_for_proposal,
                    reason=str(item.get("reason") or "该能力支持当前意图。").strip(),
                    required=self._bool_or_false(item.get("required")),
                    depends_on=self._string_list(item.get("depends_on")),
                    optional_context_from=self._string_list(item.get("optional_context_from")),
                    expected_output_schema=str(item.get("expected_output_schema") or "").strip(),
                )
                result.append(proposal)
                canonical_by_capability[capability] = proposal
                proposal_aliases[proposal_id] = proposal_id
                seen_ids.add(proposal_id)

        # Rewrite dependencies after duplicate capabilities have been folded.
        # This also handles references to an alias that appeared later in the
        # model output.
        for proposal in result:
            proposal.depends_on = list(
                dict.fromkeys(
                    canonical
                    for dependency in proposal.depends_on
                    if (canonical := proposal_aliases.get(dependency, dependency))
                    != proposal.proposal_id
                )
            )
            proposal.optional_context_from = list(
                dict.fromkeys(
                    canonical
                    for dependency in proposal.optional_context_from
                    if (canonical := proposal_aliases.get(dependency, dependency))
                    != proposal.proposal_id
                )
            )

        intent_ids = [intent.intent_id for intent in intents]

        def add_missing(
            proposal_id: str,
            capability: str,
            selected_intent_ids: list[str],
            reason: str,
            depends_on: list[str] | None = None,
            optional_context_from: list[str] | None = None,
        ) -> str:
            existing = next((item for item in result if item.capability == capability), None)
            if existing:
                existing.intent_ids = list(dict.fromkeys([*existing.intent_ids, *selected_intent_ids]))
                existing.depends_on = list(
                    dict.fromkeys(
                        dependency
                        for dependency in [*existing.depends_on, *(depends_on or [])]
                        if dependency and dependency != existing.proposal_id
                    )
                )
                existing.optional_context_from = list(
                    dict.fromkeys(
                        dependency
                        for dependency in [
                            *existing.optional_context_from,
                            *(optional_context_from or []),
                        ]
                        if dependency and dependency != existing.proposal_id
                    )
                )
                return existing.proposal_id
            unique_id = proposal_id
            suffix = 2
            while unique_id in seen_ids:
                unique_id = f"{proposal_id}_{suffix}"
                suffix += 1
            result.append(
                AgentTaskProposal(
                    proposal_id=unique_id,
                    capability=capability,  # type: ignore[arg-type]
                    intent_ids=selected_intent_ids,
                    reason=reason,
                    depends_on=depends_on or [],
                    optional_context_from=optional_context_from or [],
                )
            )
            seen_ids.add(unique_id)
            return unique_id

        if context_requests:
            add_missing("p_profile", "profile_preference", intent_ids, "用户请求需要长期画像作为受控软上下文。")

        core_proposal_id = ""
        if execution_mode == "clarify":
            core_proposal_id = add_missing("p_clarify", "clarification", intent_ids, "当前歧义阻塞后续执行。")
        elif execution_mode == "single_product":
            recommendation_intents = [
                intent.intent_id for intent in intents if intent.intent_type == "product_recommendation"
            ] or intent_ids
            core_proposal_id = add_missing(
                "p_single_product",
                "single_product_recommendation",
                recommendation_intents,
                "当前工作流包含一个商品目标。",
            )
        elif execution_mode == "multi_product":
            recommendation_intents = [
                intent.intent_id for intent in intents if intent.intent_type == "product_recommendation"
            ] or intent_ids
            core_proposal_id = add_missing(
                "p_multi_product",
                "multi_product_bundle",
                recommendation_intents,
                "当前工作流包含多个商品槽位。",
            )

        commerce_intent_ids = list(
            dict.fromkeys(
                request.intent_id
                for request in research_requests
                if request.consumer_capability == "commerce_research"
                if request.mode in {"marketplace", "social_content"}
                and any(platform in {"taobao", "douyin_ec", "xiaohongshu"} for platform in request.platforms)
            )
        )
        commerce_proposal_id = ""
        if commerce_intent_ids:
            commerce_proposal_id = add_missing(
                "p_commerce_research",
                "commerce_research",
                commerce_intent_ids,
                "用户请求需要三平台外部商品或口碑证据。",
            )

        comparison_intents = list(
            dict.fromkeys(
                [intent.intent_id for intent in intents if intent.intent_type == "product_comparison"]
                + [
                    request.intent_id
                    for request in research_requests
                    if request.consumer_capability == "comparison"
                ]
            )
        )
        if comparison_intents:
            comparison_depends_on_core = any(
                intent.intent_id in comparison_intents
                and any(
                    dependency_id in {
                        candidate.intent_id
                        for candidate in intents
                        if candidate.intent_type == "product_recommendation"
                    }
                    for dependency_id in intent.depends_on
                )
                for intent in intents
            )
            add_missing(
                "p_comparison",
                "comparison",
                comparison_intents,
                "用户明确要求商品对比。",
                [core_proposal_id] if core_proposal_id and comparison_depends_on_core else [],
                [commerce_proposal_id] if commerce_proposal_id else [],
            )
        knowledge_intents = list(
            dict.fromkeys(
                [
                    intent.intent_id
                    for intent in intents
                    if intent.intent_type in {"product_qa", "shopping_knowledge"}
                    or intent.route_basis.external_information_need == "knowledge_bridge"
                ]
                + [
                    request.intent_id
                    for request in research_requests
                    if request.consumer_capability == "knowledge_research"
                ]
            )
        )
        if knowledge_intents:
            recommendation_intent_ids = {
                intent.intent_id
                for intent in intents
                if intent.intent_type == "product_recommendation"
            }
            knowledge_depends_on_core = any(
                intent.intent_id in knowledge_intents
                and bool(set(intent.depends_on).intersection(recommendation_intent_ids))
                for intent in intents
            )
            add_missing(
                "p_knowledge",
                "knowledge_research",
                knowledge_intents,
                "用户需要商品事实或选购知识解释。",
                [core_proposal_id] if core_proposal_id and knowledge_depends_on_core else [],
            )

        self._normalize_context_proposal_references(result, context_requests)

        # Clarification is a terminal business branch. Do not let a noisy LLM
        # proposal trigger retrieval or network work before the user answers.
        if execution_mode == "clarify":
            result = [item for item in result if item.capability == "clarification"]
            for proposal in result:
                proposal.depends_on = []
                proposal.optional_context_from = []
        elif execution_mode == "direct":
            result = []
        return result

    def _normalize_context_proposal_references(
        self,
        proposals: list[AgentTaskProposal],
        context_requests: list[ContextRequest],
    ) -> None:
        """Translate Planner context IDs into proposal lineage.

        Context requests describe what should be read; task dependencies refer
        only to executable proposals. Profile data is contextual input, not a
        hard prerequisite for the user's core product task. Intent refinement
        ordering is compiled separately by the Supervisor.
        """
        profile = next(
            (proposal for proposal in proposals if proposal.capability == "profile_preference"),
            None,
        )
        if profile is None:
            return
        context_to_proposal = {
            request.request_id: profile.proposal_id
            for request in context_requests
            if request.context_type == "long_term_profile"
        }
        if not context_to_proposal:
            return

        for proposal in proposals:
            hard_dependencies: list[str] = []
            optional_dependencies = list(proposal.optional_context_from)
            for dependency in proposal.depends_on:
                resolved = context_to_proposal.get(dependency, dependency)
                if resolved == profile.proposal_id:
                    optional_dependencies.append(resolved)
                else:
                    hard_dependencies.append(resolved)
            optional_dependencies = [
                context_to_proposal.get(dependency, dependency)
                for dependency in optional_dependencies
            ]
            proposal.depends_on = list(
                dict.fromkeys(
                    dependency
                    for dependency in hard_dependencies
                    if dependency != proposal.proposal_id
                )
            )
            proposal.optional_context_from = list(
                dict.fromkeys(
                    dependency
                    for dependency in optional_dependencies
                    if dependency != proposal.proposal_id
                    and dependency not in proposal.depends_on
                )
            )

    def _sanitize_uncertainties(self, value: Any) -> list[IntentUncertainty]:
        if not isinstance(value, list):
            return []
        result: list[IntentUncertainty] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            field = str(item.get("field") or "").strip()
            description = str(item.get("description") or "").strip()
            if not field or not description:
                continue
            confidence = self._float_or_none(item.get("confidence"))
            if confidence is not None:
                confidence = max(0.0, min(1.0, confidence))
            result.append(
                IntentUncertainty(
                    field=field,
                    description=description,
                    blocking=self._bool_or_false(item.get("blocking")),
                    confidence=confidence,
                )
            )
        return result

    def _profile_usage(self, value: Any) -> str:
        usage = str(value or "").strip()
        if usage in {"intent_refinement", "ranking_only", "answer_personalization"}:
            return usage
        return "ranking_only"

    def _validate_plan_data(self, data: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        plan_type = data.get("plan_type")
        if plan_type not in self.PLAN_TYPES:
            errors.append("plan_type must be direct_answer, clarify, single_retrieval, or multi_retrieval")
        if "budget_scope" in data and data.get("budget_scope") not in {"per_item", "total", "unknown", None, ""}:
            errors.append("budget_scope must be per_item, total, or unknown")
        for field in ["summary", "vector_query", "keyword_query", "plan_reason"]:
            if field in data and data.get(field) is not None and not isinstance(data.get(field), str):
                errors.append(f"{field} must be a string")
        for field in ["budget_min", "budget_max"]:
            if data.get(field) not in {None, ""}:
                try:
                    if float(data.get(field)) < 0:
                        errors.append(f"{field} must be non-negative or null")
                except (TypeError, ValueError):
                    errors.append(f"{field} must be a number or null")
        if data.get("budget_min") not in {None, ""} and data.get("budget_max") not in {None, ""}:
            try:
                if float(data.get("budget_min")) > float(data.get("budget_max")):
                    errors.append("budget_min cannot exceed budget_max")
            except (TypeError, ValueError):
                pass
        if "need_slots" in data and not isinstance(data.get("need_slots"), list):
            errors.append("need_slots must be a list")
        if "referenced_product_ids" in data and not isinstance(data.get("referenced_product_ids"), list):
            errors.append("referenced_product_ids must be a list")
        profile_lookup = data.get("profile_lookup")
        if "profile_lookup" in data and profile_lookup is not None and not isinstance(profile_lookup, dict):
            errors.append("profile_lookup must be an object")
        elif isinstance(profile_lookup, dict) and not self._is_bool_like(profile_lookup.get("requested", False)):
            errors.append("profile_lookup.requested must be boolean")
        if "schema_version" in data and str(data.get("schema_version") or "").strip() != "2.0":
            errors.append("schema_version must be 2.0")
        if "primary_intent" in data and data.get("primary_intent") not in self.INTENT_TYPES:
            errors.append("primary_intent is invalid")
        if "execution_mode" in data and data.get("execution_mode") not in self.EXECUTION_MODES:
            errors.append("execution_mode is invalid")
        if "input_modalities" in data:
            modalities = data.get("input_modalities")
            if not isinstance(modalities, list):
                errors.append("input_modalities must be a list")
            elif any(item not in self.INPUT_MODALITIES for item in modalities):
                errors.append("input_modalities contains an invalid modality")
        for field in ["intents", "context_requests", "research_requests", "agent_proposals", "uncertainties"]:
            if field in data and not isinstance(data.get(field), list):
                errors.append(f"{field} must be a list")
        for field in ["constraints", "clarification"]:
            if field in data and not isinstance(data.get(field), dict):
                errors.append(f"{field} must be an object")
        raw_intents = data.get("intents") if isinstance(data.get("intents"), list) else []
        for index, item in enumerate(raw_intents):
            if not isinstance(item, dict):
                errors.append(f"intents[{index}] must be an object")
                continue
            intent_type = item.get("intent_type") or item.get("type")
            if intent_type not in self.INTENT_TYPES:
                errors.append(f"intents[{index}].intent_type is invalid")
        raw_proposals = data.get("agent_proposals") if isinstance(data.get("agent_proposals"), list) else []
        proposable = self.capability_catalog.proposable_names()
        for index, item in enumerate(raw_proposals):
            if not isinstance(item, dict):
                errors.append(f"agent_proposals[{index}] must be an object")
                continue
            capability = item.get("capability")
            if capability not in proposable:
                errors.append(f"agent_proposals[{index}].capability is not planner-proposable")
        return errors

    def _json_text_from_tagged_content(self, content: str) -> str:
        match = re.search(r"<json>\s*(.*?)\s*</json>", content, flags=re.DOTALL | re.IGNORECASE)
        return match.group(1) if match else content

    def _sanitize_need_slots(self, value: Any) -> list[RewriteNeedSlot]:
        if not isinstance(value, list):
            return []
        result: list[RewriteNeedSlot] = []
        for index, item in enumerate(value, start=1):
            if not isinstance(item, dict):
                continue
            try:
                result.append(
                    RewriteNeedSlot(
                        slot_id=str(item.get("slot_id") or f"s{index}").strip() or f"s{index}",
                        intent_id=str(item.get("intent_id") or "").strip(),
                        need_type=item.get("need_type") if item.get("need_type") in {"required", "optional"} else "required",
                        goal=str(item.get("goal") or item.get("product_type") or item.get("query") or "").strip(),
                        product_type=str(item.get("product_type") or item.get("goal") or "").strip(),
                        query=str(item.get("query") or item.get("goal") or item.get("product_type") or "").strip(),
                        semantic_query=str(item.get("semantic_query") or item.get("query") or "").strip(),
                        keyword_query=str(item.get("keyword_query") or item.get("query") or "").strip(),
                        soft_constraints=self._string_list(item.get("soft_constraints")),
                        exclude_terms=self._string_list(item.get("exclude_terms")),
                        min_candidates=max(1, int(item.get("min_candidates") or 1)),
                    )
                )
            except (TypeError, ValueError):
                continue
        return [slot for slot in result if slot.query or slot.goal or slot.product_type]

    def _normalize_budget_scope(self, value: Any, plan_type: str) -> str:
        scope = str(value or "").strip()
        if scope in {"per_item", "total", "unknown"}:
            return scope
        return "total" if plan_type == "multi_retrieval" else "unknown"

    def _float_or_none(self, value: Any) -> float | None:
        if value in {None, ""}:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _string_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        result: list[str] = []
        for item in value:
            text = str(item).strip()
            if text and text not in result:
                result.append(text)
        return result

    def _is_bool_like(self, value: Any) -> bool:
        if isinstance(value, bool):
            return True
        if isinstance(value, str):
            return value.strip().lower() in {"true", "false"}
        return value in {0, 1}

    def _bool_or_false(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() == "true"
        return value == 1

    def _bool_or_true(self, value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() != "false"
        return value != 0


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
