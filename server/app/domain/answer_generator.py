from __future__ import annotations

import asyncio
import inspect
import json
import re
from collections.abc import AsyncGenerator
from typing import Any

from app.db.models import Product
from app.domain.need_slot_schemas import SlotCandidate, MultiNeedSelection, MultiNeedState
from app.schemas import (
    MULTI_NEED_ALTERNATIVES_PER_SLOT,
    MULTI_NEED_PRIMARY_PER_SLOT,
    IntentPlan,
    ProductCard,
    QueryPlan,
    SINGLE_RECOMMENDATION_LIMIT,
)
from app.services.llm_client import LlmClient


class AnswerGenerator:
    def __init__(self) -> None:
        self.llm_client = LlmClient(component="AnswerGenerator")

    def product_card(self, product: Product, plan: QueryPlan) -> ProductCard:
        reason = self.reason_for(product, plan)
        return ProductCard(
            product_id=product.product_id,
            name=product.name,
            category=product.category,
            sub_category=product.sub_category,
            brand=product.brand,
            price=float(product.price),
            image_url=product.image_url,
            tags=product.tags[:6],
            rating=float(product.rating),
            reason=reason,
        )

    def reason_for(self, product: Product, plan: QueryPlan) -> str:
        matched = []
        for preference in plan.preferences + plan.scene:
            if preference in product.search_text():
                matched.append(preference)
        if plan.budget.max is not None and float(product.price) <= plan.budget.max:
            matched.append(f"价格不超过 {plan.budget.max:g} 元")
        if matched:
            return f"匹配：{'、'.join(matched[:4])}。{self._short_review(product)}"
        return product.review_summary or product.description[:80]

    async def stream_text(
        self,
        plan: QueryPlan,
        ranked_products: list[tuple[Product, float]],
        profile_narrative: str = "",
        image_attributes: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        system_prompt, user_prompt = self._single_retrieval_prompt(
            plan,
            ranked_products,
            profile_narrative=profile_narrative,
            image_attributes=image_attributes,
        )
        async for token in self._generate_stream_or_fallback(
            system_prompt,
            user_prompt,
            operation="answer_generator.single_retrieval",
        ):
            yield token

    async def stream_multi_need_text(
        self,
        state: MultiNeedState,
        selection: MultiNeedSelection,
        route: str,
        verify_reason: str,
        corrective_slot_coverage: list[dict] | None = None,
        rejected_products: list[dict] | None = None,
        combo_summary: dict | None = None,
        profile_narrative: str = "",
    ) -> AsyncGenerator[str, None]:
        system_prompt, user_prompt = self._multi_need_prompt(
            state,
            selection,
            route,
            verify_reason,
            corrective_slot_coverage,
            rejected_products,
            combo_summary,
            profile_narrative=profile_narrative,
        )
        async for token in self._generate_stream_or_fallback(
            system_prompt,
            user_prompt,
            operation="answer_generator.multi_need",
        ):
            yield token

    async def stream_direct_text(
        self,
        query: str,
        mode: str,
        reason: str,
        intent_plan: IntentPlan | None = None,
        preferred_text: str | None = None,
        extra_context: dict | None = None,
        profile_narrative: str = "",
    ) -> AsyncGenerator[str, None]:
        text = preferred_text
        if text is None:
            system_prompt, user_prompt = self._direct_prompt(
                query,
                mode,
                reason,
                intent_plan,
                extra_context,
                profile_narrative,
            )
            async for token in self._generate_stream_or_fallback(
                system_prompt,
                user_prompt,
                operation="answer_generator.direct_text",
            ):
                yield token
            return
        for token in self._chunk_text(text):
            await asyncio.sleep(0.01)
            yield token

    async def _llm_text(
        self,
        plan: QueryPlan,
        ranked_products: list[tuple[Product, float]],
        profile_narrative: str = "",
        image_attributes: dict[str, Any] | None = None,
    ) -> str:
        system_prompt, user_prompt = self._single_retrieval_prompt(
            plan,
            ranked_products,
            profile_narrative=profile_narrative,
            image_attributes=image_attributes,
        )
        return await self._generate_required(
            system_prompt,
            user_prompt,
            operation="answer_generator.single_retrieval",
        )

    def _single_retrieval_prompt(
        self,
        plan: QueryPlan,
        ranked_products: list[tuple[Product, float]],
        profile_narrative: str = "",
        image_attributes: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        if not ranked_products:
            raise RuntimeError("AnswerGenerator 没有可用于生成回答的候选商品。")
        products = []
        for product, score in ranked_products[:SINGLE_RECOMMENDATION_LIMIT]:
            products.append(
                {
                    "product_id": product.product_id,
                    "name": product.name,
                    "brand": product.brand,
                    "category": product.category,
                    "sub_category": product.sub_category,
                    "price": float(product.price),
                    "rating": float(product.rating),
                    "score": round(score, 4),
                    "description": product.description[:500],
                    "review_summary": self._short_review(product, 300),
                }
            )
        system_prompt = (
            "你是电商导购 RAG Agent。只能推荐用户提供的候选商品，"
            "不得编造商品、价格、库存、优惠或功效。"
            "并说明推荐理由来自商品描述或用户评价。"
            "单一检索回答中，final_products 里的每个商品都是 Corrective Agent 已通过的正式推荐商品；"
            f"单一检索最多推荐 {SINGLE_RECOMMENDATION_LIMIT} 个，必须按 final_recommendation_order 顺序逐一推荐全部商品，不要只挑 2-3 个，"
            "不要把其中任何商品写成候补/可考虑/备选。"
            "如果用户问的是粗品类或上位商品词，例如护肤品、衣服、鞋子、裤子、饮料、零食、电脑、手机，"
            "不要把需求误写成某一个精确子类；如果 final_products 覆盖多个 sub_category，要按不同子类/用途清楚呈现多样化选择。"
            "如果 final_products 只有一个商品，也要诚实说明本轮只有这一款通过审核，不要暗示还有其它已通过候选。"
            "如果用户没有明确性别/年龄/身份，不要假设，不要使用「男」「女」「男士」「女士」「学生」等限定词。"
            "profile_narrative 只用于表达个性化理由，不能改变 final_products，不能把画像说成用户本轮明确要求。"
            "如果 image_attributes.available=true，可以用“根据图片推测”引入颜色、风格、品类等视觉语义；"
            "必须保持不确定措辞，不能把图片推测说成商品事实，也不能覆盖商品证据。"
            "\n\n# 输出格式 (markdown, 字数不限)\n"
            "必须严格按以下 markdown 格式输出，让前端解析渲染:\n"
            "\n"
            "[一句开头总结]\n"
            "\n"
            "## 1. [商品名]\n"
            "\n"
            "**¥[价格]**　[品牌] · [子品类]\n"
            "\n"
            "*推荐理由来自商品描述和用户评价：*\n"
            "[详细推荐理由全文，字数不限，可多段]\n"
            "\n"
            "## 2. [下一个商品]\n"
            "...\n"
            "\n"
            "格式要点:\n"
            "- 每个商品用 `## 序号. 商品名` 二级标题\n"
            "- 价格用 **¥xxx** 加粗，品牌和子品类跟在后面\n"
            "- 副标 *推荐理由来自...：* 用斜体\n"
            "- 详细理由作为段落正文，不要再加 markdown 装饰\n"
            "- 不要省略任何商品，不要缩减理由文字\n"
        )
        user_prompt = json.dumps(
            {
                "query_plan": plan.model_dump(),
                "final_recommendation_order": [item["product_id"] for item in products],
                "final_products": products,
                "profile_narrative": profile_narrative[:1500],
                "image_attributes": image_attributes or {},
                "instruction": (
                    f"请按 final_recommendation_order 顺序推荐全部 {len(products)} 个 final_products。"
                    "这些商品都是正式推荐结果；不要遗漏，不要另设候补。"
                ),
            },
            ensure_ascii=False,
        )
        return system_prompt, user_prompt

    async def _llm_multi_need_text(
        self,
        state: MultiNeedState,
        selection: MultiNeedSelection,
        route: str,
        verify_reason: str,
        corrective_slot_coverage: list[dict] | None = None,
        rejected_products: list[dict] | None = None,
        combo_summary: dict | None = None,
        profile_narrative: str = "",
    ) -> str:
        system_prompt, user_prompt = self._multi_need_prompt(
            state,
            selection,
            route,
            verify_reason,
            corrective_slot_coverage,
            rejected_products,
            combo_summary,
            profile_narrative=profile_narrative,
        )
        return await self._generate_required(
            system_prompt,
            user_prompt,
            operation="answer_generator.multi_need",
        )

    def _multi_need_prompt(
        self,
        state: MultiNeedState,
        selection: MultiNeedSelection,
        route: str,
        verify_reason: str,
        corrective_slot_coverage: list[dict] | None = None,
        rejected_products: list[dict] | None = None,
        combo_summary: dict | None = None,
        profile_narrative: str = "",
    ) -> tuple[str, str]:
        if not selection.flat_candidates:
            raise RuntimeError("AnswerGenerator 没有可用于生成多需求回答的候选商品。")
        corrective_coverage_by_slot = {
            str(item.get("slot_id")): item
            for item in corrective_slot_coverage or []
            if isinstance(item, dict) and item.get("slot_id")
        }
        selected_ids = {candidate.product_id for candidate in selection.flat_candidates}
        rejected_reason_by_id = {
            str(item.get("product_id")): str(item.get("reason") or "")
            for item in rejected_products or []
            if isinstance(item, dict) and item.get("product_id")
        }
        final_products_by_slot = {}
        alternatives_by_slot = {}
        missing_slots = []
        for slot in state.slots:
            candidates = selection.selected_by_slot.get(slot.slot_id, [])
            coverage = corrective_coverage_by_slot.get(slot.slot_id)
            if candidates:
                final_products_by_slot[slot.slot_id] = {
                    "need_type": slot.need_type,
                    "goal": slot.goal,
                    "product_type": slot.product_type,
                    "source_attribution": self._slot_source_attribution(slot.goal, slot.product_type, state.original_query),
                    "corrective_coverage": coverage,
                    "products": [self._candidate_summary(candidate) for candidate in candidates],
                }
                alternative_candidates = self._slot_candidates_for_ids(
                    state,
                    slot.slot_id,
                    [
                        str(product_id)
                        for product_id in (combo_summary or {}).get("alternative_product_ids_by_slot", {}).get(slot.slot_id, [])
                    ],
                )
                if alternative_candidates:
                    alternatives_by_slot[slot.slot_id] = {
                        "need_type": slot.need_type,
                        "goal": slot.goal,
                        "product_type": slot.product_type,
                        "source_attribution": self._slot_source_attribution(slot.goal, slot.product_type, state.original_query),
                        "products": [self._candidate_summary(candidate) for candidate in alternative_candidates],
                    }
            elif slot.need_type == "required":
                missing_slots.append(
                    {
                        "slot_id": slot.slot_id,
                        "goal": slot.goal,
                        "product_type": slot.product_type,
                        "source_attribution": self._slot_source_attribution(slot.goal, slot.product_type, state.original_query),
                        "corrective_coverage": coverage,
                    }
                )
        near_miss_suggestions = []
        missing_slot_ids = {slot["slot_id"] for slot in missing_slots}
        for slot in state.slots:
            if slot.slot_id not in missing_slot_ids:
                continue
            candidates = [
                candidate
                for candidate in state.candidates_by_slot.get(slot.slot_id, [])
                if candidate.product_id not in selected_ids
            ][:2]
            if not candidates:
                continue
            near_miss_suggestions.append(
                {
                    "slot_id": slot.slot_id,
                    "goal": slot.goal,
                    "product_type": slot.product_type,
                    "source_attribution": self._slot_source_attribution(slot.goal, slot.product_type, state.original_query),
                    "suggestions": [
                        self._near_miss_summary(candidate, rejected_reason_by_id.get(candidate.product_id, ""))
                        for candidate in candidates
                    ],
                }
            )
        slot_brief = [
            {
                "slot_id": slot.slot_id,
                "need_type": slot.need_type,
                "goal": slot.goal,
                "product_type": slot.product_type,
                "source_attribution": self._slot_source_attribution(slot.goal, slot.product_type, state.original_query),
                "corrective_coverage": corrective_coverage_by_slot.get(slot.slot_id),
                "retrieval_coverage": state.coverage_by_slot.get(slot.slot_id).model_dump()
                if state.coverage_by_slot.get(slot.slot_id)
                else None,
            }
            for slot in state.slots
        ]
        is_over_budget_combo = route == "over_budget_combo"
        formal_products_by_slot = {} if is_over_budget_combo else final_products_by_slot
        over_budget_combo_by_slot = final_products_by_slot if is_over_budget_combo else {}
        answer_evidence_brief = self._multi_need_answer_evidence_brief(
            state=state,
            route=route,
            verify_reason=verify_reason,
            slot_brief=slot_brief,
            final_products_by_slot=final_products_by_slot,
            over_budget_combo_by_slot=over_budget_combo_by_slot,
            alternatives_by_slot=alternatives_by_slot,
            missing_slots=missing_slots,
            rejected_products=rejected_products or [],
            combo_summary=combo_summary or {},
        )
        system_prompt = (
            "你是电商导购 RAG Agent。你必须只基于 Answer Evidence Brief 和结构化输入回答，"
            "不得编造商品、价格、库存、优惠、功效、替代方案、检索结果或审核结果。"
            "你不是在自由导购，而是在解释一次已经完成的检索、审核和预算组合结果。\n"
            "面向用户时只输出商品名和价格；不要输出 product_id、slot_id、内部字段名、检索分数或 rerank 分数，"
            "除非用户明确询问技术链路。回答要自然、有导购感，但不要像日志转述。\n"
            "正式推荐区只能写 final_products_by_slot 里的商品；near_miss_suggestions 不是推荐结果，不能放在“已找到/推荐商品”标题下，"
            "也不能展示完整价格、评分、长描述。\n"
            "如果用户需求或 slot 是粗品类/上位商品词，例如护肤品、衣服、鞋子、裤子、饮料、零食、装备，"
            "回答时要保留这个粗品类视角；如果通过商品覆盖多个 sub_category，应按用途/子类呈现多样化选择，不要把它们改写成单一精确子类。\n"
            "只能把审核通过的候选作为可推荐或可考虑候选；被拒候选只能用于解释为什么不能推荐，不能包装成替代推荐。"
            "不得根据常识推断存在更便宜、更平价、更适合或可替换的商品。"
            "如果 Answer Evidence Brief 没有明确给出更便宜且审核通过的候选，不得暗示当前已经存在低价替代。"
            "如果建议用户放宽条件，必须表达为“放宽后可以重新检索确认是否存在更合适/更低价候选”，不能说成当前已经找到了该替代。\n"
            "original_query 才是用户原话；slots/missing_slots 可能是 IntentPlanner 为了规划自动拆出的方向。"
            "只有 source_attribution.source=user_explicit 的 slot，才可以说“你提到/你要求/你指定”。"
            "source_attribution.source=planner_inferred 的 slot 只能说“我按你的整体目标规划出的方向里...”，"
            "不要写成“您提到的 X”。\n"
            f"多需求每个 slot 主推最多 {MULTI_NEED_PRIMARY_PER_SLOT} 个；"
            f"alternatives_by_slot 是同 slot 语义通过但没有进入主推组合的备选商品，每个 slot 最多 {MULTI_NEED_ALTERNATIVES_PER_SLOT} 个，"
            "可以在主推组合之后用“同类备选/可替换项”简短展示；"
            "它们是可推荐商品，不要写成拒绝或 near-miss。\n"
            "over_budget_combo_by_slot 只在 route=over_budget_combo 时出现，表示最低完整组合候选，不是正式推荐商品；"
            "这时不要使用“已为你推荐/已找到可买组合”这类确定口吻。"
            "必须说明用户预算、当前完整组合总价、超预算金额，并说明它只是超预算完整组合候选。"
            "如果 Answer Evidence Brief 说明当前组合是本轮通过候选中的最低完整组合，不要暗示当前已有更低价完整组合。"
            "如果 Answer Evidence Brief 说明没有预算内完整组合，不要建议当前已有更便宜的完整替代方案。\n"
            "near_miss_suggestions 只用于某个 required slot 完全没有正式通过商品时，写成询问口吻："
            "“我目前只检索到相近方向，例如 X，但它不是你要的 Y；要不要放宽到这个方向？” "
            "如果 slot 已有正式通过商品，不要提该 slot 的 near-miss。\n"
            "profile_narrative 只用于表达个性化理由，不能影响正式推荐、预算组合、slot 覆盖判断；"
            "不要把画像内容说成用户本轮明确要求。\n"
            "route playbook: "
            "recommend=按 slot 分组推荐完整组合；有 total budget 时明确组合总价。"
            "over_budget_combo=说明检索到了覆盖全部 slot 的最低完整组合候选，但最低总价 X 比预算 Y 高 Z；列 over_budget_combo_by_slot 中的候选；自然询问用户是否接受超预算、提高预算、减少购买项或放宽条件，不要写成已正式推荐。"
            "partial_recommend=先推荐已覆盖 slot，再说明缺失 required slots；只有缺失 slot 才能提 near-miss 替代方向。"
            "no_product/reject_candidates=说明当前商品库没有足够匹配，可给 1-2 个 near-miss 放宽方向，但不能写成正式推荐。"
            "clarify=像真人导购一样追问一个关键问题，可给少量选项引导。"
        )
        user_prompt = json.dumps(
            {
                "original_query": state.original_query,
                "route": route,
                "verify_reason": verify_reason,
                "combo_summary": combo_summary or {},
                "answer_evidence_brief": answer_evidence_brief,
                "slots": slot_brief,
                "final_products_by_slot": formal_products_by_slot,
                "over_budget_combo_by_slot": over_budget_combo_by_slot,
                "alternatives_by_slot": alternatives_by_slot,
                "missing_slots": missing_slots,
                "near_miss_suggestions": near_miss_suggestions,
                "rejected_products": rejected_products or [],
                "profile_narrative": profile_narrative[:1500],
            },
            ensure_ascii=False,
        )
        return system_prompt, user_prompt

    async def _llm_direct_text(
        self,
        query: str,
        mode: str,
        reason: str,
        intent_plan: IntentPlan | None,
        extra_context: dict | None,
        profile_narrative: str = "",
    ) -> str:
        system_prompt, user_prompt = self._direct_prompt(
            query,
            mode,
            reason,
            intent_plan,
            extra_context,
            profile_narrative,
        )
        return await self._generate_required(
            system_prompt,
            user_prompt,
            operation="answer_generator.direct_text",
        )

    def _direct_prompt(
        self,
        query: str,
        mode: str,
        reason: str,
        intent_plan: IntentPlan | None,
        extra_context: dict | None,
        profile_narrative: str = "",
    ) -> tuple[str, str]:
        system_prompt = (
            "你是电商导购助手。根据模式回答：\n"
            "direct：回答常识性问题，不要编造当前商品库证据。\n"
            "当 extra_context.referenced_products 存在时，"
            "表示 Orchestrator 已批准基于短期上下文商品证据回答；这时只能使用 extra_context.referenced_products 和 conversation_evidence 中的商品事实，"
            "不要新增、替换或臆测其他商品，不要声称重新检索过。\n"
            "no_product：明确说明当前商品库没有足够匹配的商品证据。若 extra_context 提供 near_miss_products，可以说明只检索到哪些相近候选/替代品类，并询问用户是否接受替代；不要把它们当作正式匹配推荐。\n"
            "clarification：用户需求缺少关键商品品类。自然地反问一个澄清问题，并给出当前支持品类示例。\n"
            "若 extra_context.image_attributes.available=true，只能用“根据图片推测”表达视觉语义，不要说成确定商品事实。\n"
            "回答简洁、可信。"
        )
        user_prompt = json.dumps(
            {
                "query": query,
                "mode": mode,
                "reason": reason,
                "intent_plan": intent_plan.model_dump() if intent_plan else None,
                "extra_context": extra_context or {},
                "profile_narrative": profile_narrative[:1500],
            },
            ensure_ascii=False,
        )
        return system_prompt, user_prompt

    async def _generate_stream_or_fallback(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        operation: str,
    ) -> AsyncGenerator[str, None]:
        call = getattr(self.llm_client, "generate_stream_required", None)
        if callable(call):
            kwargs = {"operation": operation} if self._supports_parameter(call, "operation") else {}
            async for delta in call(system_prompt, user_prompt, **kwargs):
                if delta:
                    yield delta
            return

        text = await self._generate_required(system_prompt, user_prompt, operation=operation)
        for token in self._chunk_text(text):
            await asyncio.sleep(0.01)
            yield token

    async def _generate_required(self, system_prompt: str, user_prompt: str, *, operation: str) -> str:
        call = self.llm_client.generate_required
        kwargs = {"operation": operation} if self._supports_parameter(call, "operation") else {}
        return await call(system_prompt, user_prompt, **kwargs)

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

    def _multi_need_answer_evidence_brief(
        self,
        *,
        state: MultiNeedState,
        route: str,
        verify_reason: str,
        slot_brief: list[dict],
        final_products_by_slot: dict,
        over_budget_combo_by_slot: dict,
        alternatives_by_slot: dict,
        missing_slots: list[dict],
        rejected_products: list[dict],
        combo_summary: dict,
    ) -> str:
        combo_status = str(combo_summary.get("status") or "")
        is_lowest_complete_combo = route == "over_budget_combo" and combo_status == "over_budget"
        has_within_budget_complete_combo = combo_status == "within_budget"
        budget_max = combo_summary.get("budget_max")
        total_price = combo_summary.get("total_price")
        over_budget_amount = combo_summary.get("over_budget_amount")

        search_calls = state.budgets.get("search_calls")
        if search_calls is None:
            search_calls = sum(1 for call in state.tool_calls if call.action == "search_products")
        repair_attempts = sum(1 for call in state.tool_calls if "repair" in str(call.action))

        lines = [
            "Answer Evidence Brief:",
            "",
            f"用户原始需求：{state.original_query}",
            f"最终 route：{route}",
            "",
            "证据边界摘要：系统已完成本轮检索、候选审核和预算组合判断。以下商品、价格、通过/拒绝状态是本轮唯一可用证据；不要使用 Evidence Brief 之外的商品或价格。",
            "",
            "需求拆分：",
        ]
        for slot in state.slots:
            lines.append(
                f"- {slot.goal or slot.product_type}：need_type={slot.need_type}，检索 query={slot.query}，候选数={len(state.candidates_by_slot.get(slot.slot_id, []))}"
            )

        lines.extend(
            [
                "",
                "检索过程摘要：",
                f"- search 调用：{search_calls}",
                f"- repair/retry 调用：{repair_attempts}",
            ]
        )
        for item in slot_brief:
            coverage = item.get("retrieval_coverage") or {}
            if coverage:
                lines.append(
                    f"- {item.get('goal') or item.get('product_type')}：覆盖状态={coverage.get('status')}，候选数={coverage.get('candidate_count')}，尝试次数={coverage.get('attempt_count')}"
                )

        lines.extend(["", f"CorrectiveAgent 审核结论：{verify_reason or '-'}", "", "审核通过候选："])
        passed_by_slot = combo_summary.get("semantic_passed_product_ids_by_slot")
        if not isinstance(passed_by_slot, dict):
            passed_by_slot = combo_summary.get("final_combo_product_ids_by_slot")
        if not isinstance(passed_by_slot, dict):
            passed_by_slot = {
                slot_id: [product["product_id"] for product in data.get("products", []) if product.get("product_id")]
                for slot_id, data in final_products_by_slot.items()
            }
        self._append_candidates_by_slot(lines, state, passed_by_slot, default_result="通过")

        lines.extend(["", "审核拒绝候选："])
        rejected_lines = self._rejected_candidate_lines(state, rejected_products)
        lines.extend(rejected_lines or ["- 无"])

        lines.extend(["", "当前组合："])
        combo_source = over_budget_combo_by_slot if over_budget_combo_by_slot else final_products_by_slot
        if combo_source:
            for slot_id, data in combo_source.items():
                label = self._slot_label(state, slot_id, data)
                for product in data.get("products", []):
                    lines.append(
                        f"- {label}：{product.get('name')}，价格 {product.get('price')} 元"
                    )
        else:
            lines.append("- 无完整组合")

        lines.extend(
            [
                "",
                "预算判断：",
                f"- 用户预算上限：{budget_max if budget_max is not None else '未知'} 元",
                f"- 当前组合总价：{total_price if total_price is not None else '未知'} 元",
                f"- 超出预算：{over_budget_amount if over_budget_amount is not None else '未知'} 元",
                f"- 当前组合是否为本轮通过候选中的最低完整组合：{'是' if is_lowest_complete_combo else '否/不适用'}",
                f"- 本轮是否发现预算内完整组合：{'是' if has_within_budget_complete_combo else '否'}",
                "",
                "同需求备选：",
            ]
        )
        if alternatives_by_slot:
            for slot_id, data in alternatives_by_slot.items():
                label = self._slot_label(state, slot_id, data)
                selected_prices = self._selected_prices_for_slot(final_products_by_slot, slot_id)
                for product in data.get("products", []):
                    relation = self._price_relation(product.get("price"), selected_prices)
                    lines.append(
                        f"- {label}：{product.get('name')}，价格 {product.get('price')} 元，价格关系：{relation}"
                    )
        else:
            lines.append("- 无")

        if missing_slots:
            lines.extend(["", "缺失需求："])
            for slot in missing_slots:
                lines.append(f"- {slot.get('goal') or slot.get('product_type')}：没有审核通过候选")

        lines.extend(
            [
                "",
                "回答注意事项：",
                "- product_id 只用于系统追踪，最终回复用户时不要输出。",
                "- 如果没有预算内完整组合，请不要暗示当前已有更便宜的完整替代方案。",
                "- 如果某个需求下没有明确列出更便宜且审核通过的候选，请不要说可以换成更平价的该类商品。",
                "- 可以询问用户是否接受超预算、提高预算、减少购买项，或放宽条件后重新检索。",
            ]
        )
        return "\n".join(lines)

    def _append_candidates_by_slot(
        self,
        lines: list[str],
        state: MultiNeedState,
        product_ids_by_slot: dict,
        *,
        default_result: str,
    ) -> None:
        appended = False
        for slot in state.slots:
            product_ids = [str(product_id) for product_id in product_ids_by_slot.get(slot.slot_id, [])]
            candidates = self._slot_candidates_for_ids(state, slot.slot_id, product_ids)
            if not candidates:
                continue
            appended = True
            lines.append(f"- {slot.goal or slot.product_type}：")
            for candidate in candidates:
                reason = candidate.coverage_reason or "匹配该需求"
                lines.append(
                    f"  - {candidate.name}（product_id={candidate.product_id}），价格 {candidate.price} 元，审核结果={default_result}，原因={reason}"
                )
        if not appended:
            lines.append("- 无")

    def _rejected_candidate_lines(self, state: MultiNeedState, rejected_products: list[dict]) -> list[str]:
        by_id = {
            candidate.product_id: candidate
            for candidates in state.candidates_by_slot.values()
            for candidate in candidates
        }
        lines = []
        seen = set()
        for item in rejected_products:
            product_id = str(item.get("product_id") or "").strip()
            if not product_id or product_id in seen:
                continue
            seen.add(product_id)
            reason = str(item.get("reason") or "未通过审核")
            candidate = by_id.get(product_id)
            if candidate is None:
                lines.append(f"- product_id={product_id}，审核结果=拒绝，原因={reason}")
                continue
            lines.append(
                f"- {candidate.name}（product_id={candidate.product_id}），价格 {candidate.price} 元，审核结果=拒绝，原因={reason}"
            )
        return lines

    def _slot_label(self, state: MultiNeedState, slot_id: str, data: dict) -> str:
        slot = state.slot_by_id(slot_id)
        if slot is not None:
            return slot.goal or slot.product_type or slot_id
        return str(data.get("goal") or data.get("product_type") or slot_id)

    def _selected_prices_for_slot(self, final_products_by_slot: dict, slot_id: str) -> list[float]:
        prices = []
        for product in final_products_by_slot.get(slot_id, {}).get("products", []):
            try:
                prices.append(float(product.get("price")))
            except (TypeError, ValueError):
                continue
        return prices

    def _price_relation(self, price: object, selected_prices: list[float]) -> str:
        if not selected_prices:
            return "无法和当前主选比较"
        try:
            value = float(price)
        except (TypeError, ValueError):
            return "无法和当前主选比较"
        baseline = min(selected_prices)
        if value < baseline:
            return "低于当前主选价格，可能降低该需求价格"
        if value == baseline:
            return "等于当前主选价格，不能降低总价"
        return "高于当前主选价格，不能降低总价"

    def _candidate_summary(self, candidate: SlotCandidate) -> dict:
        product = candidate.product
        return {
            "product_id": product.product_id,
            "name": product.name,
            "brand": product.brand,
            "category": product.category,
            "sub_category": product.sub_category,
            "price": float(product.price),
            "rating": float(product.rating),
            "tags": product.tags[:6],
            "description": product.description[:350],
            "review_summary": self._short_review(product, 220),
            "rerank_score": candidate.rerank_score,
            "coverage_reason": candidate.coverage_reason,
        }

    def _near_miss_summary(self, candidate: SlotCandidate, rejection_reason: str) -> dict:
        return {
            "product_id": candidate.product_id,
            "name": candidate.name,
            "category": candidate.category,
            "sub_category": candidate.sub_category,
            "difference_reason": rejection_reason or candidate.coverage_reason,
        }

    def _slot_source_attribution(self, goal: str, product_type: str, original_query: str) -> dict:
        explicit_terms = [
            term
            for term in self._slot_terms(goal, product_type)
            if term and term in original_query
        ]
        return {
            "source": "user_explicit" if explicit_terms else "planner_inferred",
            "matched_terms": explicit_terms,
            "instruction": (
                "可以按用户明确提到的需求表述。"
                if explicit_terms
                else "这是系统规划出的子方向，不要写成用户明确提到。"
            ),
        }

    def _slot_terms(self, goal: str, product_type: str) -> list[str]:
        raw_terms = [goal, product_type]
        terms: list[str] = []
        for raw in raw_terms:
            for term in re.split(r"或|和|与|[\/、,，;；|｜\s]+", str(raw or "")):
                text = term.strip()
                if len(text) >= 2 and text not in terms:
                    terms.append(text)
        return terms

    def _slot_candidates_for_ids(
        self,
        state: MultiNeedState,
        slot_id: str,
        product_ids: list[str],
    ) -> list[SlotCandidate]:
        by_id = {
            candidate.product_id: candidate
            for candidate in state.candidates_by_slot.get(slot_id, [])
        }
        return [by_id[product_id] for product_id in product_ids if product_id in by_id]

    def _short_review(self, product: Product, limit: int = 180) -> str:
        text = " ".join(product.review_summary.split())
        return text[:limit] if text else product.description[:limit]

    def _chunk_text(self, text: str, size: int = 12) -> list[str]:
        return [text[index : index + size] for index in range(0, len(text), size)]
