from __future__ import annotations

import json
from itertools import product as iter_product
from typing import Any

from app.db.models import Product
from app.domain.need_slot_schemas import MultiNeedState, SlotCandidate
from app.schemas import (
    MULTI_NEED_ALTERNATIVES_PER_SLOT,
    MULTI_NEED_PRIMARY_PER_SLOT,
    IntentPlan,
    QueryPlan,
    ReflectionResult,
    RepairHint,
    SINGLE_RECOMMENDATION_LIMIT,
    SINGLE_RETRIEVAL_REVIEW_LIMIT,
)
from app.services.llm_client import LlmClient
from app.services.structured_llm import StructuredLlmValidationError, generate_validated_json


class CorrectiveAgentController:
    JSON_RESPONSE_FORMAT = {"type": "json_object"}
    FALLBACK_PLANS = {"none", "direct_answer", "clarify", "no_product"}

    def __init__(self, llm_client: LlmClient | None = None) -> None:
        self.llm_client = llm_client or LlmClient(component="CorrectiveAgent")

    async def review(
        self,
        original_query: str,
        intent_plan: IntentPlan,
        plan: QueryPlan,
        ranked: list[tuple[Product, float]],
        vector_scores: dict[str, float],
        keyword_scores: dict[str, float],
        image_attributes: dict[str, Any] | None = None,
    ) -> ReflectionResult:
        if not ranked:
            return ReflectionResult(
                has_passed_products=False,
                reason="No candidate products entered Corrective Agent review.",
                used_llm=False,
                fallback_plan="none",
                repair_hint=RepairHint(
                    repairable=True,
                    target_slot_ids=["single"],
                    failure_type="no_candidates",
                    reason="No candidate products entered Corrective Agent review.",
                ),
            )
        review_limit = self._single_review_limit(plan, ranked)
        if not self._llm_is_configured():
            passed_ids = [product.product_id for product, _ in ranked[:review_limit]]
            return ReflectionResult(
                has_passed_products=bool(passed_ids),
                reason="LLM is not configured; using ranked candidates as semantic fallback.",
                used_llm=False,
                passed_product_ids=passed_ids,
                fallback_plan="none" if passed_ids else "no_product",
            )

        candidates = [
            {
                "product_id": product.product_id,
                "name": product.name,
                "category": product.category,
                "sub_category": product.sub_category,
                "brand": product.brand,
                "price": float(product.price),
                "description": product.description[:400],
                "suitable_for": product.suitable_for,
                "avoid_for": product.avoid_for,
                "tags": product.tags,
                "review_summary": product.review_summary[:500],
                "rerank_score": round(rerank_score, 4),
                "vector_score": round(vector_scores.get(product.product_id, 0.0), 4),
                "keyword_score": round(keyword_scores.get(product.product_id, 0.0), 4),
            }
            for product, rerank_score in ranked[:review_limit]
        ]
        valid_ids = {product.product_id for product, _ in ranked}
        system_prompt = (
            "你是电商 RAG Harness 的 CorrectiveAgent（证据反射 Worker Agent）。只输出 JSON object，不要输出 Markdown。\n"
            "你的职责是审核 rerank 后的候选商品证据是否真的支撑当前用户需求；你不决定 final_route，不生成回答。\n"
            "只能依据输入候选证据，不得补充商品、价格、库存、优惠、功效或用户没有说过的约束。\n"
            "必须重点检查商品形态、商品族、核心功能、使用场景、适用对象、用户明确偏好、排除项和商品证据。\n\n"
            "## 语义审核标准\n"
            "- 不要只做字面匹配；当用户没有要求精确形态时，同一商品族、同一核心功能、同一使用部位/对象、同一使用场景的候选可以通过。\n"
            "- 不要因为候选用了相邻名称或形态名称就直接拒绝；要看商品描述、标签、适用人群、评价摘要是否能支撑需求。\n"
            "- 当用户使用粗品类或上位商品词时，例如「护肤品」「衣服」「鞋子」「裤子」「饮料」「零食」「电脑」「手机」，"
            "不要按单一精确子类审核；只要候选属于该上位商品族并且证据支持使用场景/用途，就可以通过。\n"
            "- 粗品类请求应尽量保留多个合理子类的通过商品，除非候选证据明显不匹配或违反用户约束；"
            "不要因为已经有一个子类通过，就把其它同族有效候选全部拒绝。\n"
            "- 如果候选改变了商品族、不能满足核心功能，或违反用户明确约束，必须拒绝。\n"
            "- 品牌、性别、年龄、人群、预算、材质、颜色、成分等条件，只有用户明确提出时才作为硬约束；用户没说时不要擅自当作拒绝理由。\n"
            "- 预算只用于判断明显单品超预算；不要自行计算多商品组合总价，组合和 final_route 由 Orchestrator 裁决。\n\n"
            "## 图片属性审核规则\n"
            "- 如果输入包含 image_attributes，它只是本轮图片输入的视觉语义推测，不是事实源。\n"
            "- 用户文本约束优先；图片颜色、风格、材质、场景只作为软补充，不能覆盖用户明确需求。\n"
            "- 可以结合图片相似分数、视觉语义摘要和商品文本证据判断候选是否合理，但不得假设商品证据里没有的事实。\n\n"
            "## 正反例说明\n"
            "- 反例：防晒衣/皮肤衣/外套不是防晒霜/防晒乳；手机支架不是手机；电脑主机/台式机不是笔记本电脑/平板电脑；鞋子不是裤子/帽子/背包。\n"
            "- 正例：用户说「新手化妆」时，粉底液、蜜粉、卸妆、唇妆类商品都可能匹配；用户说「户外露营」时，背包、徒步鞋、帽子、户外裤都可能匹配；用户说「运动装备」时，运动鞋、运动服装、运动配件都可能匹配。\n"
            "- 正例：用户说「护肤品，皮肤状态不好」时，精华、面霜、眼霜、面膜、化妆水等护肤子类都可能匹配；彩妆类如粉底、唇釉、眉笔通常不算护肤。\n"
            "- 正例：用户说「衣服，日常通勤穿」时，T 恤、速干上衣、卫衣等上衣子类都可能匹配；鞋子、裤子、帽子、背包通常不算衣服。\n"
            "- 这些例子是语义审核原则，不是固定映射表；最终必须以当前 query、IntentPlan、QueryPlan 和候选证据为准。\n\n"
            "## fallback_plan 边界\n"
            "- 如果有候选应该通过，把 ID 放入 passed_product_ids，并设置 fallback_plan='none'。\n"
            "- 如果当前请求本不该进入商品检索，例如身份/闲聊/系统能力问题，设置 fallback_plan='direct_answer'。\n"
            "- 如果用户确实有购物意图但商品需求缺关键条件，设置 fallback_plan='clarify'。\n"
            "- 如果商品需求明确但候选都不匹配，设置 fallback_plan='no_product'。\n"
            "- fallback_plan 只是给 Orchestrator 的反射建议，不是 final_route。\n\n"
            "返回 JSON schema: {"
            "\"reason\":\"中文原因\","
            "\"fallback_plan\":\"none|direct_answer|clarify|no_product\","
            "\"passed_product_ids\":[\"...\"],"
            "\"rejected_products\":[{\"product_id\":\"...\",\"reason\":\"中文拒绝原因\"}],"
            "\"repair_hint\":{\"repairable\":false,\"target_slot_ids\":[\"single\"],\"failure_type\":\"\",\"missing_terms\":[\"...\"],\"avoid_terms\":[\"...\"],\"reason\":\"中文诊断原因\"}"
            "}"
        )
        user_prompt = json.dumps(
            {
                "original_query": original_query,
                "intent_plan": intent_plan.model_dump(),
                "query_plan": plan.model_dump(),
                "image_attributes": image_attributes or {},
                "candidates": candidates,
            },
            ensure_ascii=False,
        )
        try:
            data = await generate_validated_json(
                self.llm_client,
                system_prompt,
                user_prompt,
                validate=lambda value: self._validate_review_data(value, valid_ids),
                error_message="Corrective Agent reflection returned invalid JSON.",
                response_format=self.JSON_RESPONSE_FORMAT,
                operation="corrective_agent.review",
            )
        except StructuredLlmValidationError:
            return ReflectionResult(
                has_passed_products=False,
                reason="Corrective Agent output validation failed; conservatively rejected candidates.",
                used_llm=True,
                rejected_products=[
                    {"product_id": product.product_id, "reason": "Corrective Agent output validation failed."}
                    for product, _ in ranked[:review_limit]
                ],
                fallback_plan="no_product",
            )

        passed_ids = [product_id for product_id in self._string_list(data.get("passed_product_ids")) if product_id in valid_ids]
        rejected_products = data.get("rejected_products") if isinstance(data.get("rejected_products"), list) else []
        fallback_plan = self._fallback_plan(data.get("fallback_plan"))
        if passed_ids:
            fallback_plan = "none"
        repair_hint = self._repair_hint(data.get("repair_hint"), fallback_plan, default_slot_ids=["single"])
        if passed_ids:
            repair_hint = RepairHint()
        return ReflectionResult(
            has_passed_products=bool(passed_ids),
            reason=str(data.get("reason") or ""),
            used_llm=True,
            passed_product_ids=passed_ids,
            rejected_products=rejected_products,
            fallback_plan=fallback_plan,
            repair_hint=repair_hint,
        )

    def _single_review_limit(self, plan: QueryPlan, ranked: list[tuple[Product, float]]) -> int:
        requested = max(SINGLE_RECOMMENDATION_LIMIT, plan.retrieval_strategy.final_top_k)
        return min(len(ranked), requested, SINGLE_RETRIEVAL_REVIEW_LIMIT)

    async def review_slots(
        self,
        original_query: str,
        intent_plan: IntentPlan,
        plan: QueryPlan,
        state: MultiNeedState,
    ) -> ReflectionResult:
        flat_candidates = [
            candidate
            for slot in state.slots
            for candidate in state.candidates_by_slot.get(slot.slot_id, [])
        ]
        if not flat_candidates:
            return ReflectionResult(
                has_passed_products=False,
                reason="No candidate products entered multi-need Corrective Agent review.",
                used_llm=False,
                combo_summary=self._empty_combo_summary(state, status="no_candidates"),
                fallback_plan="none",
                repair_hint=RepairHint(
                    repairable=True,
                    target_slot_ids=[slot.slot_id for slot in state.slots],
                    failure_type="no_candidates",
                    reason="No candidate products entered multi-need Corrective Agent review.",
                ),
            )

        fallback = self._multi_need_fallback_decision(state)
        if not self._llm_is_configured():
            return fallback

        system_prompt = (
            "你是电商多需求 RAG Harness 的 CorrectiveAgent（多需求证据反射 Worker Agent）。只输出 JSON object，不要输出 Markdown。\n"
            "你的职责是一次性审核每个 need slot 是否被候选商品语义覆盖；你不决定 final_route，不生成回答，不计算最终预算路线。\n"
            "只能依据输入候选证据，不得补充商品、价格、库存、优惠、功效或用户没有说过的约束。\n\n"
            "## slot 语义覆盖标准\n"
            "- 每个 slot 只按自己的 goal / product_type / query 和用户明确约束审核，不要把其它 slot 的商品词拼进来。\n"
            "- 商品形态、商品族、核心功能、用途、用户明确偏好、排除项和用户原话需要被候选证据支持；仅共享场景词不算覆盖。\n"
            "- 不要只做字面匹配；当用户没有要求精确形态时，同一商品族、同一核心功能、同一使用部位/对象、同一使用场景的候选可以覆盖 slot。\n"
            "- 当 slot 是粗品类或上位商品词时，例如「护肤品」「衣服」「鞋子」「裤子」「饮料」「零食」「装备」，"
            "不要按单一精确子类审核；同一上位商品族下的多个合理子类都可以覆盖该 slot。\n"
            "- 如果候选改变了商品族、不能满足 slot 核心功能，或违反用户明确约束，必须拒绝。\n"
            "- 品牌、性别、年龄、人群、预算、材质、颜色、成分等条件，只有用户明确提出时才作为硬约束；用户没说时不要擅自当作拒绝理由。\n"
            "- 如果用户明确说「女生/女款/不要男生」，男款或明显男性专用商品必须拒绝；中性商品可以通过，但 reason 要说明是中性/未标男款。\n\n"
            "## 正反例说明\n"
            "- 反例：电脑主机/台式机不是笔记本电脑/平板电脑；手机支架不是手机；防晒衣不是防晒霜；运动帽/背包不是运动鞋；运动裤不是运动鞋。\n"
            "- 正例：运动配件 slot 可以包含运动帽、运动背包等；运动服装 slot 可以包含运动裤、瑜伽裤、T 恤等；底妆 slot 可以包含粉底液/气垫；定妆 slot 可以包含蜜粉/散粉；眉妆 slot 可以包含眉笔/眉粉；唇妆 slot 可以包含唇釉/口红。\n"
            "- 唇釉不是眉妆，蜜粉不是底妆，防晒乳/卸妆产品不能覆盖彩妆 slot，除非 slot 本身就是防晒/卸妆。\n"
            "- 这些例子是语义审核原则，不是固定映射表；最终必须以当前 query、slots 和候选证据为准。\n\n"
            "## 多需求输出边界\n"
            "- partial 是合法证据状态：部分 required slot 没有语义通过商品时，必须保留已覆盖 slot 的有效商品，并说明缺口；不要因为一个 required slot 缺失而拒绝其它已覆盖 slot。\n"
            "- slot 内不匹配写入对应 slot_coverage[].rejected_product_ids 和 reason。\n"
            "- rejected_products 只放不适合任何 slot 的全局拒绝商品；不要把已在某个 slot 通过的商品写入 rejected_products。\n"
            "- slot_coverage[].selected_product_ids 是商品归属到哪个 slot 的唯一依据。\n"
            "- passed_product_ids 必须是所有 slot_coverage selected_product_ids 的去重并集。\n"
            "- 你只审核语义覆盖；预算组合、over_budget_combo、recommend、partial_recommend、no_product 等 final_route 由 Orchestrator 裁决。\n"
            "- fallback_plan 只在没有任何商品应该通过时使用：direct_answer、clarify 或 no_product；有通过商品时必须是 none。\n\n"
            "返回 JSON schema: {"
            "\"reason\":\"中文原因\","
            "\"fallback_plan\":\"none|direct_answer|clarify|no_product\","
            "\"slot_coverage\":[{\"slot_id\":\"s1\",\"status\":\"covered|missing|rejected\",\"selected_product_ids\":[\"...\"],\"rejected_product_ids\":[\"...\"],\"reason\":\"中文原因\"}],"
            "\"passed_product_ids\":[\"...\"],"
            "\"rejected_products\":[{\"product_id\":\"...\",\"reason\":\"中文拒绝原因\"}],"
            "\"repair_hint\":{\"repairable\":false,\"target_slot_ids\":[\"s1\"],\"failure_type\":\"\",\"missing_terms\":[\"...\"],\"avoid_terms\":[\"...\"],\"reason\":\"中文诊断原因\"}"
            "}"
        )
        user_prompt = json.dumps(
            {
                "original_query": original_query,
                "intent_plan": intent_plan.model_dump(),
                "query_plan": plan.model_dump(),
                "slots": [slot.model_dump() for slot in state.slots],
                "coverage_by_slot": {key: value.model_dump() for key, value in state.coverage_by_slot.items()},
                "retrieved_products_by_slot": self._candidates_by_slot_for_prompt(state),
                "slot_retrieval_results": state.budgets.get("slot_results_by_slot", {}),
                "budget_instruction": {
                    "budget_scope": intent_plan.budget_scope,
                    "budget_min": plan.budget.min if plan.budget.min is not None else intent_plan.budget_min,
                    "budget_max": plan.budget.max if plan.budget.max is not None else intent_plan.budget_max,
                    "note": "Only audit semantic fit. Orchestrator decides final_route.",
                },
            },
            ensure_ascii=False,
        )
        try:
            data = await generate_validated_json(
                self.llm_client,
                system_prompt,
                user_prompt,
                validate=lambda value: self._validate_review_slots_data(value, state),
                error_message="Multi-need Corrective Agent reflection returned invalid JSON.",
                response_format=self.JSON_RESPONSE_FORMAT,
                operation="corrective_agent.review_slots",
            )
        except StructuredLlmValidationError:
            return self._multi_need_validation_failure_decision(state)

        rejected_products = data.get("rejected_products") if isinstance(data.get("rejected_products"), list) else []
        slot_coverage = data.get("slot_coverage") if isinstance(data.get("slot_coverage"), list) else []
        explicit_passed_ids = self._string_list(data.get("passed_product_ids"))
        passed_by_slot = self._semantic_passed_by_slot(state, slot_coverage, explicit_passed_ids)
        passed_ids, combo_summary = self._build_combo_reflection(state, passed_by_slot)
        normalized_coverage = self._normalized_slot_coverage(state, slot_coverage, passed_by_slot)
        fallback_plan = self._fallback_plan(data.get("fallback_plan"))
        if passed_ids:
            fallback_plan = "none"
        missing_slot_ids = [
            str(item.get("slot_id"))
            for item in normalized_coverage
            if isinstance(item, dict) and str(item.get("status") or "") in {"missing", "rejected"}
        ]
        repair_hint = self._repair_hint(data.get("repair_hint"), fallback_plan, default_slot_ids=missing_slot_ids)
        if passed_ids and not missing_slot_ids:
            repair_hint = RepairHint()
        return ReflectionResult(
            has_passed_products=bool(passed_ids),
            reason=str(data.get("reason") or fallback.reason),
            used_llm=True,
            passed_product_ids=passed_ids,
            rejected_products=rejected_products,
            slot_coverage=normalized_coverage,
            combo_summary=combo_summary,
            fallback_plan=fallback_plan,
            repair_hint=repair_hint,
        )

    def _multi_need_fallback_decision(self, state: MultiNeedState) -> ReflectionResult:
        passed_by_slot = {
            slot.slot_id: [
                candidate.product_id
                for candidate in state.candidates_by_slot.get(slot.slot_id, [])[: max(1, slot.min_candidates)]
            ]
            for slot in state.slots
        }
        passed_ids, combo_summary = self._build_combo_reflection(state, passed_by_slot)
        missing_slot_ids = [
            slot.slot_id
            for slot in state.slots
            if slot.need_type == "required" and not passed_by_slot.get(slot.slot_id)
        ]
        return ReflectionResult(
            has_passed_products=bool(passed_ids),
            reason="LLM is not configured; using retrieved slot candidates as semantic fallback.",
            used_llm=False,
            passed_product_ids=passed_ids,
            rejected_products=[],
            slot_coverage=self._normalized_slot_coverage(state, [], passed_by_slot),
            combo_summary=combo_summary,
            fallback_plan="none" if passed_ids else "no_product",
            repair_hint=RepairHint(
                repairable=bool(missing_slot_ids),
                target_slot_ids=missing_slot_ids,
                failure_type="slot_empty",
                reason="Required slots have no fallback candidates.",
            ),
        )

    def _multi_need_validation_failure_decision(self, state: MultiNeedState) -> ReflectionResult:
        return ReflectionResult(
            has_passed_products=False,
            reason="Corrective Agent output validation failed; conservatively rejected multi-need candidates.",
            used_llm=True,
            passed_product_ids=[],
            rejected_products=[],
            slot_coverage=self._normalized_slot_coverage(state, [], {slot.slot_id: [] for slot in state.slots}),
            combo_summary=self._empty_combo_summary(state, status="validation_failed"),
            fallback_plan="no_product",
        )

    def _candidates_by_slot_for_prompt(self, state: MultiNeedState) -> dict[str, list[dict]]:
        return {
            slot.slot_id: [
                self._slot_candidate_payload(candidate)
                for candidate in state.candidates_by_slot.get(slot.slot_id, [])
            ]
            for slot in state.slots
        }

    def _semantic_passed_by_slot(
        self,
        state: MultiNeedState,
        slot_coverage: Any,
        explicit_passed_ids: list[str],
    ) -> dict[str, list[str]]:
        valid_by_slot = {
            slot.slot_id: [candidate.product_id for candidate in state.candidates_by_slot.get(slot.slot_id, [])]
            for slot in state.slots
        }
        passed_by_slot: dict[str, list[str]] = {slot.slot_id: [] for slot in state.slots}
        for item in slot_coverage if isinstance(slot_coverage, list) else []:
            if not isinstance(item, dict):
                continue
            slot_id = str(item.get("slot_id") or "").strip()
            if slot_id not in valid_by_slot:
                continue
            selected_ids = [
                product_id
                for product_id in self._string_list(item.get("selected_product_ids"))
                if product_id in valid_by_slot[slot_id]
            ]
            if str(item.get("status") or "") == "covered" and not selected_ids:
                explicit_for_slot = [product_id for product_id in explicit_passed_ids if product_id in valid_by_slot[slot_id]]
                selected_ids = explicit_for_slot[:1]
            passed_by_slot[slot_id] = self._unique([*passed_by_slot[slot_id], *selected_ids])

        if not any(passed_by_slot.values()):
            for product_id in explicit_passed_ids:
                owning_slots = [slot_id for slot_id, valid_ids in valid_by_slot.items() if product_id in valid_ids]
                if len(owning_slots) == 1:
                    slot_id = owning_slots[0]
                    passed_by_slot[slot_id] = self._unique([*passed_by_slot[slot_id], product_id])
        return passed_by_slot

    def _build_combo_reflection(
        self,
        state: MultiNeedState,
        passed_by_slot: dict[str, list[str]],
    ) -> tuple[list[str], dict]:
        required_slots = [slot for slot in state.slots if slot.need_type == "required"]
        missing_required = [slot.slot_id for slot in required_slots if not passed_by_slot.get(slot.slot_id)]
        budget_max = self._budget_max(state)
        budget_scope = state.intent_plan.budget_scope or "unknown"

        if missing_required:
            selected_by_slot = self._primary_ids_by_slot(passed_by_slot)
            passed_ids = self._flatten_selected_ids(state, selected_by_slot)
            return passed_ids, self._combo_summary(
                state,
                status="missing_required",
                selected_by_slot=selected_by_slot,
                semantic_passed_by_slot=passed_by_slot,
                budget_scope=budget_scope,
                budget_max=budget_max,
                missing_required_slot_ids=missing_required,
                needs_user_decision=False,
            )

        if budget_scope == "total" and budget_max is not None and required_slots:
            return self._select_total_budget_combo(state, passed_by_slot, budget_max)

        selected_by_slot = self._primary_ids_by_slot(passed_by_slot)
        passed_ids = self._flatten_selected_ids(state, selected_by_slot)
        return passed_ids, self._combo_summary(
            state,
            status="not_applicable" if budget_scope != "total" else "no_budget",
            selected_by_slot=selected_by_slot,
            semantic_passed_by_slot=passed_by_slot,
            budget_scope=budget_scope,
            budget_max=budget_max,
            missing_required_slot_ids=[],
            needs_user_decision=False,
        )

    def _select_total_budget_combo(
        self,
        state: MultiNeedState,
        passed_by_slot: dict[str, list[str]],
        budget_max: float,
    ) -> tuple[list[str], dict]:
        required_slots = [slot for slot in state.slots if slot.need_type == "required"]
        option_groups = [
            self._candidates_for_ids(state, slot.slot_id, passed_by_slot.get(slot.slot_id, []))
            for slot in required_slots
        ]
        combos: list[dict] = []
        for combo_candidates in iter_product(*option_groups):
            product_ids = [candidate.product_id for candidate in combo_candidates]
            if len(set(product_ids)) != len(product_ids):
                continue
            total_price = sum(float(candidate.price) for candidate in combo_candidates)
            score = sum(float(candidate.rerank_score or candidate.rrf_score or candidate.vector_score) for candidate in combo_candidates)
            selected_by_slot = {
                slot.slot_id: [candidate.product_id]
                for slot, candidate in zip(required_slots, combo_candidates, strict=True)
            }
            combos.append({"selected_by_slot": selected_by_slot, "total_price": total_price, "score": score})

        if not combos:
            return [], self._combo_summary(
                state,
                status="no_complete_combo",
                selected_by_slot={},
                semantic_passed_by_slot=passed_by_slot,
                budget_scope="total",
                budget_max=budget_max,
                missing_required_slot_ids=[slot.slot_id for slot in required_slots],
                needs_user_decision=False,
            )

        budgeted = [combo for combo in combos if combo["total_price"] <= budget_max]
        if budgeted:
            chosen = max(budgeted, key=lambda combo: (combo["score"], -combo["total_price"]))
            status = "within_budget"
            needs_user_decision = False
        else:
            chosen = min(combos, key=lambda combo: (combo["total_price"], -combo["score"]))
            status = "over_budget"
            needs_user_decision = True

        selected_by_slot = dict(chosen["selected_by_slot"])
        if status == "within_budget":
            selected_by_slot = self._add_optional_slots_with_remaining_budget(
                state,
                passed_by_slot,
                selected_by_slot,
                budget_max,
            )
        passed_ids = self._flatten_selected_ids(state, selected_by_slot)
        return passed_ids, self._combo_summary(
            state,
            status=status,
            selected_by_slot=selected_by_slot,
            semantic_passed_by_slot=passed_by_slot,
            budget_scope="total",
            budget_max=budget_max,
            total_price=self._total_price_for_ids(state, passed_ids),
            missing_required_slot_ids=[],
            needs_user_decision=needs_user_decision,
        )

    def _add_optional_slots_with_remaining_budget(
        self,
        state: MultiNeedState,
        passed_by_slot: dict[str, list[str]],
        selected_by_slot: dict[str, list[str]],
        budget_max: float,
    ) -> dict[str, list[str]]:
        selected_ids = self._flatten_selected_ids(state, selected_by_slot)
        used_price = self._total_price_for_ids(state, selected_ids)
        for slot in state.slots:
            if slot.need_type != "optional" or selected_by_slot.get(slot.slot_id):
                continue
            for candidate in self._candidates_for_ids(state, slot.slot_id, passed_by_slot.get(slot.slot_id, [])):
                if candidate.product_id in selected_ids:
                    continue
                if used_price + float(candidate.price) <= budget_max:
                    selected_by_slot[slot.slot_id] = [candidate.product_id]
                    selected_ids.append(candidate.product_id)
                    used_price += float(candidate.price)
                    break
        return selected_by_slot

    def _normalized_slot_coverage(
        self,
        state: MultiNeedState,
        slot_coverage: Any,
        passed_by_slot: dict[str, list[str]],
    ) -> list[dict]:
        coverage_by_slot: dict[str, dict] = {}
        if isinstance(slot_coverage, list):
            for item in slot_coverage:
                if not isinstance(item, dict):
                    continue
                slot_id = str(item.get("slot_id") or "").strip()
                if slot_id:
                    coverage_by_slot[slot_id] = dict(item)
        for slot in state.slots:
            selected_ids = passed_by_slot.get(slot.slot_id, [])
            item = coverage_by_slot.get(slot.slot_id, {})
            item.setdefault("slot_id", slot.slot_id)
            item["selected_product_ids"] = selected_ids
            item.setdefault("rejected_product_ids", [])
            item["status"] = "covered" if selected_ids else str(item.get("status") or "missing")
            item.setdefault("reason", "有语义通过候选。" if selected_ids else "没有语义通过候选。")
            coverage_by_slot[slot.slot_id] = item
        return [coverage_by_slot[slot.slot_id] for slot in state.slots]

    def _combo_summary(
        self,
        state: MultiNeedState,
        *,
        status: str,
        selected_by_slot: dict[str, list[str]],
        semantic_passed_by_slot: dict[str, list[str]] | None = None,
        budget_scope: str,
        budget_max: float | None,
        missing_required_slot_ids: list[str],
        needs_user_decision: bool,
        total_price: float | None = None,
    ) -> dict:
        passed_ids = self._flatten_selected_ids(state, selected_by_slot)
        semantic_passed_by_slot = {
            slot_id: self._unique(product_ids)
            for slot_id, product_ids in (semantic_passed_by_slot or selected_by_slot).items()
            if product_ids
        }
        alternative_by_slot = self._alternative_ids_by_slot(state, semantic_passed_by_slot, selected_by_slot)
        semantic_passed_ids = self._flatten_selected_ids(state, semantic_passed_by_slot)
        alternative_ids = self._flatten_selected_ids(state, alternative_by_slot)
        if total_price is None and passed_ids:
            total_price = self._total_price_for_ids(state, passed_ids)
        over_budget_amount = None
        if total_price is not None and budget_max is not None:
            over_budget_amount = max(0.0, total_price - budget_max)
        return {
            "status": status,
            "budget_scope": budget_scope,
            "budget_max": budget_max,
            "total_price": round(total_price, 2) if total_price is not None else None,
            "over_budget_amount": round(over_budget_amount, 2) if over_budget_amount is not None else None,
            "selected_product_ids_by_slot": selected_by_slot,
            "final_combo_product_ids_by_slot": selected_by_slot,
            "semantic_passed_product_ids_by_slot": semantic_passed_by_slot,
            "alternative_product_ids_by_slot": alternative_by_slot,
            "selected_product_ids": passed_ids,
            "semantic_passed_product_ids": semantic_passed_ids,
            "alternative_product_ids": alternative_ids,
            "missing_required_slot_ids": missing_required_slot_ids,
            "needs_user_decision": needs_user_decision,
        }

    def _empty_combo_summary(self, state: MultiNeedState, *, status: str) -> dict:
        return self._combo_summary(
            state,
            status=status,
            selected_by_slot={},
            semantic_passed_by_slot={},
            budget_scope=state.intent_plan.budget_scope or "unknown",
            budget_max=self._budget_max(state),
            missing_required_slot_ids=[slot.slot_id for slot in state.slots if slot.need_type == "required"],
            needs_user_decision=False,
        )

    def _primary_ids_by_slot(self, product_ids_by_slot: dict[str, list[str]]) -> dict[str, list[str]]:
        return {
            slot_id: self._unique(product_ids)[:MULTI_NEED_PRIMARY_PER_SLOT]
            for slot_id, product_ids in product_ids_by_slot.items()
            if product_ids
        }

    def _budget_max(self, state: MultiNeedState) -> float | None:
        value = state.plan.budget.max if state.plan.budget.max is not None else state.intent_plan.budget_max
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _candidates_for_ids(self, state: MultiNeedState, slot_id: str, product_ids: list[str]) -> list[SlotCandidate]:
        by_id = {candidate.product_id: candidate for candidate in state.candidates_by_slot.get(slot_id, [])}
        return [by_id[product_id] for product_id in product_ids if product_id in by_id]

    def _alternative_ids_by_slot(
        self,
        state: MultiNeedState,
        semantic_passed_by_slot: dict[str, list[str]],
        selected_by_slot: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        globally_selected_ids = set(self._flatten_selected_ids(state, selected_by_slot))
        for slot in state.slots:
            alternatives = [
                product_id
                for product_id in semantic_passed_by_slot.get(slot.slot_id, [])
                if product_id not in globally_selected_ids
            ][:MULTI_NEED_ALTERNATIVES_PER_SLOT]
            if alternatives:
                result[slot.slot_id] = alternatives
        return result

    def _flatten_selected_ids(self, state: MultiNeedState, selected_by_slot: dict[str, list[str]]) -> list[str]:
        result: list[str] = []
        for slot in state.slots:
            for product_id in selected_by_slot.get(slot.slot_id, []):
                if product_id and product_id not in result:
                    result.append(product_id)
        return result

    def _total_price_for_ids(self, state: MultiNeedState, product_ids: list[str]) -> float:
        products_by_id = {
            candidate.product_id: candidate
            for candidates in state.candidates_by_slot.values()
            for candidate in candidates
        }
        return sum(float(products_by_id[product_id].price) for product_id in self._unique(product_ids) if product_id in products_by_id)

    def _slot_candidate_payload(self, candidate: Any) -> dict:
        return {
            "product_id": candidate.product_id,
            "name": candidate.name,
            "category": candidate.category,
            "sub_category": candidate.sub_category,
            "brand": candidate.product.brand,
            "price": candidate.price,
            "description": candidate.product.description[:320],
            "tags": candidate.product.tags[:8],
            "review_summary": candidate.product.review_summary[:320],
            "vector_score": candidate.vector_score,
            "keyword_score": candidate.keyword_score,
            "rerank_score": candidate.rerank_score,
            "coverage_reason": candidate.coverage_reason,
        }

    def _validate_review_data(self, data: dict[str, Any], valid_ids: set[str]) -> list[str]:
        errors: list[str] = []
        if not isinstance(data.get("reason"), str):
            errors.append("reason must be a string.")
        fallback_plan = self._fallback_plan(data.get("fallback_plan"))
        if fallback_plan not in self.FALLBACK_PLANS:
            errors.append("fallback_plan is invalid.")
        passed_ids = data.get("passed_product_ids")
        if not isinstance(passed_ids, list) or not all(isinstance(product_id, str) for product_id in passed_ids):
            errors.append("passed_product_ids must be a string array.")
            passed_ids = []
        unknown_passed = [product_id for product_id in passed_ids if product_id not in valid_ids]
        if unknown_passed:
            errors.append(f"passed_product_ids contains unknown ids: {', '.join(unknown_passed)}.")
        self._validate_rejected_products(data.get("rejected_products"), valid_ids, errors)
        return errors

    def _validate_review_slots_data(self, data: dict[str, Any], state: MultiNeedState) -> list[str]:
        errors: list[str] = []
        if not isinstance(data.get("reason"), str):
            errors.append("reason must be a string.")
        fallback_plan = self._fallback_plan(data.get("fallback_plan"))
        if fallback_plan not in self.FALLBACK_PLANS:
            errors.append("fallback_plan is invalid.")
        valid_by_slot = {
            slot.slot_id: {candidate.product_id for candidate in state.candidates_by_slot.get(slot.slot_id, [])}
            for slot in state.slots
        }
        all_valid_ids = {product_id for product_ids in valid_by_slot.values() for product_id in product_ids}
        slot_coverage = data.get("slot_coverage")
        selected_union: list[str] = []
        if not isinstance(slot_coverage, list):
            errors.append("slot_coverage must be an array.")
        else:
            seen_slot_ids: set[str] = set()
            for index, item in enumerate(slot_coverage, start=1):
                prefix = f"slot_coverage[{index}]"
                if not isinstance(item, dict):
                    errors.append(f"{prefix} must be an object.")
                    continue
                slot_id = str(item.get("slot_id") or "").strip()
                if slot_id not in valid_by_slot:
                    errors.append(f"{prefix}.slot_id is unknown.")
                    continue
                if slot_id in seen_slot_ids:
                    errors.append(f"slot_coverage contains duplicate slot_id: {slot_id}.")
                seen_slot_ids.add(slot_id)
                status = item.get("status")
                if status not in {"covered", "missing", "rejected"}:
                    errors.append(f"{prefix}.status is invalid.")
                selected_ids = self._string_list(item.get("selected_product_ids"))
                rejected_ids = self._string_list(item.get("rejected_product_ids"))
                if not isinstance(item.get("reason"), str) or not str(item.get("reason") or "").strip():
                    errors.append(f"{prefix}.reason must be a non-empty string.")
                invalid_selected = [product_id for product_id in selected_ids if product_id not in valid_by_slot[slot_id]]
                invalid_rejected = [product_id for product_id in rejected_ids if product_id not in valid_by_slot[slot_id]]
                if invalid_selected:
                    errors.append(f"{prefix}.selected_product_ids contains ids outside the slot candidate pool.")
                if invalid_rejected:
                    errors.append(f"{prefix}.rejected_product_ids contains ids outside the slot candidate pool.")
                selected_union.extend(product_id for product_id in selected_ids if product_id not in selected_union)
            missing_slot_ids = [slot.slot_id for slot in state.slots if slot.slot_id not in seen_slot_ids]
            if missing_slot_ids:
                errors.append(f"slot_coverage misses slots: {', '.join(missing_slot_ids)}.")
        passed_ids = self._string_list(data.get("passed_product_ids"))
        unknown_passed = [product_id for product_id in passed_ids if product_id not in all_valid_ids]
        if unknown_passed:
            errors.append(f"passed_product_ids contains unknown ids: {', '.join(unknown_passed)}.")
        self._validate_rejected_products(data.get("rejected_products"), all_valid_ids, errors)
        return errors

    def _validate_rejected_products(self, value: Any, valid_ids: set[str], errors: list[str]) -> None:
        if not isinstance(value, list):
            errors.append("rejected_products must be an array.")
            return
        for index, item in enumerate(value, start=1):
            prefix = f"rejected_products[{index}]"
            if not isinstance(item, dict):
                errors.append(f"{prefix} must be an object.")
                continue
            product_id = item.get("product_id")
            reason = item.get("reason")
            if not isinstance(product_id, str) or not product_id:
                errors.append(f"{prefix}.product_id must be a string.")
            elif product_id not in valid_ids:
                errors.append(f"{prefix}.product_id is not a candidate id.")
            if not isinstance(reason, str) or not reason.strip():
                errors.append(f"{prefix}.reason must be a non-empty string.")

    def _llm_is_configured(self) -> bool:
        checker = getattr(self.llm_client, "is_configured", None)
        if checker is None:
            return True
        return bool(checker())

    def _fallback_plan(self, value: Any) -> str:
        text = str(value or "none").strip()
        return text if text in self.FALLBACK_PLANS else "none"

    def _repair_hint(self, value: Any, fallback_plan: str, *, default_slot_ids: list[str]) -> RepairHint:
        if fallback_plan in {"direct_answer", "clarify"}:
            return RepairHint()
        if not isinstance(value, dict):
            return RepairHint()
        repairable = bool(value.get("repairable"))
        target_slot_ids = self._unique([
            slot_id
            for slot_id in self._string_list(value.get("target_slot_ids"))
            if slot_id in set(default_slot_ids or ["single"])
        ])
        if repairable and not target_slot_ids:
            target_slot_ids = list(default_slot_ids or ["single"])
        return RepairHint(
            repairable=repairable,
            target_slot_ids=target_slot_ids,
            failure_type=str(value.get("failure_type") or ""),
            missing_terms=self._string_list(value.get("missing_terms")),
            avoid_terms=self._string_list(value.get("avoid_terms")),
            reason=str(value.get("reason") or ""),
        )

    def _string_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item or "").strip()]

    def _unique(self, values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            if value and value not in result:
                result.append(value)
        return result
