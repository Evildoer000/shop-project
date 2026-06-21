from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.domain.need_slot_schemas import NeedSlot
from app.schemas import IntentPlan, QueryPlan, ReflectionResult
from app.services.llm_client import LlmClient
from app.services.structured_llm import StructuredLlmValidationError, generate_validated_json


@dataclass
class RepairPlan:
    targets: list[str] = field(default_factory=list)
    queries_by_slot: dict[str, list[str]] = field(default_factory=dict)
    reason: str = ""
    used_llm: bool = False

    def summary(self) -> dict[str, Any]:
        return {
            "worker_agent": "RepairAgent",
            "targets": self.targets,
            "queries_by_slot": self.queries_by_slot,
            "reason": self.reason,
            "used_llm": self.used_llm,
        }


class RepairAgent:
    JSON_RESPONSE_FORMAT = {"type": "json_object"}

    def __init__(self, llm_client: LlmClient | None = None) -> None:
        self.llm_client = llm_client or LlmClient(component="RepairAgent")

    async def plan_repair(
        self,
        *,
        original_query: str,
        intent_plan: IntentPlan,
        plan: QueryPlan,
        slots: list[NeedSlot],
        trigger: str,
        reflection_result: ReflectionResult,
        previous_candidates: dict[str, list[dict[str, Any]]] | None = None,
    ) -> RepairPlan:
        fallback = self._fallback_plan(slots, trigger, reflection_result)
        if not self.llm_client.is_configured():
            return fallback

        system_prompt = (
            "你是电商 RAG Harness 的 RepairAgent（修复规划 Worker Agent）。只输出 JSON object，不要输出 Markdown。\n"
            "你的职责是：在 Orchestrator 已批准 repair 后，根据 CorrectiveAgent 的 reflection/repair_hint 生成检索修复计划。\n"
            "你只能生成 repair query，不执行检索、不读取商品库、不评价候选、不回答用户、不决定 final_route。\n"
            "每个 target slot 最多输出 3 条简短检索 query；query 必须保留用户明确商品形态和硬约束，并规避 reflection 中指出的错误形态。\n"
            "返回 JSON schema: {"
            "\"targets\":[\"slot_id\"],"
            "\"queries_by_slot\":{\"slot_id\":[\"query1\",\"query2\"]},"
            "\"reason\":\"中文原因\""
            "}"
        )
        user_prompt = json.dumps(
            {
                "original_query": original_query,
                "intent_plan": intent_plan.model_dump(),
                "query_plan": plan.model_dump(),
                "trigger": trigger,
                "slots": [slot.model_dump() for slot in slots],
                "reflection_result": reflection_result.model_dump(),
                "previous_candidates": previous_candidates or {},
            },
            ensure_ascii=False,
        )
        try:
            data = await generate_validated_json(
                self.llm_client,
                system_prompt,
                user_prompt,
                validate=lambda value: self._validate_repair_plan(value, {slot.slot_id for slot in slots}),
                error_message="RepairAgent returned invalid JSON.",
                response_format=self.JSON_RESPONSE_FORMAT,
                operation="repair_agent.plan_repair",
            )
        except (StructuredLlmValidationError, RuntimeError):
            return fallback

        targets = self._target_slots(data.get("targets"), slots)
        queries_by_slot = self._queries_by_slot(data.get("queries_by_slot"), targets)
        if not targets or not any(queries_by_slot.values()):
            return fallback
        return RepairPlan(
            targets=targets,
            queries_by_slot=queries_by_slot,
            reason=str(data.get("reason") or ""),
            used_llm=True,
        )

    def _fallback_plan(self, slots: list[NeedSlot], trigger: str, reflection_result: ReflectionResult) -> RepairPlan:
        hinted_targets = [
            slot_id
            for slot_id in reflection_result.repair_hint.target_slot_ids
            if any(slot.slot_id == slot_id for slot in slots)
        ]
        targets = hinted_targets or [slot.slot_id for slot in slots]
        queries_by_slot: dict[str, list[str]] = {}
        for slot in slots:
            if slot.slot_id not in targets:
                continue
            terms = [
                slot.query,
                slot.product_type,
                slot.goal,
                *slot.soft_constraints,
                *reflection_result.repair_hint.missing_terms,
            ]
            base = self._compact(" ".join(term for term in terms if term))
            alt = self._compact(" ".join(term for term in [slot.product_type, slot.goal] if term))
            queries_by_slot[slot.slot_id] = self._dedupe([base, alt, slot.query])[:3]
        return RepairPlan(
            targets=targets,
            queries_by_slot=queries_by_slot,
            reason=reflection_result.repair_hint.reason or f"deterministic repair queries for trigger={trigger}",
            used_llm=False,
        )

    def _validate_repair_plan(self, data: dict[str, Any], valid_slot_ids: set[str]) -> list[str]:
        errors: list[str] = []
        targets = data.get("targets")
        if not isinstance(targets, list) or not targets:
            errors.append("targets must be a non-empty array")
            targets = []
        invalid_targets = [str(slot_id) for slot_id in targets if str(slot_id) not in valid_slot_ids]
        if invalid_targets:
            errors.append(f"targets contains unknown slot ids: {', '.join(invalid_targets)}")
        queries_by_slot = data.get("queries_by_slot")
        if not isinstance(queries_by_slot, dict):
            errors.append("queries_by_slot must be an object")
            return errors
        for slot_id in targets:
            queries = queries_by_slot.get(str(slot_id))
            if not isinstance(queries, list) or not queries:
                errors.append(f"queries_by_slot.{slot_id} must be a non-empty array")
            elif len([query for query in queries if str(query or "").strip()]) > 3:
                errors.append(f"queries_by_slot.{slot_id} must contain at most three queries")
        return errors

    def _target_slots(self, value: Any, slots: list[NeedSlot]) -> list[str]:
        valid = {slot.slot_id for slot in slots}
        if not isinstance(value, list):
            return []
        return self._dedupe([str(slot_id).strip() for slot_id in value if str(slot_id).strip() in valid])

    def _queries_by_slot(self, value: Any, targets: list[str]) -> dict[str, list[str]]:
        if not isinstance(value, dict):
            return {}
        return {
            slot_id: self._dedupe(
                [str(query).strip() for query in value.get(slot_id, []) if str(query or "").strip()]
            )[:3]
            for slot_id in targets
        }

    def _compact(self, value: str) -> str:
        return " ".join(value.split())

    def _dedupe(self, values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            text = str(value or "").strip()
            if text and text not in result:
                result.append(text)
        return result
