from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from app.domain.agents.contracts import AgentExecutionContext, AgentResult, EvidenceRef
from app.domain.need_slot_schemas import (
    AgentToolCall,
    MultiNeedSelection,
    MultiNeedState,
    NeedSlot,
    SlotCandidate,
)
from app.domain.repair_worker import RepairPlan
from app.domain.slot_retrieval_agent import SlotRetrievalAgent, SlotRetrievalAgentResult
from app.domain.single_retrieval_worker import SingleRetrievalEvidence
from app.schemas import (
    IntentConstraintSet,
    IntentItem,
    IntentPlan,
    IntentPlanV3,
    ProductCard,
    ProfileLookupProposal,
    QueryPlan,
    ReflectionResult,
    RewriteNeedSlot,
)
from app.services.structured_llm import StructuredLlmValidationError, generate_validated_json


@dataclass
class BusinessAgentServices:
    """Dependencies injected into business Agents by the Supervisor runtime."""

    intent_planner: Any = None
    profile_lookup_tool: Any = None
    retrieval_plan_builder: Any = None
    retrieval_worker: Any = None
    product_search_tool: Any = None
    image_search_tool: Any = None
    corrective_agent: Any = None
    repair_agent: Any = None
    answer_generator: Any = None
    product_repository: Any = None
    product_detail_tool: Any = None
    commerce_search: Any = None
    commerce_product_detail: Any = None
    commerce_reviews: Any = None
    web_search: Any = None
    knowledge_llm: Any = None
    comparison_llm: Any = None
    slot_runner: Callable[[NeedSlot, QueryPlan, IntentPlan], Awaitable[SlotRetrievalAgentResult] | SlotRetrievalAgentResult] | None = None
    memory_scheduler: Callable[[AgentExecutionContext, str], Any] | None = None


class BusinessAgentHandlers:
    """Adapters from the graph contracts to the existing domain components.

    These adapters contain no routing policy.  They execute one capability and
    return structured artifacts for downstream nodes; the Supervisor decides
    which capability is present and when a retry is allowed.
    """

    MAX_KNOWLEDGE_RESEARCH_ROUNDS = 2

    def __init__(self, services: BusinessAgentServices) -> None:
        self.services = services

    def as_mapping(self) -> dict[str, Callable[[AgentExecutionContext], Any]]:
        return {
            "intent_understanding": self.intent_understanding,
            "profile_preference": self.profile_preference,
            "clarification": self.clarification,
            "single_product_recommendation": self.single_product_recommendation,
            "multi_product_bundle": self.multi_product_bundle,
            "slot_product_retrieval": self.slot_product_retrieval,
            "commerce_research": self.commerce_research,
            "comparison": self.comparison,
            "knowledge_research": self.knowledge_research,
            "evidence_verification": self.evidence_verification,
            "bundle_optimization": self.bundle_optimization,
            "repair": self.repair,
            "answer_generation": self.answer_generation,
            "memory_distillation": self.memory_distillation,
        }

    async def intent_understanding(self, context: AgentExecutionContext) -> AgentResult:
        planner = self.services.intent_planner
        if planner is None:
            return AgentResult.failure_result("planner_not_configured", "IntentUnderstandingAgent is not configured")
        planner_context = dict(context.metadata.get("planner_context") or {})
        revision = context.node.metadata.get("intent_revision")
        query = context.query
        if isinstance(revision, dict):
            planner_context["intent_revision"] = revision
            target = revision.get("target_intent") or {}
            query = str(target.get("resolved_query") or target.get("goal") or query)
        plan = await planner.plan(query, planner_context)
        artifact_key = "intent_plan"
        if isinstance(revision, dict):
            target_intent_id = str(revision.get("intent_id") or "")
            plan = self._bounded_intent_revision(
                self._supervisor_plan(context),
                plan,
                target_intent_id,
            )
            artifact_key = f"intent_plan_revision:{target_intent_id}"
        return AgentResult.success(
            {
                "intent_plan": plan.model_dump(),
                "summary": plan.summary,
                "revision_intent_id": str((revision or {}).get("intent_id") or "")
                if isinstance(revision, dict)
                else "",
            },
            artifacts={artifact_key: plan},
        )

    async def profile_preference(self, context: AgentExecutionContext) -> AgentResult:
        query = str(context.node.metadata.get("query") or context.query)
        memory = await context.execute_tool(
            "profile_lookup",
            "lookup",
            lambda tool: tool.lookup(context.user_id, query),
            input_summary={"query_length": len(query), "limit": 8},
        )
        return AgentResult.success(
            {
                "memory_count": len(memory),
                "soft_preferences": [self._memory_item(item) for item in memory],
                "usage": context.node.metadata.get("usage", "ranking_only"),
            },
            artifacts={"profile_memory": memory},
        )

    async def clarification(self, context: AgentExecutionContext) -> AgentResult:
        intent = self._target_intent(context)
        parameters = context.node.metadata.get("parameters") or {}
        missing_fields = list(parameters.get("missing_fields") or [])
        if not missing_fields and intent is not None:
            missing_fields = [item.field for item in intent.uncertainties if item.blocking]
        question = self._clarification_question(
            missing_fields,
            str(parameters.get("question_goal") or ""),
            intent.resolved_query if intent is not None else context.query,
        )
        output = {
            "required": True,
            "question": question,
            "missing_fields": missing_fields,
            "reason": str(context.node.metadata.get("reason") or "需求缺少可执行条件。"),
        }
        return AgentResult.success(
            output,
            artifacts={
                "clarification": output,
                "answer_text": question,
                "answer_tokens": [question],
                "answer_product_ids": [],
                "answer_route": "clarify",
            },
            termination_reason="clarification_emitted",
        )

    async def single_product_recommendation(self, context: AgentExecutionContext) -> AgentResult:
        override_artifact = str(context.node.metadata.get("intent_plan_artifact") or "")
        intent_plan = context.artifact(override_artifact) if override_artifact else None
        if not isinstance(intent_plan, IntentPlan):
            intent_plan = self._intent_plan(context)
        intent_plan = self._profile_aware_intent_plan(context, intent_plan)
        plan = self._query_plan(context, intent_plan=intent_plan)
        query_override = str(context.node.metadata.get("query_override") or "").strip()
        if query_override:
            intent_plan = intent_plan.model_copy(
                update={"vector_query": query_override, "keyword_query": query_override}
            )
        query = query_override or intent_plan.normalized_query or intent_plan.original_query or context.query
        image_path = context.metadata.get("image_path")
        image_attributes = None
        if image_path:
            image_attributes = await context.execute_tool(
                "image_understanding",
                "understand",
                lambda tool: tool.understand(image_path, query),
                input_summary={"query_length": len(query), "has_image": True},
            )
        repair_plan = context.artifact("repair_plan")
        retry_of = str(context.node.metadata.get("retry_of") or "")
        if repair_plan is not None and retry_of and self.services.retrieval_worker is not None:
            evidence = await context.execute_tool(
                "product_search",
                "single_repair",
                lambda _tool: self.services.retrieval_worker.run_single_repair_isolated(
                    query,
                    intent_plan,
                    plan,
                    repair_plan,
                ),
                input_summary={"query_length": len(query), "repair": True},
            )
            return AgentResult.success(
                self._single_evidence_output(evidence, intent_plan, plan),
                artifacts={"single_evidence": evidence, "query_plan": plan, "image_attributes": image_attributes},
            )
        if image_path and self.services.retrieval_worker is not None:
            evidence = await context.execute_tool(
                "image_search",
                "image_initial",
                lambda _tool: self.services.retrieval_worker.run_image_initial_isolated(
                    original_query=query,
                    intent_plan=intent_plan,
                    plan=plan,
                    image_path=image_path,
                ),
                input_summary={"query_length": len(query), "has_image": True},
            )
            artifact_key = "image_evidence"
        else:
            evidence = await context.execute_tool(
                "product_search",
                "single_initial",
                lambda _tool: self.services.retrieval_worker.run_single_initial_isolated(
                    query,
                    intent_plan,
                    plan,
                ),
                input_summary={"query_length": len(query), "repair": False},
            )
            artifact_key = "single_evidence"
        return AgentResult.success(
            self._single_evidence_output(evidence, intent_plan, plan),
            artifacts={artifact_key: evidence, "query_plan": plan, "image_attributes": image_attributes},
        )

    async def multi_product_bundle(self, context: AgentExecutionContext) -> AgentResult:
        # Dispatch and merge use the same capability but are separate graph
        # phases.  This makes the boundary visible without inventing another
        # business Agent.
        if context.node.metadata.get("phase") == "merge_slot_evidence":
            return await self._merge_bundle(context)
        slots = [self._need_slot(item) for item in self._intent_plan(context).need_slots]
        image_attributes = None
        image_path = context.metadata.get("image_path")
        if image_path:
            image_attributes = await context.execute_tool(
                "image_understanding",
                "understand",
                lambda tool: tool.understand(image_path, context.query),
                input_summary={"query_length": len(context.query), "has_image": True},
            )
        return AgentResult.success(
            {
                "phase": "dispatch",
                "slot_count": len(slots),
                "slot_ids": [slot.slot_id for slot in slots],
                "slots": [slot.model_dump() for slot in slots],
            },
            artifacts={"bundle_slots": slots, "image_attributes": image_attributes},
        )

    async def slot_product_retrieval(self, context: AgentExecutionContext) -> AgentResult:
        slot_payload = context.node.metadata.get("slot") or {}
        slot = self._need_slot(slot_payload)
        slot = self._profile_aware_slot(context, slot)
        plan = self._query_plan(context)
        intent_plan = self._intent_plan(context)
        repair_plan = context.artifact("repair_plan")
        if repair_plan is not None:
            queries = repair_plan.queries_by_slot.get(slot.slot_id, [])
            if queries:
                slot = slot.model_copy(update={"query": queries[0]})
        runner = self.services.slot_runner
        if runner is not None:
            result = await context.execute_tool(
                "product_search",
                "slot_retrieval",
                lambda _tool: runner(slot, plan, intent_plan),
                input_summary={"slot_id": slot.slot_id, "query_length": len(slot.query)},
            )
        else:
            agent = SlotRetrievalAgent(self.services.product_search_tool)
            boundary = getattr(self.services.retrieval_worker, "execution_boundary", None)
            result = await context.execute_tool(
                "product_search",
                "slot_retrieval",
                (
                    lambda _tool: boundary.run_product(
                        lambda search_tool: SlotRetrievalAgent(search_tool).run(slot, plan, intent_plan)
                    )
                    if boundary is not None
                    else lambda _tool: asyncio.to_thread(agent.run, slot, plan, intent_plan)
                ),
                input_summary={"slot_id": slot.slot_id, "query_length": len(slot.query)},
            )
        return AgentResult.success(
            self._slot_result_output(result),
            artifacts={f"slot_result:{slot.slot_id}": result},
        )

    async def commerce_research(self, context: AgentExecutionContext) -> AgentResult:
        parameters = dict(context.node.metadata.get("parameters") or {})
        requests = [
            {
                "request_id": context.node.metadata.get("planner_task_id", context.node.task_id),
                "mode": "social_content"
                if "xiaohongshu" in (parameters.get("platforms") or [])
                else "marketplace",
                **parameters,
            }
        ]
        evidence: list[EvidenceRef] = []
        results: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        available_result_count = 0
        intent_plan = self._intent_plan(context)
        for request in requests:
            mode = str(request.get("mode") or "marketplace")
            platforms = request.get("platforms") or ["taobao"]
            query = str(request.get("query") or context.query)
            for platform in platforms:
                try:
                    response = await context.execute_tool(
                        "commerce_search",
                        "search",
                        lambda tool: tool.search(platform=platform, query=query),
                        input_summary={"platform": platform, "query_length": len(query)},
                    )
                except Exception as exc:
                    errors.append({"platform": platform, "code": "tool_error", "message": str(exc)})
                    continue
                items = self._items_from_response(response) if self._response_available(response) else []
                enrichments: list[dict[str, Any]] = []
                if not items:
                    error = response.get("error") if isinstance(response, dict) else None
                    errors.append(
                        {
                            "platform": platform,
                            "request_id": request.get("request_id"),
                            "code": (error or {}).get("code", "no_external_results")
                            if isinstance(error, dict)
                            else "no_external_results",
                            "message": (error or {}).get("message", "平台搜索没有返回可用条目。")
                            if isinstance(error, dict)
                            else "平台搜索没有返回可用条目。",
                        }
                    )
                else:
                    available_result_count += 1
                for index, item in enumerate(items[:20]):
                    external_id = self._external_item_id(item)
                    source_id = external_id or self._external_item_url(item)
                    if not source_id:
                        continue
                    evidence.append(
                        EvidenceRef(
                            evidence_id=f"commerce:{platform}:{request.get('request_id')}:{index}",
                            source_type="commerce_mcp",
                            source_id=source_id,
                            claim=str(item.get("title") or item.get("name") or query),
                            summary=self._compact_json(item, 600),
                            provenance=self._commerce_provenance(response, platform=platform, mode=mode),
                        )
                    )
                if items and self._commerce_enrichment_required(request, intent_plan):
                    for item in items[:2]:
                        external_id = self._external_item_id(item)
                        if not external_id:
                            continue
                        enrichment = {"external_id": external_id}
                        for operation, tool_name, callback in (
                            (
                                "detail",
                                "commerce_product_detail",
                                lambda tool, p=platform, item_id=external_id: tool.get(
                                    platform=p,
                                    product_id=item_id,
                                ),
                            ),
                            (
                                "reviews",
                                "commerce_reviews",
                                lambda tool, p=platform, item_id=external_id: tool.list(
                                    platform=p,
                                    product_id=item_id,
                                    page=1,
                                ),
                            ),
                        ):
                            try:
                                detail_response = await context.execute_tool(
                                    tool_name,
                                    operation,
                                    callback,
                                    input_summary={"platform": platform, "external_id": external_id},
                                )
                            except Exception as exc:
                                errors.append(
                                    {
                                        "platform": platform,
                                        "external_id": external_id,
                                        "code": f"{operation}_tool_error",
                                        "message": str(exc),
                                    }
                                )
                                continue
                            enrichment[operation] = detail_response
                            if not self._response_available(detail_response):
                                continue
                            for detail_index, detail_item in enumerate(
                                self._items_from_response(detail_response)[:8]
                            ):
                                evidence.append(
                                    EvidenceRef(
                                        evidence_id=(
                                            f"commerce:{platform}:{request.get('request_id')}:"
                                            f"{external_id}:{operation}:{detail_index}"
                                        ),
                                        source_type="commerce_mcp",
                                        source_id=external_id,
                                        claim=str(
                                            detail_item.get("title")
                                            or detail_item.get("name")
                                            or f"{platform} {operation}"
                                        ),
                                        summary=self._compact_json(detail_item, 600),
                                        provenance=self._commerce_provenance(
                                            detail_response,
                                            platform=platform,
                                            mode=mode,
                                        ),
                                    )
                                )
                        enrichments.append(enrichment)
                results.append(
                    {
                        "request_id": request.get("request_id"),
                        "mode": mode,
                        "platform": platform,
                        "available": bool(items),
                        "items": items[:20],
                        "enrichments": enrichments,
                        "response_meta": self._external_response_meta(response),
                    }
                )
        return AgentResult.success(
            {
                "result_count": len(results),
                "available_result_count": available_result_count,
                "evidence_count": len(evidence),
                "errors": errors,
                "platforms": sorted({str(item.get("platform")) for item in results if item.get("platform")}),
            },
            evidence=evidence,
            artifacts={"commerce_research": results},
            termination_reason="completed" if evidence else "no_external_results",
        )

    async def comparison(self, context: AgentExecutionContext) -> AgentResult:
        plan = self._intent_plan(context)
        intent = self._target_intent(context)
        parameters = context.node.metadata.get("parameters") or {}
        upstream_evidence = self._dependency_evidence(context)
        product_ids = list(intent.referenced_product_ids if intent is not None else [])
        product_ids.extend(parameters.get("referenced_product_ids") or [])
        single = context.artifact("single_evidence") or context.artifact("image_evidence")
        if single is not None:
            product_ids.extend(product.product_id for product, _ in getattr(single, "ranked", [])[:5])
        state = context.artifact("multi_state")
        if state is not None:
            product_ids.extend(
                candidate.product_id
                for candidates in state.candidates_by_slot.values()
                for candidate in candidates[:2]
            )
        product_ids.extend(
            ref.product_id for ref in upstream_evidence if ref.product_id
        )
        product_ids = list(dict.fromkeys(product_ids))[:12]
        details: list[dict[str, Any]] = []
        if product_ids:
            products = await context.execute_tool(
                "product_detail",
                "get_by_ids",
                lambda tool: tool.get_by_ids(product_ids),
                input_summary={"product_count": len(product_ids)},
            )
            details = [self._product_payload(product) for product in products]
        external = context.artifact("commerce_research", [])
        knowledge = self._comparison_knowledge_context(
            context.artifact("knowledge_research", {})
        )
        dimensions = self._comparison_dimensions(details)
        referenced_ids = set(product_ids)
        analysis = await self._reason_over_comparison(
            context,
            products=details,
            external=external,
            knowledge=knowledge,
            upstream_evidence=upstream_evidence,
            fallback_dimensions=dimensions,
        )
        product_evidence = self._comparison_evidence(details, referenced_ids=referenced_ids)
        comparison_evidence = self._comparison_analysis_evidence(
            analysis,
            supporting_evidence_ids=[
                ref.evidence_id
                for ref in [*product_evidence, *upstream_evidence]
            ],
        )
        return AgentResult.success(
            {
                "product_count": len(details),
                "dimensions": analysis["dimensions"],
                "winner_by_goal": analysis["winner_by_goal"],
                "unknowns": analysis["unknowns"],
                "used_llm": analysis["used_llm"],
                "reflection": analysis.get("reflection", {}),
                "external_available": self._commerce_results_available(external),
                "knowledge_available": bool(knowledge),
            },
            evidence=[*product_evidence, *comparison_evidence],
            artifacts={
                "comparison": {
                    "products": details,
                    "external": external,
                    "knowledge": knowledge,
                    **analysis,
                }
            },
        )

    async def knowledge_research(self, context: AgentExecutionContext) -> AgentResult:
        parameters = dict(context.node.metadata.get("parameters") or {})
        intent = self._target_intent(context)
        referenced_product_ids = list(
            dict.fromkeys(
                [
                    *[
                        str(item)
                        for item in (intent.referenced_product_ids if intent else [])
                    ],
                    *[
                        str(item)
                        for item in (parameters.get("referenced_product_ids") or [])
                    ],
                    *[str(item) for item in (parameters.get("product_ids") or [])],
                ]
            )
        )
        local_products: list[dict[str, Any]] = []
        local_product_evidence: list[EvidenceRef] = []
        missing_product_ids: list[str] = []
        if referenced_product_ids:
            if self.services.product_detail_tool is None:
                missing_product_ids = list(referenced_product_ids)
            else:
                products = await context.execute_tool(
                    "product_detail",
                    "get_by_ids",
                    lambda tool: tool.get_by_ids(referenced_product_ids),
                    input_summary={"product_count": len(referenced_product_ids)},
                )
                loaded_ids = set()
                for product in products or []:
                    payload = self._knowledge_product_payload(product)
                    product_id = str(payload["product_id"])
                    loaded_ids.add(product_id)
                    local_products.append(payload)
                    local_product_evidence.append(
                        EvidenceRef(
                            evidence_id=f"product_detail:{product_id}",
                            source_type="local_product_catalog",
                            source_id=product_id,
                            product_id=product_id,
                            claim=f"商品详情：{payload['name']}",
                            summary=self._compact_json(payload, 1_000),
                            verified=True,
                            provenance={
                                "source_agent": "product_knowledge_agent",
                                "evidence_role": "local_product_detail",
                            },
                        )
                    )
                missing_product_ids = [
                    product_id
                    for product_id in referenced_product_ids
                    if product_id not in loaded_ids
                ]
        use_web = self._knowledge_requires_web(
            intent,
            parameters,
            referenced_product_ids,
        )
        requests = [
            {
                "request_id": context.node.metadata.get("planner_task_id", context.node.task_id),
                "mode": "web_general",
                **parameters,
            }
        ]
        responses: list[dict[str, Any]] = []
        claims: list[dict[str, Any]] = []
        expansions: list[str] = []
        research_rounds: list[dict[str, Any]] = []
        reasoning: dict[str, Any] = {
            "candidate_concepts": [],
            "grounded_concepts": [],
            "risks": [],
            "used_llm": False,
            "coverage": {
                "sufficient": False,
                "missing_aspects": ["web_evidence"],
                "followup_query": "",
                "reason": "No web research round was executed.",
            },
        }
        if local_product_evidence and not use_web:
            reasoning["coverage"] = {
                "sufficient": True,
                "missing_aspects": [],
                "followup_query": "",
                "reason": "Local product details were loaded from the catalog.",
            }
        for request in requests if use_web else []:
            if request.get("mode") not in {None, "web_general"}:
                continue
            initial_query = str(request.get("query") or context.query).strip()
            freshness = str(request.get("freshness") or "recent")
            current_query = initial_query
            searched_queries: set[str] = set()
            for round_index in range(1, self.MAX_KNOWLEDGE_RESEARCH_ROUNDS + 1):
                normalized_query = " ".join(current_query.split())
                if not normalized_query or normalized_query in searched_queries:
                    break
                searched_queries.add(normalized_query)
                response = await context.execute_tool(
                    "web_search",
                    f"search_round_{round_index}",
                    lambda tool, query=normalized_query, freshness_value=freshness: tool.search(
                        query=query,
                        freshness=freshness_value,
                    ),
                    input_summary={
                        "query": normalized_query[:200],
                        "query_length": len(normalized_query),
                        "freshness": freshness,
                        "research_round": round_index,
                    },
                )
                responses.append(response)
                for term in self._response_candidate_terms(response):
                    if term not in expansions:
                        expansions.append(term)
                new_claim_count = self._append_knowledge_claims(
                    claims,
                    response,
                    research_query=normalized_query,
                    research_round=round_index,
                )
                reasoning = await self._reason_over_knowledge(
                    context,
                    claims=claims,
                    structured_terms=expansions,
                    research_query=initial_query,
                    research_round=round_index,
                    allow_followup=round_index < self.MAX_KNOWLEDGE_RESEARCH_ROUNDS,
                    search_available=isinstance(response, dict)
                    and bool(response.get("available")),
                )
                coverage = dict(reasoning.get("coverage") or {})
                round_summary = {
                    "round": round_index,
                    "query": normalized_query,
                    "available": self._response_available(response),
                    "new_claim_count": new_claim_count,
                    "total_claim_count": len(claims),
                    "coverage": coverage,
                    "stop_reason": "",
                }
                research_rounds.append(round_summary)

                if bool(coverage.get("sufficient")):
                    round_summary["stop_reason"] = "coverage_sufficient"
                    break
                if round_index >= self.MAX_KNOWLEDGE_RESEARCH_ROUNDS:
                    round_summary["stop_reason"] = "max_rounds_reached"
                    break
                followup_query = " ".join(
                    str(coverage.get("followup_query") or "").split()
                )[:500]
                if not followup_query:
                    round_summary["stop_reason"] = "no_followup_query"
                    break
                if followup_query in searched_queries:
                    round_summary["stop_reason"] = "duplicate_followup_query"
                    break
                round_summary["stop_reason"] = "followup_scheduled"
                current_query = followup_query

        expansions = reasoning["candidate_concepts"]
        web_evidence = [
            EvidenceRef(
                evidence_id=f"web:{index}",
                source_type="web_search",
                source_id=str(item.get("source_id") or item.get("url") or ""),
                claim=str(item.get("title") or "知识资料"),
                summary=str(item.get("snippet") or "")[:600],
                provenance={
                    "url": item.get("url"),
                    "evidence_role": "external_observation",
                    "research_round": item.get("research_round", 1),
                    "research_query": item.get("research_query", ""),
                },
            )
            for index, item in enumerate(claims)
        ]
        concept_proposals = self._knowledge_concept_proposals(
            reasoning,
            web_evidence=web_evidence,
        )
        knowledge_evidence = self._knowledge_analysis_evidence(
            reasoning,
            web_evidence=web_evidence,
        )
        available = bool(web_evidence or local_product_evidence)
        source_modes = []
        if local_product_evidence:
            source_modes.append("local_product_detail")
        if web_evidence:
            source_modes.append("web_search")
        return AgentResult.success(
            {
                "available": available,
                "claim_count": len(claims),
                "product_detail_count": len(local_products),
                "product_details": local_products,
                "product_ids": [item["product_id"] for item in local_products],
                "missing_product_ids": missing_product_ids,
                "source_modes": source_modes,
                "candidate_concepts": expansions[:12],
                "concept_proposals": concept_proposals,
                "grounded_concepts": reasoning["grounded_concepts"],
                "risks": reasoning["risks"],
                "used_llm": reasoning["used_llm"],
                "research_round_count": len(research_rounds),
                "research_rounds": research_rounds,
                "coverage": reasoning.get("coverage", {}),
                "status": "completed" if available else "unavailable",
            },
            evidence=[*local_product_evidence, *web_evidence, *knowledge_evidence],
            artifacts={
                "knowledge_research": {
                    "responses": responses,
                    "product_details": local_products,
                    "product_evidence": [
                        item.compact() for item in local_product_evidence
                    ],
                    "local_product_ids": [item["product_id"] for item in local_products],
                    "missing_product_ids": missing_product_ids,
                    "claims": claims,
                    "query_expansions": expansions[:12],
                    "candidate_concepts": expansions[:12],
                    "concept_proposals": concept_proposals,
                    "grounded_concepts": reasoning["grounded_concepts"],
                    "risks": reasoning["risks"],
                    "used_llm": reasoning["used_llm"],
                    "research_plan": {
                        "initial_query": str(requests[0].get("query") or context.query),
                        "freshness": str(requests[0].get("freshness") or "recent"),
                        "max_rounds": self.MAX_KNOWLEDGE_RESEARCH_ROUNDS,
                        "source_modes": source_modes,
                    },
                    "research_rounds": research_rounds,
                    "coverage": reasoning.get("coverage", {}),
                }
            },
            termination_reason=(
                "completed"
                if available
                else "product_detail_unavailable"
                if referenced_product_ids and not use_web
                else "web_search_unavailable"
            ),
        )

    def _knowledge_requires_web(
        self,
        intent: IntentItem | None,
        parameters: dict[str, Any],
        referenced_product_ids: list[str],
    ) -> bool:
        mode = str(parameters.get("knowledge_mode") or "").strip().lower()
        if mode in {"local", "local_product", "product_detail"}:
            return False
        if mode in {"web", "web_knowledge", "mixed"}:
            return True
        trigger_type = str(parameters.get("trigger_type") or "none")
        external_need = (
            intent.route_basis.external_information_need
            if intent is not None
            else "none"
        )
        if trigger_type in {
            "explicit_web_request",
            "freshness_required",
            "knowledge_bridge",
        }:
            return True
        if external_need in {"explicit_web", "freshness_required", "knowledge_bridge"}:
            return True
        if intent is not None and intent.intent_type == "shopping_knowledge":
            return True
        return not referenced_product_ids

    def _knowledge_product_payload(self, product: Any) -> dict[str, Any]:
        return {
            "product_id": str(getattr(product, "product_id", "")),
            "name": str(getattr(product, "name", "")),
            "brand": str(getattr(product, "brand", "")),
            "category": str(getattr(product, "category", "")),
            "sub_category": str(getattr(product, "sub_category", "") or ""),
            "price": float(getattr(product, "price", 0) or 0),
            "stock": int(getattr(product, "stock", 0) or 0),
            "description": str(getattr(product, "description", "") or "")[:1_200],
            "specs": getattr(product, "specs", {}) or {},
            "ingredients_or_material": str(
                getattr(product, "ingredients_or_material", "") or ""
            )[:800],
            "suitable_for": str(getattr(product, "suitable_for", "") or "")[:600],
            "avoid_for": str(getattr(product, "avoid_for", "") or "")[:600],
            "tags": list(getattr(product, "tags", []) or [])[:12],
            "rating": float(getattr(product, "rating", 0) or 0),
            "review_summary": str(getattr(product, "review_summary", "") or "")[:800],
        }

    async def evidence_verification(self, context: AgentExecutionContext) -> AgentResult:
        intent_plan = self._intent_plan(context)
        plan = self._query_plan(context)
        single = context.artifact("single_evidence") or context.artifact("image_evidence")
        state = context.artifact("multi_state")
        candidate_pool: list[dict[str, Any]] = []
        target_intent = self._target_intent(context)
        if (
            target_intent is not None
            and target_intent.intent_type == "product_recommendation"
            and state is None
        ):
            single, candidate_pool = await self._merge_single_candidate_pool(
                context,
                single,
                intent_plan,
            )
        external_refs = self._dependency_evidence(context)
        auxiliary_review = await self._review_auxiliary_evidence(
            context,
            intent_plan=intent_plan,
            evidence=external_refs,
        )
        if state is not None and self.services.corrective_agent is not None:
            reflection = await self.services.corrective_agent.review_slots(
                context.query,
                intent_plan,
                plan,
                state,
                system_prompt_prefix=str(context.metadata.get("agent_system_prompt") or ""),
            )
            verified_auxiliary = self._verified_auxiliary_evidence(
                external_refs,
                reflection,
                passed_evidence_ids=set(auxiliary_review["passed_evidence_ids"]),
            )
            return AgentResult.success(
                self._reflection_output(reflection),
                evidence=verified_auxiliary,
                artifacts={
                    "reflection": reflection,
                    "verified_state": state,
                    "verified_external_evidence": verified_auxiliary,
                    "auxiliary_evidence_review": auxiliary_review,
                },
            )
        if single is not None and self.services.corrective_agent is not None:
            reflection = await self.services.corrective_agent.review(
                context.query,
                intent_plan,
                plan,
                single.ranked,
                single.vector_scores,
                single.keyword_scores,
                candidate_sources=single.candidate_sources,
                image_attributes=self._image_attributes(context.artifact("image_attributes") or context.image_attributes),
                system_prompt_prefix=str(context.metadata.get("agent_system_prompt") or ""),
            )
            verified_auxiliary = self._verified_auxiliary_evidence(
                external_refs,
                reflection,
                passed_evidence_ids=set(auxiliary_review["passed_evidence_ids"]),
            )
            return AgentResult.success(
                {
                    **self._reflection_output(reflection),
                    "candidate_pool": candidate_pool,
                },
                evidence=verified_auxiliary,
                artifacts={
                    "reflection": reflection,
                    "verified_evidence": single,
                    "recommendation_candidate_pool": candidate_pool,
                    "verified_external_evidence": verified_auxiliary,
                    "auxiliary_evidence_review": auxiliary_review,
                },
            )
        # Comparison/knowledge-only routes still get a deterministic evidence
        # boundary. They may be answerable, but no candidate is silently marked
        # as a product recommendation.
        verified_auxiliary = self._verified_auxiliary_evidence(
            external_refs,
            ReflectionResult(has_passed_products=False),
            passed_evidence_ids=set(auxiliary_review["passed_evidence_ids"]),
        )
        reason = (
            "外部或本地事实证据已进入回答边界；本轮不产生商品推荐通过集合。"
            if verified_auxiliary
            else "本轮没有可验证的外部证据，回答必须明确说明资料能力不可用。"
        )
        return AgentResult.success(
            {
                "has_passed_products": False,
                "evidence_count": len(verified_auxiliary),
                "status": "non_product_evidence",
            },
            evidence=verified_auxiliary,
            artifacts={
                "reflection": ReflectionResult(
                    has_passed_products=False,
                    reason=reason,
                    fallback_plan="direct_answer",
                ),
                "verified_external_evidence": verified_auxiliary,
                "auxiliary_evidence_review": auxiliary_review,
            },
        )

    async def bundle_optimization(self, context: AgentExecutionContext) -> AgentResult:
        state = context.artifact("verified_state") or context.artifact("multi_state")
        reflection = context.artifact("reflection") or ReflectionResult()
        if state is None:
            return AgentResult.success({"selected_count": 0, "status": "not_applicable"})
        selection = self._selection_from_reflection(state, reflection)
        return AgentResult.success(
            {
                "selected_by_slot": {
                    slot_id: [candidate.product_id for candidate in candidates]
                    for slot_id, candidates in selection.selected_by_slot.items()
                },
                "route": selection.route,
                "reason": selection.reason,
            },
            artifacts={"selection": selection},
        )

    async def repair(self, context: AgentExecutionContext) -> AgentResult:
        if self.services.repair_agent is None:
            return AgentResult.failure_result("repair_not_configured", "RepairAgent is not configured", retryable=False)
        reflection = context.artifact("reflection")
        if not isinstance(reflection, ReflectionResult):
            reflection = ReflectionResult(
                has_passed_products=False,
                reason=str(context.node.metadata.get("reason") or "上游节点失败。"),
            )
        slots = self._slots_for_context(context)
        plan = self._query_plan(context)
        repair_plan = await self.services.repair_agent.plan_repair(
            original_query=context.query,
            intent_plan=self._intent_plan(context),
            plan=plan,
            slots=slots,
            trigger=str(context.node.metadata.get("reason") or "evidence_verification_failed"),
            reflection_result=reflection,
            system_prompt_prefix=str(context.metadata.get("agent_system_prompt") or ""),
        )
        return AgentResult.success(repair_plan.summary(), artifacts={"repair_plan": repair_plan})

    async def answer_generation(self, context: AgentExecutionContext) -> AgentResult:
        generator = self.services.answer_generator
        if generator is None:
            return AgentResult.failure_result("answer_generator_not_configured", "AnswerGenerator is not configured")
        supervisor_plan = self._supervisor_plan(context)
        branch_terminals = dict(
            context.graph.metadata.get("branch_terminal_nodes") or {}
        )
        profile_narrative = str(context.metadata.get("profile_narrative") or "")
        if not profile_narrative:
            profile_items = self._artifact_in_any_branch(context, "profile_memory") or []
            profile_narrative = "；".join(
                f"{item.get('key')}:{item.get('value')}"
                for item in profile_items
                if isinstance(item, dict)
            )
        tokens: list[str] = []
        product_ids: list[str] = []
        cards: list[dict[str, Any]] = []
        branch_results: list[dict[str, Any]] = []
        branch_routes: list[str] = []
        multiple = len(supervisor_plan.intents) > 1

        for index, intent in enumerate(supervisor_plan.intents, start=1):
            terminal_id = str(branch_terminals.get(intent.intent_id) or "")
            if multiple:
                heading = f"\n\n### {index}. {intent.goal}\n"
                tokens.append(heading)
                await context.emit_client_event({"type": "token", "content": heading})

            branch_failed = self._branch_failed(context, intent.intent_id, terminal_id)
            policy_status = (
                context.graph.metadata.get("intent_statuses", {})
                .get(intent.intent_id, {})
                .get("status", "ready")
            )
            intent_plan = self._projection_for_intent(intent, context.query)
            plan = (
                self.services.retrieval_plan_builder.plan(intent_plan)
                if self.services.retrieval_plan_builder is not None
                else QueryPlan()
            )
            reflection = self._artifact_in_branch(
                context,
                terminal_id,
                "reflection",
            )
            if not isinstance(reflection, ReflectionResult):
                reflection = ReflectionResult()
            single = self._artifact_in_branch(
                context,
                terminal_id,
                "verified_evidence",
            ) or self._artifact_in_branch(context, terminal_id, "single_evidence")
            state = self._artifact_in_branch(
                context,
                terminal_id,
                "verified_state",
            ) or self._artifact_in_branch(context, terminal_id, "multi_state")
            selection = self._artifact_in_branch(context, terminal_id, "selection")
            clarification = self._artifact_in_branch(
                context,
                terminal_id,
                "clarification",
            )
            verified_external = self._artifact_in_branch(
                context,
                terminal_id,
                "verified_external_evidence",
            ) or []
            auxiliary_evidence = [
                item.compact() for item in verified_external if isinstance(item, EvidenceRef)
            ]
            auxiliary_context = {
                "verified_external_evidence": auxiliary_evidence,
                "comparison": self._artifact_in_branch(
                    context,
                    terminal_id,
                    "comparison",
                )
                or {},
                "knowledge_research": self._artifact_in_branch(
                    context,
                    terminal_id,
                    "knowledge_research",
                )
                or {},
                "commerce_research": self._artifact_in_branch(
                    context,
                    terminal_id,
                    "commerce_research",
                )
                or [],
                "recommendation_selection": {
                    "policy": intent.recommendation_policy.model_dump(),
                    "verified_product_ids": reflection.verified_product_ids
                    or reflection.passed_product_ids,
                    "selected_product_ids": reflection.selected_product_ids,
                    "quantity_status": reflection.quantity_status,
                    "reference_decisions": reflection.reference_decisions,
                },
            }
            branch_cards: list[dict[str, Any]] = []
            branch_product_ids: list[str] = []

            if policy_status == "unsupported":
                route = "unsupported"
                stream = generator.stream_direct_text(
                    intent.resolved_query,
                    "direct",
                    "该意图没有形成通过策略检查的可执行任务。",
                    intent_plan,
                    preferred_text="这个任务暂时无法执行，因为没有形成可用的专家任务路线。",
                )
            elif branch_failed:
                route = "failed"
                stream = generator.stream_direct_text(
                    intent.resolved_query,
                    "direct",
                    "该意图分支执行失败，其他意图不受影响。",
                    intent_plan,
                    preferred_text="这个任务本轮没有完成；其它任务的结果仍然有效。",
                )
            elif isinstance(clarification, dict) and clarification.get("question"):
                route = "clarify"
                stream = generator.stream_direct_text(
                    intent.resolved_query,
                    "clarification",
                    str(clarification.get("reason") or "需要补充一个关键信息。"),
                    intent_plan,
                    preferred_text=str(clarification["question"]),
                )
            elif state is not None:
                if not isinstance(selection, MultiNeedSelection):
                    selection = self._selection_from_reflection(state, reflection)
                route = selection.route
                branch_product_ids = [
                    candidate.product_id for candidate in selection.flat_candidates
                ]
                branch_cards = [
                    generator.product_card(candidate.product, plan).model_dump()
                    for candidate in selection.flat_candidates
                ]
                if selection.flat_candidates and route not in {"no_product", "clarify"}:
                    stream = generator.stream_multi_need_text(
                        state,
                        selection,
                        route,
                        reflection.reason,
                        reflection.slot_coverage,
                        reflection.rejected_products,
                        reflection.combo_summary,
                        profile_narrative=profile_narrative,
                        extra_context=auxiliary_context,
                        system_prompt_prefix=str(
                            context.metadata.get("agent_system_prompt") or ""
                        ),
                    )
                else:
                    stream = generator.stream_direct_text(
                        intent.resolved_query,
                        "no_product",
                        reflection.reason or "没有足够可靠的商品证据。",
                        intent_plan,
                        extra_context=auxiliary_context,
                        profile_narrative=profile_narrative,
                    )
            elif single is not None:
                selected_ids = list(reflection.selected_product_ids)
                if (
                    not selected_ids
                    and reflection.quantity_status == "not_applicable"
                ):
                    selected_ids = list(reflection.passed_product_ids)
                ranked_by_id = {
                    product.product_id: (product, score)
                    for product, score in single.ranked
                }
                ranked = [
                    ranked_by_id[product_id]
                    for product_id in selected_ids
                    if product_id in ranked_by_id
                ]
                branch_product_ids = [product.product_id for product, _ in ranked]
                branch_cards = [
                    generator.product_card(product, plan).model_dump()
                    for product, _ in ranked
                ]
                route = "recommend" if ranked else "no_product"
                if ranked:
                    stream = generator.stream_text(
                        plan,
                        ranked,
                        profile_narrative=profile_narrative,
                        image_attributes=self._image_attributes(
                            self._artifact_in_branch(
                                context,
                                terminal_id,
                                "image_attributes",
                            )
                            or context.image_attributes
                        ),
                        extra_context=auxiliary_context,
                        system_prompt_prefix=str(
                            context.metadata.get("agent_system_prompt") or ""
                        ),
                    )
                else:
                    stream = generator.stream_direct_text(
                        intent.resolved_query,
                        "no_product",
                        reflection.reason or "没有足够可靠的商品证据。",
                        intent_plan,
                        extra_context=auxiliary_context,
                        profile_narrative=profile_narrative,
                    )
            else:
                route = "direct_answer"
                reason = (
                    reflection.reason
                    or intent.route_basis.reason
                    or "本轮使用已校验的知识或对比证据回答。"
                )
                stream = generator.stream_direct_text(
                    intent.resolved_query,
                    "direct",
                    reason,
                    intent_plan,
                    extra_context={
                        **auxiliary_context,
                        "external_evidence_status": (
                            "verified" if auxiliary_evidence else "unavailable_or_empty"
                        ),
                    },
                    profile_narrative=profile_narrative,
                    system_prompt_prefix=str(
                        context.metadata.get("agent_system_prompt") or ""
                    ),
                )

            if branch_cards:
                await context.emit_client_event(
                    {"type": "product_cards", "products": branch_cards}
                )
            async for token in stream:
                token = str(token)
                tokens.append(token)
                await context.emit_client_event({"type": "token", "content": token})
            cards.extend(branch_cards)
            product_ids.extend(branch_product_ids)
            branch_routes.append(route)
            branch_results.append(
                {
                    "intent_id": intent.intent_id,
                    "goal": intent.goal,
                    "route": route,
                    "terminal_node_id": terminal_id,
                    "product_ids": branch_product_ids,
                    "failed": branch_failed,
                }
            )

        answer_text = "".join(tokens)
        route = branch_routes[0] if len(branch_routes) == 1 else "multi_intent"
        cards = list({item.get("product_id"): item for item in cards if item.get("product_id")}.values())
        product_ids = list(dict.fromkeys(product_ids))
        return AgentResult.success(
            {
                "route": route,
                "answer_length": len(answer_text),
                "product_ids": product_ids,
                "cards": cards,
                "branches": branch_results,
            },
            artifacts={
                "answer_text": answer_text,
                "answer_tokens": tokens,
                "answer_cards": cards,
                "answer_product_ids": product_ids,
                "answer_route": route,
                "answer_branches": branch_results,
            },
        )

    async def memory_distillation(self, context: AgentExecutionContext) -> AgentResult:
        answer = str(context.artifact("answer_text") or "")
        scheduler = self.services.memory_scheduler
        if scheduler is not None:
            value = scheduler(context, answer)
            if asyncio.iscoroutine(value):
                await value
        return AgentResult.success({"scheduled": scheduler is not None, "answer_length": len(answer)}, termination_reason="scheduled")

    async def _merge_bundle(self, context: AgentExecutionContext) -> AgentResult:
        slots = self._slots_for_context(context)
        state = MultiNeedState(
            original_query=context.query,
            intent_plan=self._intent_plan(context),
            plan=self._query_plan(context),
            global_constraints=[],
            slots=slots,
            budgets={"slot_task_mode": "supervisor_task_graph", "parallel_slot_agents": len(slots)},
        )
        for slot in slots:
            result = context.artifact(f"slot_result:{slot.slot_id}")
            if result is None:
                state.candidates_by_slot[slot.slot_id] = []
                continue
            state.candidates_by_slot[slot.slot_id] = list(result.candidates)
            state.tool_calls.extend(result.tool_calls)
            state.budgets.setdefault("slot_results_by_slot", {})[slot.slot_id] = result.slot_result
        coordinator = getattr(self.services.retrieval_worker, "multi_need_coordinator", None)
        if coordinator is not None:
            coordinator.verify_coverage(state)
            state.final_signal = coordinator._partial_or_no_product_signal(state, "slot agents completed")
        return AgentResult.success(
            {
                "slot_count": len(state.slots),
                "candidate_count": sum(len(items) for items in state.candidates_by_slot.values()),
                "coverage": {key: value.model_dump() for key, value in state.coverage_by_slot.items()},
            },
            artifacts={"multi_state": state},
        )

    def _projection_for_intent(
        self,
        intent: IntentItem,
        fallback_query: str,
    ) -> IntentPlan:
        budget_min, budget_max, budget_scope = self._intent_budget(intent)
        slots = [self._rewrite_slot(intent, need) for need in intent.product_needs]
        if intent.intent_type == "social_chat":
            plan_type = "direct_answer"
        elif any(item.blocking for item in intent.uncertainties) and not intent.product_needs:
            plan_type = "clarify"
        elif len(slots) >= 2:
            plan_type = "multi_retrieval"
        else:
            plan_type = "single_retrieval"
        query = intent.resolved_query or fallback_query
        return IntentPlan(
            original_query=query,
            normalized_query=query,
            summary=intent.goal,
            constraints=IntentConstraintSet(
                budget_min=budget_min,
                budget_max=budget_max,
                budget_scope=budget_scope,
                items=list(intent.constraints),
            ),
            plan_type=plan_type,
            vector_query=query if plan_type not in {"direct_answer", "clarify"} else "",
            keyword_query=query if plan_type not in {"direct_answer", "clarify"} else "",
            budget_min=budget_min,
            budget_max=budget_max,
            budget_scope=budget_scope,
            need_slots=slots,
            referenced_product_ids=list(intent.referenced_product_ids),
            recommendation_policy=intent.recommendation_policy.model_copy(deep=True),
            plan_reason=intent.route_basis.reason,
        )

    def _artifact_in_branch(
        self,
        context: AgentExecutionContext,
        terminal_node_id: str,
        key: str,
    ) -> Any:
        if not terminal_node_id or context.graph.get_node(terminal_node_id) is None:
            return None
        node_ids = [
            *context.graph.ancestor_node_ids(terminal_node_id),
            terminal_node_id,
        ]
        for node_id in reversed(node_ids):
            scoped = context.artifacts.get(f"artifacts:{node_id}")
            if isinstance(scoped, dict) and key in scoped:
                return scoped[key]
        return None

    def _artifact_in_any_branch(
        self,
        context: AgentExecutionContext,
        key: str,
    ) -> Any:
        for node_id in reversed(context.graph.topological_order()):
            scoped = context.artifacts.get(f"artifacts:{node_id}")
            if isinstance(scoped, dict) and key in scoped:
                return scoped[key]
        return None

    def _branch_failed(
        self,
        context: AgentExecutionContext,
        intent_id: str,
        terminal_node_id: str,
    ) -> bool:
        if not terminal_node_id or context.graph.get_node(terminal_node_id) is None:
            return False
        node_ids = {
            *context.graph.ancestor_node_ids(terminal_node_id),
            terminal_node_id,
        }
        for node_id in node_ids:
            node = context.graph.require_node(node_id)
            if intent_id not in node.intent_ids or not node.required:
                continue
            if node.status == "skipped" and node.metadata.get("skipped_reason") in {
                "dependency_failed",
                "optional_branch_failed",
            }:
                return True
            if node.status not in {"failed", "timeout", "cancelled"}:
                continue
            replacement_id = str(node.metadata.get("superseded_by") or "")
            replacement = context.graph.get_node(replacement_id) if replacement_id else None
            if replacement is None or replacement.status != "succeeded":
                return True
        return False

    def _query_plan(
        self,
        context: AgentExecutionContext,
        *,
        intent_plan: IntentPlan | None = None,
    ) -> QueryPlan:
        if isinstance(context.query_plan, QueryPlan):
            return context.query_plan
        artifact_plan = context.artifact("query_plan")
        if isinstance(artifact_plan, QueryPlan):
            return artifact_plan
        if self.services.retrieval_plan_builder is not None:
            return self.services.retrieval_plan_builder.plan(intent_plan or self._intent_plan(context))
        return QueryPlan()

    def _intent_plan(self, context: AgentExecutionContext) -> IntentPlan:
        supervisor_plan = self._supervisor_plan(context)
        intent = self._target_intent(context, supervisor_plan)
        if intent is None:
            return IntentPlan(original_query=context.query, normalized_query=context.query)

        parameters = context.node.metadata.get("parameters") or {}
        selected_need_ids = set(parameters.get("product_need_ids") or [])
        needs = [
            need
            for need in intent.product_needs
            if not selected_need_ids or need.need_id in selected_need_ids
        ]
        budget_min, budget_max, budget_scope = self._intent_budget(intent)
        constraints = IntentConstraintSet(
            budget_min=budget_min,
            budget_max=budget_max,
            budget_scope=budget_scope,
            items=list(intent.constraints),
        )
        slots = [self._rewrite_slot(intent, need) for need in needs]
        capability = context.node.capability
        if capability == "multi_product_bundle" or len(slots) >= 2:
            plan_type = "multi_retrieval"
        elif capability == "clarification":
            plan_type = "clarify"
        else:
            plan_type = "single_retrieval"
        query = str(parameters.get("query") or intent.resolved_query or intent.goal).strip()
        return IntentPlan(
            original_query=intent.resolved_query or context.query,
            normalized_query=query,
            summary=intent.goal,
            constraints=constraints,
            plan_type=plan_type,
            vector_query=query if plan_type not in {"clarify", "direct_answer"} else "",
            keyword_query=query if plan_type not in {"clarify", "direct_answer"} else "",
            budget_min=budget_min,
            budget_max=budget_max,
            budget_scope=budget_scope,
            need_slots=slots,
            referenced_product_ids=list(
                dict.fromkeys(
                    [
                        *intent.referenced_product_ids,
                        *(parameters.get("referenced_product_ids") or []),
                    ]
                )
            ),
            recommendation_policy=intent.recommendation_policy.model_copy(deep=True),
            profile_lookup=ProfileLookupProposal(),
            plan_reason=str(context.node.metadata.get("reason") or intent.route_basis.reason),
        )

    def _supervisor_plan(self, context: AgentExecutionContext) -> IntentPlanV3:
        if isinstance(context.intent_plan, IntentPlanV3):
            return context.intent_plan
        value = context.artifacts.get("intent_plan")
        if isinstance(value, IntentPlanV3):
            return value
        return IntentPlanV3(
            original_query=context.query,
            normalized_query=context.query,
            intents=[],
            task_proposals=[],
        )

    def _target_intent(
        self,
        context: AgentExecutionContext,
        plan: IntentPlanV3 | None = None,
    ) -> IntentItem | None:
        resolved = plan or self._supervisor_plan(context)
        selected = set(context.node.intent_ids)
        return next(
            (intent for intent in resolved.intents if intent.intent_id in selected),
            None,
        )

    def _bounded_intent_revision(
        self,
        original: IntentPlanV3,
        revised: IntentPlanV3,
        target_intent_id: str,
    ) -> IntentPlanV3:
        original_intent = next(
            (item for item in original.intents if item.intent_id == target_intent_id),
            None,
        )
        revised_intent = next(
            (item for item in revised.intents if item.intent_id == target_intent_id),
            revised.intents[0] if revised.intents else None,
        )
        if original_intent is None or revised_intent is None:
            return revised
        bounded_intent = revised_intent.model_copy(
            update={
                "intent_id": target_intent_id,
                "priority": original_intent.priority,
                "goal": original_intent.goal,
                "constraints": original_intent.constraints,
                "referenced_product_ids": original_intent.referenced_product_ids,
            }
        )
        allowed_tasks = [
            task
            for task in revised.task_proposals
            if task.capability != "knowledge_research"
        ]
        allowed_ids = {task.task_id for task in allowed_tasks}
        bounded_tasks = [
            task.model_copy(
                update={
                    "intent_ids": [target_intent_id],
                    "depends_on": [
                        value for value in task.depends_on if value in allowed_ids
                    ],
                    "optional_context_from": [
                        value
                        for value in task.optional_context_from
                        if value in allowed_ids
                    ],
                }
            )
            for task in allowed_tasks
        ]
        return IntentPlanV3(
            original_query=original.original_query,
            normalized_query=bounded_intent.resolved_query,
            summary=revised.summary,
            intents=[bounded_intent],
            task_proposals=bounded_tasks,
        )

    def _profile_aware_intent_plan(
        self,
        context: AgentExecutionContext,
        intent_plan: IntentPlan,
    ) -> IntentPlan:
        terms = self._profile_terms(context)
        if not terms:
            return intent_plan
        semantic_query = " ".join(
            part for part in [intent_plan.vector_query.strip(), "软偏好 " + " ".join(terms)] if part
        )
        return intent_plan.model_copy(update={"vector_query": semantic_query})

    def _profile_aware_slot(self, context: AgentExecutionContext, slot: NeedSlot) -> NeedSlot:
        terms = self._profile_terms(context)
        if not terms:
            return slot
        soft_constraints = list(dict.fromkeys([*slot.soft_constraints, *(f"长期软偏好:{term}" for term in terms)]))
        return slot.model_copy(update={"soft_constraints": soft_constraints})

    def _profile_terms(self, context: AgentExecutionContext) -> list[str]:
        memory = context.artifact("profile_memory") or []
        result: list[str] = []
        for item in memory[:6]:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key") or "").strip()
            value = str(item.get("value") or "").strip()
            if key and value:
                result.append(f"{key}:{value}")
        return result

    def _slots_for_context(self, context: AgentExecutionContext) -> list[NeedSlot]:
        slots: list[NeedSlot] = []
        selected_intents = set(context.node.intent_ids)
        for node in context.graph.nodes:
            if selected_intents and not selected_intents.intersection(node.intent_ids):
                continue
            payload = node.metadata.get("slot")
            if isinstance(payload, dict):
                slot = self._need_slot(payload)
                if slot.slot_id not in {item.slot_id for item in slots}:
                    slots.append(slot)
        if not slots:
            slots = [self._need_slot(item) for item in self._intent_plan(context).need_slots]
        if not slots:
            intent_plan = self._intent_plan(context)
            hard_constraints = [
                f"{item.name}={item.value}"
                for item in intent_plan.constraints.items
                if item.strength == "hard"
            ]
            soft_constraints = [
                f"{item.name}={item.value}"
                for item in intent_plan.constraints.items
                if item.strength == "soft"
            ]
            slots = [
                NeedSlot(
                    slot_id="single",
                    need_type="required",
                    goal=intent_plan.normalized_query or intent_plan.summary or context.query or "单商品推荐",
                    product_type="",
                    query=intent_plan.vector_query or intent_plan.keyword_query or context.query,
                    hard_constraints=hard_constraints,
                    soft_constraints=soft_constraints,
                    min_candidates=1,
                )
            ]
        return slots

    def _intent_budget(self, intent: IntentItem) -> tuple[float | None, float | None, str]:
        minimum: float | None = None
        maximum: float | None = None
        scope = "unknown"
        for item in intent.constraints:
            name = item.name.strip().lower()
            value = item.value
            if name in {"budget_scope", "预算范围"} and str(value) in {
                "per_item",
                "total",
                "unknown",
            }:
                scope = str(value)
                continue
            try:
                number = float(value) if not isinstance(value, list) else None
            except (TypeError, ValueError):
                number = None
            if number is None:
                continue
            if name in {"budget_min", "min_budget", "最低预算", "预算下限"}:
                minimum = number
            elif name in {
                "budget_max",
                "max_budget",
                "budget",
                "最高预算",
                "预算上限",
                "预算",
            }:
                maximum = number
        return minimum, maximum, scope

    def _rewrite_slot(self, intent: IntentItem, need: Any) -> RewriteNeedSlot:
        constraints = [*intent.constraints, *need.constraints]
        soft_constraints = [
            f"{item.name}={item.value}"
            for item in constraints
            if item.strength == "soft"
        ]
        hard_terms = [
            f"{item.name}={item.value}"
            for item in constraints
            if item.strength == "hard"
        ]
        query = " ".join(
            dict.fromkeys(
                value
                for value in [need.product_type, need.goal, *hard_terms, *soft_constraints]
                if str(value).strip()
            )
        )
        return RewriteNeedSlot(
            slot_id=need.need_id,
            intent_id=intent.intent_id,
            need_type=need.priority,
            goal=need.goal,
            product_type=need.product_type,
            query=query or intent.resolved_query,
            semantic_query=query or intent.resolved_query,
            keyword_query=query or intent.resolved_query,
            soft_constraints=soft_constraints,
            exclude_terms=list(need.exclusions),
            min_candidates=1,
        )

    def _need_slot(self, payload: Any) -> NeedSlot:
        if isinstance(payload, NeedSlot):
            return payload.model_copy(deep=True)
        value = dict(payload or {})
        return NeedSlot(
            slot_id=str(value.get("slot_id") or "slot"),
            need_type=value.get("need_type") if value.get("need_type") in {"required", "optional"} else "required",
            goal=str(value.get("goal") or value.get("product_type") or value.get("query") or "商品"),
            product_type=str(value.get("product_type") or ""),
            query=str(value.get("query") or value.get("semantic_query") or value.get("keyword_query") or ""),
            hard_constraints=list(value.get("hard_constraints") or []),
            soft_constraints=list(value.get("soft_constraints") or []),
            exclude_terms=list(value.get("exclude_terms") or []),
            min_candidates=max(1, int(value.get("min_candidates") or 1)),
        )

    def _single_evidence_output(self, evidence: Any, intent_plan: IntentPlan, plan: QueryPlan) -> dict[str, Any]:
        summary = evidence.summary(intent_plan, plan) if hasattr(evidence, "summary") else {}
        summary = dict(summary)
        summary["candidate_ids"] = [product.product_id for product, _ in getattr(evidence, "ranked", [])]
        summary["candidate_count"] = len(getattr(evidence, "ranked", []))
        return summary

    async def _merge_single_candidate_pool(
        self,
        context: AgentExecutionContext,
        evidence: Any,
        intent_plan: IntentPlan,
    ) -> tuple[Any, list[dict[str, Any]]]:
        policy = intent_plan.recommendation_policy
        include_context = policy.candidate_source in {
            "context_only",
            "context_plus_new",
        }
        include_retrieved = policy.candidate_source in {
            "new_only",
            "context_plus_new",
        }
        referenced_ids = list(dict.fromkeys(intent_plan.referenced_product_ids))
        context_products: list[Any] = []
        if include_context and referenced_ids:
            context_products = await context.execute_tool(
                "product_detail",
                "get_referenced_candidates",
                lambda tool: tool.get_by_ids(referenced_ids),
                input_summary={
                    "product_count": len(referenced_ids),
                    "candidate_source": policy.candidate_source,
                },
            )

        retrieved_ranked = list(getattr(evidence, "ranked", []) or [])
        retrieved_scores = {
            product.product_id: float(score)
            for product, score in retrieved_ranked
        }
        retrieved_by_id = {
            product.product_id: product for product, _ in retrieved_ranked
        }
        context_by_id = {
            product.product_id: product for product in context_products
        }

        ordered_ids: list[str] = []
        if include_context:
            ordered_ids.extend(
                product_id
                for product_id in referenced_ids
                if product_id in context_by_id
            )
        if include_retrieved:
            ordered_ids.extend(product.product_id for product, _ in retrieved_ranked)
        ordered_ids = list(dict.fromkeys(ordered_ids))

        candidate_sources: dict[str, list[str]] = {}
        ranked: list[tuple[Any, float]] = []
        for product_id in ordered_ids:
            sources: list[str] = []
            if product_id in context_by_id:
                sources.append("context")
            if product_id in retrieved_by_id:
                sources.append("retrieved")
            candidate_sources[product_id] = sources
            product = context_by_id.get(product_id) or retrieved_by_id[product_id]
            ranked.append((product, retrieved_scores.get(product_id, 0.0)))

        merged = copy.copy(evidence) if evidence is not None else SingleRetrievalEvidence()
        merged.ranked = ranked
        merged.candidate_sources = candidate_sources
        merged.vector_scores = dict(getattr(evidence, "vector_scores", {}) or {})
        merged.keyword_scores = dict(getattr(evidence, "keyword_scores", {}) or {})
        merged.structured_products = self._merge_products(
            context_products if include_context else [],
            list(getattr(evidence, "structured_products", []) or [])
            if include_retrieved
            else [],
        )
        merged.score_filtered_products = [product for product, _ in ranked]
        merged.hybrid_ranked_products = [product for product, _ in ranked]
        if ranked:
            merged.failure_trigger = ""

        candidate_pool = [
            {
                "product_id": product.product_id,
                "name": product.name,
                "sources": candidate_sources.get(product.product_id, []),
                "is_explicit_reference": "context"
                in candidate_sources.get(product.product_id, []),
                "reference_policy": (
                    policy.reference_policy
                    if "context" in candidate_sources.get(product.product_id, [])
                    else "none"
                ),
                "rerank_score": round(float(score), 4),
            }
            for product, score in ranked
        ]
        return merged, candidate_pool

    def _merge_products(self, *groups: list[Any]) -> list[Any]:
        merged: dict[str, Any] = {}
        for group in groups:
            for product in group:
                product_id = str(getattr(product, "product_id", "") or "")
                if product_id and product_id not in merged:
                    merged[product_id] = product
        return list(merged.values())

    def _slot_result_output(self, result: SlotRetrievalAgentResult) -> dict[str, Any]:
        return {
            "slot_id": result.slot.slot_id,
            "candidate_ids": [candidate.product_id for candidate in result.candidates],
            "candidate_count": len(result.candidates),
            "search_calls": result.search_calls,
            "termination_reason": result.termination_reason,
            "slot_result": result.slot_result,
        }

    def _reflection_output(self, reflection: ReflectionResult) -> dict[str, Any]:
        return {
            "has_passed_products": reflection.has_passed_products,
            "verified_product_ids": reflection.verified_product_ids,
            "passed_product_ids": reflection.passed_product_ids,
            "selected_product_ids": reflection.selected_product_ids,
            "quantity_status": reflection.quantity_status,
            "reference_decisions": reflection.reference_decisions,
            "rejected_count": len(reflection.rejected_products),
            "slot_coverage": reflection.slot_coverage,
            "fallback_plan": reflection.fallback_plan,
            "repairable": reflection.repair_hint.repairable,
            "reason": reflection.reason,
        }

    def _selection_from_reflection(self, state: MultiNeedState, reflection: ReflectionResult) -> MultiNeedSelection:
        selected_by_slot: dict[str, list[SlotCandidate]] = {}
        by_id = {
            candidate.product_id: candidate
            for candidates in state.candidates_by_slot.values()
            for candidate in candidates
        }
        mapping = (reflection.combo_summary or {}).get("final_combo_product_ids_by_slot", {})
        for slot in state.slots:
            ids = mapping.get(slot.slot_id) or [
                item.get("selected_product_ids", [])[0]
                for item in reflection.slot_coverage
                if item.get("slot_id") == slot.slot_id and item.get("selected_product_ids")
            ]
            if not ids:
                ids = [product_id for product_id in reflection.passed_product_ids if product_id in {candidate.product_id for candidate in state.candidates_by_slot.get(slot.slot_id, [])}][:1]
            selected_by_slot[slot.slot_id] = [by_id[product_id] for product_id in ids if product_id in by_id]
        route = "recommend" if all(selected_by_slot.get(slot.slot_id) for slot in state.slots if slot.need_type == "required") else "partial_recommend"
        if not any(selected_by_slot.values()):
            route = "no_product"
        return MultiNeedSelection(selected_by_slot=selected_by_slot, route=route, reason=reflection.reason)

    def _route(self, context: AgentExecutionContext, reflection: ReflectionResult, state: Any, selection: Any) -> str:
        intent_plan = self._intent_plan(context)
        if intent_plan.plan_type == "clarify":
            return "clarify"
        if intent_plan.plan_type == "direct_answer":
            return "direct_answer"
        if isinstance(selection, MultiNeedSelection):
            return selection.route
        if reflection.has_passed_products:
            return "recommend"
        return reflection.fallback_plan if reflection.fallback_plan != "none" else "no_product"

    def _memory_item(self, item: Any) -> dict[str, Any]:
        return {
            "key": item.get("key", "") if isinstance(item, dict) else "",
            "value": item.get("value", "") if isinstance(item, dict) else "",
            "confidence": item.get("confidence") if isinstance(item, dict) else None,
            "source": item.get("source", "") if isinstance(item, dict) else "",
        }

    def _clarification_question(
        self,
        missing: list[str],
        question_goal: str,
        query: str,
    ) -> str:
        if missing:
            return f"为了更准确推荐，请先告诉我：{missing[0]}。"
        if question_goal.strip():
            return question_goal.strip()
        return "你想购买哪一类商品，主要使用场景或预算是多少？"

    def _product_payload(self, product: Any) -> dict[str, Any]:
        return {
            "product_id": product.product_id,
            "name": product.name,
            "brand": product.brand,
            "category": product.category,
            "sub_category": product.sub_category,
            "price": float(product.price),
            "rating": float(product.rating),
            "description": str(product.description or "")[:400],
            "suitable_for": str(product.suitable_for or "")[:300],
            "avoid_for": str(product.avoid_for or "")[:300],
        }

    def _comparison_dimensions(self, products: list[dict[str, Any]]) -> list[dict[str, Any]]:
        dimensions = ["price", "rating", "brand", "suitable_for", "avoid_for"]
        return [{"name": name, "values": [{"product_id": item["product_id"], "value": item.get(name)} for item in products]} for name in dimensions]

    def _comparison_evidence(
        self,
        products: list[dict[str, Any]],
        *,
        referenced_ids: set[str] | None = None,
    ) -> list[EvidenceRef]:
        referenced_ids = referenced_ids or set()
        return [
            EvidenceRef(
                evidence_id=f"product:{item['product_id']}",
                source_type="local_product_catalog",
                source_id=item["product_id"],
                claim=item["name"],
                summary=self._compact_json(item, 500),
                product_id=item["product_id"],
                verified=True,
                provenance={
                    "evidence_role": (
                        "referenced_context"
                        if item["product_id"] in referenced_ids
                        else "comparison_candidate"
                    )
                },
            )
            for item in products
        ]

    def _comparison_analysis_evidence(
        self,
        analysis: dict[str, Any],
        *,
        supporting_evidence_ids: list[str],
    ) -> list[EvidenceRef]:
        if not analysis.get("used_llm") or not supporting_evidence_ids:
            return []
        result: list[EvidenceRef] = []
        for index, winner in enumerate(analysis.get("winner_by_goal", [])[:12]):
            if not isinstance(winner, dict):
                continue
            product_id = str(winner.get("product_id") or "")
            goal = str(winner.get("goal") or "").strip()
            reason = str(winner.get("reason") or "").strip()
            if not product_id or not (goal or reason):
                continue
            result.append(
                EvidenceRef(
                    evidence_id=f"comparison:conclusion:{index}",
                    source_type="agent_derived",
                    source_id="comparison_agent",
                    product_id=product_id,
                    claim=f"{goal}: {product_id}",
                    summary=reason,
                    provenance={
                        "source_agent": "comparison_agent",
                        "supporting_evidence_ids": supporting_evidence_ids,
                        "evidence_role": "comparison_conclusion",
                    },
                )
            )
        return result

    def _knowledge_analysis_evidence(
        self,
        reasoning: dict[str, Any],
        *,
        web_evidence: list[EvidenceRef],
    ) -> list[EvidenceRef]:
        if not reasoning.get("used_llm"):
            return []
        result: list[EvidenceRef] = []
        for index, concept in enumerate(reasoning.get("grounded_concepts", [])[:12]):
            if not isinstance(concept, dict):
                continue
            value = str(concept.get("concept") or "").strip()
            evidence_indexes = [
                evidence_index
                for evidence_index in concept.get("evidence_indexes", [])
                if isinstance(evidence_index, int) and 0 <= evidence_index < len(web_evidence)
            ]
            supporting_ids = [web_evidence[evidence_index].evidence_id for evidence_index in evidence_indexes]
            if not value or not supporting_ids:
                continue
            result.append(
                EvidenceRef(
                    evidence_id=f"knowledge:concept:{index}",
                    source_type="agent_derived",
                    source_id="product_knowledge_agent",
                    claim=f"候选商品概念：{value}",
                    summary="该概念由联网资料抽取，仍需 Supervisor 审批和本地商品检索校验。",
                    provenance={
                        "source_agent": "product_knowledge_agent",
                        "supporting_evidence_ids": supporting_ids,
                        "evidence_role": "knowledge_candidate_concept",
                    },
                )
            )
        return result

    def _knowledge_concept_proposals(
        self,
        reasoning: dict[str, Any],
        *,
        web_evidence: list[EvidenceRef],
    ) -> list[dict[str, Any]]:
        proposals: list[dict[str, Any]] = []
        risks = [str(item).strip() for item in reasoning.get("risks", []) if str(item).strip()][:8]
        for index, concept in enumerate(reasoning.get("grounded_concepts", [])[:12]):
            if not isinstance(concept, dict):
                continue
            value = str(concept.get("concept") or "").strip()
            evidence_indexes = [
                item
                for item in concept.get("evidence_indexes", [])
                if isinstance(item, int) and 0 <= item < len(web_evidence)
            ]
            supporting_ids = list(dict.fromkeys(web_evidence[item].evidence_id for item in evidence_indexes))
            if not value or not supporting_ids:
                continue
            proposals.append(
                {
                    "proposal_id": f"knowledge_concept:{index}",
                    "concept": value,
                    "concept_type": str(concept.get("concept_type") or "other"),
                    "supporting_evidence_ids": supporting_ids,
                    "source": str(concept.get("source") or "product_knowledge_agent"),
                    "risk_flags": risks,
                    "approval": "pending_supervisor",
                }
            )
        return proposals

    def _candidate_terms(self, title: str, snippet: str) -> list[str]:
        # Keep this conservative: only short noun-like phrases supplied by the
        # search result become proposals, never hard constraints.
        values = []
        for value in [title, snippet]:
            for token in value.replace("，", " ").replace("。", " ").split():
                token = token.strip("，。；;:：()（）")
                if 2 <= len(token) <= 12 and token not in values:
                    values.append(token)
        return values[:4]

    def _response_candidate_terms(self, response: Any) -> list[str]:
        if not isinstance(response, dict):
            return []
        values: list[Any] = []
        for key in ("query_expansions", "candidate_concepts", "keywords", "entities"):
            raw = response.get(key)
            if isinstance(raw, list):
                values.extend(raw)
        result: list[str] = []
        for value in values:
            if isinstance(value, dict):
                value = value.get("name") or value.get("text") or value.get("value")
            term = str(value or "").strip()
            if 2 <= len(term) <= 24 and term not in result:
                result.append(term)
        return result[:12]

    def _append_knowledge_claims(
        self,
        claims: list[dict[str, Any]],
        response: Any,
        *,
        research_query: str,
        research_round: int,
    ) -> int:
        if not self._response_available(response):
            return 0
        known_sources = {
            str(item.get("source_id") or item.get("url") or "").strip().casefold()
            for item in claims
            if str(item.get("source_id") or item.get("url") or "").strip()
        }
        added = 0
        for item in self._items_from_response(response)[:10]:
            title = str(item.get("title") or item.get("name") or "").strip()
            snippet = str(
                item.get("snippet")
                or item.get("summary")
                or item.get("description")
                or item.get("content")
                or ""
            ).strip()
            url = str(item.get("url") or item.get("link") or "").strip()
            source_id = str(
                item.get("source_id")
                or url
                or item.get("id")
                or ""
            ).strip()
            source_key = source_id.casefold()
            if not source_key or source_key in known_sources or not (title or snippet):
                continue
            claims.append(
                {
                    "title": title[:240],
                    "snippet": snippet[:1200],
                    "url": url,
                    "source_id": source_id,
                    "research_query": research_query[:500],
                    "research_round": research_round,
                }
            )
            known_sources.add(source_key)
            added += 1
            if len(claims) >= 16:
                break
        return added

    def _items_from_response(self, response: Any) -> list[dict[str, Any]]:
        return self._find_items(response, depth=0)

    def _find_items(self, response: Any, *, depth: int) -> list[dict[str, Any]]:
        if depth > 7 or not isinstance(response, dict):
            return []
        for key in (
            "items",
            "item_list",
            "products",
            "product_list",
            "notes",
            "note_list",
            "comments",
            "comment_list",
            "results",
            "list",
        ):
            nested = response.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
        for key in ("data", "result", "result_data", "payload", "raw"):
            nested = self._find_items(response.get(key), depth=depth + 1)
            if nested:
                return nested
        if self._external_item_id(response):
            return [response]
        return []

    def _dependency_evidence(self, context: AgentExecutionContext) -> list[EvidenceRef]:
        result: list[EvidenceRef] = []
        seen: set[str] = set()
        for dependency_id in dict.fromkeys(
            [*context.node.depends_on, *context.node.input_refs]
        ):
            refs = context.artifact(f"evidence:{dependency_id}", [])
            for ref in refs:
                if not isinstance(ref, EvidenceRef) or not ref.evidence_id or ref.evidence_id in seen:
                    continue
                if not ref.source_type or not ref.source_id:
                    continue
                result.append(ref)
                seen.add(ref.evidence_id)
        return result

    def _verified_auxiliary_evidence(
        self,
        evidence: list[EvidenceRef],
        reflection: ReflectionResult,
        *,
        passed_evidence_ids: set[str] | None = None,
    ) -> list[EvidenceRef]:
        passed_ids = set(reflection.passed_product_ids)
        approved_evidence_ids = passed_evidence_ids or set()
        result: list[EvidenceRef] = []
        for ref in evidence:
            is_referenced_context = (
                ref.verified
                and ref.source_type == "local_product_catalog"
                and ref.provenance.get("evidence_role") == "referenced_context"
            )
            if (
                ref.source_type == "local_product_catalog"
                and ref.product_id
                and ref.product_id not in passed_ids
                and not is_referenced_context
            ):
                continue
            if not ref.product_id and not self._auxiliary_source_is_traceable(ref):
                continue
            if ref.evidence_id not in approved_evidence_ids:
                continue
            if ref.source_type == "agent_derived":
                supporting = {
                    str(item)
                    for item in ref.provenance.get("supporting_evidence_ids", [])
                    if str(item)
                }
                if not supporting or not supporting.issubset(approved_evidence_ids):
                    continue
            result.append(
                EvidenceRef(
                    evidence_id=ref.evidence_id,
                    source_type=ref.source_type,
                    source_id=ref.source_id,
                    claim=ref.claim,
                    summary=ref.summary,
                    product_id=ref.product_id,
                    slot_id=ref.slot_id,
                    score=ref.score,
                    verified=True,
                    provenance={
                        **ref.provenance,
                        "verification_scope": (
                            "catalog_fact"
                            if ref.source_type == "local_product_catalog"
                            else "source_and_lineage"
                        ),
                    },
                )
            )
        return result

    async def _review_auxiliary_evidence(
        self,
        context: AgentExecutionContext,
        *,
        intent_plan: IntentPlan,
        evidence: list[EvidenceRef],
    ) -> dict[str, Any]:
        if not evidence:
            return {
                "passed_evidence_ids": [],
                "rejected_evidence": [],
                "reason": "没有辅助证据需要审核。",
                "used_llm": False,
            }
        corrective = self.services.corrective_agent
        if corrective is None or not hasattr(corrective, "review_auxiliary"):
            return {
                "passed_evidence_ids": [
                    ref.evidence_id
                    for ref in evidence
                    if ref.verified
                    and ref.source_type == "local_product_catalog"
                    and ref.provenance.get("evidence_role") == "referenced_context"
                ],
                "rejected_evidence": [],
                "reason": "辅助证据审核器不可用，只保留本地引用商品事实。",
                "used_llm": False,
            }
        return await corrective.review_auxiliary(
            context.query,
            intent_plan,
            [ref.compact() for ref in evidence],
            system_prompt_prefix=str(context.metadata.get("agent_system_prompt") or ""),
        )

    def _response_available(self, response: Any) -> bool:
        return (
            isinstance(response, dict)
            and bool(response.get("available"))
            and bool(self._items_from_response(response))
        )

    def _commerce_results_available(self, results: Any) -> bool:
        return isinstance(results, list) and any(
            isinstance(item, dict) and bool(item.get("available")) and bool(item.get("items"))
            for item in results
        )

    def _comparison_knowledge_context(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        claims = value.get("claims") if isinstance(value.get("claims"), list) else []
        concepts = (
            value.get("grounded_concepts")
            if isinstance(value.get("grounded_concepts"), list)
            else []
        )
        risks = value.get("risks") if isinstance(value.get("risks"), list) else []
        if not claims and not concepts and not risks:
            return {}
        return {
            "claims": claims[:12],
            "grounded_concepts": concepts[:12],
            "risks": risks[:8],
        }

    def _external_item_id(self, item: Any) -> str:
        if not isinstance(item, dict):
            return ""
        for key in ("item_id", "itemId", "product_id", "productId", "note_id", "noteId", "id"):
            value = str(item.get(key) or "").strip()
            if value:
                return value
        return ""

    def _external_item_url(self, item: Any) -> str:
        if not isinstance(item, dict):
            return ""
        for key in ("url", "item_url", "itemUrl", "detail_url", "share_url", "shareUrl"):
            value = str(item.get(key) or "").strip()
            if value:
                return value
        return ""

    def _commerce_enrichment_required(self, request: dict[str, Any], plan: IntentPlan) -> bool:
        if str(request.get("mode") or "") == "social_content":
            return True
        query = str(request.get("query") or "")
        return any(term in query for term in ("评价", "评论", "口碑", "对比", "区别"))

    def _commerce_provenance(
        self,
        response: Any,
        *,
        platform: str,
        mode: str,
    ) -> dict[str, Any]:
        payload = response if isinstance(response, dict) else {}
        return {
            "platform": platform,
            "mode": mode,
            "endpoint_id": str(payload.get("endpoint_id") or ""),
            "collected_at": str(payload.get("collected_at") or ""),
            "evidence_role": "external_observation",
        }

    def _external_response_meta(self, response: Any) -> dict[str, Any]:
        payload = response if isinstance(response, dict) else {}
        return {
            "available": bool(payload.get("available")),
            "operation": str(payload.get("operation") or ""),
            "platform": str(payload.get("platform") or ""),
            "endpoint_id": str(payload.get("endpoint_id") or ""),
            "collected_at": str(payload.get("collected_at") or ""),
            "latency_ms": payload.get("latency_ms"),
            "error": payload.get("error") if not payload.get("available") else None,
        }

    def _auxiliary_source_is_traceable(self, ref: EvidenceRef) -> bool:
        if ref.source_type == "local_product_catalog":
            return ref.verified and bool(ref.source_id)
        if ref.source_type == "commerce_mcp":
            return bool(
                ref.source_id
                and ref.provenance.get("platform")
                and ref.provenance.get("endpoint_id")
            )
        if ref.source_type == "web_search":
            return bool(ref.source_id and (ref.claim or ref.summary))
        if ref.source_type == "agent_derived":
            supporting = ref.provenance.get("supporting_evidence_ids")
            return bool(
                ref.provenance.get("source_agent")
                and isinstance(supporting, list)
                and supporting
            )
        return False

    async def _reason_over_knowledge(
        self,
        context: AgentExecutionContext,
        *,
        claims: list[dict[str, Any]],
        structured_terms: list[str],
        research_query: str,
        research_round: int,
        allow_followup: bool,
        search_available: bool,
    ) -> dict[str, Any]:
        grounded_fallback = []
        for term in list(dict.fromkeys(structured_terms))[:12]:
            indexes = [
                index
                for index, claim in enumerate(claims[:12])
                if term in f"{claim.get('title', '')} {claim.get('snippet', '')}"
            ]
            if indexes:
                grounded_fallback.append(
                    {
                        "concept": term,
                        "concept_type": self._fallback_concept_type(term),
                        "evidence_indexes": indexes[:3],
                        "source": "structured_search_response",
                    }
                )
        fallback = {
            "candidate_concepts": [item["concept"] for item in grounded_fallback],
            "grounded_concepts": grounded_fallback,
            "risks": [],
            "used_llm": False,
            "coverage": {
                "sufficient": False,
                "missing_aspects": ["coverage_assessment" if claims else "web_evidence"],
                "followup_query": "",
                "reason": (
                    "Coverage could not be assessed because the reasoning model is unavailable."
                    if claims
                    else "No traceable web evidence was returned."
                ),
            },
        }
        client = self.services.knowledge_llm
        if (
            client is None
            or not search_available
            or not getattr(client, "is_configured", lambda: False)()
        ):
            return fallback
        evidence = [
            {
                "index": index,
                "title": str(item.get("title") or "")[:240],
                "snippet": str(item.get("snippet") or "")[:700],
                "url": str(item.get("url") or "")[:500],
                "research_round": item.get("research_round", 1),
                "research_query": str(item.get("research_query") or "")[:500],
            }
            for index, item in enumerate(claims[:16])
        ]
        system_prompt = str(context.metadata.get("agent_system_prompt") or "") + (
            "\n## 本任务结构化约束\n"
            "你正在执行有界的 Plan-Execute-Reflection：根据累计搜索证据提取概念，并判断资料是否足以回答分配给你的知识目标。"
            "对每个概念必须分类为 product_type、ingredient、efficacy 或 other。"
            "只有可购买商品类别才标 product_type；症状、效果、成分、形容词、品牌营销词和治疗结论"
            "不能标 product_type。每个概念必须在引用证据正文中明确出现，并引用 evidence_indexes。"
            "coverage.sufficient 只能依据当前给定网页证据判断；不得用模型记忆补足证据。"
            "若 search_evidence 为空，concepts 必须为空且 coverage.sufficient 必须为 false，但仍可生成补搜词。"
            "若证据不足且允许补搜，missing_aspects 要指出缺口，followup_query 必须是一个更聚焦且不重复的搜索词；"
            "若证据已足够或不允许补搜，followup_query 必须为空字符串。"
            "输出 JSON：{\"concepts\":[{\"concept\":\"...\",\"concept_type\":\"product_type\",\"evidence_indexes\":[0]}],"
            "\"risks\":[\"...\"],\"coverage\":{\"sufficient\":false,\"missing_aspects\":[\"...\"],"
            "\"followup_query\":\"...\",\"reason\":\"...\"}}。"
        )
        user_prompt = json.dumps(
            {
                "assigned_research_goal": research_query,
                "original_user_query": context.query,
                "research_round": research_round,
                "allow_followup": allow_followup,
                "search_evidence": evidence,
                "structured_candidate_terms": structured_terms[:12],
            },
            ensure_ascii=False,
        )
        try:
            data = await generate_validated_json(
                client,
                system_prompt,
                user_prompt,
                validate=lambda value: self._validate_knowledge_reasoning(
                    value,
                    len(evidence),
                    allow_followup=allow_followup,
                ),
                error_message="ProductKnowledgeAgent returned invalid grounded concepts.",
                response_format={"type": "json_object"},
                operation=f"product_knowledge_agent.reflect_round_{research_round}",
            )
        except (StructuredLlmValidationError, RuntimeError):
            return fallback
        grounded = []
        for item in data.get("concepts", []):
            concept = str(item.get("concept") or "").strip()
            indexes = list(dict.fromkeys(int(index) for index in item.get("evidence_indexes", [])))
            grounded.append(
                {
                    "concept": concept,
                    "concept_type": str(item.get("concept_type") or "other"),
                    "evidence_indexes": indexes,
                    "evidence_urls": [evidence[index]["url"] for index in indexes if evidence[index]["url"]],
                    "source": "knowledge_research_llm",
                }
            )
        raw_coverage = dict(data.get("coverage") or {})
        sufficient = bool(raw_coverage.get("sufficient")) and bool(evidence)
        missing_aspects = list(
            dict.fromkeys(
                str(item).strip()
                for item in raw_coverage.get("missing_aspects", [])
                if str(item).strip()
            )
        )[:8]
        followup_query = ""
        if allow_followup and not sufficient:
            followup_query = " ".join(
                str(raw_coverage.get("followup_query") or "").split()
            )[:500]
        return {
            "candidate_concepts": [item["concept"] for item in grounded],
            "grounded_concepts": grounded,
            "risks": [str(item).strip() for item in data.get("risks", []) if str(item).strip()][:8],
            "used_llm": True,
            "coverage": {
                "sufficient": sufficient,
                "missing_aspects": missing_aspects,
                "followup_query": followup_query,
                "reason": str(raw_coverage.get("reason") or "").strip()[:500],
            },
        }

    def _validate_knowledge_reasoning(
        self,
        data: dict[str, Any],
        evidence_count: int,
        *,
        allow_followup: bool,
    ) -> list[str]:
        errors: list[str] = []
        concepts = data.get("concepts")
        if not isinstance(concepts, list):
            return ["concepts must be an array"]
        for index, item in enumerate(concepts):
            if not isinstance(item, dict):
                errors.append(f"concepts[{index}] must be an object")
                continue
            concept = str(item.get("concept") or "").strip()
            refs = item.get("evidence_indexes")
            concept_type = str(item.get("concept_type") or "")
            if not (2 <= len(concept) <= 24):
                errors.append(f"concepts[{index}].concept must contain 2..24 characters")
            if concept_type not in {"product_type", "ingredient", "efficacy", "other"}:
                errors.append(f"concepts[{index}].concept_type is invalid")
            if not isinstance(refs, list) or not refs:
                errors.append(f"concepts[{index}].evidence_indexes must be non-empty")
                continue
            if any(not isinstance(ref, int) or ref < 0 or ref >= evidence_count for ref in refs):
                errors.append(f"concepts[{index}] references unknown evidence index")
        if data.get("risks") is not None and not isinstance(data.get("risks"), list):
            errors.append("risks must be an array")
        coverage = data.get("coverage")
        if not isinstance(coverage, dict):
            errors.append("coverage must be an object")
            return errors
        sufficient = coverage.get("sufficient")
        missing_aspects = coverage.get("missing_aspects")
        followup_query = coverage.get("followup_query")
        reason = coverage.get("reason")
        if not isinstance(sufficient, bool):
            errors.append("coverage.sufficient must be a boolean")
        if not isinstance(missing_aspects, list) or any(
            not isinstance(item, str) for item in (missing_aspects or [])
        ):
            errors.append("coverage.missing_aspects must be an array of strings")
        if not isinstance(followup_query, str):
            errors.append("coverage.followup_query must be a string")
        if not isinstance(reason, str) or not reason.strip():
            errors.append("coverage.reason must be a non-empty string")
        if sufficient is False and allow_followup and (
            not isinstance(followup_query, str) or not followup_query.strip()
        ):
            errors.append("coverage.followup_query is required when evidence is insufficient")
        return errors

    def _fallback_concept_type(self, concept: str) -> str:
        product_suffixes = (
            "霜", "乳", "膏", "胶", "液", "油", "膜", "贴", "喷雾", "精华",
            "洁面", "洗面奶", "防晒", "面膜", "洗发水", "护发素", "沐浴露",
            "裤", "袜", "鞋", "衣", "包", "杯", "灯", "耳机", "手机", "电脑",
        )
        return "product_type" if any(concept.endswith(suffix) for suffix in product_suffixes) else "other"

    async def _reason_over_comparison(
        self,
        context: AgentExecutionContext,
        *,
        products: list[dict[str, Any]],
        external: Any,
        knowledge: dict[str, Any],
        upstream_evidence: list[EvidenceRef],
        fallback_dimensions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        fallback = {
            "dimensions": fallback_dimensions,
            "winner_by_goal": [],
            "unknowns": [],
            "used_llm": False,
            "reflection": {
                "performed": False,
                "status": "skipped",
                "changed": False,
                "issue_count": 0,
                "issues": [],
                "reason": "Comparison reasoning was not available.",
            },
        }
        client = self.services.comparison_llm
        if client is None or not products or not getattr(client, "is_configured", lambda: False)():
            return fallback
        valid_ids = {str(item.get("product_id") or "") for item in products}
        system_prompt = str(context.metadata.get("agent_system_prompt") or "") + (
            "\n## 本任务结构化约束\n"
            "只能比较 input_products 中的商品，不得发现或新增商品。外部平台信息只作为观察，"
            "网页知识只能来自上游 ProductKnowledgeAgent，二者都不得覆盖本地目录中的价格、规格或库存。"
            "不得自行搜索网页。输出 JSON："
            "{\"dimensions\":[{\"name\":\"...\",\"values\":[{\"product_id\":\"...\","
            "\"value\":\"...\",\"evidence\":\"local|external|unknown\"}]}],"
            "\"winner_by_goal\":[{\"goal\":\"...\",\"product_id\":\"...\",\"reason\":\"...\"}],"
            "\"unknowns\":[\"...\"]}。"
        )
        evidence_package = {
            "user_query": context.query,
            "input_products": products,
            "external_observations": external if self._commerce_results_available(external) else [],
            "knowledge_observations": knowledge,
            "upstream_evidence": [ref.compact() for ref in upstream_evidence[:30]],
        }
        user_prompt = json.dumps(
            evidence_package,
            ensure_ascii=False,
            default=str,
        )
        try:
            data = await generate_validated_json(
                client,
                system_prompt,
                user_prompt,
                validate=lambda value: self._validate_comparison_reasoning(value, valid_ids),
                error_message="ComparisonAgent returned invalid structured comparison.",
                response_format={"type": "json_object"},
                operation="comparison_agent.compare",
            )
        except (StructuredLlmValidationError, RuntimeError) as exc:
            return {
                **fallback,
                "reflection": {
                    **fallback["reflection"],
                    "reason": f"Initial comparison failed: {type(exc).__name__}",
                },
            }
        draft = {
            "dimensions": data.get("dimensions", []),
            "winner_by_goal": data.get("winner_by_goal", []),
            "unknowns": [str(item).strip() for item in data.get("unknowns", []) if str(item).strip()][:20],
        }
        reflection_prompt = str(context.metadata.get("agent_system_prompt") or "") + (
            "\n## 对比结果证据自检\n"
            "你现在只审核并修正 comparison_draft，不重新检索、不联网、不新增商品。"
            "逐项检查维度值、证据类型和 winner_by_goal 的理由是否能由 evidence_package 支撑。"
            "本地价格、规格和库存只能来自 input_products；平台观察和网页知识不能冒充本地商品事实。"
            "证据不足时应改为 unknown、写入 unknowns，或删除不成立的胜出结论。"
            "输出 JSON：{\"issues\":[{\"type\":\"unsupported_claim\",\"location\":\"...\",\"reason\":\"...\"}],"
            "\"changed\":true,\"reflection_reason\":\"...\",\"final_comparison\":{"
            "\"dimensions\":[{\"name\":\"...\",\"values\":[{\"product_id\":\"...\",\"value\":\"...\","
            "\"evidence\":\"local|external|unknown\"}]}],\"winner_by_goal\":[{\"goal\":\"...\","
            "\"product_id\":\"...\",\"reason\":\"...\"}],\"unknowns\":[\"...\"]}}。"
        )
        reflection_user_prompt = json.dumps(
            {
                "evidence_package": evidence_package,
                "comparison_draft": draft,
            },
            ensure_ascii=False,
            default=str,
        )
        try:
            reflected = await generate_validated_json(
                client,
                reflection_prompt,
                reflection_user_prompt,
                validate=lambda value: self._validate_comparison_reflection(value, valid_ids),
                error_message="ComparisonAgent returned invalid reflection output.",
                max_retries=0,
                response_format={"type": "json_object"},
                operation="comparison_agent.reflect",
            )
        except (StructuredLlmValidationError, RuntimeError) as exc:
            return {
                **draft,
                "used_llm": True,
                "reflection": {
                    "performed": True,
                    "status": "failed",
                    "changed": False,
                    "issue_count": 0,
                    "issues": [],
                    "reason": f"Reflection failed; retained the validated draft: {type(exc).__name__}",
                },
            }
        final_comparison = dict(reflected.get("final_comparison") or {})
        issues = [
            {
                "type": str(item.get("type") or "").strip()[:80],
                "location": str(item.get("location") or "").strip()[:160],
                "reason": str(item.get("reason") or "").strip()[:500],
            }
            for item in reflected.get("issues", [])[:20]
            if isinstance(item, dict)
        ]
        changed = final_comparison != draft
        return {
            "dimensions": final_comparison.get("dimensions", []),
            "winner_by_goal": final_comparison.get("winner_by_goal", []),
            "unknowns": [
                str(item).strip()
                for item in final_comparison.get("unknowns", [])
                if str(item).strip()
            ][:20],
            "used_llm": True,
            "reflection": {
                "performed": True,
                "status": "completed",
                "changed": changed,
                "reported_changed": bool(reflected.get("changed")),
                "issue_count": len(issues),
                "issues": issues,
                "reason": str(reflected.get("reflection_reason") or "").strip()[:500],
            },
        }

    def _validate_comparison_reasoning(
        self,
        data: dict[str, Any],
        valid_product_ids: set[str],
    ) -> list[str]:
        errors: list[str] = []
        dimensions = data.get("dimensions")
        winners = data.get("winner_by_goal")
        unknowns = data.get("unknowns")
        if not isinstance(dimensions, list):
            errors.append("dimensions must be an array")
            dimensions = []
        if not isinstance(winners, list):
            errors.append("winner_by_goal must be an array")
            winners = []
        if unknowns is not None and not isinstance(unknowns, list):
            errors.append("unknowns must be an array")
        referenced_ids: list[str] = []
        for dimension_index, dimension in enumerate(dimensions):
            if not isinstance(dimension, dict) or not str(dimension.get("name") or "").strip():
                errors.append("each dimension requires a name")
                continue
            values = dimension.get("values")
            if not isinstance(values, list):
                errors.append("each dimension requires values")
                continue
            for value_index, value in enumerate(values):
                if not isinstance(value, dict):
                    errors.append(f"dimensions[{dimension_index}].values[{value_index}] must be an object")
                    continue
                product_id = str(value.get("product_id") or "")
                referenced_ids.append(product_id)
                if "value" not in value:
                    errors.append(f"dimensions[{dimension_index}].values[{value_index}] requires value")
                if value.get("evidence") not in {"local", "external", "unknown"}:
                    errors.append(
                        f"dimensions[{dimension_index}].values[{value_index}].evidence is invalid"
                    )
        for winner_index, item in enumerate(winners):
            if not isinstance(item, dict):
                errors.append(f"winner_by_goal[{winner_index}] must be an object")
                continue
            referenced_ids.append(str(item.get("product_id") or ""))
            if not str(item.get("goal") or "").strip():
                errors.append(f"winner_by_goal[{winner_index}].goal is required")
            if not str(item.get("reason") or "").strip():
                errors.append(f"winner_by_goal[{winner_index}].reason is required")
        if isinstance(unknowns, list) and any(not isinstance(item, str) for item in unknowns):
            errors.append("unknowns must contain strings")
        invalid_ids = sorted({item for item in referenced_ids if item not in valid_product_ids})
        if invalid_ids:
            errors.append(f"comparison references unknown product IDs: {invalid_ids}")
        return errors

    def _validate_comparison_reflection(
        self,
        data: dict[str, Any],
        valid_product_ids: set[str],
    ) -> list[str]:
        errors: list[str] = []
        issues = data.get("issues")
        if not isinstance(issues, list):
            errors.append("issues must be an array")
        else:
            for index, item in enumerate(issues):
                if not isinstance(item, dict):
                    errors.append(f"issues[{index}] must be an object")
                    continue
                if not str(item.get("type") or "").strip():
                    errors.append(f"issues[{index}].type is required")
                if not str(item.get("reason") or "").strip():
                    errors.append(f"issues[{index}].reason is required")
        if not isinstance(data.get("changed"), bool):
            errors.append("changed must be a boolean")
        if not str(data.get("reflection_reason") or "").strip():
            errors.append("reflection_reason is required")
        final_comparison = data.get("final_comparison")
        if not isinstance(final_comparison, dict):
            errors.append("final_comparison must be an object")
        else:
            errors.extend(
                self._validate_comparison_reasoning(
                    final_comparison,
                    valid_product_ids,
                )
            )
        return errors

    def _compact_json(self, value: Any, limit: int) -> str:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            text = str(value)
        return text[:limit]

    def _image_attributes(self, value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        if hasattr(value, "model_dump"):
            return value.model_dump()
        return value if isinstance(value, dict) else None
