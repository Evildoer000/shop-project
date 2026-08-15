from __future__ import annotations

import asyncio
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
from app.schemas import IntentPlan, ProductCard, QueryPlan, ReflectionResult
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
        profile_memory = context.artifact("profile_memory")
        if profile_memory:
            planner_context["profile_memory"] = profile_memory
        plan = await planner.plan(context.query, planner_context)
        topology_locked = context.node.node_id == "system:intent_refinement"
        if topology_locked:
            plan = self._bounded_profile_refinement(self._intent_plan(context), plan)
        return AgentResult.success(
            {
                "intent_plan": plan.model_dump(),
                "summary": plan.summary,
                "topology_locked": topology_locked,
            },
            artifacts={"intent_plan": plan},
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
        proposal = getattr(context.intent_plan, "clarification", None)
        question = self._clarification_question(proposal, context.query)
        output = {
            "required": True,
            "question": question,
            "missing_fields": list(getattr(proposal, "missing_fields", []) or []),
            "reason": str(getattr(proposal, "reason", "") or "需求缺少可执行条件。"),
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
        query = query_override or context.query or intent_plan.original_query
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
        slots = [self._need_slot(item) for item in context.intent_plan.need_slots]
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
        requests = context.node.metadata.get("research_requests") or []
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
        product_ids = list(plan.referenced_product_ids)
        product_ids.extend(
            product_id
            for intent in plan.intents
            for product_id in intent.referenced_product_ids
        )
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
        upstream_evidence = self._dependency_evidence(context)
        web = None
        web_request = next(
            (
                request
                for request in context.node.metadata.get("research_requests", [])
                if request.get("mode") == "web_general"
            ),
            None,
        )
        if web_request is not None:
            web_query = str(web_request.get("query") or context.query)
            web = await context.execute_tool(
                "web_search",
                "search",
                lambda tool: tool.search(
                    query=web_query,
                    freshness=str(web_request.get("freshness") or "recent"),
                ),
                input_summary={"query_length": len(web_query), "freshness": "recent"},
            )
        dimensions = self._comparison_dimensions(details)
        referenced_ids = set(plan.referenced_product_ids)
        referenced_ids.update(
            product_id
            for intent in plan.intents
            for product_id in intent.referenced_product_ids
        )
        analysis = await self._reason_over_comparison(
            context,
            products=details,
            external=external,
            web=web,
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
                "external_available": self._commerce_results_available(external),
                "web_available": self._response_available(web),
            },
            evidence=[*product_evidence, *comparison_evidence],
            artifacts={
                "comparison": {
                    "products": details,
                    "external": external,
                    "web": web,
                    **analysis,
                }
            },
        )

    async def knowledge_research(self, context: AgentExecutionContext) -> AgentResult:
        requests = context.node.metadata.get("research_requests") or []
        responses: list[dict[str, Any]] = []
        claims: list[dict[str, Any]] = []
        expansions: list[str] = []
        local_evidence: list[EvidenceRef] = []
        referenced_product_ids = list(self._intent_plan(context).referenced_product_ids)
        if referenced_product_ids:
            products = await context.execute_tool(
                "product_detail",
                "get_by_ids",
                lambda tool: tool.get_by_ids(referenced_product_ids[:12]),
                input_summary={"product_count": min(len(referenced_product_ids), 12)},
            )
            for product in products:
                payload = self._product_payload(product)
                local_evidence.append(
                    EvidenceRef(
                        evidence_id=f"product:{product.product_id}:knowledge",
                        source_type="local_product_catalog",
                        source_id=product.product_id,
                        product_id=product.product_id,
                        claim=product.name,
                        summary=self._compact_json(payload, 600),
                        verified=True,
                        provenance={"evidence_role": "referenced_context"},
                    )
                )
        for request in requests:
            if request.get("mode") not in {None, "web_general"}:
                continue
            research_query = str(request.get("query") or context.query)
            freshness = str(request.get("freshness") or "recent")
            response = await context.execute_tool(
                "web_search",
                "search",
                lambda tool: tool.search(query=research_query, freshness=freshness),
                input_summary={"query_length": len(research_query), "freshness": freshness},
            )
            responses.append(response)
            for term in self._response_candidate_terms(response):
                if term not in expansions:
                    expansions.append(term)
            for item in self._items_from_response(response)[:10]:
                title = str(item.get("title") or item.get("name") or "").strip()
                snippet = str(item.get("snippet") or item.get("summary") or item.get("description") or "").strip()
                source_id = str(
                    item.get("url")
                    or item.get("source_id")
                    or item.get("id")
                    or ""
                ).strip()
                if (title or snippet) and source_id and self._response_available(response):
                    claims.append(
                        {
                            "title": title,
                            "snippet": snippet,
                            "url": item.get("url"),
                            "source_id": source_id,
                        }
                    )
        reasoning = await self._reason_over_knowledge(
            context,
            claims=claims,
            structured_terms=expansions,
        )
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
        available = bool(local_evidence or web_evidence)
        return AgentResult.success(
            {
                "available": available,
                "claim_count": len(claims),
                "candidate_concepts": expansions[:12],
                "concept_proposals": concept_proposals,
                "grounded_concepts": reasoning["grounded_concepts"],
                "risks": reasoning["risks"],
                "used_llm": reasoning["used_llm"],
                "status": "completed" if available else "unavailable",
            },
            evidence=[*local_evidence, *web_evidence, *knowledge_evidence],
            artifacts={
                "knowledge_research": {
                    "responses": responses,
                    "claims": claims,
                    "local_product_evidence_count": len(local_evidence),
                    "query_expansions": expansions[:12],
                    "candidate_concepts": expansions[:12],
                    "concept_proposals": concept_proposals,
                    "grounded_concepts": reasoning["grounded_concepts"],
                    "risks": reasoning["risks"],
                    "used_llm": reasoning["used_llm"],
                }
            },
            termination_reason="completed" if available else "web_search_unavailable",
        )

    async def evidence_verification(self, context: AgentExecutionContext) -> AgentResult:
        intent_plan = self._intent_plan(context)
        plan = self._query_plan(context)
        single = context.artifact("single_evidence") or context.artifact("image_evidence")
        state = context.artifact("multi_state")
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
                image_attributes=self._image_attributes(context.artifact("image_attributes") or context.image_attributes),
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
                    "verified_evidence": single,
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
        intent_plan = self._intent_plan(context)
        plan = self._query_plan(context)
        reflection = context.artifact("reflection") or ReflectionResult()
        single = context.artifact("verified_evidence") or context.artifact("single_evidence")
        state = context.artifact("verified_state") or context.artifact("multi_state")
        selection = context.artifact("selection")
        auxiliary_evidence = [
            item.compact()
            for item in context.artifact("verified_external_evidence", [])
            if isinstance(item, EvidenceRef)
        ]
        auxiliary_context = {"verified_external_evidence": auxiliary_evidence}
        route = self._route(context, reflection, state, selection)
        profile_narrative = str(context.metadata.get("profile_narrative") or "")
        if not profile_narrative:
            profile_items = context.artifact("profile_memory") or []
            profile_narrative = "；".join(
                f"{item.get('key')}:{item.get('value')}"
                for item in profile_items
                if isinstance(item, dict)
            )
        tokens: list[str] = []
        product_ids: list[str] = []
        cards: list[dict[str, Any]] = []
        if state is not None:
            if not isinstance(selection, MultiNeedSelection):
                selection = self._selection_from_reflection(state, reflection)
            product_ids = [candidate.product_id for candidate in selection.flat_candidates]
            cards = [generator.product_card(candidate.product, plan).model_dump() for candidate in selection.flat_candidates]
            if selection.flat_candidates and route not in {"no_product", "clarify", "direct_answer"}:
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
                    system_prompt_prefix=str(context.metadata.get("agent_system_prompt") or ""),
                )
            else:
                stream = generator.stream_direct_text(
                    context.query,
                    "clarification" if route == "clarify" else ("direct" if route == "direct_answer" else "no_product"),
                    reflection.reason or "没有足够可靠的商品证据。",
                    intent_plan,
                    profile_narrative=profile_narrative,
                    system_prompt_prefix=str(context.metadata.get("agent_system_prompt") or ""),
                )
        elif single is not None:
            passed = set(reflection.passed_product_ids)
            ranked = [(product, score) for product, score in single.ranked if product.product_id in passed]
            product_ids = [product.product_id for product, _ in ranked]
            cards = [generator.product_card(product, plan).model_dump() for product, _ in ranked]
            if ranked and route == "recommend":
                stream = generator.stream_text(
                    plan,
                    ranked,
                    profile_narrative=profile_narrative,
                    image_attributes=self._image_attributes(context.artifact("image_attributes") or context.image_attributes),
                    extra_context=auxiliary_context,
                    system_prompt_prefix=str(context.metadata.get("agent_system_prompt") or ""),
                )
            else:
                stream = generator.stream_direct_text(
                    context.query,
                    "clarification" if route == "clarify" else ("direct" if route == "direct_answer" else "no_product"),
                    reflection.reason or "没有足够可靠的商品证据。",
                    intent_plan,
                    profile_narrative=str(context.metadata.get("profile_narrative") or ""),
                    system_prompt_prefix=str(context.metadata.get("agent_system_prompt") or ""),
                )
        else:
            stream = generator.stream_direct_text(
                context.query,
                "clarification" if route == "clarify" else "direct",
                str(reflection.reason or intent_plan.plan_reason or "本轮使用知识或外部证据回答。"),
                intent_plan,
                extra_context={
                    **auxiliary_context,
                    "external_evidence_status": (
                        "verified" if auxiliary_evidence else "unavailable_or_empty"
                    ),
                },
                profile_narrative=profile_narrative,
                system_prompt_prefix=str(context.metadata.get("agent_system_prompt") or ""),
            )
        if cards:
            await context.emit_client_event({"type": "product_cards", "products": cards})
        async for token in stream:
            token = str(token)
            tokens.append(token)
            await context.emit_client_event({"type": "token", "content": token})
        answer_text = "".join(tokens)
        return AgentResult.success(
            {"route": route, "answer_length": len(answer_text), "product_ids": product_ids, "cards": cards},
            artifacts={"answer_text": answer_text, "answer_tokens": tokens, "answer_cards": cards, "answer_product_ids": product_ids, "answer_route": route},
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
        if isinstance(context.intent_plan, IntentPlan):
            return context.intent_plan
        return IntentPlan(original_query=context.query)

    def _bounded_profile_refinement(self, original: IntentPlan, refined: IntentPlan) -> IntentPlan:
        """Allow profile-informed query tuning without changing approved topology."""
        original_intent_ids = {item.intent_id for item in original.intents}
        refined_intents = [item for item in refined.intents if item.intent_id in original_intent_ids]
        return refined.model_copy(
            update={
                "primary_intent": original.primary_intent,
                "intents": refined_intents or original.intents,
                "execution_mode": original.execution_mode,
                "input_modalities": original.input_modalities,
                "constraints": original.constraints,
                "context_requests": original.context_requests,
                "clarification": original.clarification,
                "research_requests": original.research_requests,
                "agent_proposals": original.agent_proposals,
                "need_slots": original.need_slots,
                "referenced_product_ids": original.referenced_product_ids,
                "profile_lookup": original.profile_lookup,
                "plan_type": original.plan_type,
                "budget_min": original.budget_min,
                "budget_max": original.budget_max,
                "budget_scope": original.budget_scope,
            }
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
        for node in context.graph.nodes:
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
            "passed_product_ids": reflection.passed_product_ids,
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
        if context.intent_plan.plan_type == "clarify":
            return "clarify"
        if context.intent_plan.plan_type == "direct_answer":
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

    def _clarification_question(self, proposal: Any, query: str) -> str:
        missing = list(getattr(proposal, "missing_fields", []) or [])
        if missing:
            return f"为了更准确推荐，请先告诉我：{missing[0]}。"
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
                    source_id="knowledge_research_agent",
                    claim=f"候选商品概念：{value}",
                    summary="该概念由联网资料抽取，仍需 Supervisor 审批和本地商品检索校验。",
                    provenance={
                        "source_agent": "knowledge_research_agent",
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
                    "source": str(concept.get("source") or "knowledge_research_agent"),
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
        if any(intent.intent_type == "product_comparison" for intent in plan.intents):
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
        }
        client = self.services.knowledge_llm
        if client is None or not claims or not getattr(client, "is_configured", lambda: False)():
            return fallback
        evidence = [
            {
                "index": index,
                "title": str(item.get("title") or "")[:240],
                "snippet": str(item.get("snippet") or "")[:700],
                "url": str(item.get("url") or "")[:500],
            }
            for index, item in enumerate(claims[:12])
        ]
        system_prompt = str(context.metadata.get("agent_system_prompt") or "") + (
            "\n## 本任务结构化约束\n"
            "从给定搜索证据中识别可供本地商品检索使用的商品概念。"
            "对每个概念必须分类为 product_type、ingredient、efficacy 或 other。"
            "只有可购买商品类别才标 product_type；症状、效果、成分、形容词、品牌营销词和治疗结论"
            "不能标 product_type。每个概念必须在引用证据正文中明确出现，并引用 evidence_indexes。"
            "输出 JSON：{\"concepts\":[{\"concept\":\"...\",\"concept_type\":\"product_type\",\"evidence_indexes\":[0]}],"
            "\"risks\":[\"...\"]}。"
        )
        user_prompt = json.dumps(
            {
                "user_goal": context.query,
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
                validate=lambda value: self._validate_knowledge_reasoning(value, len(evidence)),
                error_message="KnowledgeResearchAgent returned invalid grounded concepts.",
                response_format={"type": "json_object"},
                operation="knowledge_research_agent.extract_grounded_concepts",
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
        return {
            "candidate_concepts": [item["concept"] for item in grounded],
            "grounded_concepts": grounded,
            "risks": [str(item).strip() for item in data.get("risks", []) if str(item).strip()][:8],
            "used_llm": True,
        }

    def _validate_knowledge_reasoning(self, data: dict[str, Any], evidence_count: int) -> list[str]:
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
        web: Any,
        fallback_dimensions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        fallback = {
            "dimensions": fallback_dimensions,
            "winner_by_goal": [],
            "unknowns": [],
            "used_llm": False,
        }
        client = self.services.comparison_llm
        if client is None or not products or not getattr(client, "is_configured", lambda: False)():
            return fallback
        valid_ids = {str(item.get("product_id") or "") for item in products}
        system_prompt = str(context.metadata.get("agent_system_prompt") or "") + (
            "\n## 本任务结构化约束\n"
            "只能比较 input_products 中的商品，不得发现或新增商品。外部平台信息只作为观察，"
            "不得覆盖本地目录中的价格、规格或库存。输出 JSON："
            "{\"dimensions\":[{\"name\":\"...\",\"values\":[{\"product_id\":\"...\","
            "\"value\":\"...\",\"evidence\":\"local|external|unknown\"}]}],"
            "\"winner_by_goal\":[{\"goal\":\"...\",\"product_id\":\"...\",\"reason\":\"...\"}],"
            "\"unknowns\":[\"...\"]}。"
        )
        user_prompt = json.dumps(
            {
                "user_query": context.query,
                "input_products": products,
                "external_observations": external if self._commerce_results_available(external) else [],
                "web_observations": web if self._response_available(web) else None,
            },
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
        except (StructuredLlmValidationError, RuntimeError):
            return fallback
        return {
            "dimensions": data.get("dimensions", []),
            "winner_by_goal": data.get("winner_by_goal", []),
            "unknowns": [str(item).strip() for item in data.get("unknowns", []) if str(item).strip()][:20],
            "used_llm": True,
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
        for dimension in dimensions:
            if not isinstance(dimension, dict) or not str(dimension.get("name") or "").strip():
                errors.append("each dimension requires a name")
                continue
            values = dimension.get("values")
            if not isinstance(values, list):
                errors.append("each dimension requires values")
                continue
            referenced_ids.extend(
                str(value.get("product_id") or "")
                for value in values
                if isinstance(value, dict)
            )
        referenced_ids.extend(
            str(item.get("product_id") or "")
            for item in winners
            if isinstance(item, dict)
        )
        invalid_ids = sorted({item for item in referenced_ids if item not in valid_product_ids})
        if invalid_ids:
            errors.append(f"comparison references unknown product IDs: {invalid_ids}")
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
