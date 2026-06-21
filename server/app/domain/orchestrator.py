from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy.orm import Session

from app.db.session import get_sessionmaker
from app.domain.answer_generator import AnswerGenerator
from app.domain.corrective_agent import CorrectiveAgentController
from app.domain.image_retrieval_worker import ImageRetrievalWorker
from app.domain.image_search_tool import ImageSearchTool
from app.domain.input_processor import InputProcessor
from app.domain.intent_planner import IntentPlanner
from app.domain.memory import ConversationContext, MemoryManager, SessionSummarizer
from app.domain.multi_need_retrieval_coordinator import MultiNeedRetrievalCoordinator
from app.domain.need_slot_schemas import MultiNeedSelection, MultiNeedState, NeedSlot, SlotCandidate
from app.domain.product_search_tool import ProductSearchTool
from app.domain.profile_lookup_tool import ProfileLookupTool
from app.domain.reranker import build_reranker
from app.domain.repair_worker import RepairAgent, RepairPlan
from app.domain.retrieval_worker import RetrievalWorker
from app.domain.retrieval_plan_builder import RetrievalPlanBuilder
from app.domain.single_retrieval_worker import SingleRetrievalEvidence, SingleRetrievalWorker
from app.domain.task_lifecycle import OrchestratorDecision, TurnTaskState
from app.domain.trajectory_logger import TrajectoryLogger
from app.harness import EvidenceBundle, EvidenceCandidate, EvidenceSlot, HarnessRuntime
from app.observability.langfuse_tracer import LangfuseTracer
from app.rag.llamaindex_milvus import LlamaIndexMilvusRetriever
from app.schemas import (
    ChatStreamRequest,
    DecisionTrace,
    ImageAttributes,
    IntentPlan,
    MULTI_NEED_PRODUCT_CARD_LIMIT,
    ProductCard,
    QueryPlan,
    ReflectionResult,
    RewriteNeedSlot,
    SINGLE_RECOMMENDATION_LIMIT,
    SINGLE_RETRIEVAL_REVIEW_LIMIT,
)
from app.services.image_attribute_extractor import ImageAttributeExtractor
from app.services.product_repository import ProductRepository
from app.services.structured_llm import StructuredLlmValidationError


logger = logging.getLogger(__name__)

CLIENT_TRACE_DROP_KEYS = {
    "original_query",
    "query",
    "query_understanding",
    "normalized_query",
    "message",
}


class EcommerceOrchestrator:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.input_processor = InputProcessor()
        self.intent_planner = IntentPlanner()
        self.image_attribute_extractor = ImageAttributeExtractor()
        self.corrective_agent = CorrectiveAgentController()
        self.retrieval_plan_builder = RetrievalPlanBuilder()
        self.memory_manager = MemoryManager(db)
        self.product_repository = ProductRepository(db)
        self.profile_lookup_tool = ProfileLookupTool(db)
        self.retriever = LlamaIndexMilvusRetriever()
        self.reranker = build_reranker()
        self.answer_generator = AnswerGenerator()
        self.product_search_tool = ProductSearchTool(self.product_repository, self.retriever, self.reranker)
        self.image_search_tool = ImageSearchTool(self.product_repository)
        self.single_retrieval_worker = SingleRetrievalWorker(self.product_search_tool)
        self.image_retrieval_worker = ImageRetrievalWorker(self.image_search_tool)
        self.repair_agent = RepairAgent()
        self.multi_need_coordinator = MultiNeedRetrievalCoordinator(self.product_search_tool)
        self.retrieval_worker = RetrievalWorker(
            product_search_tool=self.product_search_tool,
            single_retrieval_worker=self.single_retrieval_worker,
            multi_need_coordinator=self.multi_need_coordinator,
            image_retrieval_worker=self.image_retrieval_worker,
        )
        self.harness = HarnessRuntime.from_settings()
        self.budget_manager = self.harness.budget_manager
        self.trace_recorder = self.harness.trace_recorder
        self.tool_registry = self.harness.tool_registry
        self.evidence_cache = self.harness.evidence_cache
        self.tool_registry.register(
            "product_search",
            self.product_search_tool,
            description="文字商品检索原子能力",
        )
        self.tool_registry.register(
            "image_search",
            self.image_search_tool,
            description="单图商品相似度检索原子能力",
        )
        self.tool_registry.register(
            "profile_lookup",
            self.profile_lookup_tool,
            description="长期画像读取原子能力",
        )
        self.trajectory_logger = TrajectoryLogger()
        self.langfuse_tracer = LangfuseTracer()

    async def stream(self, request: ChatStreamRequest) -> AsyncGenerator[dict, None]:
        normalized_input = self.input_processor.normalize(request)
        query = normalized_input.text
        image_path = normalized_input.image_path
        image_attributes = ImageAttributes()
        image_attributes_task: asyncio.Task[ImageAttributes] | None = None
        image_fast_path = image_path is not None and not query
        task = self._new_task(request)
        answer_parts: list[str] = []
        product_ids: list[str] = []
        profile_narrative = ""
        trace_run = self.langfuse_tracer.start_run(
            "ecommerce_rag_chat",
            input_payload={
                "message": request.message,
                "normalized_query": query,
                "image_id": request.image_id,
                "image_path_resolved": bool(image_path),
            },
            user_id=request.user_id,
            session_id=request.session_id,
            metadata={"endpoint": "chat_stream"},
        )
        task.add_step("input", "succeeded", output_summary={"normalized_query": query, "image_path_resolved": bool(image_path)})
        await trace_run.span("input", output_payload={"normalized_query": query, "image_path_resolved": bool(image_path)})
        yield self._trace_event("input", "已标准化用户输入")
        await asyncio.sleep(0.01)

        conversation_context = self.memory_manager.build_context(request.user_id, request.session_id)
        profile_narrative = conversation_context.long_term_narrative
        await trace_run.span("context_assembler", output_payload=conversation_context.trace_payload())

        if request.image_id and image_path is None:
            async for event in self._stream_missing_image(
                request=request,
                query=query,
                task=task,
                trace_run=trace_run,
                profile_narrative=profile_narrative,
            ):
                yield event
            return

        if image_path is not None and image_fast_path:
            for chunk in self._visible_text_chunks("我先按图片相似度找候选，同时提取图片线索。"):
                yield self._agent_update(stage="planner", title="检索图片", content_delta=chunk, done=False)
            image_attributes_task = self._start_image_attribute_task(image_path, query, trace_run, task)
        elif image_path is not None:
            for chunk in self._visible_text_chunks("我先识别图片里的商品类型、颜色和风格。"):
                yield self._agent_update(stage="planner", title="理解图片", content_delta=chunk, done=False)
            image_attributes = await self._extract_image_attributes(image_path, query, trace_run, task)
            for chunk in self._visible_text_chunks(self._image_attribute_update_text(image_attributes)):
                yield self._agent_update(stage="planner", title="理解图片", content_delta=chunk, done=False)
            yield self._trace_event(
                "image_attribute_extraction",
                "ImageAttributeExtractor 已完成图片属性理解",
                image_attributes=self._image_attribute_trace_payload(image_attributes),
            )

        if image_fast_path:
            intent_plan = self._image_only_intent_plan(None)
        else:
            self.budget_manager.record_planner_call(task)
            try:
                intent_plan = None
                planner_context = self._planner_context(
                    conversation_context,
                    request.session_id,
                    include_long_term=False,
                    image_attributes=image_attributes if image_path is not None else None,
                )
                yield self._agent_update(
                    stage="planner",
                    title="理解需求",
                    content_delta="正在理解你的需求和约束。",
                    done=False,
                )
                async for planner_event in self.intent_planner.stream_plan_with_summary(query, planner_context):
                    if getattr(planner_event, "kind", "") == "summary_delta":
                        content = str(getattr(planner_event, "content", "") or "")
                        if content:
                            yield self._agent_update(
                                stage="planner",
                                title="理解需求",
                                content_delta=content,
                                done=False,
                            )
                    elif getattr(planner_event, "kind", "") == "plan":
                        intent_plan = getattr(planner_event, "intent_plan", None)
                yield self._agent_update(
                    stage="planner",
                    title="理解需求",
                    content_delta="需求理解完成，准备进入检索。",
                    done=True,
                )
                if intent_plan is None:
                    raise StructuredLlmValidationError(
                        "IntentPlanner streaming did not return a plan.",
                        errors=["missing IntentPlan after tagged stream"],
                        data=None,
                        content="",
                    )
            except StructuredLlmValidationError as exc:
                async for event in self._stream_planner_failure(
                    request=request,
                    query=query,
                    task=task,
                    trace_run=trace_run,
                    error=exc,
                ):
                    yield event
                return
        self._update_planner_proposal(task, intent_plan)
        task.add_step("intent_planning", "succeeded", output_summary=intent_plan.model_dump())
        self.decide_intent_plan(task, intent_plan)
        await trace_run.span(
            "intent_planning",
            input_payload={"query": query, "context": conversation_context.trace_payload()},
            output_payload=intent_plan.model_dump(),
        )
        yield self._trace_event(
            "intent_planning",
            "Orchestrator 已构造图片快路径计划" if image_fast_path else "IntentPlanner 已输出最小声明式计划",
            intent_plan=intent_plan.model_dump(),
        )
        await asyncio.sleep(0.01)

        if image_path is not None:
            image_path_decision = self.decide_image_retrieval_path(task, intent_plan, image_path_resolved=True)
            if image_path_decision.approved:
                intent_plan = intent_plan.model_copy(
                    update={
                        "plan_type": "image_retrieval",
                        "plan_reason": image_path_decision.reason,
                    }
                )
                self._update_planner_proposal(task, intent_plan)

        if intent_plan.profile_lookup.requested:
            decision = self.decide_profile_lookup(task, intent_plan)
            profile_memory = self.profile_lookup_tool.lookup(
                request.user_id,
                intent_plan.profile_lookup.query or query,
            )
            self.budget_manager.record_tool_call(task)
            await trace_run.span(
                "profile_lookup",
                input_payload={
                    "query": intent_plan.profile_lookup.query or query,
                    "reason": intent_plan.profile_lookup.reason,
                    "approved": decision.approved,
                },
                output_payload={"profile_memory": profile_memory},
            )
            yield self._trace_event(
                "profile_lookup",
                "Orchestrator 已批准读取画像作为软偏好",
                profile_lookup={
                    "requested": True,
                    "found": bool(profile_memory),
                    "query": intent_plan.profile_lookup.query,
                    "reason": intent_plan.profile_lookup.reason,
                },
            )
            profile_narrative = self._merge_profile_narrative(profile_narrative, profile_memory)
            if profile_memory and self.budget_manager.can_call_planner(task):
                self.budget_manager.record_planner_call(task)
                intent_plan = await self.intent_planner.plan(
                    query,
                    self._planner_context(
                        conversation_context,
                        request.session_id,
                        include_long_term=False,
                        profile_memory=profile_memory,
                        image_attributes=image_attributes if image_path is not None else None,
                    ),
                )
                self._update_planner_proposal(task, intent_plan)
                await trace_run.span(
                    "intent_planning_profile_refine",
                    input_payload={"query": query, "profile_memory": profile_memory},
                    output_payload=intent_plan.model_dump(),
                )
                if image_path is not None:
                    image_path_decision = self.decide_image_retrieval_path(task, intent_plan, image_path_resolved=True)
                    if image_path_decision.approved:
                        intent_plan = intent_plan.model_copy(
                            update={
                                "plan_type": "image_retrieval",
                                "plan_reason": image_path_decision.reason,
                            }
                        )
                        self._update_planner_proposal(task, intent_plan)

        if intent_plan.referenced_product_ids:
            referenced_products = self.product_repository.get_by_ids(intent_plan.referenced_product_ids)
            self.budget_manager.record_tool_call(task)
            previous_decision = self.decide_previous_evidence_answer(
                task,
                intent_plan,
                conversation_context,
                referenced_products,
            )
            if previous_decision.approved:
                loaded_ids = [product.product_id for product in referenced_products]
                reason = previous_decision.reason
                self.budget_manager.record_answer_call(task)
                self.decide_execution_path(task, intent_plan)
                trace = DecisionTrace(
                    query_understanding=intent_plan.model_dump(),
                    image_attributes=self._image_attributes_payload(image_attributes),
                    retrieval_summary={
                        "route": "direct_answer",
                        "answer_mode": "context_evidence",
                        "reason": reason,
                        "referenced_product_ids": loaded_ids,
                        "loaded_product_ids": loaded_ids,
                    },
                    route="direct_answer",
                    failure_stage="none",
                    candidate_counts={},
                    stages=[
                        self._stage("input", "passed", "已接收并标准化用户输入。"),
                        self._stage("intent_planning", "passed", "IntentPlanner 已输出上下文商品引用。"),
                        self._stage("context_evidence_answer", "stopped", reason, product_ids=loaded_ids),
                    ],
                    final_reason=reason,
                )
                self._finish_trace(trace, task, route="direct_answer")
                yield self._decision_trace_event(trace)
                context_cards = [
                    self.answer_generator.product_card(product, QueryPlan())
                    for product in referenced_products
                ]
                if context_cards:
                    yield {"type": "product_cards", "products": [card.model_dump() for card in context_cards]}
                async for token in self.answer_generator.stream_direct_text(
                    query,
                    "direct",
                    reason,
                    intent_plan,
                    extra_context={
                        "referenced_products": [self._product_detail_for_answer(product) for product in referenced_products],
                        "conversation_evidence": previous_decision.decision_summary.get("conversation_evidence", {}),
                    },
                    profile_narrative=profile_narrative,
                ):
                    answer_parts.append(token)
                    yield {"type": "token", "content": token}
                product_ids = loaded_ids
                self._schedule_memory_update(
                    request=request,
                    query=query,
                    answer_text="".join(answer_parts),
                    route="direct_answer",
                    product_ids=product_ids,
                    intent_plan=intent_plan,
                    decision_trace=trace.model_dump(),
                )
                await trace_run.end(
                    output_payload=self._trace_output(
                        route="direct_answer",
                        reason=reason,
                        products=self._products_brief_from_products(referenced_products),
                    ),
                    metadata=self._trace_metadata(),
                )
                yield {"type": "done"}
                return

        self.decide_execution_path(task, intent_plan)
        if intent_plan.plan_type in {"direct_answer", "clarify"}:
            mode = "clarification" if intent_plan.plan_type == "clarify" else "direct"
            route = "clarify" if intent_plan.plan_type == "clarify" else "direct_answer"
            reason = intent_plan.plan_reason or ("需要先澄清商品需求。" if route == "clarify" else "本轮不需要检索商品库。")
            self.budget_manager.record_answer_call(task)
            trace = DecisionTrace(
                query_understanding=intent_plan.model_dump(),
                image_attributes=self._image_attributes_payload(image_attributes),
                retrieval_summary={"route": route, "answer_mode": mode, "reason": reason},
                route=route,
                failure_stage="none",
                candidate_counts={},
                stages=[
                    self._stage("input", "passed", "已接收并标准化用户输入。"),
                    self._stage("intent_planning", "stopped", reason),
                ],
                final_reason=reason,
            )
            self._finish_trace(trace, task, route=route)
            yield self._decision_trace_event(trace)
            async for token in self.answer_generator.stream_direct_text(
                query,
                mode,
                reason,
                intent_plan,
                profile_narrative=profile_narrative,
            ):
                answer_parts.append(token)
                yield {"type": "token", "content": token}
            self._schedule_memory_update(
                request=request,
                query=query,
                answer_text="".join(answer_parts),
                route=route,
                product_ids=[],
                intent_plan=intent_plan,
                decision_trace=trace.model_dump(),
            )
            await trace_run.end(
                output_payload=self._trace_output(route=route, reason=reason),
                metadata=self._trace_metadata(),
            )
            yield {"type": "done"}
            return

        plan = self._retrieval_plan_builder().plan(intent_plan)
        if intent_plan.plan_type == "image_retrieval" and image_path is not None:
            async for event in self._stream_image_retrieval(
                request=request,
                query=query or "用户上传图片找相似商品",
                intent_plan=intent_plan,
                plan=plan,
                image_path=image_path,
                image_attributes=image_attributes,
                image_attributes_task=image_attributes_task,
                image_fast_path=image_fast_path,
                task=task,
                trace_run=trace_run,
                profile_narrative=profile_narrative,
            ):
                yield event
            return

        if intent_plan.plan_type == "multi_retrieval":
            slots = self._need_slots_from_intent_plan(intent_plan, plan)
            if len(slots) >= 2:
                async for event in self._stream_multi_need(
                    request=request,
                    query=query,
                    intent_plan=intent_plan,
                    plan=plan,
                    slots=slots,
                    task=task,
                    trace_run=trace_run,
                    profile_narrative=profile_narrative,
                ):
                    yield event
                return

        async for event in self._stream_single_retrieval(
            request=request,
            query=query,
            intent_plan=intent_plan,
            plan=plan,
            task=task,
            trace_run=trace_run,
            profile_narrative=profile_narrative,
        ):
            yield event

    async def _stream_single_retrieval(
        self,
        *,
        request: ChatStreamRequest,
        query: str,
        intent_plan: IntentPlan,
        plan: QueryPlan,
        task: TurnTaskState,
        trace_run: Any,
        profile_narrative: str,
    ) -> AsyncGenerator[dict, None]:
        answer_parts: list[str] = []
        yield self._agent_update(
            stage="retrieval",
            title="检索商品",
            content_delta="正在从商品库召回候选商品。",
            done=False,
        )
        await asyncio.sleep(0.01)
        evidence = self.retrieval_worker.run_single_initial(query, intent_plan, plan)
        self.budget_manager.record_tool_call(task, evidence.tool_call_count)
        yield self._trace_event(
            "single_retrieval_worker_execution",
            "RetrievalWorker 已产出单检索初次候选证据",
            single_retrieval=evidence.summary(intent_plan, plan),
        )
        await trace_run.span("single_retrieval_worker_execution", output_payload=evidence.summary(intent_plan, plan))
        yield self._agent_update(
            stage="retrieval",
            title="检索商品",
            content_delta=f"已召回 {evidence.after_rerank} 款候选，开始校验证据。",
            done=True,
        )

        self.budget_manager.record_corrective_call(task)
        yield self._agent_update(
            stage="corrective",
            title="校验证据",
            content_delta="正在核对候选是否符合需求、预算和排除项。",
            done=False,
        )
        await asyncio.sleep(0.01)
        reflection = await self.corrective_agent.review(
            query,
            intent_plan,
            plan,
            evidence.ranked,
            evidence.vector_scores,
            evidence.keyword_scores,
        )
        yield self._trace_event(
            "corrective_reflection",
            "CorrectiveAgent 已完成单检索证据反射",
            reflection_result=reflection.model_dump(),
        )
        yield self._corrective_update_event(reflection)

        trigger = self._repair_trigger(evidence.failure_trigger, reflection, default="corrective_rejected_candidates")
        repair_plans: list[RepairPlan] = []
        repair_evidences: list[SingleRetrievalEvidence] = []
        while self._should_consider_repair(reflection, trigger):
            repair_decision = self.decide_repair(task, trigger)
            if not repair_decision.approved:
                break
            self.budget_manager.record_repair_attempt(task)
            repair_plan = await self._run_repair_agent(
                original_query=query,
                intent_plan=intent_plan,
                plan=plan,
                slots=[self._single_repair_slot(query, intent_plan, plan)],
                trigger=trigger or "corrective_rejected_candidates",
                reflection_result=reflection,
                previous_candidates=self._single_previous_candidates(evidence.ranked),
            )
            if not repair_plan or not any(repair_plan.queries_by_slot.values()):
                break
            repair_plans.append(repair_plan)
            yield self._trace_event(
                "repair_plan_generated",
                "RepairAgent 已生成被批准的 RepairPlan",
                repair_plan=repair_plan.summary(),
            )
            repair_evidence = self.retrieval_worker.run_single_repair(query, intent_plan, plan, repair_plan)
            repair_evidences.append(repair_evidence)
            self.budget_manager.record_tool_call(task, repair_evidence.tool_call_count)
            yield self._trace_event(
                "repair_search_executed",
                "RetrievalWorker 已按 RepairPlan 执行修复检索",
                repair_retrieval=repair_evidence.summary(intent_plan, plan),
            )
            merged_ranked = self._merge_ranked_evidence(evidence.ranked, repair_evidence.ranked)
            merged_vector_scores = self._merge_score_maps(evidence.vector_scores, repair_evidence.vector_scores)
            merged_keyword_scores = self._merge_score_maps(evidence.keyword_scores, repair_evidence.keyword_scores)
            self.budget_manager.record_corrective_call(task)
            reflection = await self.corrective_agent.review(
                query,
                intent_plan,
                plan,
                merged_ranked,
                merged_vector_scores,
                merged_keyword_scores,
            )
            evidence.ranked = merged_ranked
            evidence.vector_scores = merged_vector_scores
            evidence.keyword_scores = merged_keyword_scores
            trigger = self._repair_trigger("", reflection, default="corrective_rejected_candidates")
            yield self._trace_event(
                "corrective_reflection",
                "CorrectiveAgent 已完成 repair 后统一反射",
                reflection_result=reflection.model_dump(),
            )
            yield self._corrective_update_event(reflection)

        final_route = self.decide_final_route_from_reflection(task, reflection, task.execution_path or "single_retrieval").selected
        passed_ids = set(reflection.passed_product_ids)
        final_ranked = [
            (product, score)
            for product, score in evidence.ranked
            if product.product_id in passed_ids
        ][:SINGLE_RECOMMENDATION_LIMIT]
        if not final_ranked and final_route == "recommend":
            final_ranked = evidence.ranked[:SINGLE_RECOMMENDATION_LIMIT]
        product_ids = [product.product_id for product, _ in final_ranked]
        candidate_counts = evidence.counts(after_corrective=len(product_ids))
        retrieval_summary = {
            **evidence.summary(intent_plan, plan),
            "route": final_route,
            "passed_product_ids": reflection.passed_product_ids,
            "rejected_products": reflection.rejected_products,
            "fallback_plan": reflection.fallback_plan,
            "reflection_result": reflection.model_dump(),
        }
        if repair_plans:
            retrieval_summary["repair_plans"] = [repair_plan.summary() for repair_plan in repair_plans]
        if repair_evidences:
            retrieval_summary["repair_retrievals"] = [
                repair_evidence.summary(intent_plan, plan) for repair_evidence in repair_evidences
            ]
        trace = DecisionTrace(
            query_understanding=intent_plan.model_dump(),
            filters=plan.filters,
            retrieval_summary=retrieval_summary,
            route=final_route,
            failure_stage="none" if final_route in {"recommend", "direct_answer", "clarify"} else "corrective_reflection",
            candidate_counts=candidate_counts,
            stages=[
                self._stage("input", "passed", "已接收并标准化用户输入。"),
                self._stage("intent_planning", "passed", "IntentPlanner 已输出检索计划。"),
                self._stage("single_retrieval_worker_execution", "passed", "SingleRetrievalWorker 已完成证据获取。"),
                self._stage("corrective_reflection", "passed", reflection.reason, product_ids=product_ids),
            ],
            final_reason=reflection.reason,
        )
        self._finish_trace(trace, task, route=final_route)
        yield self._decision_trace_event(trace)

        if final_route == "recommend" and final_ranked:
            cards = [self.answer_generator.product_card(product, plan) for product, _ in final_ranked]
            yield {"type": "product_cards", "products": [card.model_dump() for card in cards]}
            await asyncio.sleep(0.01)
            yield self._agent_update(
                stage="answer",
                title="生成回答",
                content_delta="商品卡片已准备好，正在整理推荐解释。",
                done=False,
            )
            self.budget_manager.record_answer_call(task)
            async for token in self.answer_generator.stream_text(plan, final_ranked, profile_narrative=profile_narrative):
                answer_parts.append(token)
                yield {"type": "token", "content": token}
        else:
            self.budget_manager.record_answer_call(task)
            mode = "clarification" if final_route == "clarify" else ("direct" if final_route == "direct_answer" else "no_product")
            yield self._agent_update(
                stage="answer",
                title="生成回答",
                content_delta="正在整理回复。",
                done=False,
            )
            async for token in self.answer_generator.stream_direct_text(
                query,
                mode,
                reflection.reason or "没有足够匹配的商品证据。",
                intent_plan,
                extra_context=self._single_near_miss_context(evidence.ranked, reflection),
                profile_narrative=profile_narrative,
            ):
                answer_parts.append(token)
                yield {"type": "token", "content": token}

        self._schedule_memory_update(
            request=request,
            query=query,
            answer_text="".join(answer_parts),
            route=final_route,
            product_ids=product_ids,
            intent_plan=intent_plan,
            decision_trace=trace.model_dump(),
            evidence_bundle=self._single_evidence_bundle(
                request=request,
                task=task,
                query=query,
                execution_path="single_retrieval",
                final_route=final_route,
                ranked=evidence.ranked,
                displayed_ranked=final_ranked,
                reflection=reflection,
                trace=trace,
            ),
        )
        await trace_run.end(
            output_payload=self._trace_output(
                route=final_route,
                reason=trace.final_reason,
                products=self._products_brief_from_ranked(final_ranked),
            ),
            metadata=self._trace_metadata(),
        )
        yield {"type": "done"}

    async def _stream_image_retrieval(
        self,
        *,
        request: ChatStreamRequest,
        query: str,
        intent_plan: IntentPlan,
        plan: QueryPlan,
        image_path: Any,
        image_attributes: ImageAttributes | None,
        image_attributes_task: asyncio.Task[ImageAttributes] | None,
        image_fast_path: bool,
        task: TurnTaskState,
        trace_run: Any,
        profile_narrative: str,
    ) -> AsyncGenerator[dict, None]:
        answer_parts: list[str] = []
        yield self._agent_update(
            stage="retrieval",
            title="检索商品",
            content_delta="正在用图片相似度召回候选商品。",
            done=False,
        )
        await asyncio.sleep(0.01)
        evidence = self.retrieval_worker.run_image_initial(
            original_query=query,
            intent_plan=intent_plan,
            plan=plan,
            image_path=image_path,
        )
        self.budget_manager.record_tool_call(task, evidence.tool_call_count)
        yield self._trace_event(
            "image_retrieval_worker_execution",
            "ImageRetrievalWorker 已基于单图产出候选证据",
            image_retrieval=evidence.summary(intent_plan, plan),
        )
        await trace_run.span("image_retrieval_worker_execution", output_payload=evidence.summary(intent_plan, plan))
        self.decide_image_relevance(task, evidence)
        yield self._agent_update(
            stage="retrieval",
            title="检索商品",
            content_delta=f"已找到 {evidence.after_rerank} 款图片相似候选，开始校验证据。",
            done=True,
        )
        if image_attributes_task is not None:
            yield self._agent_update(
                stage="planner",
                title="理解图片",
                content_delta="图片候选已召回，等待图片语义用于证据校验。",
                done=False,
            )
            image_attributes = await self._image_attributes_from_task_or_placeholder(
                image_attributes_task,
                wait_seconds=self._image_attribute_corrective_wait_seconds(),
            )
            yield self._trace_event(
                "image_attribute_extraction",
                (
                    "ImageAttributeExtractor 已完成图片属性理解"
                    if image_attributes.available
                    else "图片快路径未等待完整图片属性理解"
                ),
                image_attributes=self._image_attribute_trace_payload(image_attributes),
            )
            if image_attributes.available:
                for chunk in self._visible_text_chunks(self._image_attribute_update_text(image_attributes)):
                    yield self._agent_update(stage="planner", title="理解图片", content_delta=chunk, done=True)
            else:
                yield self._agent_update(
                    stage="planner",
                    title="检索图片",
                    content_delta="图片语义未及时返回，继续用图片相似证据校验。",
                    done=True,
                )

        self.budget_manager.record_corrective_call(task)
        yield self._agent_update(
            stage="corrective",
            title="校验证据",
            content_delta="正在核对图片相似候选是否适合推荐。",
            done=False,
        )
        await asyncio.sleep(0.01)
        reflection = await self.corrective_agent.review(
            query,
            intent_plan,
            plan,
            evidence.ranked,
            evidence.vector_scores,
            evidence.keyword_scores,
            image_attributes=self._image_attributes_for_prompt(image_attributes),
        )
        yield self._trace_event(
            "corrective_reflection",
            "CorrectiveAgent 已完成图片候选证据反射",
            reflection_result=reflection.model_dump(),
        )
        yield self._corrective_update_event(reflection)
        final_route = self.decide_final_route_from_reflection(task, reflection, "image_retrieval").selected
        passed_ids = set(reflection.passed_product_ids)
        final_ranked = [
            (product, score)
            for product, score in evidence.ranked
            if product.product_id in passed_ids
        ][:SINGLE_RECOMMENDATION_LIMIT]
        if not final_ranked and final_route == "recommend":
            final_ranked = evidence.ranked[:SINGLE_RECOMMENDATION_LIMIT]
        product_ids = [product.product_id for product, _ in final_ranked]
        trace = DecisionTrace(
            query_understanding=intent_plan.model_dump(),
            image_attributes=self._image_attributes_payload(image_attributes),
            filters=plan.filters,
            retrieval_summary={
                **evidence.summary(intent_plan, plan),
                "image_attributes": self._image_attributes_payload(image_attributes),
                "route": final_route,
                "answer_mode": "image_retrieval",
                "passed_product_ids": reflection.passed_product_ids,
                "rejected_products": reflection.rejected_products,
                "fallback_plan": reflection.fallback_plan,
                "reflection_result": reflection.model_dump(),
            },
            route=final_route,
            failure_stage="none" if final_route == "recommend" else "corrective_reflection",
            candidate_counts=evidence.counts(after_corrective=len(product_ids)),
            stages=[
                self._stage("input", "passed", "已接收并标准化用户输入。", image_path_resolved=True),
                self._stage(
                    "intent_planning",
                    "passed",
                    "Orchestrator 已构造图片快路径计划。"
                    if image_fast_path
                    else "IntentPlanner proposal 已进入图片检索裁决。",
                ),
                self._stage("image_retrieval_worker_execution", "passed", "ImageRetrievalWorker 已完成单图证据获取。"),
                self._stage("corrective_reflection", "passed", reflection.reason, product_ids=product_ids),
            ],
            final_reason=reflection.reason,
        )
        self._finish_trace(trace, task, route=final_route)
        yield self._decision_trace_event(trace)
        if final_route == "recommend" and final_ranked:
            cards = [self.answer_generator.product_card(product, plan) for product, _ in final_ranked]
            yield {"type": "product_cards", "products": [card.model_dump() for card in cards]}
            await asyncio.sleep(0.01)
            yield self._agent_update(
                stage="answer",
                title="生成回答",
                content_delta="相似商品卡片已准备好，正在整理解释。",
                done=False,
            )
            self.budget_manager.record_answer_call(task)
            async for token in self.answer_generator.stream_text(
                plan,
                final_ranked,
                profile_narrative=profile_narrative,
                image_attributes=self._image_attributes_for_prompt(image_attributes),
            ):
                answer_parts.append(token)
                yield {"type": "token", "content": token}
        else:
            self.budget_manager.record_answer_call(task)
            mode = "clarification" if final_route == "clarify" else ("direct" if final_route == "direct_answer" else "no_product")
            yield self._agent_update(
                stage="answer",
                title="生成回答",
                content_delta="正在整理回复。",
                done=False,
            )
            async for token in self.answer_generator.stream_direct_text(
                query,
                mode,
                reflection.reason or "图片检索没有得到足够可靠的商品证据。",
                intent_plan,
                extra_context={
                    **self._single_near_miss_context(evidence.ranked, reflection),
                    "image_attributes": self._image_attributes_for_prompt(image_attributes),
                },
                profile_narrative=profile_narrative,
            ):
                answer_parts.append(token)
                yield {"type": "token", "content": token}

        self._schedule_memory_update(
            request=request,
            query=query,
            answer_text="".join(answer_parts),
            route=final_route,
            product_ids=product_ids,
            intent_plan=intent_plan,
            decision_trace=trace.model_dump(),
            evidence_bundle=self._single_evidence_bundle(
                request=request,
                task=task,
                query=query,
                execution_path="image_retrieval",
                final_route=final_route,
                ranked=evidence.ranked,
                displayed_ranked=final_ranked,
                reflection=reflection,
                trace=trace,
            ),
        )
        await trace_run.end(
            output_payload=self._trace_output(
                route=final_route,
                reason=trace.final_reason,
                products=self._products_brief_from_ranked(final_ranked),
            ),
            metadata=self._trace_metadata(),
        )
        yield {"type": "done"}

    async def _stream_multi_need(
        self,
        *,
        request: ChatStreamRequest,
        query: str,
        intent_plan: IntentPlan,
        plan: QueryPlan,
        slots: list[NeedSlot],
        task: TurnTaskState,
        trace_run: Any,
        profile_narrative: str = "",
    ) -> AsyncGenerator[dict, None]:
        answer_parts: list[str] = []
        yield self._agent_update(
            stage="retrieval",
            title="检索商品",
            content_delta=f"正在为 {len(slots)} 个需求分别召回候选。",
            done=False,
        )
        await asyncio.sleep(0.01)
        state = await self.retrieval_worker.run_multi_initial(query, intent_plan, plan, slots)
        self.budget_manager.record_tool_call(task, int(state.budgets.get("search_calls", 0)))
        await trace_run.span("multi_need_retrieval", output_payload=self._multi_need_trace(state))
        yield self._trace_event(
            "multi_need_retrieval",
            "RetrievalWorker 已完成多需求初次检索",
            multi_need=self._multi_need_trace(state),
        )
        yield self._agent_update(
            stage="retrieval",
            title="检索商品",
            content_delta="多需求候选已召回，开始校验每个需求是否被覆盖。",
            done=True,
        )

        self.budget_manager.record_corrective_call(task)
        yield self._agent_update(
            stage="corrective",
            title="校验证据",
            content_delta="正在逐项核对候选商品和需求槽位。",
            done=False,
        )
        await asyncio.sleep(0.01)
        reflection = await self.corrective_agent.review_slots(query, intent_plan, plan, state)
        await trace_run.span("corrective_reflection", output_payload=reflection.model_dump())
        yield self._trace_event(
            "corrective_reflection",
            "CorrectiveAgent 已完成多需求证据反射",
            reflection_result=reflection.model_dump(),
        )
        yield self._corrective_update_event(reflection)

        repair_plans: list[RepairPlan] = []
        trigger = self._repair_trigger("", reflection, default="slot_rejected")
        while self._should_consider_repair(reflection, trigger):
            repair_decision = self.decide_repair(task, trigger)
            if not repair_decision.approved:
                break
            self.budget_manager.record_repair_attempt(task)
            repair_plan = await self._run_repair_agent(
                original_query=query,
                intent_plan=intent_plan,
                plan=plan,
                slots=state.slots,
                trigger=trigger,
                reflection_result=reflection,
                previous_candidates=self._multi_previous_candidates(state),
            )
            if not repair_plan or not any(repair_plan.queries_by_slot.values()):
                break
            repair_plans.append(repair_plan)
            state.budgets.setdefault("internal_actions", []).append(
                {"action": "repair_plan_generated", **repair_plan.summary()}
            )
            yield self._trace_event(
                "repair_plan_generated",
                "RepairAgent 已为多需求失败 slot 生成 RepairPlan",
                repair_plan=repair_plan.summary(),
            )
            state = self.retrieval_worker.run_multi_repair(state, repair_plan)
            self.budget_manager.record_tool_call(task, self._repair_plan_query_count(repair_plan))
            yield self._trace_event(
                "repair_search_executed",
                "RetrievalWorker 已按 RepairPlan 执行多需求修复检索并合并证据",
                multi_need=self._multi_need_trace(state),
            )
            self.budget_manager.record_corrective_call(task)
            reflection = await self.corrective_agent.review_slots(query, intent_plan, plan, state)
            trigger = self._repair_trigger("", reflection, default="slot_rejected")
            await trace_run.span("corrective_reflection_after_repair", output_payload=reflection.model_dump())
            yield self._trace_event(
                "corrective_reflection",
                "CorrectiveAgent 已完成 repair 后多需求统一反射",
                reflection_result=reflection.model_dump(),
            )
            yield self._corrective_update_event(reflection)

        final_route = self.decide_final_route_from_reflection(task, reflection, "multi_retrieval", multi_need_state=state).selected
        selection = self._selection_from_reflection(state, reflection)
        selection.route = final_route  # type: ignore[assignment]
        selection.reason = reflection.reason
        cards = self._card_candidates_from_reflection(state, selection, reflection)
        product_ids = [candidate.product_id for candidate in cards]
        trace = self._multi_need_decision_trace(state, selection, reflection, task)
        if repair_plans:
            trace.retrieval_summary["repair_plans"] = [repair_plan.summary() for repair_plan in repair_plans]
        self._finish_trace(trace, task, route=final_route)
        yield self._decision_trace_event(trace)
        if cards:
            product_cards = [self.answer_generator.product_card(candidate.product, plan) for candidate in cards]
            yield {"type": "product_cards", "products": [card.model_dump() for card in product_cards]}
            await asyncio.sleep(0.01)
            yield self._agent_update(
                stage="answer",
                title="生成回答",
                content_delta="组合商品卡片已准备好，正在整理解释。",
                done=False,
            )

        self.budget_manager.record_answer_call(task)
        if not selection.flat_candidates and final_route in {"no_product", "clarify", "direct_answer"}:
            mode = "clarification" if final_route == "clarify" else ("direct" if final_route == "direct_answer" else "no_product")
            yield self._agent_update(
                stage="answer",
                title="生成回答",
                content_delta="正在整理回复。",
                done=False,
            )
            async for token in self.answer_generator.stream_direct_text(
                query,
                mode,
                reflection.reason or "多需求检索没有得到足够可靠的商品证据。",
                intent_plan,
                extra_context={"reflection_result": reflection.model_dump(), "multi_need_trace": self._multi_need_trace(state)},
                profile_narrative=profile_narrative,
            ):
                answer_parts.append(token)
                yield {"type": "token", "content": token}
        else:
            async for token in self.answer_generator.stream_multi_need_text(
                state,
                selection,
                final_route,
                reflection.reason,
                reflection.slot_coverage,
                reflection.rejected_products,
                reflection.combo_summary,
                profile_narrative=profile_narrative,
            ):
                answer_parts.append(token)
                yield {"type": "token", "content": token}

        self._schedule_memory_update(
            request=request,
            query=query,
            answer_text="".join(answer_parts),
            route=final_route,
            product_ids=product_ids,
            intent_plan=intent_plan,
            decision_trace=trace.model_dump(),
            evidence_bundle=self._multi_evidence_bundle(
                request=request,
                task=task,
                query=query,
                final_route=final_route,
                state=state,
                displayed_candidates=cards,
                reflection=reflection,
                trace=trace,
            ),
        )
        await trace_run.end(
            output_payload=self._trace_output(
                route=final_route,
                reason=trace.final_reason,
                products=self._products_brief_from_slot_candidates(cards),
            ),
            metadata=self._trace_metadata(),
        )
        yield {"type": "done"}

    def _multi_need_decision_trace(
        self,
        state: MultiNeedState,
        selection: MultiNeedSelection,
        reflection: ReflectionResult,
        task: TurnTaskState | None = None,
    ) -> DecisionTrace:
        final_route = selection.route
        passed_ids = reflection.passed_product_ids
        candidate_counts = self._multi_need_counts(state, selection)
        return DecisionTrace(
            query_understanding=state.intent_plan.model_dump(),
            filters=state.plan.filters,
            retrieval_summary={
                "route": final_route,
                "passed_product_ids": passed_ids,
                "rejected_products": reflection.rejected_products,
                "slot_coverage": reflection.slot_coverage,
                "combo_summary": reflection.combo_summary,
                "fallback_plan": reflection.fallback_plan,
                "reflection_result": reflection.model_dump(),
            },
            multi_need_trace=self._multi_need_trace(state),
            route=final_route,
            failure_stage="none",
            candidate_counts=candidate_counts,
            stages=[
                self._stage("input", "passed", "已接收并标准化用户输入。"),
                self._stage("intent_planning", "passed", "IntentPlanner 已输出多需求计划。"),
                self._stage("multi_need_retrieval", "passed", state.termination_reason),
                self._stage("corrective_reflection", "passed", reflection.reason, product_ids=passed_ids),
            ],
            final_reason=reflection.reason,
        )

    def _multi_need_trace(self, state: MultiNeedState) -> dict:
        return {
            "stop_reason": state.termination_reason,
            "decision_steps": state.budgets.get("decision_steps", 0),
            "search_calls": state.budgets.get("search_calls", 0),
            "coverage_by_slot": {slot_id: value.model_dump() for slot_id, value in state.coverage_by_slot.items()},
            "slot_results_by_slot": state.budgets.get("slot_results_by_slot", {}),
            "tool_calls": [call.model_dump() for call in state.tool_calls],
        }

    def _selection_from_reflection(self, state: MultiNeedState, reflection_result: ReflectionResult) -> MultiNeedSelection:
        selected_ids_by_slot = reflection_result.combo_summary.get("final_combo_product_ids_by_slot")
        if not isinstance(selected_ids_by_slot, dict):
            selected_ids_by_slot = reflection_result.combo_summary.get("selected_product_ids_by_slot")
        if not isinstance(selected_ids_by_slot, dict):
            passed_ids = set(reflection_result.passed_product_ids)
            selected_ids_by_slot = {
                slot.slot_id: [
                    candidate.product_id
                    for candidate in state.candidates_by_slot.get(slot.slot_id, [])
                    if candidate.product_id in passed_ids
                ][:1]
                for slot in state.slots
            }
        selected_by_slot: dict[str, list[SlotCandidate]] = {}
        for slot in state.slots:
            selected_by_slot[slot.slot_id] = self._slot_candidates_for_ids(
                state,
                slot.slot_id,
                [str(product_id) for product_id in selected_ids_by_slot.get(slot.slot_id, [])],
            )
        return MultiNeedSelection(
            selected_by_slot={slot_id: candidates for slot_id, candidates in selected_by_slot.items() if candidates},
            rejected_candidates=reflection_result.rejected_products,
            route=self._final_route_from_reflection(reflection_result, "multi_retrieval"),  # type: ignore[arg-type]
            reason=reflection_result.reason,
        )

    def _card_candidates_from_reflection(
        self,
        state: MultiNeedState,
        selection: MultiNeedSelection,
        reflection_result: ReflectionResult,
    ) -> list[SlotCandidate]:
        candidates = self._dedupe_slot_candidates(selection.flat_candidates)
        alternative_by_slot = reflection_result.combo_summary.get("alternative_product_ids_by_slot")
        if isinstance(alternative_by_slot, dict):
            for slot in state.slots:
                candidates.extend(
                    self._slot_candidates_for_ids(
                        state,
                        slot.slot_id,
                        [str(product_id) for product_id in alternative_by_slot.get(slot.slot_id, [])],
                    )
                )
        return self._dedupe_slot_candidates(candidates)[:MULTI_NEED_PRODUCT_CARD_LIMIT]

    def _slot_candidates_for_ids(self, state: MultiNeedState, slot_id: str, product_ids: list[str]) -> list[SlotCandidate]:
        by_id = {candidate.product_id: candidate for candidate in state.candidates_by_slot.get(slot_id, [])}
        return [by_id[product_id] for product_id in product_ids if product_id in by_id]

    def _dedupe_slot_candidates(self, candidates: list[SlotCandidate]) -> list[SlotCandidate]:
        result: list[SlotCandidate] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate.product_id not in seen:
                result.append(candidate)
                seen.add(candidate.product_id)
        return result

    async def _run_repair_agent(
        self,
        *,
        original_query: str,
        intent_plan: IntentPlan,
        plan: QueryPlan,
        slots: list[NeedSlot],
        trigger: str,
        reflection_result: ReflectionResult,
        previous_candidates: dict[str, list[dict[str, Any]]],
    ) -> RepairPlan:
        return await self.repair_agent.plan_repair(
            original_query=original_query,
            intent_plan=intent_plan,
            plan=plan,
            slots=slots,
            trigger=trigger,
            reflection_result=reflection_result,
            previous_candidates=previous_candidates,
        )

    def _single_repair_slot(self, query: str, intent_plan: IntentPlan, plan: QueryPlan) -> NeedSlot:
        slot_query = intent_plan.vector_query or intent_plan.keyword_query or intent_plan.original_query or query
        return NeedSlot(
            slot_id="single",
            goal=slot_query,
            product_type="",
            query=slot_query,
            hard_constraints=list(plan.filters),
            soft_constraints=[*plan.preferences, *plan.scene],
            exclude_terms=list(plan.exclude),
            min_candidates=plan.retrieval_strategy.final_top_k,
        )

    def _single_previous_candidates(self, ranked: list[tuple[Any, float]]) -> dict[str, list[dict[str, Any]]]:
        return {
            "single": [
                {
                    "product_id": product.product_id,
                    "name": product.name,
                    "category": product.category,
                    "sub_category": product.sub_category,
                    "price": float(product.price),
                    "score": round(score, 4),
                }
                for product, score in ranked[:SINGLE_RETRIEVAL_REVIEW_LIMIT]
            ]
        }

    def _multi_previous_candidates(self, state: MultiNeedState) -> dict[str, list[dict[str, Any]]]:
        return {
            slot.slot_id: [
                {
                    "product_id": candidate.product_id,
                    "name": candidate.name,
                    "category": candidate.category,
                    "sub_category": candidate.sub_category,
                    "price": candidate.price,
                    "score": candidate.rerank_score,
                }
                for candidate in state.candidates_by_slot.get(slot.slot_id, [])
            ]
            for slot in state.slots
        }

    def _repair_plan_query_count(self, repair_plan: RepairPlan) -> int:
        return sum(len(queries) for queries in repair_plan.queries_by_slot.values())

    def _repair_trigger(self, failure_trigger: str, reflection: ReflectionResult, *, default: str) -> str:
        if failure_trigger:
            return failure_trigger
        if reflection.repair_hint.failure_type:
            return reflection.repair_hint.failure_type
        if reflection.repair_hint.target_slot_ids:
            return "slot_rejected" if len(reflection.repair_hint.target_slot_ids) > 1 else "corrective_rejected_candidates"
        if not reflection.has_passed_products:
            return default
        return ""

    def _should_consider_repair(self, reflection: ReflectionResult, trigger: str) -> bool:
        if reflection.has_passed_products and not reflection.repair_hint.target_slot_ids:
            return False
        if reflection.fallback_plan in {"direct_answer", "clarify"}:
            return False
        return bool(trigger and reflection.repair_hint.repairable)

    def _single_retrieval_worker(self) -> SingleRetrievalWorker:
        return self.single_retrieval_worker

    def _retrieval_plan_builder(self) -> RetrievalPlanBuilder:
        return self.retrieval_plan_builder

    def _merge_ranked_evidence(
        self,
        left: list[tuple[Any, float]],
        right: list[tuple[Any, float]],
    ) -> list[tuple[Any, float]]:
        by_id: dict[str, tuple[Any, float]] = {}
        for product, score in [*left, *right]:
            current = by_id.get(product.product_id)
            if current is None or score > current[1]:
                by_id[product.product_id] = (product, score)
        return sorted(by_id.values(), key=lambda item: item[1], reverse=True)[:SINGLE_RETRIEVAL_REVIEW_LIMIT]

    def _merge_score_maps(self, left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
        result = dict(left)
        for key, value in right.items():
            result[key] = max(result.get(key, 0.0), value)
        return result

    def _new_task(self, request: ChatStreamRequest) -> TurnTaskState:
        return TurnTaskState(turn_id=uuid.uuid4().hex, user_id=request.user_id, session_id=request.session_id)

    async def _extract_image_attributes(
        self,
        image_path: Any,
        query: str,
        trace_run: Any,
        task: TurnTaskState,
    ) -> ImageAttributes:
        extractor = getattr(self, "image_attribute_extractor", None)
        if extractor is None:
            attributes = ImageAttributes(
                available=False,
                uncertainty_note="ImageAttributeExtractor is not initialized.",
            )
        else:
            attributes = await extractor.extract(image_path, query)
        output_payload = self._image_attribute_trace_payload(attributes)
        task.add_step(
            "image_attribute_extraction",
            "succeeded" if attributes.available else "skipped",
            output_summary=output_payload,
        )
        await trace_run.span(
            "image_attribute_extraction",
            input_payload={"image_path_resolved": True},
            output_payload=output_payload,
        )
        return attributes

    def _start_image_attribute_task(
        self,
        image_path: Any,
        query: str,
        trace_run: Any,
        task: TurnTaskState,
    ) -> asyncio.Task[ImageAttributes]:
        image_task = asyncio.create_task(self._extract_image_attributes(image_path, query, trace_run, task))
        image_task.add_done_callback(self._log_image_attribute_task_result)
        return image_task

    async def _image_attributes_from_task_or_placeholder(
        self,
        image_task: asyncio.Task[ImageAttributes],
        *,
        wait_seconds: float,
    ) -> ImageAttributes:
        try:
            if image_task.done():
                return image_task.result()
            return await asyncio.wait_for(asyncio.shield(image_task), timeout=max(0.0, wait_seconds))
        except TimeoutError:
            return ImageAttributes(
                available=False,
                uncertainty_note="ImageAttributeExtractor is still running; image similarity evidence was used first.",
            )
        except Exception as exc:
            logger.warning("ImageAttributeExtractor background task failed: %s", exc)
            return ImageAttributes(
                available=False,
                uncertainty_note=f"ImageAttributeExtractor failed: {str(exc)[:160]}",
            )

    def _image_attribute_corrective_wait_seconds(self) -> float:
        settings = getattr(getattr(self, "image_attribute_extractor", None), "settings", None)
        timeout = getattr(settings, "vlm_timeout_seconds", 10)
        try:
            return max(1.0, float(timeout))
        except (TypeError, ValueError):
            return 10.0

    def _log_image_attribute_task_result(self, image_task: asyncio.Task[ImageAttributes]) -> None:
        try:
            image_task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.warning("ImageAttributeExtractor background task failed: %s", exc)

    def _agent_update(
        self,
        *,
        stage: str,
        title: str,
        content_delta: str = "",
        done: bool = False,
    ) -> dict[str, Any]:
        return {
            "type": "agent_update",
            "stage": stage,
            "title": title,
            "content_delta": content_delta,
            "done": done,
        }

    def _visible_text_chunks(self, text: str, *, size: int = 9) -> list[str]:
        compact = " ".join(str(text or "").split())
        if not compact:
            return []
        return [compact[index : index + size] for index in range(0, len(compact), size)]

    def _image_attribute_update_text(self, attributes: ImageAttributes) -> str:
        if not attributes.available:
            return "图片理解耗时较久，我先用图片相似度检索。"
        parts: list[str] = []
        if attributes.product_type_guess:
            parts.append(attributes.product_type_guess)
        if attributes.colors:
            parts.append("、".join(attributes.colors[:3]))
        if attributes.style_tags:
            parts.append("、".join(attributes.style_tags[:3]))
        summary = "，".join(parts)
        if not summary:
            summary = attributes.retrieval_query or attributes.category_guess or "这张图片里的商品特征"
        return f"图片看起来像{summary}，我会用这些视觉线索辅助检索。"

    def _image_attribute_trace_payload(self, attributes: ImageAttributes) -> dict[str, Any]:
        payload = attributes.model_dump()
        telemetry = getattr(self.image_attribute_extractor, "last_call", None)
        if telemetry is not None:
            call = telemetry.model_dump()
            payload["model_status"] = call.get("status")
            payload["model_latency_ms"] = call.get("latency_ms")
            payload["model_usage"] = call.get("usage") or {}
            payload["model_error_type"] = call.get("error_type")
        return payload

    def _corrective_update_event(self, reflection: ReflectionResult) -> dict[str, Any]:
        return self._agent_update(
            stage="corrective",
            title="校验证据",
            content_delta=self._build_corrective_update(reflection),
            done=True,
        )

    def _build_corrective_update(self, reflection: ReflectionResult) -> str:
        passed_count = len(reflection.passed_product_ids)
        rejected_count = len(self._rejected_product_ids(reflection))
        slot_count = len(reflection.slot_coverage)
        if passed_count > 0 and slot_count > 1:
            return f"已校验多项需求，保留 {passed_count} 款可进入组合推荐的候选。"
        if passed_count > 0:
            return f"已校验候选商品，保留 {passed_count} 款更符合需求的选项。"
        if rejected_count > 0:
            return f"已校验候选商品，{rejected_count} 款未通过约束检查，继续给出稳妥结果。"
        if reflection.fallback_plan == "clarify":
            return "已校验证据，需要先补充关键信息再继续推荐。"
        if reflection.fallback_plan == "direct_answer":
            return "已确认本轮更适合直接回答，不进入商品推荐。"
        return "已完成候选证据校验，正在整理最终回复。"

    def _decision_trace_event(self, trace: DecisionTrace) -> dict[str, Any]:
        return {"type": "decision_trace", "trace": self._client_safe_trace_payload(trace)}

    def _trace_output(
        self,
        *,
        route: str,
        reason: str,
        products: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        product_briefs = products or []
        return {
            "route": route,
            "reason": reason,
            "products": product_briefs,
            "product_count": len(product_briefs),
            "model_call_count": len(self._model_telemetry_payloads()),
        }

    def _trace_metadata(self) -> dict[str, Any]:
        model_calls = self._model_telemetry_payloads()
        return {
            "model_usage_summary": self._model_usage_summary(model_calls),
            "model_calls": model_calls,
        }

    def _model_telemetry_payloads(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for name in [
            "intent_planner",
            "image_attribute_extractor",
            "corrective_agent",
            "repair_agent",
            "answer_generator",
        ]:
            component = getattr(self, name, None)
            if component is None:
                continue
            llm_client = getattr(component, "llm_client", None)
            if llm_client is not None and hasattr(llm_client, "telemetry_payloads"):
                payloads.extend(llm_client.telemetry_payloads())
                continue
            if hasattr(component, "telemetry_payloads"):
                payloads.extend(component.telemetry_payloads())
        return payloads

    def _model_usage_summary(self, model_calls: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        latency_ms = 0.0
        estimated_cost = 0.0
        has_cost = False
        for call in model_calls:
            usage = call.get("usage") if isinstance(call.get("usage"), dict) else {}
            prompt_tokens += self._usage_token_count(usage, "prompt_tokens", "input_tokens")
            completion_tokens += self._usage_token_count(usage, "completion_tokens", "output_tokens")
            total_tokens += self._usage_token_count(usage, "total_tokens")
            latency_ms += float(call.get("latency_ms") or 0.0)
            cost = call.get("estimated_cost")
            if isinstance(cost, (int, float)):
                estimated_cost += float(cost)
                has_cost = True
        return {
            "call_count": len(model_calls),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency_ms": round(latency_ms, 2),
            "estimated_cost": round(estimated_cost, 8) if has_cost else None,
        }

    def _usage_token_count(self, usage: dict[str, Any], *keys: str) -> int:
        for key in keys:
            value = usage.get(key)
            if isinstance(value, (int, float)):
                return int(value)
        return 0

    def _products_brief_from_ranked(self, ranked: list[tuple[Any, float]]) -> list[dict[str, Any]]:
        return [self._compact_product_evidence(product) for product, _ in ranked]

    def _products_brief_from_products(self, products: list[Any]) -> list[dict[str, Any]]:
        return [self._compact_product_evidence(product) for product in products]

    def _products_brief_from_slot_candidates(self, candidates: list[SlotCandidate]) -> list[dict[str, Any]]:
        return [self._compact_product_evidence(candidate.product) for candidate in candidates]

    def _trace_event(self, stage: str, content: str, **payload: Any) -> dict[str, Any]:
        return self._drop_client_trace_fields(
            {
                "type": "trace",
                "stage": stage,
                "content": content,
                **payload,
            }
        )

    def _client_safe_trace_payload(self, trace: DecisionTrace) -> dict[str, Any]:
        return self._drop_client_trace_fields(trace.model_dump())

    def _drop_client_trace_fields(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: self._drop_client_trace_fields(item)
                for key, item in value.items()
                if not self._is_client_sensitive_trace_key(key)
            }
        if isinstance(value, list):
            return [self._drop_client_trace_fields(item) for item in value]
        return value

    def _is_client_sensitive_trace_key(self, key: str) -> bool:
        normalized = key.lower()
        return (
            normalized in CLIENT_TRACE_DROP_KEYS
            or normalized.endswith("_query")
            or normalized.endswith("_queries")
            or normalized == "queries_by_slot"
        )

    def _planner_context(
        self,
        conversation_context: ConversationContext,
        session_id: str,
        *,
        include_long_term: bool = True,
        profile_memory: list[dict[str, Any]] | None = None,
        image_attributes: ImageAttributes | None = None,
    ) -> dict[str, Any]:
        context = conversation_context.to_rewrite_context(
            include_long_term=include_long_term,
            profile_memory=profile_memory,
        )
        if image_attributes is not None:
            context["image_attributes"] = image_attributes.model_dump()
            context.setdefault("priority", []).append("image_attributes")
        recent_evidence = self.evidence_cache.compact_recent(session_id)
        if recent_evidence:
            context["recent_evidence"] = recent_evidence
            context.setdefault("priority", []).append("recent_evidence")
        return context

    def _merge_profile_narrative(self, current: str, profile_memory: list[dict[str, Any]]) -> str:
        if not profile_memory:
            return current
        lines = []
        for item in profile_memory[:6]:
            key = str(item.get("key") or "").strip()
            value = str(item.get("value") or "").strip()
            if key and value:
                lines.append(f"- {key}: {value}")
        if not lines:
            return current
        profile_text = "本轮读取到的长期偏好（只能作为软偏好解释，不能覆盖当前需求）：\n" + "\n".join(lines)
        return "\n".join(part for part in [current.strip(), profile_text] if part)

    def _image_only_intent_plan(self, image_attributes: ImageAttributes | None = None) -> IntentPlan:
        visual_query = ""
        if image_attributes is not None and image_attributes.available:
            visual_query = image_attributes.retrieval_query.strip()
        return IntentPlan(
            original_query="",
            plan_type="image_retrieval",
            vector_query=visual_query,
            keyword_query=visual_query,
            need_slots=[],
            plan_reason=(
                "用户只上传了图片，Orchestrator 将结合图片属性推测进入单图图片检索路径。"
                if visual_query
                else "用户只上传了图片，Orchestrator 将进入单图图片检索路径。"
            ),
        )

    def _image_attributes_payload(self, image_attributes: ImageAttributes | dict[str, Any] | None) -> dict[str, Any]:
        if image_attributes is None:
            return {}
        if isinstance(image_attributes, ImageAttributes):
            return image_attributes.model_dump()
        if isinstance(image_attributes, dict):
            return dict(image_attributes)
        return {}

    def _image_attributes_for_prompt(self, image_attributes: ImageAttributes | dict[str, Any] | None) -> dict[str, Any]:
        payload = self._image_attributes_payload(image_attributes)
        return payload if payload.get("available") else {}

    async def _stream_missing_image(
        self,
        *,
        request: ChatStreamRequest,
        query: str,
        task: TurnTaskState,
        trace_run: Any,
        profile_narrative: str,
    ) -> AsyncGenerator[dict, None]:
        reason = "请求包含 image_id，但服务器没有找到对应上传图片。"
        intent_plan = IntentPlan(original_query=query, plan_type="clarify", plan_reason=reason)
        self._update_planner_proposal(task, intent_plan)
        task.add_decision(
            OrchestratorDecision(
                decision="image_input",
                approved=False,
                selected="missing_image",
                reason=reason,
                proposal_summary={"image_id": request.image_id},
            )
        )
        self.decide_execution_path(task, intent_plan)
        final_route = "clarify"
        self.budget_manager.record_answer_call(task)
        trace = DecisionTrace(
            query_understanding=intent_plan.model_dump(),
            retrieval_summary={"route": final_route, "answer_mode": "clarification", "reason": reason},
            route=final_route,
            failure_stage="input",
            failure_reason=reason,
            candidate_counts={},
            stages=[
                self._stage("input", "failed", reason, image_id=request.image_id),
            ],
            final_reason=reason,
        )
        self._finish_trace(trace, task, route=final_route)
        yield self._decision_trace_event(trace)
        answer_parts: list[str] = []
        async for token in self.answer_generator.stream_direct_text(
            query,
            "clarification",
            "图片没有上传成功或已经过期，请重新选择一张图片再发送。",
            intent_plan,
            profile_narrative=profile_narrative,
        ):
            answer_parts.append(token)
            yield {"type": "token", "content": token}
        self._schedule_memory_update(
            request=request,
            query=query,
            answer_text="".join(answer_parts),
            route=final_route,
            product_ids=[],
            intent_plan=intent_plan,
            decision_trace=trace.model_dump(),
        )
        await trace_run.end(
            output_payload=self._trace_output(route=final_route, reason=reason),
            metadata=self._trace_metadata(),
        )
        yield {"type": "done"}

    async def _stream_planner_failure(
        self,
        *,
        request: ChatStreamRequest,
        query: str,
        task: TurnTaskState,
        trace_run: Any,
        error: StructuredLlmValidationError,
    ) -> AsyncGenerator[dict, None]:
        reason = "IntentPlanner 结构化输出在修复重试后仍不合法，已停止本轮检索路径。"
        answer_text = "我这轮没有稳定理解你的请求，已经停止检索流程，避免给你乱推荐。请稍后再试，或者换个说法重新问我。"
        task.add_step(
            "intent_planning",
            "failed",
            input_summary={"query": query},
            output_summary={"content_preview": error.content[:500], "parsed_data": error.data or {}},
            decision_summary={"validation_errors": error.errors, "retry_exhausted": True},
            error_type="validation_failed",
            error_message=reason,
        )
        task.add_decision(
            OrchestratorDecision(
                decision="intent_plan",
                approved=False,
                selected="planner_failed",
                reason=reason,
                decision_summary={"validation_errors": error.errors, "retry_exhausted": True},
            )
        )
        task.mark_failed("validation_failed", reason)
        await trace_run.span(
            "intent_planning",
            input_payload={"query": query},
            output_payload={
                "status": "failed",
                "reason": reason,
                "validation_errors": error.errors,
                "content_preview": error.content[:500],
            },
        )
        trace = DecisionTrace(
            query_understanding={"original_query": query},
            retrieval_summary={"route": "planner_failed", "reason": reason},
            route="planner_failed",
            failure_stage="intent_planning",
            failure_reason=reason,
            candidate_counts={},
            stages=[
                self._stage("input", "passed", "已接收并标准化用户输入。"),
                self._stage("intent_planning", "failed", reason, validation_errors=error.errors),
            ],
            final_reason=reason,
        )
        self.trace_recorder.apply_failure_trace(trace, task)
        yield self._trace_event("intent_planning", reason)
        yield self._decision_trace_event(trace)
        yield {"type": "token", "content": answer_text}
        self._schedule_memory_update(
            request=request,
            query=query,
            answer_text=answer_text,
            route="planner_failed",
            product_ids=[],
            intent_plan=IntentPlan(original_query=query, plan_type="direct_answer", plan_reason=reason),
            decision_trace=trace.model_dump(),
        )
        await trace_run.end(
            output_payload=self._trace_output(route="planner_failed", reason=reason),
            metadata=self._trace_metadata(),
        )
        yield {"type": "done"}

    def _update_planner_proposal(self, task: TurnTaskState, intent_plan: IntentPlan) -> None:
        task.planner_proposal = {
            "original_query": intent_plan.original_query,
            "summary": intent_plan.summary,
            "plan_type": intent_plan.plan_type,
            "vector_query": intent_plan.vector_query,
            "keyword_query": intent_plan.keyword_query,
            "budget_min": intent_plan.budget_min,
            "budget_max": intent_plan.budget_max,
            "budget_scope": intent_plan.budget_scope,
            "need_slots": [slot.model_dump() for slot in intent_plan.need_slots],
            "referenced_product_ids": intent_plan.referenced_product_ids,
            "profile_lookup": intent_plan.profile_lookup.model_dump(),
            "plan_reason": intent_plan.plan_reason,
        }

    def decide_intent_plan(self, task: TurnTaskState, intent_plan: IntentPlan) -> OrchestratorDecision:
        return task.add_decision(
            OrchestratorDecision(
                decision="intent_plan",
                approved=True,
                selected=intent_plan.plan_type,
                reason=intent_plan.plan_reason or "IntentPlanner produced a syntactically valid plan.",
                proposal_summary=intent_plan.model_dump(),
            )
        )

    def decide_profile_lookup(self, task: TurnTaskState, intent_plan: IntentPlan) -> OrchestratorDecision:
        return task.add_decision(
            OrchestratorDecision(
                decision="profile_lookup",
                approved=bool(intent_plan.profile_lookup.requested),
                reason=intent_plan.profile_lookup.reason or "Planner requested profile lookup.",
                proposal_summary=intent_plan.profile_lookup.model_dump(),
            )
        )

    def decide_previous_evidence_answer(
        self,
        task: TurnTaskState,
        intent_plan: IntentPlan,
        conversation_context: ConversationContext,
        referenced_products: list[Any],
    ) -> OrchestratorDecision:
        requested_ids = [product_id for product_id in intent_plan.referenced_product_ids if product_id]
        recent_ids = self._recent_context_product_ids(conversation_context)
        cache_evidence = self.evidence_cache.compact_recent(task.session_id)
        cache_product_ids = {
            product_id
            for bundle in cache_evidence
            for product_id in [
                *bundle.get("displayed_product_ids", []),
                *bundle.get("selected_product_ids", []),
                *bundle.get("candidate_product_ids", []),
            ]
            if product_id
        }
        loaded_ids = {product.product_id for product in referenced_products}
        missing_from_context = [product_id for product_id in requested_ids if product_id not in recent_ids]
        missing_from_db = [product_id for product_id in requested_ids if product_id not in loaded_ids]
        plan_allows_direct_context_answer = intent_plan.plan_type == "direct_answer"
        approved = (
            plan_allows_direct_context_answer
            and bool(requested_ids)
            and not missing_from_context
            and not missing_from_db
        )
        reason = (
            "Planner referenced products from recent context and DB details were loaded."
            if approved
            else (
                "Planner proposed retrieval/update path; referenced products are context anchors, not a reason to skip retrieval."
                if not plan_allows_direct_context_answer
                else "Referenced product evidence is missing; fall back to the planned path."
            )
        )
        return task.add_decision(
            OrchestratorDecision(
                decision="previous_evidence_answer",
                approved=approved,
                reason=reason,
                proposal_summary={"referenced_product_ids": requested_ids},
                decision_summary={
                    "recent_context_product_ids": sorted(recent_ids),
                    "recent_cache_product_ids": sorted(cache_product_ids),
                    "loaded_product_ids": sorted(loaded_ids),
                    "missing_from_context": missing_from_context,
                    "missing_from_db": missing_from_db,
                    "plan_allows_direct_context_answer": plan_allows_direct_context_answer,
                    "conversation_evidence": self._conversation_evidence_for_products(conversation_context, requested_ids),
                    "recent_evidence": cache_evidence,
                },
            )
        )

    def decide_image_retrieval_path(
        self,
        task: TurnTaskState,
        intent_plan: IntentPlan,
        *,
        image_path_resolved: bool,
    ) -> OrchestratorDecision:
        direct_non_product = intent_plan.plan_type == "direct_answer" and bool(intent_plan.original_query.strip())
        approved = image_path_resolved and not direct_non_product
        reason = (
            "单图输入已解析，Orchestrator 批准进入 image_retrieval 执行路径。"
            if approved
            else "文字计划是非商品直接回答，本轮不让图片输入覆盖 direct_answer。"
        )
        return task.add_decision(
            OrchestratorDecision(
                decision="image_retrieval_path",
                approved=approved,
                selected="image_retrieval" if approved else intent_plan.plan_type,
                reason=reason,
                proposal_summary={"plan_type": intent_plan.plan_type, "image_path_resolved": image_path_resolved},
            )
        )

    def decide_image_relevance(self, task: TurnTaskState, evidence: Any) -> OrchestratorDecision:
        approved = bool(getattr(evidence, "ranked", []))
        reason = (
            "图片检索返回了候选证据，继续交给 CorrectiveAgent reflection。"
            if approved
            else "图片检索没有达到最低相关阈值，后续由 CorrectiveAgent/Orchestrator 裁决 no_product 或 clarify。"
        )
        return task.add_decision(
            OrchestratorDecision(
                decision="image_relevance",
                approved=approved,
                selected="candidate_evidence" if approved else "low_relevance",
                reason=reason,
                decision_summary={
                    "candidate_count": len(getattr(evidence, "ranked", [])),
                    "max_image_score": getattr(evidence, "max_image_score", 0.0),
                },
            )
        )

    def decide_execution_path(self, task: TurnTaskState, intent_plan: IntentPlan) -> OrchestratorDecision:
        return task.add_decision(
            OrchestratorDecision(
                decision="execution_path",
                selected=intent_plan.plan_type,
                reason=f"Orchestrator approved plan_type={intent_plan.plan_type}.",
                proposal_summary={"plan_type": intent_plan.plan_type},
            )
        )

    def decide_repair(self, task: TurnTaskState, trigger: str) -> OrchestratorDecision:
        repairable = trigger in {
            "score_filter_empty",
            "no_candidates",
            "corrective_rejected_candidates",
            "slot_empty",
            "slot_weak",
            "slot_rejected",
            "constraint_mismatch",
        }
        approved = repairable and self.budget_manager.can_repair(task)
        return task.add_decision(
            OrchestratorDecision(
                decision="repair",
                approved=approved,
                internal_decision="repair" if approved else "",
                reason="Repair approved by Orchestrator." if approved else "Repair not applicable or budget exhausted.",
                proposal_summary={"trigger": trigger},
                decision_summary=self.budget_manager.repair_snapshot(task),
            )
        )

    def decide_final_route_from_reflection(
        self,
        task: TurnTaskState,
        reflection_result: ReflectionResult,
        execution_path: str,
        multi_need_state: MultiNeedState | None = None,
    ) -> OrchestratorDecision:
        route = self._final_route_from_reflection(reflection_result, execution_path, multi_need_state=multi_need_state)
        return task.add_decision(
            OrchestratorDecision(
                decision="final_route",
                selected=route,
                reason=reflection_result.reason,
                proposal_summary={"reflection_result": reflection_result.model_dump()},
            )
        )

    def _final_route_from_reflection(
        self,
        reflection_result: ReflectionResult,
        execution_path: str,
        *,
        multi_need_state: MultiNeedState | None = None,
    ) -> str:
        if reflection_result.has_passed_products:
            if execution_path == "multi_retrieval":
                status = str(reflection_result.combo_summary.get("status") or "")
                if status == "over_budget":
                    return "over_budget_combo"
                if status in {"missing_required", "no_complete_combo"} and self._has_route_blocking_missing_required_slots(
                    reflection_result,
                    multi_need_state,
                ):
                    return "partial_recommend"
            return "recommend"
        if reflection_result.fallback_plan != "none":
            return reflection_result.fallback_plan
        return "no_product"

    def _has_missing_required_slots(self, reflection_result: ReflectionResult) -> bool:
        combo_summary = reflection_result.combo_summary if isinstance(reflection_result.combo_summary, dict) else {}
        return any(str(slot_id or "").strip() for slot_id in combo_summary.get("missing_required_slot_ids") or [])

    def _has_route_blocking_missing_required_slots(
        self,
        reflection_result: ReflectionResult,
        multi_need_state: MultiNeedState | None,
    ) -> bool:
        combo_summary = reflection_result.combo_summary if isinstance(reflection_result.combo_summary, dict) else {}
        missing_ids = [str(slot_id or "").strip() for slot_id in combo_summary.get("missing_required_slot_ids") or [] if str(slot_id or "").strip()]
        if not missing_ids:
            return False
        if multi_need_state is None:
            return True
        query = str(multi_need_state.intent_plan.original_query or "").strip()
        slots_by_id = {slot.slot_id: slot for slot in multi_need_state.slots}
        return any(self._slot_explicitly_requested(slots_by_id.get(slot_id), query) for slot_id in missing_ids)

    def _slot_explicitly_requested(self, slot: NeedSlot | None, query: str) -> bool:
        if slot is None or not query:
            return False
        generic_terms = {
            "商品",
            "产品",
            "用品",
            "装备",
            "套装",
            "清单",
            "组合",
            "配齐",
            "配件",
            "设备",
            "基础",
            "轻量",
            "户外",
            "露营",
            "拍照",
            "徒步",
            "通勤",
            "训练",
            "健身房",
            "日常",
            "适合",
            "新手",
            "春天",
            "预算",
            "整套",
            "一套",
            "几件",
            "推荐",
        }
        text = " ".join([slot.goal, slot.product_type, slot.query])
        normalized = text
        for char in "（）()/、，,;；|和或":
            normalized = normalized.replace(char, " ")
        terms = {
            term.strip()
            for term in normalized.split()
            if len(term.strip()) >= 2 and term.strip() not in generic_terms
        }
        return any(term in query for term in terms)

    def _finish_trace(self, trace: DecisionTrace, task: TurnTaskState, *, route: str) -> None:
        self.trace_recorder.finish_trace(
            trace,
            task,
            route=route,
        )

    def _agent_path_from_trace(self, trace: DecisionTrace, task: TurnTaskState) -> list[dict[str, Any]]:
        return self.trace_recorder.agent_path(trace, task)

    def _recent_context_product_ids(self, conversation_context: ConversationContext) -> set[str]:
        return {
            product_id
            for turn in conversation_context.recent_turns
            for product_id in turn.product_ids
            if product_id
        }

    def _conversation_evidence_for_products(
        self,
        conversation_context: ConversationContext,
        product_ids: list[str],
    ) -> dict[str, Any]:
        requested = set(product_ids)
        turns = []
        for turn in conversation_context.recent_turns:
            if requested.intersection(turn.product_ids):
                turns.append(turn.compact())
        return {"turns": turns}

    def _product_detail_for_answer(self, product: Any) -> dict[str, Any]:
        return {
            "product_id": product.product_id,
            "name": product.name,
            "brand": product.brand,
            "category": product.category,
            "sub_category": product.sub_category,
            "price": float(product.price),
            "rating": float(product.rating),
            "description": product.description[:500],
            "specs": product.specs,
            "suitable_for": product.suitable_for,
            "avoid_for": product.avoid_for,
            "tags": product.tags[:12],
            "review_summary": product.review_summary[:500],
        }

    def _compact_product_evidence(self, product: Any) -> dict[str, Any]:
        return {
            "product_id": product.product_id,
            "name": product.name,
            "brand": product.brand,
            "category": product.category,
            "sub_category": product.sub_category,
            "price": float(product.price),
            "rating": float(product.rating),
        }

    def _need_slots_from_intent_plan(self, intent_plan: IntentPlan, plan: QueryPlan) -> list[NeedSlot]:
        slots: list[NeedSlot] = []
        for index, slot in enumerate(intent_plan.need_slots, start=1):
            slots.append(
                NeedSlot(
                    slot_id=slot.slot_id or f"s{index}",
                    need_type=slot.need_type,
                    goal=slot.goal,
                    product_type=slot.product_type,
                    query=slot.query or slot.goal or slot.product_type,
                    hard_constraints=self._slot_hard_constraints(intent_plan, plan),
                    soft_constraints=self._dedupe([*slot.soft_constraints, *plan.preferences, *plan.scene]),
                    exclude_terms=self._dedupe([*slot.exclude_terms, *plan.exclude]),
                    min_candidates=max(1, int(slot.min_candidates or 1)),
                )
            )
        return slots

    def _slot_hard_constraints(self, intent_plan: IntentPlan, plan: QueryPlan) -> list[str]:
        constraints = list(plan.filters)
        if intent_plan.budget_scope == "total":
            constraints = [item for item in constraints if not item.startswith("price <=") and not item.startswith("price >=")]
        return constraints

    def _multi_need_counts(self, state: MultiNeedState, selection: MultiNeedSelection) -> dict[str, int]:
        total_candidates = sum(len(candidates) for candidates in state.candidates_by_slot.values())
        return {
            "multi_need_slot_candidates": total_candidates,
            "after_corrective": len(selection.flat_candidates),
            "search_calls": int(state.budgets.get("search_calls", 0)),
        }

    def _single_near_miss_context(
        self,
        ranked: list[tuple[Any, float]],
        reflection_result: ReflectionResult,
    ) -> dict[str, Any]:
        rejected_ids = {str(item.get("product_id")) for item in reflection_result.rejected_products if isinstance(item, dict)}
        near_miss = [
            {
                "product_id": product.product_id,
                "name": product.name,
                "price": float(product.price),
                "reason": next(
                    (
                        str(item.get("reason") or "")
                        for item in reflection_result.rejected_products
                        if isinstance(item, dict) and item.get("product_id") == product.product_id
                    ),
                    "",
                ),
            }
            for product, _ in ranked[:3]
            if product.product_id in rejected_ids or not rejected_ids
            ]
        return {"near_miss_products": near_miss, "reflection_result": reflection_result.model_dump()}

    def _single_evidence_bundle(
        self,
        *,
        request: ChatStreamRequest,
        task: TurnTaskState,
        query: str,
        execution_path: str,
        final_route: str,
        ranked: list[tuple[Any, float]],
        displayed_ranked: list[tuple[Any, float]],
        reflection: ReflectionResult,
        trace: DecisionTrace,
    ) -> EvidenceBundle | None:
        if not ranked and not displayed_ranked:
            return None
        displayed_product_ids = [product.product_id for product, _ in displayed_ranked]
        display_order = {product_id: index for index, product_id in enumerate(displayed_product_ids, start=1)}
        rejected_product_ids = self._rejected_product_ids(reflection)
        return EvidenceBundle(
            user_id=request.user_id,
            session_id=request.session_id,
            turn_id=task.turn_id,
            query=query,
            execution_path=execution_path,
            final_route=final_route,
            displayed_product_ids=displayed_product_ids,
            selected_product_ids=list(reflection.passed_product_ids or displayed_product_ids),
            candidate_product_ids=[product.product_id for product, _ in ranked],
            rejected_product_ids=rejected_product_ids,
            candidates=[
                EvidenceCandidate(
                    product_id=product.product_id,
                    score=float(score),
                    stage=execution_path,
                    reason=self._rejected_reason(reflection, product.product_id),
                    display_order=display_order.get(product.product_id),
                    compact_product=self._compact_product_evidence(product),
                )
                for product, score in ranked
            ],
            reflection_summary=self._reflection_summary(reflection),
            trace_summary=self._compact_trace_for_memory(trace.model_dump()),
        )

    def _multi_evidence_bundle(
        self,
        *,
        request: ChatStreamRequest,
        task: TurnTaskState,
        query: str,
        final_route: str,
        state: MultiNeedState,
        displayed_candidates: list[SlotCandidate],
        reflection: ReflectionResult,
        trace: DecisionTrace,
    ) -> EvidenceBundle | None:
        all_candidates = [
            candidate
            for slot in state.slots
            for candidate in state.candidates_by_slot.get(slot.slot_id, [])
        ]
        if not all_candidates and not displayed_candidates:
            return None
        displayed_product_ids = [candidate.product_id for candidate in displayed_candidates]
        display_order = {product_id: index for index, product_id in enumerate(displayed_product_ids, start=1)}
        rejected_product_ids = self._rejected_product_ids(reflection)
        return EvidenceBundle(
            user_id=request.user_id,
            session_id=request.session_id,
            turn_id=task.turn_id,
            query=query,
            execution_path="multi_retrieval",
            final_route=final_route,
            displayed_product_ids=displayed_product_ids,
            selected_product_ids=list(reflection.passed_product_ids or displayed_product_ids),
            candidate_product_ids=[candidate.product_id for candidate in all_candidates],
            rejected_product_ids=rejected_product_ids,
            candidates=[
                EvidenceCandidate(
                    product_id=candidate.product_id,
                    score=float(candidate.rerank_score),
                    slot_id=slot_id,
                    stage="multi_retrieval",
                    reason=self._rejected_reason(reflection, candidate.product_id) or candidate.coverage_reason,
                    display_order=display_order.get(candidate.product_id),
                    compact_product=self._compact_product_evidence(candidate.product),
                )
                for slot_id, candidates in state.candidates_by_slot.items()
                for candidate in candidates
            ],
            slots=[
                EvidenceSlot(
                    slot_id=slot.slot_id,
                    goal=slot.goal,
                    selected_product_ids=[
                        candidate.product_id
                        for candidate in displayed_candidates
                        if candidate in state.candidates_by_slot.get(slot.slot_id, [])
                    ],
                    candidate_product_ids=[
                        candidate.product_id for candidate in state.candidates_by_slot.get(slot.slot_id, [])
                    ],
                    rejected_product_ids=[
                        product_id
                        for product_id in rejected_product_ids
                        if product_id in {candidate.product_id for candidate in state.candidates_by_slot.get(slot.slot_id, [])}
                    ],
                    coverage_status=state.coverage_by_slot.get(slot.slot_id).status
                    if state.coverage_by_slot.get(slot.slot_id)
                    else "",
                    reason=state.coverage_by_slot.get(slot.slot_id).reason
                    if state.coverage_by_slot.get(slot.slot_id)
                    else "",
                )
                for slot in state.slots
            ],
            reflection_summary=self._reflection_summary(reflection),
            trace_summary=self._compact_trace_for_memory(trace.model_dump()),
        )

    def _reflection_summary(self, reflection: ReflectionResult) -> dict[str, Any]:
        return {
            "has_passed_products": reflection.has_passed_products,
            "passed_product_ids": reflection.passed_product_ids,
            "fallback_plan": reflection.fallback_plan,
            "slot_coverage": reflection.slot_coverage,
            "combo_summary": reflection.combo_summary,
            "reason": reflection.reason,
        }

    def _rejected_product_ids(self, reflection: ReflectionResult) -> list[str]:
        return [
            str(item.get("product_id"))
            for item in reflection.rejected_products
            if isinstance(item, dict) and item.get("product_id")
        ]

    def _rejected_reason(self, reflection: ReflectionResult, product_id: str) -> str:
        for item in reflection.rejected_products:
            if isinstance(item, dict) and item.get("product_id") == product_id:
                return str(item.get("reason") or "")
        return ""

    def _schedule_memory_update(
        self,
        *,
        request: ChatStreamRequest,
        query: str,
        answer_text: str,
        route: str,
        product_ids: list[str],
        intent_plan: IntentPlan,
        decision_trace: dict[str, Any],
        evidence_bundle: EvidenceBundle | None = None,
    ) -> None:
        task = asyncio.create_task(
            self._write_memory_update(
                request=request,
                query=query,
                answer_text=answer_text,
                route=route,
                product_ids=product_ids,
                intent_plan=intent_plan,
                decision_trace=decision_trace,
                evidence_bundle=evidence_bundle,
            )
        )
        task.add_done_callback(self._log_memory_task_result)

    async def _write_memory_update(
        self,
        *,
        request: ChatStreamRequest,
        query: str,
        answer_text: str,
        route: str,
        product_ids: list[str],
        intent_plan: IntentPlan,
        decision_trace: dict[str, Any],
        evidence_bundle: EvidenceBundle | None = None,
    ) -> None:
        SessionLocal = get_sessionmaker()
        db = SessionLocal()
        try:
            manager = MemoryManager(db)
            selected_products = self._selected_product_snapshots(db, product_ids)
            trace_summary = self._compact_trace_for_memory(decision_trace)
            if selected_products:
                trace_summary["selected_products"] = selected_products
            if evidence_bundle is not None:
                evidence_bundle.trace_summary = dict(trace_summary)
                self.evidence_cache.put_turn_evidence(evidence_bundle)
            manager.append_turn(
                user_id=request.user_id,
                session_id=request.session_id,
                user_message=query,
                assistant_message=answer_text,
                route=route,
                product_ids=product_ids,
                rewrite_summary=intent_plan.model_dump(),
                trace_summary=trace_summary,
            )
            await manager.summarize_if_needed(request.user_id, request.session_id, SessionSummarizer())
        finally:
            db.close()

    def _selected_product_snapshots(self, db: Session, product_ids: list[str]) -> list[dict[str, Any]]:
        products = ProductRepository(db).get_by_ids(product_ids)
        return [self._compact_product_evidence(product) for product in products]

    def _log_memory_task_result(self, task: asyncio.Task) -> None:
        try:
            task.result()
        except Exception as exc:
            logger.warning("memory update failed: %s", exc, exc_info=True)

    def _compact_trace_for_memory(self, decision_trace: dict[str, Any]) -> dict[str, Any]:
        summary = decision_trace.get("retrieval_summary") or {}
        return {
            "route": decision_trace.get("route"),
            "passed_product_ids": summary.get("passed_product_ids") or [],
            "slot_coverage": summary.get("slot_coverage") or [],
            "combo_summary": summary.get("combo_summary") or {},
            "fallback_plan": summary.get("fallback_plan"),
        }

    def _dedupe(self, values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            text = str(value or "").strip()
            if text and text not in result:
                result.append(text)
        return result

    def _stage(self, name: str, status: str, reason: str, **details: object) -> dict[str, object]:
        clean_details = {key: value for key, value in details.items() if value not in (None, "", [])}
        return self.trace_recorder.stage(name, status, reason, **clean_details)
