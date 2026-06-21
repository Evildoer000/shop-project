from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Base, ConversationTurn, SessionMemoryState, UserMemory
from app.services.llm_client import LlmClient


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROFILE_MD_DIR = "data/profiles"
RECENT_TURNS_LIMIT = 12
SUMMARY_TRIGGER_PENDING_TURNS = 6
SUMMARY_OVERLAP_TURNS = 2
SUMMARY_MAX_TOKENS = 500
ASSISTANT_MESSAGE_MAX_CHARS = 1600
TURN_PRODUCT_IDS_LIMIT = 30
TURN_SELECTED_PRODUCTS_LIMIT = 30


@dataclass
class ConversationTurnView:
    turn_id: int
    user_message: str
    assistant_message: str
    route: str
    product_ids: list[str] = field(default_factory=list)
    rewrite_summary: dict[str, Any] = field(default_factory=dict)
    trace_summary: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: ConversationTurn) -> "ConversationTurnView":
        return cls(
            turn_id=row.turn_id,
            user_message=row.user_message,
            assistant_message=row.assistant_message,
            route=row.route,
            product_ids=list(row.product_ids or []),
            rewrite_summary=dict(row.rewrite_summary or {}),
            trace_summary=dict(row.trace_summary or {}),
        )

    def model_dump(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "user_message": self.user_message,
            "assistant_message": self.assistant_message,
            "route": self.route,
            "product_ids": self.product_ids,
            "rewrite_summary": self.rewrite_summary,
            "trace_summary": self.trace_summary,
        }

    def compact(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "user": self.user_message,
            "assistant": self.assistant_message[:600],
            "route": self.route,
            "product_ids": self.product_ids[:TURN_PRODUCT_IDS_LIMIT],
            "selected_products": self._compact_selected_products(),
            "rewrite": {
                key: self.rewrite_summary.get(key)
                for key in [
                    "action",
                    "product_type",
                    "categories",
                    "vector_query",
                    "keyword_query",
                    "negative_terms",
                    "preferences",
                    "budget_max",
                    "budget_scope",
                    "need_slots",
                    "slot_plan_reason",
                ]
                if self.rewrite_summary.get(key) not in (None, "", [])
            },
        }

    def _compact_selected_products(self) -> list[dict[str, Any]]:
        selected_products = self.trace_summary.get("selected_products")
        if not isinstance(selected_products, list):
            return []
        compacted = []
        for item in selected_products[:TURN_SELECTED_PRODUCTS_LIMIT]:
            if not isinstance(item, dict):
                continue
            compacted.append(
                {
                    key: item.get(key)
                    for key in ["product_id", "name", "brand", "category", "sub_category", "price", "rating"]
                    if item.get(key) not in (None, "", [])
                }
            )
        return compacted


@dataclass
class ConversationContext:
    session_summary: str = ""
    pending_summary_turns: list[ConversationTurnView] = field(default_factory=list)
    recent_turns: list[ConversationTurnView] = field(default_factory=list)
    long_term_profile: list[str] = field(default_factory=list)
    long_term_narrative: str = ""

    def to_rewrite_context(
        self,
        *,
        include_long_term: bool = True,
        profile_memory: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        # recent_turns 是按 turn_id 升序 (旧→新). 给每条加 n_turns_ago, 让 LLM 看出"上一轮"vs"早期".
        # n_turns_ago=0 → 上一轮 (recent_turns 末尾), 数字越大越早.
        recent_compact = []
        recent_count = len(self.recent_turns)
        for index, turn in enumerate(self.recent_turns):
            item = turn.compact()
            item["n_turns_ago"] = recent_count - 1 - index
            recent_compact.append(item)

        # pending_summary_turns 在 recent 之前 → n_turns_ago 必然更大
        pending_compact = []
        pending_count = len(self.pending_summary_turns)
        for index, turn in enumerate(self.pending_summary_turns):
            item = turn.compact()
            item["n_turns_ago"] = recent_count + (pending_count - 1 - index)
            pending_compact.append(item)

        payload = {
            "session_summary": self.session_summary,
            "pending_summary_turns": pending_compact,
            "recent_turns": recent_compact,
            "priority": [
                "current_query",
                "recent_turns[n_turns_ago=0]",  # 上一轮: 指代"那个/上次/刚才"应优先绑这里
                "recent_turns[n_turns_ago>=1]",  # 更早的轮: 仅省略式追问/续问/延续话题时启用
                "pending_summary_turns",
                "session_summary",
            ],
        }
        if include_long_term:
            payload["long_term_profile"] = self.long_term_profile
            payload["priority"].append("long_term_profile")
        if profile_memory is not None:
            payload["profile_memory"] = profile_memory
            payload["priority"].append("profile_memory")
        return payload

    def trace_payload(self) -> dict[str, Any]:
        return {
            "session_summary": self.session_summary,
            "pending_turn_count": len(self.pending_summary_turns),
            "recent_turn_count": len(self.recent_turns),
            "long_term_profile": self.long_term_profile,
            "long_term_narrative_chars": len(self.long_term_narrative),
            "pending_summary_turns": [turn.compact() for turn in self.pending_summary_turns],
            "recent_turns": [turn.compact() for turn in self.recent_turns],
        }


class SessionSummarizer:
    def __init__(self, llm_client: LlmClient | None = None) -> None:
        self.llm_client = llm_client or LlmClient(component="SessionSummarizer")

    async def summarize(
        self,
        old_summary: str,
        pending_turns: list[ConversationTurnView],
        overlap_turns: list[ConversationTurnView],
    ) -> str | None:
        if not pending_turns:
            return None
        system_prompt = (
            "你是电商导购会话摘要器。只输出精简中文 Markdown，不要输出解释。\n"
            "摘要上限约 500 tokens，越短越好。\n"
            "只保留会话目标、当前商品范围、已确认约束、排除项、未完成追问。\n"
            "不要流水账复述每轮对话，不要记录一次性噪声。\n"
            "overlap_turns 只用于理解衔接，除非改变会话主题，不要重复写入仍在 recent window 的细节。\n\n"
            "## 必须保留的结构化锚点 (用于跨摘要边界的指代追问)\n"
            "- 每个被摘要的轮次, 如果该轮 product_ids 非空, 必须在摘要里保留 turn_id 和 product_ids 列表.\n"
            "- 推荐写法: 在话题段落末尾用方括号附上, 例如：\n"
            "    跑鞋 (轻量通勤风格) [t1: p_run_a, p_run_b, p_run_c; t2: p_run_d]\n"
            "    防晒 (敏感肌, 不要酒精) [t4: p_sun_a, p_sun_b]\n"
            "- 如果 turn 是 image 轮 (用户上传过图片), 在 turn_id 后加 (img), 例如 [t2(img): p_run_d].\n"
            "- 这些锚点不计入 500 token 上限的'内容部分', 它们是后续轮次能正确指代'最早那双''第一次问的'等表达的唯一来源.\n"
            "- 不要省略 product_ids; 即使你认为该轮不重要, 也要保留 ID 串, 以备后续指代."
        )
        user_prompt = json.dumps(
            {
                "old_session_summary": old_summary,
                "pending_summary_turns": [turn.compact() for turn in pending_turns],
                "overlap_recent_turns": [turn.compact() for turn in overlap_turns],
                "output_contract": "500 tokens 内的滚动 session_summary",
            },
            ensure_ascii=False,
        )
        content = await self.llm_client.generate(
            system_prompt,
            user_prompt,
            operation="session_summarizer.summarize",
        )
        if not content:
            return None
        return self._trim_summary(content.strip())

    def _trim_summary(self, text: str) -> str:
        words = text.split()
        if len(words) > SUMMARY_MAX_TOKENS:
            return " ".join(words[:SUMMARY_MAX_TOKENS])
        return text[:2400]


class MemoryManager:
    _session_tables_checked = False

    def __init__(self, db: Session) -> None:
        self.db = db
        self._ensure_session_tables()

    def _ensure_session_tables(self) -> None:
        if MemoryManager._session_tables_checked:
            return
        Base.metadata.create_all(
            bind=self.db.get_bind(),
            tables=[ConversationTurn.__table__, SessionMemoryState.__table__],
        )
        MemoryManager._session_tables_checked = True

    def load_memory(self, user_id: str) -> list[str]:
        rows = self.db.scalars(select(UserMemory).where(UserMemory.user_id == user_id)).all()
        return [f"{row.key}:{row.value}" for row in rows]

    def build_context(self, user_id: str, session_id: str) -> ConversationContext:
        state = self._get_state(user_id, session_id)
        turns = self._load_turns(user_id, session_id)
        recent = turns[-RECENT_TURNS_LIMIT:]
        recent_start_id = recent[0].turn_id if recent else None
        summarized_through = state.summarized_through_turn_id if state else 0
        pending = [
            turn
            for turn in turns
            if turn.turn_id > summarized_through
            and (recent_start_id is None or turn.turn_id < recent_start_id)
        ]
        profile = self.load_memory(user_id)
        return ConversationContext(
            session_summary=(state.session_summary if state else "") or "",
            pending_summary_turns=pending,
            recent_turns=recent,
            long_term_profile=profile,
            long_term_narrative=self.load_profile_narrative(user_id),
        )

    def load_profile_narrative(self, user_id: str) -> str:
        path = self._profile_md_path(user_id)
        if not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8")[:4000]
        except Exception as exc:
            LOGGER.warning("read profile narrative failed for %s: %s", user_id, exc)
            return ""

    def append_turn(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        assistant_message: str,
        route: str,
        product_ids: list[str],
        rewrite_summary: dict[str, Any],
        trace_summary: dict[str, Any],
    ) -> ConversationTurn:
        row = ConversationTurn(
            user_id=user_id,
            session_id=session_id,
            user_message=user_message,
            assistant_message=self._truncate(assistant_message, ASSISTANT_MESSAGE_MAX_CHARS),
            route=route,
            product_ids=product_ids[:20],
            rewrite_summary=self._json_safe(rewrite_summary),
            trace_summary=self._json_safe(trace_summary),
        )
        self.db.add(row)
        self.db.commit()
        self.db.refresh(row)
        return row

    async def summarize_if_needed(
        self,
        user_id: str,
        session_id: str,
        summarizer: SessionSummarizer | None = None,
    ) -> dict[str, Any]:
        context = self.build_context(user_id, session_id)
        if len(context.pending_summary_turns) < SUMMARY_TRIGGER_PENDING_TURNS:
            return {
                "updated": False,
                "reason": "pending_below_threshold",
                "pending_count": len(context.pending_summary_turns),
            }
        state = self._get_or_create_state(user_id, session_id)
        overlap = context.recent_turns[:SUMMARY_OVERLAP_TURNS]
        new_summary = await (summarizer or SessionSummarizer()).summarize(
            state.session_summary or "",
            context.pending_summary_turns,
            overlap,
        )
        if not new_summary:
            return {
                "updated": False,
                "reason": "summarizer_failed",
                "pending_count": len(context.pending_summary_turns),
            }
        state.session_summary = new_summary
        state.summarized_through_turn_id = max(turn.turn_id for turn in context.pending_summary_turns)
        self.db.commit()
        return {
            "updated": True,
            "pending_count": len(context.pending_summary_turns),
            "summarized_through_turn_id": state.summarized_through_turn_id,
            "summary": new_summary,
        }

    def upsert_explicit_preferences(self, user_id: str, query: str) -> list[str]:
        created: list[str] = []
        explicit_pairs = {
            "肤质": ["油皮", "干皮", "混合皮", "混油皮", "敏感肌"],
            "偏好": ["清爽", "保湿", "轻量", "轻便", "通勤", "户外", "无糖", "不太甜"],
            "禁忌": ["乳糖不耐", "坚果过敏"],
        }
        for key, values in explicit_pairs.items():
            for value in values:
                if value in query and self._is_stable_memory_statement(query, key, value):
                    exists = self.db.scalar(
                        select(UserMemory).where(
                            UserMemory.user_id == user_id,
                            UserMemory.key == key,
                            UserMemory.value == value,
                        )
                    )
                    if exists is None:
                        self.db.add(
                            UserMemory(
                                user_id=user_id,
                                memory_type="preference" if key != "禁忌" else "avoid",
                                key=key,
                                value=value,
                                source="explicit",
                            )
                        )
                        created.append(f"{key}:{value}")
        if created:
            self.db.commit()
        return created

    def _get_or_create_state(self, user_id: str, session_id: str) -> SessionMemoryState:
        row = self._get_state(user_id, session_id)
        if row is not None:
            return row
        row = SessionMemoryState(user_id=user_id, session_id=session_id)
        self.db.add(row)
        self.db.commit()
        self.db.refresh(row)
        return row

    def _get_state(self, user_id: str, session_id: str) -> SessionMemoryState | None:
        return self.db.scalar(
            select(SessionMemoryState).where(
                SessionMemoryState.user_id == user_id,
                SessionMemoryState.session_id == session_id,
            )
        )

    def _load_turns(self, user_id: str, session_id: str) -> list[ConversationTurnView]:
        rows = self.db.scalars(
            select(ConversationTurn)
            .where(ConversationTurn.user_id == user_id, ConversationTurn.session_id == session_id)
            .order_by(ConversationTurn.turn_id.asc())
        ).all()
        return [ConversationTurnView.from_row(row) for row in rows]

    def _is_stable_memory_statement(self, query: str, key: str, value: str) -> bool:
        stable_markers = ["我是", "我属于", "我喜欢", "我不喜欢", "我一般", "我通常", "以后", "记住", "别给我"]
        if any(marker in query for marker in stable_markers):
            return True
        if key == "偏好" and value in {"通勤", "户外", "轻便", "无糖", "不太甜"}:
            return any(marker in query for marker in ["喜欢", "偏好", "一般", "通常", "以后", "记住"])
        return False

    def _truncate(self, text: str, max_chars: int) -> str:
        text = " ".join((text or "").split())
        return text[:max_chars]

    def _json_safe(self, value: dict[str, Any]) -> dict[str, Any]:
        return json.loads(json.dumps(value or {}, ensure_ascii=False, default=str))

    def _profile_md_path(self, user_id: str) -> Path:
        safe_user_id = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in user_id)
        return PROJECT_ROOT / PROFILE_MD_DIR / f"{safe_user_id}.md"
