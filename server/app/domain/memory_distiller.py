from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.db.models import (
    Product,
    SessionMemoryState,
    UserBrandAffinity,
    UserEvent,
    UserMemory,
    UserProductAffinity,
)
from app.services.llm_client import LlmClient


LOGGER = logging.getLogger(__name__)

# Harness boundary:
# LongTermDistiller is an offline/asynchronous Memory subsystem. Its LLM calls
# are not part of the online turn decision chain, must not run inside the
# current chat turn, and must not directly change the current IntentPlan,
# CorrectiveAgent reflection, execution_path, or final_route. Online turns may
# only read already-distilled memory through Orchestrator-approved lookup, and
# that evidence remains a soft preference.

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROFILE_MD_DIR = "data/profiles"
DISTILL_AGE_DAYS = int(os.environ.get("DISTILL_AGE_DAYS", "3"))
MAX_SUMMARIES_PER_USER = 5
MAX_BRANDS_DISTILL = 5
MAX_PRODUCTS_DISTILL = 10

DB_SYSTEM_PROMPT = """你是用户画像抽取器，从聊天摘要和行为数据里抽取稳定偏好。

只输出 JSON，不要 Markdown 代码块，不要解释。
JSON 顶层格式：{"memories": [...]}。
每条 memory 含 key、value、confidence，confidence 是 0 到 1 的数字。

允许的 key：
- 常买品牌 / 关注品牌
- 关注品类
- 已拥有品类
- 价位偏好
- 肤质 / 口味禁忌

不要抽：
- 一次性需求
- 临时话题
- old_memories 里已有且没有新证据的内容

置信度：
- buy 行为 >= 1 次：0.9
- cart_add 或多次 click：0.7
- chat 显式声明：1.0
- chat 和行为互相印证：0.95
"""

MD_SYSTEM_PROMPT = """你是用户画像撰写器。给这个用户写一份持续更新的画像 markdown。

硬约束：
- 保留 old_profile_md 里仍然成立的部分，在末尾整合新信号
- 不要罗列具体商品 ID，用品牌+品类描述
- 总长不超过 1500 字，超出就压缩老内容
- 用 ## 分段，每段 1-3 行
- 输出纯 markdown，不要任何前后说明
"""


class LongTermDistiller:
    def __init__(self, db: Session, llm_client: LlmClient | None = None) -> None:
        self.db = db
        self.llm_client = llm_client or LlmClient(component="LongTermDistiller")
        self.profiles_dir = PROJECT_ROOT / PROFILE_MD_DIR
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    async def run_daily(self) -> dict[str, int]:
        users = self._find_pending_users()
        ok = 0
        failed = 0
        for user_id in users:
            try:
                await self._distill_one(user_id)
                ok += 1
            except Exception as exc:
                LOGGER.warning("distill failed for %s: %s", user_id, exc, exc_info=True)
                failed += 1
        return {"ok": ok, "failed": failed, "total": len(users)}

    def _find_pending_users(self) -> list[str]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=DISTILL_AGE_DAYS)
        rows = self.db.execute(
            select(SessionMemoryState.user_id)
            .where(
                SessionMemoryState.updated_at < cutoff,
                SessionMemoryState.distilled_at.is_(None),
            )
            .distinct()
        ).all()
        return [str(row[0]) for row in rows]

    async def _distill_one(self, user_id: str) -> None:
        inputs = self._gather_inputs(user_id)
        if not inputs["session_summaries"] and not inputs["top_brands"] and not inputs["top_products"]:
            self._mark_sessions_distilled(user_id)
            self.db.commit()
            return

        await self._distill_to_db(user_id, inputs)
        await self._distill_to_md(user_id, inputs)
        self._mark_sessions_distilled(user_id)
        self.db.commit()

    def _gather_inputs(self, user_id: str) -> dict[str, Any]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=DISTILL_AGE_DAYS)
        summaries = self.db.scalars(
            select(SessionMemoryState)
            .where(
                SessionMemoryState.user_id == user_id,
                SessionMemoryState.updated_at < cutoff,
                SessionMemoryState.distilled_at.is_(None),
                SessionMemoryState.session_summary != "",
            )
            .order_by(desc(SessionMemoryState.updated_at))
            .limit(MAX_SUMMARIES_PER_USER)
        ).all()
        top_brands = self.db.scalars(
            select(UserBrandAffinity)
            .where(UserBrandAffinity.user_id == user_id)
            .order_by(desc(UserBrandAffinity.affinity))
            .limit(MAX_BRANDS_DISTILL)
        ).all()
        top_products = self.db.scalars(
            select(UserProductAffinity)
            .where(UserProductAffinity.user_id == user_id)
            .order_by(desc(UserProductAffinity.affinity))
            .limit(MAX_PRODUCTS_DISTILL)
        ).all()
        product_meta = self._fetch_product_meta([row.product_id for row in top_products])
        old_memories = self.db.scalars(select(UserMemory).where(UserMemory.user_id == user_id)).all()
        return {
            "session_summaries": [
                {"summary": row.session_summary, "session_id": row.session_id}
                for row in summaries
            ],
            "top_brands": [
                {
                    "brand": row.brand,
                    "click_count": row.click_count,
                    "cart_count": row.cart_count,
                    "buy_count": row.buy_count,
                    "affinity": round(float(row.affinity), 3),
                }
                for row in top_brands
            ],
            "top_products": [
                {
                    "name": product_meta.get(row.product_id, {}).get("name", row.product_id),
                    "brand": product_meta.get(row.product_id, {}).get("brand", ""),
                    "sub_category": product_meta.get(row.product_id, {}).get("sub_category", ""),
                    "click": row.click_count,
                    "cart": row.cart_count,
                    "buy": row.buy_count,
                    "owned": row.owned,
                }
                for row in top_products
            ],
            "owned_categories": sorted(self._load_owned_subcats(user_id)),
            "price_stats": self._load_price_stats(user_id),
            "old_memories": [
                {
                    "key": row.key,
                    "value": row.value,
                    "source": row.source,
                    "confidence": float(row.confidence),
                }
                for row in old_memories
            ],
            "old_profile_md": self._load_profile_md(user_id),
        }

    def _load_owned_subcats(self, user_id: str) -> set[str]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=90)
        rows = self.db.execute(
            select(Product.sub_category)
            .join(UserEvent, UserEvent.product_id == Product.product_id)
            .where(
                UserEvent.user_id == user_id,
                UserEvent.event_type == "buy",
                UserEvent.created_at >= cutoff,
            )
            .distinct()
        ).all()
        return {str(row[0]) for row in rows if row[0]}

    def _load_price_stats(self, user_id: str) -> dict[str, float] | None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=90)
        rows = self.db.execute(
            select(Product.price)
            .join(UserEvent, UserEvent.product_id == Product.product_id)
            .where(
                UserEvent.user_id == user_id,
                UserEvent.event_type == "buy",
                UserEvent.created_at >= cutoff,
            )
        ).all()
        prices = sorted(float(row[0]) for row in rows if row[0] is not None)
        if not prices:
            return None
        return {
            "n_buys": float(len(prices)),
            "median": prices[len(prices) // 2],
            "min": prices[0],
            "max": prices[-1],
        }

    def _fetch_product_meta(self, product_ids: list[str]) -> dict[str, dict[str, str]]:
        if not product_ids:
            return {}
        rows = self.db.scalars(select(Product).where(Product.product_id.in_(product_ids))).all()
        return {
            row.product_id: {
                "name": row.name,
                "brand": row.brand,
                "sub_category": row.sub_category or "",
            }
            for row in rows
        }

    def _load_profile_md(self, user_id: str) -> str:
        path = self.profile_path(user_id)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")[:4000]

    async def _distill_to_db(self, user_id: str, inputs: dict[str, Any]) -> None:
        if not self.llm_client.is_configured():
            LOGGER.info("LLM not configured, skip DB distill for %s", user_id)
            return
        content = await self.llm_client.generate(
            DB_SYSTEM_PROMPT,
            json.dumps(
                {
                    "session_summaries": inputs["session_summaries"],
                    "top_brands": inputs["top_brands"],
                    "top_products": inputs["top_products"],
                    "owned_categories": inputs["owned_categories"],
                    "price_stats": inputs["price_stats"],
                    "old_memories": inputs["old_memories"],
                },
                ensure_ascii=False,
            ),
            response_format={"type": "json_object"},
            operation="long_term_distiller.db_memory",
        )
        if not content:
            return
        for memory in self._parse_db_output(content):
            self._upsert_distilled_memory(user_id, memory)

    def _upsert_distilled_memory(self, user_id: str, memory: dict[str, Any]) -> None:
        key = str(memory.get("key") or "").strip()
        value = str(memory.get("value") or "").strip()
        if not key or not value:
            return
        confidence = Decimal(str(round(float(memory.get("confidence") or 0.7), 2)))
        existing = self.db.scalar(
            select(UserMemory).where(
                UserMemory.user_id == user_id,
                UserMemory.key == key,
                UserMemory.value == value,
            )
        )
        if existing is not None:
            if existing.source == "explicit":
                existing.confidence = max(existing.confidence, confidence)
                return
            existing.confidence = confidence
            existing.source = "distilled"
            return
        self.db.add(
            UserMemory(
                user_id=user_id,
                memory_type=self._memory_type_from_key(key),
                key=key,
                value=value,
                confidence=confidence,
                source="distilled",
            )
        )

    async def _distill_to_md(self, user_id: str, inputs: dict[str, Any]) -> None:
        text: str | None = None
        if self.llm_client.is_configured():
            content = await self.llm_client.generate(
                MD_SYSTEM_PROMPT,
                json.dumps(
                    {
                        "old_profile_md": inputs["old_profile_md"],
                        "session_summaries": inputs["session_summaries"],
                        "top_brands": inputs["top_brands"],
                        "top_products": inputs["top_products"],
                        "owned_categories": inputs["owned_categories"],
                        "price_stats": inputs["price_stats"],
                    },
                    ensure_ascii=False,
                ),
                operation="long_term_distiller.profile_md",
            )
            if content:
                text = self._strip_fence(content)
        if not text:
            text = self._render_fallback_md(user_id, inputs)
        self.profile_path(user_id).write_text(text[:4000], encoding="utf-8")
        self.profile_path(user_id, suffix=".json").write_text(
            json.dumps(
                {
                    "user_id": user_id,
                    "last_distilled_at": datetime.now(timezone.utc).isoformat(),
                    "source_session_ids": [row["session_id"] for row in inputs["session_summaries"]],
                    "size_chars": len(text),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _mark_sessions_distilled(self, user_id: str) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=DISTILL_AGE_DAYS)
        now = datetime.now(timezone.utc)
        rows = self.db.scalars(
            select(SessionMemoryState).where(
                SessionMemoryState.user_id == user_id,
                SessionMemoryState.updated_at < cutoff,
                SessionMemoryState.distilled_at.is_(None),
            )
        ).all()
        for row in rows:
            row.distilled_at = now

    def profile_path(self, user_id: str, suffix: str = ".md") -> Path:
        safe_user_id = re.sub(r"[^A-Za-z0-9_.-]", "_", user_id)
        return self.profiles_dir / f"{safe_user_id}{suffix}"

    @staticmethod
    def _render_fallback_md(user_id: str, inputs: dict[str, Any]) -> str:
        lines = [f"# 用户画像 (id={user_id})", ""]
        if inputs.get("old_profile_md"):
            lines.extend([str(inputs["old_profile_md"]).strip(), "", "---", ""])
        if inputs.get("top_brands"):
            lines.append("## 关注品牌")
            for item in inputs["top_brands"][:5]:
                lines.append(f"- {item['brand']}: affinity={item['affinity']}")
            lines.append("")
        if inputs.get("top_products"):
            lines.append("## 关注商品")
            for item in inputs["top_products"][:5]:
                owned = "，已购" if item.get("owned") else ""
                lines.append(f"- {item['name']}（{item['brand']}，{item['sub_category']}{owned}）")
            lines.append("")
        if inputs.get("owned_categories"):
            lines.append("## 已拥有品类")
            for category in inputs["owned_categories"]:
                lines.append(f"- {category}")
            lines.append("")
        if inputs.get("price_stats"):
            price = inputs["price_stats"]
            lines.append("## 价位偏好")
            lines.append(f"- 中位数 ¥{price['median']:.0f}，区间 ¥{price['min']:.0f}~¥{price['max']:.0f}")
            lines.append("")
        if inputs.get("session_summaries"):
            lines.append("## 近期会话主题")
            for item in inputs["session_summaries"][:3]:
                summary = str(item.get("summary") or "").strip()
                if summary:
                    lines.append(f"- {summary[:200]}")
        return "\n".join(lines).strip()

    @staticmethod
    def _parse_db_output(content: str) -> list[dict[str, Any]]:
        text = LongTermDistiller._strip_fence(content)
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return []
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return []
        result = []
        for item in data.get("memories") or []:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key") or "").strip()
            value = str(item.get("value") or "").strip()
            if not key or not value:
                continue
            try:
                confidence = float(item.get("confidence") or 0.7)
            except (TypeError, ValueError):
                confidence = 0.7
            result.append({"key": key, "value": value, "confidence": max(0.0, min(1.0, confidence))})
        return result

    @staticmethod
    def _strip_fence(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json|markdown|md)?", "", stripped).strip()
            stripped = re.sub(r"```$", "", stripped).strip()
        return stripped

    @staticmethod
    def _memory_type_from_key(key: str) -> str:
        if key in {"禁忌", "口味禁忌"}:
            return "avoid"
        if key == "已拥有品类":
            return "owned"
        return "preference"
