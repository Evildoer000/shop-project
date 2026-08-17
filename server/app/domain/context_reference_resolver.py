from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


class ContextReferenceResolver:
    """Build and resolve trusted product references for one user session."""

    _LATEST_MARKERS = (
        "刚才",
        "刚刚",
        "前面",
        "之前",
        "这些",
        "它们",
        "上述",
        "上面",
        "推荐的",
        "提到的",
    )
    _PREVIOUS_TURN_MARKERS = ("上一轮", "上轮")
    _ORDINALS = {
        "第一款": 1,
        "第一个": 1,
        "第一件": 1,
        "第二款": 2,
        "第二个": 2,
        "第二件": 2,
        "第三款": 3,
        "第三个": 3,
        "第三件": 3,
        "第四款": 4,
        "第四个": 4,
        "第四件": 4,
        "第五款": 5,
        "第五个": 5,
        "第五件": 5,
    }

    def build_context(
        self,
        ledger: list[dict[str, Any]],
        *,
        query: str = "",
    ) -> dict[str, Any]:
        entries = self._normalize_ledger(ledger)
        groups: dict[str, dict[str, Any]] = {}
        by_turn: dict[int, list[dict[str, Any]]] = defaultdict(list)
        by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entry in entries:
            by_turn[int(entry["turn_id"])].append(entry)
            category = str(entry.get("category") or "").strip()
            if category:
                by_category[category].append(entry)

        for turn_id, products in by_turn.items():
            self._add_group(
                groups,
                f"turn:{turn_id}",
                f"会话第 {turn_id} 轮展示的商品",
                products,
                source="conversation_turn",
            )

        if entries:
            latest_product_turn = max(by_turn)
            latest_products = by_turn[latest_product_turn]
            self._add_group(
                groups,
                "latest_product_turn",
                "最近一次展示商品的会话轮次",
                latest_products,
                source="latest_product_turn",
            )
            for index, product in enumerate(latest_products[:10], start=1):
                self._add_group(
                    groups,
                    f"latest:item:{index}",
                    f"最近一次展示商品中的第 {index} 个",
                    [product],
                    source="latest_product_ordinal",
                )
            if len(latest_products) >= 2:
                self._add_group(
                    groups,
                    "latest:first_two",
                    "最近一次展示商品中的前两个",
                    latest_products[:2],
                    source="latest_product_ordinal",
                )
            self._add_group(
                groups,
                "session:all_products",
                "本会话历史中展示过的全部商品",
                entries,
                source="session_product_history",
            )

        previous_turn_products = [
            entry for entry in entries if bool(entry.get("is_latest_conversation_turn"))
        ]
        if previous_turn_products:
            self._add_group(
                groups,
                "previous_turn",
                "紧邻当前请求的上一轮展示商品",
                previous_turn_products,
                source="previous_conversation_turn",
            )

        for category, products in by_category.items():
            self._add_group(
                groups,
                f"category:{category}",
                f"本会话历史中的 {category} 商品",
                products,
                source="session_category",
            )

        for entry in entries:
            product_id = str(entry["product_id"])
            self._add_group(
                groups,
                f"product:{product_id}",
                f"历史商品：{entry.get('name') or product_id}",
                [entry],
                source="explicit_product",
            )

        matched_products = self._name_matches(query, entries)
        if matched_products:
            self._add_group(
                groups,
                "query_named_products",
                "当前问题中明确点名的历史商品",
                matched_products,
                source="explicit_name_match",
            )

        payload = {
            "groups": list(groups.values()),
            "suggested_keys": self.suggest_keys(query, groups),
            "trusted_product_ids": self._dedupe(
                [str(entry["product_id"]) for entry in entries]
            ),
        }
        return payload

    def resolve_keys(
        self,
        reference_context: dict[str, Any] | None,
        keys: list[str],
    ) -> tuple[list[str], list[str]]:
        groups = self._group_map(reference_context)
        product_ids: list[str] = []
        unknown_keys: list[str] = []
        for key in self._dedupe(keys):
            group = groups.get(key)
            if group is None:
                unknown_keys.append(key)
                continue
            product_ids.extend(str(item) for item in group.get("product_ids") or [])
        return self._dedupe(product_ids), unknown_keys

    def suggest_keys(
        self,
        text: str,
        groups_or_context: dict[str, Any] | None,
    ) -> list[str]:
        groups = (
            groups_or_context
            if groups_or_context
            and all(isinstance(value, dict) for value in groups_or_context.values())
            and "groups" not in groups_or_context
            else self._group_map(groups_or_context)
        )
        normalized = self._normalized(text)
        suggestions: list[str] = []
        if "query_named_products" in groups:
            suggestions.append("query_named_products")
        for marker in self._PREVIOUS_TURN_MARKERS:
            if marker in normalized and "previous_turn" in groups:
                suggestions.append("previous_turn")
                break
        if "前两个" in normalized and "latest:first_two" in groups:
            suggestions.append("latest:first_two")
        for marker, index in self._ORDINALS.items():
            key = f"latest:item:{index}"
            if marker in normalized and key in groups:
                suggestions.append(key)
        if any(marker in normalized for marker in self._LATEST_MARKERS):
            if "latest_product_turn" in groups:
                suggestions.append("latest_product_turn")
        for key, group in groups.items():
            if not key.startswith("category:"):
                continue
            category = key.removeprefix("category:")
            if category and category in normalized and any(
                marker in normalized for marker in self._LATEST_MARKERS
            ):
                suggestions.append(key)
        return self._dedupe(suggestions)

    def _normalize_ledger(
        self,
        ledger: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        seen: set[tuple[int, str]] = set()
        for raw in ledger:
            if not isinstance(raw, dict):
                continue
            product_id = str(raw.get("product_id") or "").strip()
            try:
                turn_id = int(raw.get("turn_id"))
            except (TypeError, ValueError):
                continue
            if not product_id or (turn_id, product_id) in seen:
                continue
            seen.add((turn_id, product_id))
            try:
                display_order = int(raw.get("display_order") or 0)
            except (TypeError, ValueError):
                display_order = 0
            normalized.append(
                {
                    "turn_id": turn_id,
                    "product_id": product_id,
                    "name": str(raw.get("name") or "").strip(),
                    "display_order": display_order,
                    "role": str(raw.get("role") or "selected").strip(),
                    "slot_id": str(raw.get("slot_id") or "").strip(),
                    "category": str(raw.get("category") or "").strip(),
                    "is_latest_conversation_turn": bool(
                        raw.get("is_latest_conversation_turn")
                    ),
                }
            )
        normalized.sort(key=lambda item: (item["turn_id"], item["display_order"]))
        return normalized

    def _add_group(
        self,
        groups: dict[str, dict[str, Any]],
        key: str,
        label: str,
        products: list[dict[str, Any]],
        *,
        source: str,
    ) -> None:
        compact_products: list[dict[str, Any]] = []
        product_ids: list[str] = []
        for product in products:
            product_id = str(product.get("product_id") or "").strip()
            if not product_id or product_id in product_ids:
                continue
            product_ids.append(product_id)
            compact_products.append(
                {
                    field: product.get(field)
                    for field in (
                        "turn_id",
                        "product_id",
                        "name",
                        "display_order",
                        "role",
                        "slot_id",
                        "category",
                    )
                    if product.get(field) not in (None, "")
                }
            )
        if not product_ids:
            return
        groups[key] = {
            "key": key,
            "label": label,
            "source": source,
            "product_ids": product_ids,
            "products": compact_products,
        }

    def _name_matches(
        self,
        query: str,
        entries: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        normalized_query = self._normalized(query)
        if not normalized_query:
            return []
        matches: list[dict[str, Any]] = []
        for entry in reversed(entries):
            name = self._normalized(str(entry.get("name") or ""))
            if len(name) >= 2 and name in normalized_query:
                matches.append(entry)
        matches.reverse()
        return matches

    def _group_map(
        self,
        reference_context: dict[str, Any] | None,
    ) -> dict[str, dict[str, Any]]:
        if not isinstance(reference_context, dict):
            return {}
        raw_groups = reference_context.get("groups")
        if not isinstance(raw_groups, list):
            return {}
        return {
            str(item.get("key")): item
            for item in raw_groups
            if isinstance(item, dict) and item.get("key")
        }

    def _normalized(self, value: str) -> str:
        return re.sub(r"[\W_]+", "", str(value or "").lower(), flags=re.UNICODE)

    def _dedupe(self, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value for value in values if value))
