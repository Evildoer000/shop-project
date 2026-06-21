from __future__ import annotations

import re

from app.schemas import IntentPlan, QueryBudget, QueryPlan, QueryRetrievalStrategy


class RetrievalPlanBuilder:
    """Build a retrieval plan from the Orchestrator-approved IntentPlan."""

    TOP_LEVEL_CATEGORIES = {"美妆护肤", "数码电子", "服饰运动", "食品饮料"}
    OUT_OF_CATALOG_PRODUCT_TERMS = ("电脑主机", "台式电脑", "台式机", "主机", "手机支架")
    CATEGORY_KEYWORDS = {
        "精华": ["精华", "精华液", "安瓶", "原液"],
        "化妆水": ["化妆水", "爽肤水", "柔肤水", "toner"],
        "防晒": ["防晒", "防晒霜", "防晒乳", "不泛白", "隔离"],
        "面霜": ["面霜", "修护霜", "保湿霜", "乳液", "润肤霜"],
        "洁面": ["洗面奶", "洁面", "洁面乳"],
        "蜜粉": ["蜜粉", "粉饼", "散粉", "定妆粉"],
        "唇釉": ["唇釉", "口红", "唇膏", "唇彩"],
        "眼霜": ["眼霜", "眼胶", "眼精华"],
        "卸妆": ["卸妆", "卸妆水", "卸妆油", "卸妆乳", "化妆棉"],
        "面膜": ["面膜", "补水面膜", "晒后", "面贴膜", "涂抹面膜"],
        "眉笔": ["眉笔", "眉粉", "染眉膏"],
        "粉底液": ["粉底", "粉底液", "粉底霜", "气垫", "bb霜", "隔离霜"],
        "智能手机": ["手机", "iphone", "安卓", "智能手机"],
        "笔记本电脑": ["笔记本", "电脑", "laptop", "笔记本电脑"],
        "平板电脑": ["平板", "pad", "ipad", "平板电脑"],
        "真无线耳机": ["耳机", "蓝牙", "降噪", "真无线", "tws"],
        "短袖T恤": ["短袖", "t恤", "T恤"],
        "速干T恤": ["速干", "运动T恤"],
        "运动短裤": ["短裤", "运动短裤"],
        "运动长裤": ["运动裤", "长裤", "运动长裤"],
        "卫衣": ["卫衣", "套头衫", "连帽衫"],
        "跑步鞋": ["跑鞋", "跑步鞋", "慢跑"],
        "篮球鞋": ["篮球鞋", "球鞋"],
        "徒步鞋": ["徒步", "户外鞋", "登山鞋"],
        "瑜伽裤": ["瑜伽", "瑜伽裤", "打底裤"],
        "户外裤": ["户外裤", "户外运动裤", "登山裤"],
        "背包": ["背包", "书包", "双肩包", "旅行包"],
        "帽子": ["帽子", "遮阳帽", "棒球帽", "鸭舌帽"],
        "咖啡": ["咖啡", "拿铁", "美式", "速溶"],
        "茶饮": ["茶", "茶饮", "茶叶", "绿茶", "红茶", "乌龙茶", "花茶"],
        "碳酸饮料": ["可乐", "汽水", "苏打水", "碳酸饮料", "雪碧"],
        "功能饮料": ["功能饮料", "运动饮料", "能量饮料", "红牛", "电解质"],
        "牛奶": ["牛奶", "纯牛奶", "鲜奶", "脱脂奶"],
        "酸奶": ["酸奶", "酸奶饮品"],
        "坚果/零食": ["坚果", "零食", "薯片", "饼干", "巧克力"],
        "方便食品": ["方便面", "速食", "泡面", "自热", "方便食品"],
        "调味品": ["调味品", "酱油", "醋", "味精", "料酒", "蚝油", "食用油", "生抽", "老抽", "鸡精"],
    }
    TOP_CATEGORY_HINTS = {
        "服饰运动": ["鞋", "衣", "裤", "帽", "包", "穿", "运动", "户外", "露营", "服装"],
        "食品饮料": ["食品", "食物", "吃", "零食", "饮料", "咖啡", "茶", "奶", "速食", "泡面"],
        "数码电子": ["数码", "手机", "电脑", "主机", "台式", "耳机", "平板", "支架", "充电", "蓝牙"],
        "美妆护肤": ["护肤", "美妆", "化妆", "防晒", "洁面", "面膜", "粉底", "唇"],
    }
    PREFERENCE_KEYWORDS = [
        "油皮",
        "干皮",
        "敏感肌",
        "清爽",
        "保湿",
        "轻量",
        "通勤",
        "户外",
        "夏天",
        "不闷",
        "不泛白",
        "性价比",
        "耐用",
        "降噪",
        "续航",
        "性能",
    ]
    EXCLUDE_PATTERNS = [
        r"不要([^，。?.]+)",
        r"不想要([^，。?.]+)",
        r"避开([^，。?.]+)",
        r"避免([^，。?.]+)",
        r"排除([^，。?.]+)",
        r"不选([^，。?.]+)",
        r"不能有([^，。?.]+)",
    ]
    SCENE_KEYWORDS = ["三亚", "旅行", "高温", "户外", "通勤", "运动", "夏天", "露营", "跑步"]

    def plan(self, intent_plan: IntentPlan) -> QueryPlan:
        source_text = self._source_text(intent_plan)
        categories = self._categories_from_text(source_text)
        budget = QueryBudget(min=intent_plan.budget_min, max=intent_plan.budget_max)
        if budget.max is None:
            budget.max = self._extract_budget_max(source_text)
        exclude = self._unique(self._extract_excludes(self._exclude_source_text(intent_plan)))
        preferences = [
            word
            for word in self.PREFERENCE_KEYWORDS
            if word in source_text and word not in exclude
        ]
        scene = [word for word in self.SCENE_KEYWORDS if word in source_text]

        is_comparison = "对比" in intent_plan.original_query or "比较" in intent_plan.original_query
        plan = QueryPlan(
            intent="comparison" if is_comparison else "recommendation",
            categories=categories,
            scene=scene,
            budget=budget,
            preferences=self._unique(preferences),
            exclude=exclude,
            compare_targets=(
                self._extract_compare_targets(
                    intent_plan.original_query,
                    categories,
                    self._out_of_catalog_targets(source_text),
                )
                if is_comparison
                else []
            ),
            cart_action=self._extract_cart_action(intent_plan.original_query),
            need_clarification=intent_plan.plan_type == "clarify",
            clarification_question=(
                "你想优先找哪一类商品？比如防晒、洗面奶、耳机、外套或跑鞋。"
                if intent_plan.plan_type == "clarify"
                else None
            ),
        )
        plan.retrieval_strategy = self._retrieval_strategy(plan)
        plan.filters = self._filters(plan)
        return plan

    def _source_text(self, intent_plan: IntentPlan) -> str:
        return self._compact(
            " ".join(
                part
                for part in [
                    intent_plan.original_query,
                    intent_plan.keyword_query,
                    intent_plan.vector_query,
                    *[slot.goal for slot in intent_plan.need_slots],
                    *[slot.product_type for slot in intent_plan.need_slots],
                    *[slot.query for slot in intent_plan.need_slots],
                ]
                if part
            )
        )

    def _exclude_source_text(self, intent_plan: IntentPlan) -> str:
        parts = [
            intent_plan.original_query,
            intent_plan.keyword_query,
            intent_plan.vector_query,
            *[slot.goal for slot in intent_plan.need_slots],
            *[slot.product_type for slot in intent_plan.need_slots],
            *[slot.query for slot in intent_plan.need_slots],
        ]
        return "。".join(part.strip() for part in parts if part and part.strip())

    def _categories_from_text(self, text: str) -> list[str]:
        matches: list[str] = []
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            if category in text or any(keyword and keyword in text for keyword in keywords):
                matches.append(category)
        for category, hints in self.TOP_CATEGORY_HINTS.items():
            if category in text or any(hint and hint in text for hint in hints):
                matches.append(category)
        return self._unique(matches)

    def _out_of_catalog_targets(self, text: str) -> list[str]:
        return self._drop_nested_terms([term for term in self.OUT_OF_CATALOG_PRODUCT_TERMS if term in text])

    def _retrieval_strategy(self, plan: QueryPlan) -> QueryRetrievalStrategy:
        strategy = QueryRetrievalStrategy()
        if plan.intent == "comparison":
            strategy.vector_top_k = 16
            strategy.keyword_top_k = 16
        if len(plan.exclude) >= 2:
            strategy.vector_top_k = max(strategy.vector_top_k, 16)
            strategy.keyword_top_k = max(strategy.keyword_top_k, 16)
        return strategy

    def _filters(self, plan: QueryPlan) -> list[str]:
        filters: list[str] = ["stock is unknown or stock > 0"]
        if plan.budget.min is not None:
            filters.append(f"price >= {plan.budget.min:g}")
        if plan.budget.max is not None:
            filters.append(f"price <= {plan.budget.max:g}")
        for excluded in plan.exclude:
            filters.append(f"exclude not matched: {excluded}")
        return filters

    def _extract_budget_max(self, query: str) -> float | None:
        patterns = [
            r"预算\s*(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*(?:元|块)?\s*(?:以内|以下|内|不超过)",
        ]
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                return float(match.group(1))
        if re.search(r"(不要|别|不想).{0,4}太贵", query):
            return 300.0
        return None

    def _extract_excludes(self, query: str) -> list[str]:
        excludes: list[str] = []
        for pattern in self.EXCLUDE_PATTERNS:
            for match in re.finditer(pattern, query):
                value = match.group(1).strip()
                if value:
                    excludes.append(value)
        return excludes

    def _extract_compare_targets(
        self,
        query: str,
        matched_categories: list[str] | None = None,
        out_of_catalog_targets: list[str] | None = None,
    ) -> list[str]:
        targets = []
        if "第一个" in query:
            targets.append("first")
        if "第二个" in query:
            targets.append("second")
        product_targets = self._ordered_terms_in_text(query, [*(matched_categories or []), *(out_of_catalog_targets or [])])
        targets.extend(term for term in product_targets if term not in targets)
        return targets

    def _ordered_terms_in_text(self, text: str, terms: list[str]) -> list[str]:
        positions = []
        for term in self._drop_nested_terms(self._unique(terms)):
            position = text.find(term)
            if position != -1:
                positions.append((position, -len(term), term))
        return [term for _, _, term in sorted(positions)]

    def _drop_nested_terms(self, values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            if not any(value != existing and value in existing for existing in result):
                result.append(value)
        return result

    def _extract_cart_action(self, query: str) -> str | None:
        if "购物车" in query and any(word in query for word in ["看", "看看", "查询"]):
            return "view"
        if any(word in query for word in ["加购", "加入购物车"]):
            return "add"
        if "删" in query:
            return "delete"
        if "数量" in query or "改成" in query:
            return "update_quantity"
        if "下单" in query:
            return "checkout"
        return None

    def _compact(self, text: str) -> str:
        return " ".join(text.split())

    def _unique(self, values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            text = value.strip()
            if text and text not in result:
                result.append(text)
        return result
