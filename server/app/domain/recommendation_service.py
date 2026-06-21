from __future__ import annotations

import math
import random
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Product, UserBrandAffinity, UserEvent, UserProductAffinity
from app.schemas import QueryPlan, RecommendationCard, RecommendationResponse
from app.services.product_repository import ProductRepository


COLD_THRESHOLD = 5
WARM_THRESHOLD = 30

# ─── 时间衰减 ───
HALF_LIFE_DAYS = 30                    # affinity 半衰期 (30 天前的偏好权重打 5 折)
RECENT_BOOST_HOURS_1 = 1               # 1 小时内事件 ×3 倍权
RECENT_BOOST_HOURS_24 = 24             # 24 小时内 ×2 倍权

# ─── 曝光抑制 ───
EXPOSURE_WINDOW_HOURS = 48              # 看 48 小时内的曝光
EXPOSURE_MAX_DAMPING = 0.40             # 最多扣 40% 分数
EXPOSURE_DAMPING_SATURATION = 5         # 见过 5+ 次饱和

# ─── MMR 多样性 ───
MMR_LAMBDA = 0.7                        # 0.7 = 70% 相关性 + 30% 多样性
MMR_POOL_SIZE = 40                      # 候选池大小 (从 top 40 里挑 size 件)

# ─── 类目硬约束 ───
MAX_PER_SUB_CATEGORY = 3
MAX_PER_BRAND = 2

# ─── 探索 ───
EXPLORE_SLOT_INDEX = 5                  # 在 top-5 之后插一件全随机商品
EXPLORE_SCORE_FALLBACK = 0.05
EXPLORE_EXTRA_SLOTS = (12, 18)          # 额外探索槽位

# ─── 刷新随机性 ───
# 给每个商品分数加 [0, REFRESH_NOISE] 之间的均匀噪声, 让排名在刷新之间能洗牌.
# 0.30 = 中段同档位商品会换位置, top 1-2 若分差很大仍会被锁定 (符合预期).
REFRESH_NOISE = 0.30


class RecommendationService:
    """商城推荐服务（RecommendationService）。

    给商城首页返回个性化/热门商品流。
    流程: score → 时间衰减 → 曝光抑制 → MMR re-rank → 类目硬约束 → 探索注入
    """

    def __init__(self, db: Session, product_repository: ProductRepository | None = None) -> None:
        self.db = db
        self.product_repository = product_repository or ProductRepository(db)

    # =========================================================
    # 主入口
    # =========================================================

    def get_home_recommendations(self, user_id: str, size: int = 24) -> RecommendationResponse:
        size = max(1, min(int(size or 24), 60))
        products = self.product_repository.list_for_plan(QueryPlan(), limit=max(200, size * 8))
        total_events = self._count_events(user_id)
        stage = self._stage(total_events)
        if not products:
            return RecommendationResponse(products=[], stage=stage, total_events=total_events)

        # ========== ① 时间衰减 + 近期加权: 重新计算 affinity ==========
        now = datetime.now(timezone.utc)
        brand_affinity = self._brand_affinity_decayed(user_id, now)
        product_affinity = self._product_affinity_decayed(user_id, now)
        subcat_affinity = self._subcat_affinity(product_affinity, products)
        event_counts = self._event_counts(user_id)

        # ========== ② 打分 (score_product) + 刷新噪声 ==========
        # 每次请求生成新的 RNG, 让同档位商品在不同次刷新间洗牌
        rng = random.Random()
        scored: list[tuple[Product, float, str]] = []
        for product in products:
            score, reason = self._score_product(
                product,
                stage=stage,
                brand_affinity=brand_affinity,
                product_affinity=product_affinity,
                subcat_affinity=subcat_affinity,
                event_counts=event_counts,
            )
            score += rng.uniform(0, REFRESH_NOISE)
            scored.append((product, score, reason))

        scored.sort(key=lambda item: item[1], reverse=True)

        # ========== ③ 曝光抑制 (用 user_events 的 impression 历史) ==========
        impressions = self._exposure_data(user_id, now)
        damped: list[tuple[Product, float, str]] = []
        for product, score, reason in scored:
            damping = self._exposure_damping(product.product_id, impressions)
            new_score = score * (1.0 - damping)
            new_reason = reason
            damped.append((product, new_score, new_reason))
        damped.sort(key=lambda item: item[1], reverse=True)

        # ========== ④ MMR re-rank (在前 N 个候选里挑 size 件多样的) ==========
        pool = damped[:MMR_POOL_SIZE]
        mmr_chosen = self._mmr_rerank(pool, size=size, lambda_=MMR_LAMBDA)

        # ========== ⑤ 类目硬约束 ==========
        constrained = self._enforce_category_constraints(mmr_chosen, fallback_pool=damped, size=size)

        # ========== ⑥ 探索注入 (top-5/12/18 各插 1 件用户没见过的) ==========
        if stage != "cold":
            constrained = self._inject_exploration(
                constrained,
                all_products=products,
                impressions=impressions,
                rng=rng,
            )

        return RecommendationResponse(
            products=[self._product_to_card(p, s, r) for p, s, r in constrained[:size]],
            stage=stage,
            total_events=total_events,
        )

    # =========================================================
    # ① 时间衰减 + 近期加权 affinity
    # =========================================================

    def _brand_affinity_decayed(self, user_id: str, now: datetime) -> dict[str, float]:
        """重算每个 brand 的 effective affinity (从 user_events 重新算, 带时间衰减)."""
        rows = self.db.execute(
            select(
                Product.brand,
                UserEvent.event_type,
                UserEvent.created_at,
            )
            .join(UserEvent, UserEvent.product_id == Product.product_id)
            .where(UserEvent.user_id == user_id)
            .where(UserEvent.event_type.in_(["click", "cart_add", "buy", "favorite"]))
        ).all()
        affinity: dict[str, float] = defaultdict(float)
        for brand, event_type, created_at in rows:
            if not brand:
                continue
            base = _EVENT_WEIGHT.get(event_type, 0.0)
            if base == 0:
                continue
            weight = base * self._recent_multiplier(created_at, now) * self._time_decay(created_at, now)
            affinity[brand] += weight
        return {brand: a for brand, a in affinity.items() if a > 0}

    def _product_affinity_decayed(self, user_id: str, now: datetime) -> dict[str, float]:
        rows = self.db.execute(
            select(
                UserEvent.product_id,
                UserEvent.event_type,
                UserEvent.created_at,
            )
            .where(UserEvent.user_id == user_id)
            .where(UserEvent.event_type.in_(["click", "cart_add", "buy", "favorite"]))
        ).all()
        affinity: dict[str, float] = defaultdict(float)
        for product_id, event_type, created_at in rows:
            base = _EVENT_WEIGHT.get(event_type, 0.0)
            if base == 0:
                continue
            weight = base * self._recent_multiplier(created_at, now) * self._time_decay(created_at, now)
            affinity[product_id] += weight
        return {pid: a for pid, a in affinity.items() if a > 0}

    @staticmethod
    def _time_decay(event_at: datetime, now: datetime) -> float:
        """指数时间衰减, 半衰期 30 天."""
        if event_at is None:
            return 1.0
        if event_at.tzinfo is None:
            event_at = event_at.replace(tzinfo=timezone.utc)
        delta_days = max(0.0, (now - event_at).total_seconds() / 86400.0)
        return 0.5 ** (delta_days / HALF_LIFE_DAYS)

    @staticmethod
    def _recent_multiplier(event_at: datetime, now: datetime) -> float:
        """近期事件加权: 1h 内 ×3, 24h 内 ×2, 否则 ×1."""
        if event_at is None:
            return 1.0
        if event_at.tzinfo is None:
            event_at = event_at.replace(tzinfo=timezone.utc)
        delta_hours = (now - event_at).total_seconds() / 3600.0
        if delta_hours < RECENT_BOOST_HOURS_1:
            return 3.0
        if delta_hours < RECENT_BOOST_HOURS_24:
            return 2.0
        return 1.0

    # =========================================================
    # ② 打分 (跟原版一样, 但 affinity 来自衰减后的)
    # =========================================================

    def _score_product(
        self,
        product: Product,
        *,
        stage: str,
        brand_affinity: dict[str, float],
        product_affinity: dict[str, float],
        subcat_affinity: dict[str, float],
        event_counts: dict[str, dict[str, int]],
    ) -> tuple[float, str]:
        hotness = self._hotness(product)
        if stage == "cold":
            return hotness, "热门商品"

        score = hotness * 0.35
        reasons: list[str] = []
        brand_score = brand_affinity.get(product.brand, 0.0)
        if brand_score > 0:
            score += min(brand_score, 3.0) * 0.28
            reasons.append(f"你常关注 {product.brand}")

        product_score = product_affinity.get(product.product_id, 0.0)
        if product_score > 0:
            score += min(product_score, 3.0) * 0.24
            reasons.append("你看过相关商品")

        subcat = product.sub_category or ""
        subcat_score = subcat_affinity.get(subcat, 0.0)
        if subcat_score > 0:
            score += min(subcat_score, 3.0) * 0.18
            reasons.append(f"你关注 {subcat}")

        counts = event_counts.get(product.product_id, {})
        if counts.get("favorite", 0) > 0:
            score += 0.25
            reasons.append("你收藏过")
        if counts.get("detail_view", 0) > 0:
            score += 0.12

        if stage == "warmup":
            score += self._category_exploration_bonus(product)

        return score, " · ".join(dict.fromkeys(reasons).keys()) or "为你推荐"

    # =========================================================
    # ③ 曝光抑制
    # =========================================================

    def _exposure_data(self, user_id: str, now: datetime) -> dict[str, tuple[int, datetime]]:
        """读 user_events 里 impression 事件, 返回 {product_id: (count, last_at)}."""
        cutoff = now - timedelta(hours=EXPOSURE_WINDOW_HOURS)
        rows = self.db.execute(
            select(
                UserEvent.product_id,
                func.count(UserEvent.event_id),
                func.max(UserEvent.created_at),
            )
            .where(UserEvent.user_id == user_id)
            .where(UserEvent.event_type == "impression")
            .where(UserEvent.created_at >= cutoff)
            .group_by(UserEvent.product_id)
        ).all()
        return {pid: (int(count or 0), last_at) for pid, count, last_at in rows}

    @staticmethod
    def _exposure_damping(product_id: str, impressions: dict[str, tuple[int, datetime]]) -> float:
        """已经看过 N 次的商品减分, 返回 [0, EXPOSURE_MAX_DAMPING]."""
        data = impressions.get(product_id)
        if data is None:
            return 0.0
        count, _ = data
        if count <= 0:
            return 0.0
        # 见过越多次减越多, count >= EXPOSURE_DAMPING_SATURATION 后到上限
        ratio = min(count / EXPOSURE_DAMPING_SATURATION, 1.0)
        return ratio * EXPOSURE_MAX_DAMPING

    # =========================================================
    # ④ MMR re-rank (Maximal Marginal Relevance)
    # =========================================================

    def _mmr_rerank(
        self,
        pool: list[tuple[Product, float, str]],
        size: int,
        lambda_: float,
    ) -> list[tuple[Product, float, str]]:
        """贪心 MMR: 每次选 (相关性高 - 跟已选商品相似度高) 最大的."""
        if not pool:
            return []
        chosen: list[tuple[Product, float, str]] = []
        remaining = list(pool)
        # 第 1 个: 最高分商品
        chosen.append(remaining.pop(0))
        while remaining and len(chosen) < size:
            best_idx = -1
            best_mmr = float("-inf")
            for i, (product, score, reason) in enumerate(remaining):
                # 跟已选商品的最大相似度
                max_sim = max(self._similarity(product, c[0]) for c in chosen)
                mmr = lambda_ * score - (1.0 - lambda_) * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i
            if best_idx < 0:
                break
            chosen.append(remaining.pop(best_idx))
        return chosen

    @staticmethod
    def _similarity(a: Product, b: Product) -> float:
        """商品相似度 (粗略). 同 sub_cat=0.6, 同 brand=0.3, 同 cat=0.2."""
        if a.product_id == b.product_id:
            return 1.0
        sim = 0.0
        if a.sub_category and a.sub_category == b.sub_category:
            sim += 0.6
        if a.brand and a.brand == b.brand:
            sim += 0.3
        if a.category == b.category:
            sim += 0.2
        return min(sim, 1.0)

    # =========================================================
    # ⑤ 类目硬约束
    # =========================================================

    @staticmethod
    def _enforce_category_constraints(
        chosen: list[tuple[Product, float, str]],
        fallback_pool: list[tuple[Product, float, str]],
        size: int,
    ) -> list[tuple[Product, float, str]]:
        """同 sub_category ≤ 3, 同 brand ≤ 2; 不够时从 fallback_pool 补."""
        result: list[tuple[Product, float, str]] = []
        per_subcat: dict[str, int] = defaultdict(int)
        per_brand: dict[str, int] = defaultdict(int)
        used_ids: set[str] = set()

        def _try_add(item: tuple[Product, float, str]) -> bool:
            product = item[0]
            if product.product_id in used_ids:
                return False
            sc = product.sub_category or ""
            br = product.brand or ""
            if per_subcat[sc] >= MAX_PER_SUB_CATEGORY:
                return False
            if per_brand[br] >= MAX_PER_BRAND:
                return False
            result.append(item)
            per_subcat[sc] += 1
            per_brand[br] += 1
            used_ids.add(product.product_id)
            return True

        # 第 1 轮: 严格按硬约束
        for item in chosen:
            if len(result) >= size:
                break
            _try_add(item)

        # 第 2 轮: 还差就从 fallback 补 (放宽顺序, 先补不冲突的)
        if len(result) < size:
            for item in fallback_pool:
                if len(result) >= size:
                    break
                _try_add(item)

        # 第 3 轮: 还差就强行塞 (无视约束)
        if len(result) < size:
            for item in fallback_pool:
                if len(result) >= size:
                    break
                if item[0].product_id not in used_ids:
                    result.append(item)
                    used_ids.add(item[0].product_id)

        return result

    # =========================================================
    # ⑥ 探索注入
    # =========================================================

    def _inject_exploration(
        self,
        chosen: list[tuple[Product, float, str]],
        all_products: list[Product],
        impressions: dict[str, tuple[int, datetime]],
        rng: random.Random,
    ) -> list[tuple[Product, float, str]]:
        """在 top-5/12/18 各插 1 件用户从未见过的商品 (每次刷新真随机)."""
        chosen_ids = {item[0].product_id for item in chosen}
        unseen = [
            p for p in all_products
            if p.product_id not in impressions and p.product_id not in chosen_ids
        ]
        if not unseen:
            return chosen
        # 多个槽位 (5, 12, 18) 各塞一件不同的探索商品, 不重样
        slots = [s for s in (EXPLORE_SLOT_INDEX, *EXPLORE_EXTRA_SLOTS) if s <= len(chosen)]
        if not slots:
            return chosen
        # 随机抽足够多的探索品 (但别超过 unseen 库存)
        sample_n = min(len(slots), len(unseen))
        if sample_n == 0:
            return chosen
        explores = rng.sample(unseen, sample_n)
        new_list = list(chosen)
        # 从后往前插, 不打乱前面 slot 的索引
        for slot, prod in zip(reversed(slots[:sample_n]), reversed(explores)):
            insert_at = min(slot, len(new_list))
            new_list.insert(insert_at, (prod, EXPLORE_SCORE_FALLBACK, "💡 看看新选择"))
        return new_list

    # =========================================================
    # 工具方法
    # =========================================================

    def _count_events(self, user_id: str) -> int:
        return int(self.db.scalar(select(func.count(UserEvent.event_id)).where(UserEvent.user_id == user_id)) or 0)

    @staticmethod
    def _stage(total_events: int) -> str:
        if total_events < COLD_THRESHOLD:
            return "cold"
        if total_events < WARM_THRESHOLD:
            return "warmup"
        return "warm"

    def _subcat_affinity(self, product_affinity: dict[str, float], products: list[Product]) -> dict[str, float]:
        product_by_id = {product.product_id: product for product in products}
        result: dict[str, float] = defaultdict(float)
        for product_id, affinity in product_affinity.items():
            product = product_by_id.get(product_id)
            if product is not None and product.sub_category:
                result[product.sub_category] += affinity
        return dict(result)

    def _event_counts(self, user_id: str) -> dict[str, dict[str, int]]:
        rows = self.db.execute(
            select(UserEvent.product_id, UserEvent.event_type, func.count(UserEvent.event_id))
            .where(UserEvent.user_id == user_id)
            .group_by(UserEvent.product_id, UserEvent.event_type)
        ).all()
        result: dict[str, dict[str, int]] = defaultdict(dict)
        for product_id, event_type, count in rows:
            result[str(product_id)][str(event_type)] = int(count or 0)
        return dict(result)

    @staticmethod
    def _hotness(product: Product) -> float:
        rating = float(product.rating or Decimal("0")) / 5.0
        sales = float(product.sales or 0)
        price = float(product.price or Decimal("0"))
        price_penalty = 0.0 if price <= 0 else min(math.log10(price + 1) / 8.0, 0.35)
        return rating * 0.7 + min(math.log1p(sales) / 10.0, 1.0) * 0.3 - price_penalty

    @staticmethod
    def _category_exploration_bonus(product: Product) -> float:
        seed = sum(ord(char) for char in f"{product.category}:{product.product_id}")
        return (seed % 17) / 100.0

    @staticmethod
    def _product_to_card(product: Product, score: float, reason: str) -> RecommendationCard:
        return RecommendationCard(
            product_id=product.product_id,
            name=product.name,
            category=product.category,
            sub_category=product.sub_category,
            brand=product.brand,
            price=float(product.price or Decimal("0")),
            image_url=product.image_url,
            tags=list(product.tags or [])[:6],
            rating=float(product.rating or Decimal("0")),
            reason=reason,
            score=round(score, 4),
        )


# 事件权重 (跟 EventService 一致, 但这里用于推荐计算)
_EVENT_WEIGHT = {
    "click": 0.30,
    "favorite": 0.50,
    "cart_add": 0.60,
    "cart_remove": -0.30,
    "buy": 1.00,
}


def _simple_seed(items) -> int:
    """从 set 算个简单 hash 当种子."""
    s = 0
    for it in items:
        for c in str(it):
            s = (s * 31 + ord(c)) & 0x7FFFFFFF
    return s
