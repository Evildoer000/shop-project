"""Plan 注意力诊断脚本 (Layer 1, IntentPlanner 单元).

构造 16 轮对话历史 + 1 个当前 query, 直接调 IntentPlanner.plan(),
验证 plan 是否被早期上下文/长期画像污染. 9 个 case 复用同一份主剧本,
只换 current_query / 局部 patch / 长期画像.

Usage:
    cd server
    .venv/bin/python -m scripts.plan_attention_test            # 跑全部 9 个
    .venv/bin/python -m scripts.plan_attention_test --case A,B # 只跑 A B
    .venv/bin/python -m scripts.plan_attention_test --verbose  # 打印 plan 完整字段 + 喂给 LLM 的 context
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from app.domain.intent_planner import IntentPlanner
from app.domain.memory import ConversationContext, ConversationTurnView
from app.schemas import ImageAttributes, IntentPlan


# =====================================================================
# 主剧本 (16 轮)
#   T1-T8 → session_summary
#   T9-T16 → recent_turns (RECENT_TURNS_LIMIT=8)
# =====================================================================


def _turn(
    turn_id: int,
    user: str,
    assistant: str,
    route: str = "recommend",
    product_ids: Optional[list[str]] = None,
    rewrite: Optional[dict] = None,
    selected_products: Optional[list[dict]] = None,
) -> ConversationTurnView:
    return ConversationTurnView(
        turn_id=turn_id,
        user_message=user,
        assistant_message=assistant,
        route=route,
        product_ids=product_ids or [],
        rewrite_summary=rewrite or {},
        trace_summary={"selected_products": selected_products or []},
    )


# 5 张图占位 (Layer 1 不真跑 VLM, image 只出现在 user_message 文本里 + image_attributes)
_IMG_RUN_A = "[图片#shoe_a 跑鞋A]"
_IMG_COFFEE = "[图片#coffee 咖啡豆]"
_IMG_LAPTOP = "[图片#laptop_a Macbook]"
_IMG_SONY = "[图片#sony 索尼耳机]"
_IMG_RUN_B = "[图片#shoe_b 跑鞋B]"


def build_master_script() -> tuple[str, list[ConversationTurnView]]:
    """返回 (session_summary 文字, T9-T16 这 8 个 ConversationTurnView).

    主剧本: T1-T8 已被 SessionSummarizer 压缩进 session_summary;
    T9-T16 在 recent_turns 里. 当前 query (T17) 由各 case 提供.
    """
    # 模拟 SessionSummarizer 的新输出: 自然语言摘要 + 结构化锚点
    # (改造后的 prompt 强制要求每个非空 turn 保留 [t#: pid...] 锚点)
    session_summary = (
        "用户先后聊过：\n"
        "- 跑鞋（喜欢轻量通勤风格） [t1: p_run_a, p_run_b, p_run_c; t2(img): p_run_d]\n"
        "- 防晒（敏感肌、不要酒精） [t4: p_sun_a, p_sun_b]\n"
        "- 咖啡豆（让助手记住喝咖啡口味） [t6: p_coffee_a, p_coffee_b, p_coffee_c]\n"
        "- 之后准备聊手机但没展开。"
    )

    # T9-T16 (8 个 turn 进 recent_turns)
    recent = [
        _turn(
            9,
            f"{_IMG_LAPTOP} 想买这种轻薄本",
            "推荐 1 款轻薄本: MacBook Air M3，符合便携场景。",
            product_ids=["p_lap_a"],
            rewrite={"product_type": "笔记本电脑", "categories": ["笔记本"]},
            selected_products=[{"product_id": "p_lap_a", "name": "MacBook Air M3", "brand": "Apple"}],
        ),
        _turn(
            10,
            "预算 8000",
            "在 8000 内推荐 3 款轻薄本: ThinkPad X1 Carbon / 联想小新 Pro / 华硕灵耀 14。",
            product_ids=["p_lap_b", "p_lap_c", "p_lap_d"],
            rewrite={"product_type": "笔记本电脑", "budget_max": 8000, "budget_scope": "per_item"},
            selected_products=[
                {"product_id": "p_lap_b", "name": "ThinkPad X1 Carbon", "brand": "Lenovo"},
                {"product_id": "p_lap_c", "name": "小新 Pro 16", "brand": "Lenovo"},
                {"product_id": "p_lap_d", "name": "灵耀 14", "brand": "Asus"},
            ],
        ),
        _turn(
            11,
            "推荐一款降噪耳机",
            "推荐 1 款降噪耳机: Sony WH-1000XM5。",
            product_ids=["p_phone_a"],
            rewrite={"product_type": "降噪耳机"},
            selected_products=[{"product_id": "p_phone_a", "name": "Sony WH-1000XM5", "brand": "Sony"}],
        ),
        _turn(
            12,
            "2000 以内",
            "在 2000 内推荐 2 款降噪耳机: Bose QC45 / 索尼 WH-CH720N。",
            product_ids=["p_phone_a", "p_phone_b"],
            rewrite={"product_type": "降噪耳机", "budget_max": 2000, "budget_scope": "per_item"},
            selected_products=[
                {"product_id": "p_phone_a", "name": "Bose QC45", "brand": "Bose"},
                {"product_id": "p_phone_b", "name": "WH-CH720N", "brand": "Sony"},
            ],
        ),
        _turn(
            13,
            f"{_IMG_SONY} 类似这个的",
            "推荐 1 款相似耳机: 索尼 WH-CH520。",
            product_ids=["p_phone_c"],
            rewrite={"product_type": "降噪耳机"},
            selected_products=[{"product_id": "p_phone_c", "name": "WH-CH520", "brand": "Sony"}],
        ),
        _turn(
            14,
            "你叫什么",
            "我是智购助手，可以帮你筛选商品、对比规格、记录偏好。",
            route="direct_answer",
            product_ids=[],
            rewrite={},
        ),
        _turn(
            15,
            f"{_IMG_RUN_B} 再看下这种",
            "推荐 3 款相似跑鞋: Nike Pegasus 41 / Asics Nimbus 26 / HOKA Clifton 9。",
            product_ids=["p_run_x", "p_run_y", "p_run_z"],
            rewrite={"product_type": "跑鞋", "categories": ["跑步鞋"]},
            selected_products=[
                {"product_id": "p_run_x", "name": "Nike Pegasus 41", "brand": "Nike"},
                {"product_id": "p_run_y", "name": "Asics Nimbus 26", "brand": "Asics"},
                {"product_id": "p_run_z", "name": "HOKA Clifton 9", "brand": "HOKA"},
            ],
        ),
        _turn(
            16,
            "你能再推荐一双轻量的吗",
            "推荐 1 款轻量跑鞋: Saucony Endorphin Speed 4。",
            product_ids=["p_run_w"],
            rewrite={"product_type": "跑鞋", "soft_constraints": ["轻量"]},
            selected_products=[{"product_id": "p_run_w", "name": "Endorphin Speed 4", "brand": "Saucony"}],
        ),
    ]
    return session_summary, recent


# =====================================================================
# Case patches (针对个别 case 微调主剧本)
# =====================================================================


def patch_case_d(summary: str, recent: list[ConversationTurnView]) -> tuple[str, list[ConversationTurnView]]:
    """D: 幽灵继承. 让 session_summary + 多轮 recent 反复说敏感肌/酒精/300, 测当前 query 是否会被污染."""
    new_summary = summary + " 用户反复强调：敏感肌、不要酒精、预算 300。"
    new_recent = copy.deepcopy(recent)
    # T9 改成"敏感肌不要酒精预算300"的护肤上下文
    new_recent[0] = _turn(
        9,
        "再来一款敏感肌防晒，不要酒精，预算 300",
        "敏感肌不含酒精的防晒在 300 内: 薇诺娜清透防晒 / 理肤泉特护防晒。",
        product_ids=["p_sun_c", "p_sun_d"],
        rewrite={
            "product_type": "防晒",
            "soft_constraints": ["敏感肌"],
            "exclude_terms": ["酒精"],
            "budget_max": 300,
            "budget_scope": "per_item",
        },
        selected_products=[
            {"product_id": "p_sun_c", "name": "薇诺娜清透防晒", "brand": "薇诺娜"},
            {"product_id": "p_sun_d", "name": "理肤泉特护防晒", "brand": "理肤泉"},
        ],
    )
    new_recent[1] = _turn(
        10,
        "再帮我看看面霜，敏感肌不要酒精预算 300",
        "敏感肌不含酒精的面霜: 薇诺娜舒敏面霜 / 理肤泉特安霜。",
        product_ids=["p_cream_a", "p_cream_b"],
        rewrite={
            "product_type": "面霜",
            "soft_constraints": ["敏感肌"],
            "exclude_terms": ["酒精"],
            "budget_max": 300,
        },
    )
    return new_summary, new_recent


def patch_case_f(summary: str, recent: list[ConversationTurnView]) -> tuple[str, list[ConversationTurnView]]:
    """F: product_ids 截 10 测试.

    用不规律 ID 断 LLM 按命名规律编造的路径:
      pids = [p_xk39, p_z7q2, p_a1m8, p_q4w0, p_h6n5, p_b2v3, p_e9r7, p_t8u1,
              p_i5o2, p_p3l4, p_d6f8, p_g0j1]
    第 11 个真实 ID 是 p_d6f8. 它会因为 product_ids[:10] 和 selected_products[:10]
    被双双截掉. assistant_message 也只列前 10 个商品名, 让 LLM 真的看不到 11/12 的任何信息.
    """
    new_recent = copy.deepcopy(recent)
    pids = [
        "p_xk39", "p_z7q2", "p_a1m8", "p_q4w0", "p_h6n5", "p_b2v3",
        "p_e9r7", "p_t8u1", "p_i5o2", "p_p3l4", "p_d6f8", "p_g0j1",
    ]
    names = [
        "Nike Pegasus 41", "Asics Nimbus 26", "HOKA Clifton 9", "Saucony Endorphin Speed",
        "New Balance 1080v13", "Brooks Ghost 16", "Mizuno Wave Rider", "Adidas Boston 12",
        "Puma Velocity Nitro 3", "Salomon Glide Max",
        # 下面这两个 (第 11/12) 不会出现在 assistant_message 里
        "On Cloudmonster", "Diadora Mythos Blueshield",
    ]
    selected = [{"product_id": pid, "name": name, "brand": "Various"} for pid, name in zip(pids, names)]
    # assistant_message 只列前 10 个商品名 (跟 product_ids[:10] 一致)
    new_recent[6] = _turn(
        15,
        f"{_IMG_RUN_B} 再看下这种",
        "推荐 12 款相似跑鞋: " + " / ".join(names[:10]) + " 等。",
        product_ids=pids,
        rewrite={"product_type": "跑鞋", "categories": ["跑步鞋"]},
        selected_products=selected,
    )
    new_recent[7] = _turn(
        16,
        "你刚才推了好多双",
        "是的，刚才一次推荐了 12 款，可以告诉我哪些方向你最关注。",
        route="direct_answer",
        product_ids=[],
        rewrite={},
    )
    return summary, new_recent


# =====================================================================
# 期望 (机器可判)
# =====================================================================


def _has_any(text: str, words: list[str]) -> bool:
    text = text or ""
    return any(w in text for w in words)


def _all_in(items: list[str], allowed: set[str]) -> bool:
    return all(i in allowed for i in items)


def expect_a(plan: IntentPlan) -> tuple[bool, str]:
    """A: 上一轮短距离指代 'T17: 那双蓝色的多少钱'.
    PASS 路径 (任一即可):
      1) direct_answer + ref ⊆ {p_run_w/x/y/z} (T15/T16) — 真找回
      2) 走检索 + query 含跑鞋词 + 不含污染词 + 不绑早期 ref — 保守但正确
      3) clarify — 也能接受 (谨慎风格)
    FAIL: ref 含 T1-T8 的早期跑鞋 (绑错时间) 或 query 含护肤/敏感肌词.
    """
    later = {"p_run_w", "p_run_x", "p_run_y", "p_run_z"}
    earlier = {"p_run_a", "p_run_b", "p_run_c", "p_run_d"}
    forbidden = ["敏感肌", "酒精", "护肤", "面霜", "防晒"]

    if any(rid in earlier for rid in plan.referenced_product_ids):
        return False, f"ref 绑了 T1/T2 的早期跑鞋: {plan.referenced_product_ids} (时间锚点错位)"
    contaminated = [w for w in forbidden if w in (plan.vector_query or "") + (plan.keyword_query or "")]
    if contaminated:
        return False, f"query 被无关上下文污染: {contaminated}"

    if plan.referenced_product_ids:
        if all(rid in later for rid in plan.referenced_product_ids):
            return True, f"ok, ref 真找回 T15/T16: {plan.referenced_product_ids}"
        return False, f"ref 含非 later/earlier 的产品: {plan.referenced_product_ids}"
    if plan.plan_type == "clarify":
        return True, "clarify (保守但可接受)"
    if plan.plan_type in {"single_retrieval", "multi_retrieval"} and _has_any(
        plan.vector_query + plan.keyword_query, ["跑鞋", "跑步"]
    ):
        return True, "走检索 + query 含跑鞋, 未绑错 ref (保守通过)"
    return False, f"未识别跑鞋目标: vector={plan.vector_query!r}"


def expect_b(plan: IntentPlan) -> tuple[bool, str]:
    """B: 跨摘要边界指代. T17='我最开始问的那双跑鞋还有别的色吗'.
    '最开始' 应指 T1 (p_run_a/b/c), 不是 T15/T16.

    PASS 三条路径:
      1) ref ⊆ {p_run_a, p_run_b, p_run_c, p_run_d}  (T1/T2 真找回)
      2) clarify                                       (识别"最开始"已丢)
      3) 走检索 + 不绑错 ref                            (query 含跑鞋, ref 不含 T15/T16)

    FAIL: ref 含 T15/T16 的 p_run_w/x/y/z (典型时间锚点错位).
    """
    earliest = {"p_run_a", "p_run_b", "p_run_c", "p_run_d"}
    later = {"p_run_w", "p_run_x", "p_run_y", "p_run_z"}
    if any(rid in later for rid in plan.referenced_product_ids):
        return False, (
            f"ref 抓到了 T15/T16 的近期跑鞋 {plan.referenced_product_ids}, "
            "用户问的是'最开始那双' (T1). 时间锚点错位 (issue ① ⑥)."
        )
    if any(rid in earliest for rid in plan.referenced_product_ids):
        return True, f"ok, ref 真找回了 T1/T2 的早期跑鞋: {plan.referenced_product_ids}"
    if plan.plan_type == "clarify":
        return True, "clarify (正确识别早期信息已丢)"
    if not _has_any(plan.vector_query + plan.keyword_query, ["跑鞋", "跑步"]):
        return False, f"检索但 query 不含跑鞋词: vector={plan.vector_query!r}"
    if plan.referenced_product_ids:
        return False, f"ref 给了既非早期也非 T15-16 的产品: {plan.referenced_product_ids}"
    return True, "ok (走检索, query 含跑鞋词, 未绑错 ref)"


def expect_c(plan: IntentPlan) -> tuple[bool, str]:
    """C: T17='那个'. 歧义指代, 期望 clarify 或 ref=[]."""
    if plan.plan_type == "clarify":
        return True, "clarify (正确识别歧义)"
    if not plan.referenced_product_ids:
        return True, "未填充 ref (正确)"
    return False, f"在歧义指代下绑了 ref: {plan.referenced_product_ids}"


def expect_d(plan: IntentPlan) -> tuple[bool, str]:
    """D: T17='推荐一款无线鼠标'. query 不该污染敏感肌/酒精/护肤词."""
    forbidden = ["敏感肌", "酒精", "护肤", "防晒", "面霜"]
    contaminated = []
    for w in forbidden:
        if w in (plan.vector_query or "") or w in (plan.keyword_query or ""):
            contaminated.append(w)
    if contaminated:
        return False, f"query 被护肤上下文污染: 出现 {contaminated}"
    if plan.budget_max == 300:
        return False, "误继承上下文预算 300"
    if not _has_any(plan.vector_query + plan.keyword_query, ["鼠标", "无线鼠标"]):
        return False, f"vector_query 不含鼠标主目标: {plan.vector_query!r}"
    return True, "ok"


def expect_e(plan: IntentPlan) -> tuple[bool, str]:
    """E: T17='推荐一台笔记本电脑', long_term_profile 含'肤质=敏感肌'. 不该出现皮肤词."""
    forbidden = ["敏感肌", "油皮", "肤质", "护肤"]
    contaminated = [w for w in forbidden if w in (plan.vector_query or "") or w in (plan.keyword_query or "")]
    if contaminated:
        return False, f"vector_query 被长期画像污染: {contaminated}"
    if not _has_any(plan.vector_query + plan.keyword_query, ["笔记本", "电脑"]):
        return False, f"vector_query 不含笔记本: {plan.vector_query!r}"
    return True, "ok"


def expect_f(plan: IntentPlan) -> tuple[bool, str]:
    """F: T17='第 11 个的颜色'. 真实第 11 个 ID = p_d6f8 (不规律, 防 LLM 编造)."""
    truth = "p_d6f8"
    truncated_ids = {"p_xk39", "p_z7q2", "p_a1m8", "p_q4w0", "p_h6n5",
                     "p_b2v3", "p_e9r7", "p_t8u1", "p_i5o2", "p_p3l4"}  # 前 10 个 (LLM 看到的)
    if truth in plan.referenced_product_ids:
        return True, f"ok, ref 真包含第 11 个 {truth}"
    if any(rid in truncated_ids for rid in plan.referenced_product_ids):
        return False, f"ref 给了前 10 名内的 ID (绑错位置), 第 11 名 {truth} 已丢失. ref={plan.referenced_product_ids}"
    if not plan.referenced_product_ids:
        return False, f"ref=[]; product_ids[:10] 截断后第 11 名 {truth} 无法指代 (issue ⑤ 复现)"
    return False, f"ref 包含编造/无关 ID: {plan.referenced_product_ids} (期望 {truth})"


def expect_g(plan: IntentPlan) -> tuple[bool, str]:
    """G: 跨多轮图片指代. T17='你最早给我看的那张图的鞋什么牌子'.
    最早的图是 T2 (跑鞋图 A → p_run_d), 不是 T15.

    PASS 三条路径:
      1) ref 严格 ⊆ {p_run_d}  (LLM 真找回早期 image-轮)
      2) clarify              (LLM 识别"最早那张图"信息已丢)
      3) 走检索 + 不绑错 ref   (vector_query 含跑鞋, ref 不含 T9-T16 的产品)

    FAIL: ref 含 T9-T16 的产品 (尤其 p_run_w/x/y/z, 这是抓错时间锚点的典型表现).
    """
    later_run_ids = {"p_run_w", "p_run_x", "p_run_y", "p_run_z"}
    if any(rid in later_run_ids for rid in plan.referenced_product_ids):
        return False, (
            f"ref 抓到了 T15/T16 的近期跑鞋 {plan.referenced_product_ids}, "
            "用户问的是'最早那张图' (T2 → p_run_d). 时间锚点错位."
        )
    if "p_run_d" in plan.referenced_product_ids:
        return True, "ok, ref 真找回了 T2 的 p_run_d"
    if plan.plan_type == "clarify":
        return True, "clarify (正确识别早期图片信息已丢)"
    if not _has_any(plan.vector_query + plan.keyword_query, ["跑鞋", "跑步"]):
        return False, f"检索但 query 不含跑鞋词: vector={plan.vector_query!r}"
    if plan.referenced_product_ids:
        return False, f"ref 给了非 T2 也非 T15-16 的产品: {plan.referenced_product_ids}"
    return True, "ok (走检索, query 含跑鞋词, 未绑错 ref)"


def expect_h(plan: IntentPlan) -> tuple[bool, str]:
    """H: 双图叠加 + T17='这个比那个好吗' + 当前轮带新跑鞋图.
    PASS 路径:
      1) direct_answer + ref ⊆ T15/T16 跑鞋 — 真找回
      2) clarify — 双图比较保守询问可接受
    FAIL: ref 含早期跑鞋 (T1/T2) 或非跑鞋产品.
    """
    later = {"p_run_w", "p_run_x", "p_run_y", "p_run_z"}
    earlier = {"p_run_a", "p_run_b", "p_run_c", "p_run_d"}
    if any(rid in earlier for rid in plan.referenced_product_ids):
        return False, f"ref 绑了 T1/T2 早期跑鞋, 时间锚点错位: {plan.referenced_product_ids}"
    if plan.referenced_product_ids:
        if all(rid in later for rid in plan.referenced_product_ids):
            return True, f"ok, ref 找回 T15/T16: {plan.referenced_product_ids}"
        return False, f"ref 含非跑鞋产品: {plan.referenced_product_ids}"
    if plan.plan_type == "clarify":
        return True, "clarify (双图比较保守询问可接受)"
    return False, f"既无 ref 也未 clarify: plan_type={plan.plan_type}"


def expect_i(plan: IntentPlan) -> tuple[bool, str]:
    """I: 当前轮纯图 (image-only). 在 Layer 1 直接走的 IntentPlanner, 没有 orchestrator 的 image-only 短路.
    我们用 query='' + image_attributes 测 plan 是否被前面护肤上下文污染."""
    forbidden = ["敏感肌", "酒精", "护肤", "面霜"]
    contaminated = [w for w in forbidden if w in (plan.vector_query or "") or w in (plan.keyword_query or "")]
    if contaminated:
        return False, f"vector_query 被前面上下文污染: {contaminated}"
    return True, "ok"


# =====================================================================
# Case 定义
# =====================================================================


@dataclass
class TestCase:
    case_id: str
    title: str
    issues: list[str]
    current_query: str
    image_attributes: Optional[ImageAttributes] = None
    long_term_profile: list[str] = field(default_factory=list)
    fixture_patch: Optional[Callable[[str, list[ConversationTurnView]], tuple[str, list[ConversationTurnView]]]] = None
    expect: Callable[[IntentPlan], tuple[bool, str]] = field(default=lambda p: (True, ""))


def build_cases() -> list[TestCase]:
    return [
        TestCase(
            case_id="A",
            title="上一轮短距离指代",
            issues=["② recent_turns 内部 recency 不清", "⑤ product_ids 截 10"],
            current_query="那双蓝色的多少钱",
            expect=expect_a,
        ),
        TestCase(
            case_id="B",
            title="跨摘要边界指代",
            issues=["① RECENT_TURNS=8 截断", "⑥ session_summary 文字摘要丢 product_id"],
            current_query="我最开始问的那双跑鞋还有别的色吗",
            expect=expect_b,
        ),
        TestCase(
            case_id="C",
            title="多次主题切换的歧义指代",
            issues=["② recent_turns 平级"],
            current_query="那个",
            expect=expect_c,
        ),
        TestCase(
            case_id="D",
            title="幽灵继承 (敏感肌/酒精/300 反复出现)",
            issues=["④ rewrite_summary 全字段累积"],
            current_query="推荐一款无线鼠标",
            fixture_patch=patch_case_d,
            expect=expect_d,
        ),
        TestCase(
            case_id="E",
            title="long-term profile 污染",
            issues=["⑧ long_term_profile 平铺全量"],
            current_query="推荐一台笔记本电脑",
            long_term_profile=["肤质:敏感肌油皮", "偏好:不要酒精", "口味:三顿半冷萃"],
            expect=expect_e,
        ),
        TestCase(
            case_id="F",
            title="product_ids 截 10 后丢失",
            issues=["⑤ compact 截 product_ids[:10]"],
            current_query="第 11 个的颜色",
            fixture_patch=patch_case_f,
            expect=expect_f,
        ),
        TestCase(
            case_id="G",
            title="跨多轮的图片轮指代",
            issues=["① 跨摘要", "⑥ summary 丢图片 product_id"],
            current_query="你最早给我看的那张图的鞋什么牌子",
            expect=expect_g,
        ),
        TestCase(
            case_id="H",
            title="双图叠加 + 历史指代",
            issues=["image_attributes 优先级", "ref 找回历史轮 product_id"],
            current_query="这个比那个好吗",
            image_attributes=ImageAttributes(
                available=True,
                category_guess="服饰运动",
                product_type_guess="跑鞋",
                colors=["蓝色"],
                style_tags=["竞速"],
                retrieval_query="蓝色竞速跑鞋",
                confidence=0.85,
            ),
            expect=expect_h,
        ),
        TestCase(
            case_id="I",
            title="当前轮纯图切换主题 (Layer-1 用 query 空串模拟)",
            issues=["纯图被 recent_turns 污染"],
            current_query="",
            image_attributes=ImageAttributes(
                available=True,
                category_guess="服饰运动",
                product_type_guess="跑鞋",
                colors=["白色"],
                style_tags=["简约"],
                retrieval_query="白色简约跑鞋",
                confidence=0.80,
            ),
            expect=expect_i,
        ),
    ]


# =====================================================================
# Runner
# =====================================================================


async def run_case(planner: IntentPlanner, case: TestCase, verbose: bool) -> tuple[bool, str, Optional[IntentPlan]]:
    summary, recent = build_master_script()
    if case.fixture_patch:
        summary, recent = case.fixture_patch(summary, recent)
    ctx = ConversationContext(
        session_summary=summary,
        pending_summary_turns=[],
        recent_turns=recent,
        long_term_profile=case.long_term_profile,
    )
    rewrite_ctx = ctx.to_rewrite_context()
    if case.image_attributes is not None:
        rewrite_ctx["image_attributes"] = case.image_attributes.model_dump()
        rewrite_ctx.setdefault("priority", []).append("image_attributes")

    if verbose:
        print(f"[{case.case_id}] context payload (size={len(json.dumps(rewrite_ctx, ensure_ascii=False))}):")
        print(json.dumps(rewrite_ctx, ensure_ascii=False, indent=2)[:2000])
        print("---")

    try:
        plan = await planner.plan(case.current_query, rewrite_ctx)
    except Exception as exc:  # noqa: BLE001
        return False, f"LLM 调用异常: {exc}", None
    passed, reason = case.expect(plan)
    return passed, reason, plan


def render(case: TestCase, plan: Optional[IntentPlan], passed: bool, reason: str, verbose: bool) -> None:
    icon = "✅" if passed else "❌"
    print(f"\n[{case.case_id}] {case.title}")
    print(f"  current_query : {case.current_query!r}")
    if case.image_attributes:
        print(f"  image_attrs   : {case.image_attributes.retrieval_query!r}")
    if plan is None:
        print(f"  {icon} ERROR — {reason}")
        return
    print(f"  plan_type     : {plan.plan_type}")
    print(f"  vector_query  : {plan.vector_query!r}")
    print(f"  keyword_query : {plan.keyword_query!r}")
    print(f"  ref_product_ids: {plan.referenced_product_ids}")
    if plan.budget_max is not None:
        print(f"  budget        : [{plan.budget_min}, {plan.budget_max}] {plan.budget_scope}")
    if plan.need_slots:
        print(f"  need_slots    : {[s.product_type for s in plan.need_slots]}")
    if verbose:
        print(f"  plan_reason   : {plan.plan_reason}")
    print(f"  {icon} {'PASS' if passed else 'FAIL'} — {reason}")
    if not passed:
        print(f"     suspect issues: {', '.join(case.issues)}")


async def main_async(case_filter: Optional[set[str]], verbose: bool) -> int:
    all_cases = build_cases()
    cases = [c for c in all_cases if (case_filter is None or c.case_id in case_filter)]
    if not cases:
        print(f"no cases matched filter {case_filter}")
        return 1
    planner = IntentPlanner()
    pass_n = fail_n = 0
    issue_hits: dict[str, int] = {}
    for case in cases:
        passed, reason, plan = await run_case(planner, case, verbose)
        render(case, plan, passed, reason, verbose)
        if passed:
            pass_n += 1
        else:
            fail_n += 1
            for issue in case.issues:
                issue_hits[issue] = issue_hits.get(issue, 0) + 1
    print()
    print("=" * 60)
    print(f"总计: {pass_n} PASS / {fail_n} FAIL  (case 总数 {len(cases)})")
    if issue_hits:
        print("被复现的 issue:")
        for issue, n in sorted(issue_hits.items(), key=lambda kv: -kv[1]):
            print(f"  ×{n}  {issue}")
    print("=" * 60)
    return 0 if fail_n == 0 else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="IntentPlanner attention diagnostic")
    parser.add_argument("--case", type=str, default=None, help="A,B,C... 逗号分隔, 默认全部")
    parser.add_argument("--verbose", "-v", action="store_true", help="打印 plan 完整字段 + 喂给 LLM 的 context")
    args = parser.parse_args()
    case_filter = None
    if args.case:
        case_filter = {c.strip().upper() for c in args.case.split(",") if c.strip()}
    return asyncio.run(main_async(case_filter, args.verbose))


if __name__ == "__main__":
    sys.exit(main())
