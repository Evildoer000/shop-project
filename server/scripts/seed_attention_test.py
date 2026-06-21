"""把 16 轮固定剧本注入 conversation_turn / session_memory_state.

让你在 Android 客户端进入第 17 轮就能直接发测试 query, 后端 plan 拉到的 history
就是这份预置剧本.

Usage:
    cd server
    .venv/bin/python -m scripts.seed_attention_test --variant main      # 主剧本
    .venv/bin/python -m scripts.seed_attention_test --variant d         # 幽灵继承场景
    .venv/bin/python -m scripts.seed_attention_test --variant f         # product_ids 截断场景
    .venv/bin/python -m scripts.seed_attention_test --variant e         # long-term profile 污染
    .venv/bin/python -m scripts.seed_attention_test --list              # 看每个 session 对应哪些测试 query
    .venv/bin/python -m scripts.seed_attention_test --reset             # 清掉所有测试 session

session_id 约定:
    plan_test_main   — 主剧本 (case A/B/C/G/H/I 共用)
    plan_test_d      — 幽灵继承 (case D)
    plan_test_e      — long-term profile (case E)
    plan_test_f      — product_ids 截断 (case F)

user_id 固定 plan_test_user.
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable

from sqlalchemy import delete

from app.db.models import (
    Base,
    ConversationTurn,
    SessionMemoryState,
    UserMemory,
)
from app.db.session import get_engine, get_sessionmaker

USER_ID = "plan_test_user"
SESS_MAIN = "plan_test_main"
SESS_D = "plan_test_d"
SESS_E = "plan_test_e"
SESS_F = "plan_test_f"

ALL_SESSIONS = [SESS_MAIN, SESS_D, SESS_E, SESS_F]

# 5 张图占位
_IMG_RUN_A = "[图片#shoe_a 跑鞋A]"
_IMG_COFFEE = "[图片#coffee 咖啡豆]"
_IMG_LAPTOP = "[图片#laptop_a Macbook]"
_IMG_SONY = "[图片#sony 索尼耳机]"
_IMG_RUN_B = "[图片#shoe_b 跑鞋B]"


def _t(
    user: str,
    assistant: str,
    *,
    route: str = "recommend",
    product_ids: list[str] | None = None,
    rewrite: dict | None = None,
    selected_products: list[dict] | None = None,
) -> dict:
    return {
        "user_message": user,
        "assistant_message": assistant,
        "route": route,
        "product_ids": product_ids or [],
        "rewrite_summary": rewrite or {},
        "trace_summary": {"selected_products": selected_products or []},
    }


# =============== 主剧本 (T1..T16) ===============


def turns_main() -> list[dict]:
    return [
        _t(
            "想买双跑鞋",
            "为你推荐 3 款热门跑鞋: Nike Pegasus 41/Adidas Ultraboost 5/Asics Gel-Kayano",
            product_ids=["p_run_a", "p_run_b", "p_run_c"],
            rewrite={"product_type": "跑鞋", "categories": ["跑步鞋"]},
            selected_products=[
                {"product_id": "p_run_a", "name": "Nike Pegasus 41", "brand": "Nike"},
                {"product_id": "p_run_b", "name": "Adidas Ultraboost 5", "brand": "Adidas"},
                {"product_id": "p_run_c", "name": "Asics Gel-Kayano", "brand": "Asics"},
            ],
        ),
        _t(
            f"{_IMG_RUN_A} 类似这种风格",
            "推荐 1 款类似风格的跑鞋: HOKA Clifton 9。",
            product_ids=["p_run_d"],
            rewrite={"product_type": "跑鞋"},
            selected_products=[{"product_id": "p_run_d", "name": "HOKA Clifton 9", "brand": "HOKA"}],
        ),
        _t(
            "再来一款防晒",
            "好的，请告诉我使用场景与肤质偏好。",
            route="clarify",
        ),
        _t(
            "敏感肌、不要酒精",
            "为你推荐 2 款敏感肌防晒: 薇诺娜清透防晒/理肤泉特护防晒。",
            product_ids=["p_sun_a", "p_sun_b"],
            rewrite={
                "product_type": "防晒",
                "soft_constraints": ["敏感肌"],
                "exclude_terms": ["酒精"],
            },
            selected_products=[
                {"product_id": "p_sun_a", "name": "薇诺娜清透防晒", "brand": "薇诺娜"},
                {"product_id": "p_sun_b", "name": "理肤泉特护防晒", "brand": "理肤泉"},
            ],
        ),
        _t(
            f"{_IMG_COFFEE}",
            "看到了一袋咖啡豆图片，能告诉我冲煮偏好或喜欢的口味吗？",
            route="clarify",
        ),
        _t(
            "推荐 3 款这种豆",
            "推荐 3 款类似豆: 三顿半数字星球/瑰夏/曼特宁。",
            product_ids=["p_coffee_a", "p_coffee_b", "p_coffee_c"],
            rewrite={"product_type": "咖啡豆"},
            selected_products=[
                {"product_id": "p_coffee_a", "name": "三顿半数字星球", "brand": "三顿半"},
                {"product_id": "p_coffee_b", "name": "瑰夏", "brand": "蓝瓶"},
                {"product_id": "p_coffee_c", "name": "曼特宁", "brand": "illy"},
            ],
        ),
        _t(
            "你能记住我喝咖啡的口味吗",
            "可以，会作为软偏好用在后续推荐里。",
            route="direct_answer",
        ),
        _t(
            "再聊聊手机",
            "你大概的用机场景和预算是？",
            route="clarify",
        ),
        # ↑ T1-T8 (这 8 轮会进 session_summary)
        _t(
            f"{_IMG_LAPTOP} 想买这种轻薄本",
            "推荐 1 款轻薄本: MacBook Air M3。",
            product_ids=["p_lap_a"],
            rewrite={"product_type": "笔记本电脑", "categories": ["笔记本"]},
            selected_products=[{"product_id": "p_lap_a", "name": "MacBook Air M3", "brand": "Apple"}],
        ),
        _t(
            "预算 8000",
            "在 8000 内推荐 3 款轻薄本: ThinkPad X1 Carbon/小新 Pro 16/灵耀 14。",
            product_ids=["p_lap_b", "p_lap_c", "p_lap_d"],
            rewrite={"product_type": "笔记本电脑", "budget_max": 8000, "budget_scope": "per_item"},
            selected_products=[
                {"product_id": "p_lap_b", "name": "ThinkPad X1 Carbon", "brand": "Lenovo"},
                {"product_id": "p_lap_c", "name": "小新 Pro 16", "brand": "Lenovo"},
                {"product_id": "p_lap_d", "name": "灵耀 14", "brand": "Asus"},
            ],
        ),
        _t(
            "推荐一款降噪耳机",
            "推荐 1 款降噪耳机: Sony WH-1000XM5。",
            product_ids=["p_phone_a"],
            rewrite={"product_type": "降噪耳机"},
            selected_products=[{"product_id": "p_phone_a", "name": "Sony WH-1000XM5", "brand": "Sony"}],
        ),
        _t(
            "2000 以内",
            "在 2000 内推荐 2 款降噪耳机: Bose QC45/索尼 WH-CH720N。",
            product_ids=["p_phone_a", "p_phone_b"],
            rewrite={"product_type": "降噪耳机", "budget_max": 2000, "budget_scope": "per_item"},
            selected_products=[
                {"product_id": "p_phone_a", "name": "Bose QC45", "brand": "Bose"},
                {"product_id": "p_phone_b", "name": "WH-CH720N", "brand": "Sony"},
            ],
        ),
        _t(
            f"{_IMG_SONY} 类似这个的",
            "推荐 1 款相似耳机: 索尼 WH-CH520。",
            product_ids=["p_phone_c"],
            rewrite={"product_type": "降噪耳机"},
            selected_products=[{"product_id": "p_phone_c", "name": "WH-CH520", "brand": "Sony"}],
        ),
        _t(
            "你叫什么",
            "我是智购助手，可以帮你筛选商品、对比规格、记录偏好。",
            route="direct_answer",
        ),
        _t(
            f"{_IMG_RUN_B} 再看下这种",
            "推荐 3 款相似跑鞋: Nike Pegasus 41/Asics Nimbus 26/HOKA Clifton 9。",
            product_ids=["p_run_x", "p_run_y", "p_run_z"],
            rewrite={"product_type": "跑鞋", "categories": ["跑步鞋"]},
            selected_products=[
                {"product_id": "p_run_x", "name": "Nike Pegasus 41", "brand": "Nike"},
                {"product_id": "p_run_y", "name": "Asics Nimbus 26", "brand": "Asics"},
                {"product_id": "p_run_z", "name": "HOKA Clifton 9", "brand": "HOKA"},
            ],
        ),
        _t(
            "你能再推荐一双轻量的吗",
            "推荐 1 款轻量跑鞋: Saucony Endorphin Speed 4。",
            product_ids=["p_run_w"],
            rewrite={"product_type": "跑鞋", "soft_constraints": ["轻量"]},
            selected_products=[{"product_id": "p_run_w", "name": "Endorphin Speed 4", "brand": "Saucony"}],
        ),
    ]


SUMMARY_MAIN = (
    "用户先后聊过：\n"
    "- 跑鞋（喜欢轻量通勤风格） [t1: p_run_a, p_run_b, p_run_c; t2(img): p_run_d]\n"
    "- 防晒（敏感肌、不要酒精） [t4: p_sun_a, p_sun_b]\n"
    "- 咖啡豆（让助手记住喝咖啡口味） [t6: p_coffee_a, p_coffee_b, p_coffee_c]\n"
    "- 之后准备聊手机但没展开。"
)


# =============== Variant: D 幽灵继承 ===============


def turns_d() -> list[dict]:
    base = turns_main()
    base[8] = _t(
        "再来一款敏感肌防晒，不要酒精，预算 300",
        "敏感肌不含酒精的防晒在 300 内: 薇诺娜清透防晒/理肤泉特护防晒。",
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
    base[9] = _t(
        "再帮我看看面霜，敏感肌不要酒精预算 300",
        "敏感肌不含酒精的面霜: 薇诺娜舒敏面霜/理肤泉特安霜。",
        product_ids=["p_cream_a", "p_cream_b"],
        rewrite={
            "product_type": "面霜",
            "soft_constraints": ["敏感肌"],
            "exclude_terms": ["酒精"],
            "budget_max": 300,
        },
    )
    return base


SUMMARY_D = SUMMARY_MAIN + " 用户反复强调：敏感肌、不要酒精、预算 300。"


# =============== Variant: F 截 10 ===============


def turns_f() -> list[dict]:
    base = turns_main()
    pids = [
        "p_xk39", "p_z7q2", "p_a1m8", "p_q4w0", "p_h6n5", "p_b2v3",
        "p_e9r7", "p_t8u1", "p_i5o2", "p_p3l4", "p_d6f8", "p_g0j1",
    ]
    names = [
        "Nike Pegasus 41", "Asics Nimbus 26", "HOKA Clifton 9", "Saucony Endorphin Speed",
        "New Balance 1080v13", "Brooks Ghost 16", "Mizuno Wave Rider", "Adidas Boston 12",
        "Puma Velocity Nitro 3", "Salomon Glide Max",
        "On Cloudmonster", "Diadora Mythos Blueshield",
    ]
    selected = [{"product_id": pid, "name": name, "brand": "Various"} for pid, name in zip(pids, names)]
    base[14] = _t(
        f"{_IMG_RUN_B} 再看下这种",
        "推荐 12 款相似跑鞋: " + " / ".join(names[:10]) + " 等。",
        product_ids=pids,
        rewrite={"product_type": "跑鞋", "categories": ["跑步鞋"]},
        selected_products=selected,
    )
    base[15] = _t(
        "你刚才推了好多双",
        "是的，刚才一次推荐了 12 款，可以告诉我哪些方向你最关注。",
        route="direct_answer",
    )
    return base


# =============== Variant: E long-term profile ===============


def turns_e() -> list[dict]:
    return turns_main()


# =============== 写入 ===============


def _wipe_session(db, session_id: str) -> None:
    db.execute(delete(ConversationTurn).where(ConversationTurn.user_id == USER_ID, ConversationTurn.session_id == session_id))
    db.execute(delete(SessionMemoryState).where(SessionMemoryState.user_id == USER_ID, SessionMemoryState.session_id == session_id))
    db.commit()


def _wipe_user_memory() -> None:
    db = get_sessionmaker()()
    try:
        db.execute(delete(UserMemory).where(UserMemory.user_id == USER_ID))
        db.commit()
    finally:
        db.close()


def _seed(session_id: str, turns: list[dict], summary: str, summarized_through: int = 8) -> None:
    Base.metadata.create_all(
        bind=get_engine(),
        tables=[ConversationTurn.__table__, SessionMemoryState.__table__, UserMemory.__table__],
    )
    db = get_sessionmaker()()
    try:
        _wipe_session(db, session_id)
        for turn in turns:
            db.add(
                ConversationTurn(
                    user_id=USER_ID,
                    session_id=session_id,
                    user_message=turn["user_message"],
                    assistant_message=turn["assistant_message"],
                    route=turn["route"],
                    product_ids=turn["product_ids"],
                    rewrite_summary=turn["rewrite_summary"],
                    trace_summary=turn["trace_summary"],
                )
            )
        db.add(
            SessionMemoryState(
                user_id=USER_ID,
                session_id=session_id,
                session_summary=summary,
                summarized_through_turn_id=summarized_through,
            )
        )
        db.commit()
        print(f"  ✅ {session_id}: 已写 {len(turns)} 轮 + summary (覆盖 t1..t{summarized_through})")
    finally:
        db.close()


def _seed_user_memory_for_e() -> None:
    db = get_sessionmaker()()
    try:
        db.execute(delete(UserMemory).where(UserMemory.user_id == USER_ID))
        for key, value in [
            ("肤质", "敏感肌油皮"),
            ("偏好", "不要酒精"),
            ("口味", "三顿半冷萃"),
        ]:
            db.add(
                UserMemory(
                    user_id=USER_ID,
                    memory_type="long_term_profile",
                    key=key,
                    value=value,
                )
            )
        db.commit()
        print(f"  ✅ user_memory: 写入 {USER_ID} 的长期画像 (肤质/偏好/口味)")
    finally:
        db.close()


VARIANT_BUILDERS: dict[str, Callable[[], tuple[str, str, list[dict], str]]] = {
    "main": lambda: (SESS_MAIN, "主剧本", turns_main(), SUMMARY_MAIN),
    "d": lambda: (SESS_D, "幽灵继承 (T9-T10 改成护肤)", turns_d(), SUMMARY_D),
    "e": lambda: (SESS_E, "long-term profile 污染", turns_e(), SUMMARY_MAIN),
    "f": lambda: (SESS_F, "product_ids 截断 (T15 推 12 款)", turns_f(), SUMMARY_MAIN),
}


def cmd_seed(variant: str) -> None:
    builder = VARIANT_BUILDERS[variant]
    sess_id, label, turns, summary = builder()
    print(f"[{variant}] {label}")
    _seed(sess_id, turns, summary)
    if variant == "e":
        _seed_user_memory_for_e()
    _print_query_hints(variant)


def cmd_reset() -> None:
    db = get_sessionmaker()()
    try:
        for sid in ALL_SESSIONS:
            _wipe_session(db, sid)
            print(f"  🗑  cleaned {sid}")
    finally:
        db.close()
    _wipe_user_memory()
    print(f"  🗑  cleaned user_memory for {USER_ID}")


def cmd_list() -> None:
    print(f"user_id: {USER_ID}\n")
    for variant, builder in VARIANT_BUILDERS.items():
        sess_id, label, _, _ = builder()
        print(f"[{variant}] {sess_id}  — {label}")
        _print_query_hints(variant, indent="    ")
        print()


def _print_query_hints(variant: str, indent: str = "  ") -> None:
    queries = {
        "main": [
            ("A", "那双蓝色的多少钱", "→ 应给 T15/T16 跑鞋的价格 (Nike Pegasus / Endorphin), 不要给 T1 的 3 双"),
            ("B", "我最开始问的那双跑鞋还有别的色吗", "→ 应找 T1 的 3 双 (Pegasus 41 / Ultraboost 5 / Gel-Kayano)"),
            ("C", "那个", "→ 应 clarify, 询问指代哪个商品"),
            ("G", "你最早给我看的那张图的鞋什么牌子", "→ 应找 T2 的 HOKA Clifton 9 (跑鞋图 A)"),
            ("H", "[上传一张鞋图] 这个比那个好吗", "→ 应识别新图 + T15 的跑鞋, 或 clarify 让你说清楚"),
            ("I", "[上传一张防晒图] (不打字)", "→ 应基于图片切到防晒话题, 不被前面跑鞋污染"),
        ],
        "d": [
            ("D", "推荐一款无线鼠标", "→ vector_query 应该是'无线鼠标', 绝不能含'敏感肌/酒精/护肤/防晒'"),
        ],
        "e": [
            ("E", "推荐一台笔记本电脑", "→ vector_query 应该只含'笔记本/电脑', 不能含'敏感肌/油皮/肤质'"),
        ],
        "f": [
            ("F", "第 11 个的颜色", "→ 应找回 On Cloudmonster (T15 的第 11 件); 修复前会乱绑前 10 名内的"),
        ],
    }
    for case, q, expect in queries.get(variant, []):
        print(f"{indent}case {case}: {q}")
        print(f"{indent}  期望: {expect}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=list(VARIANT_BUILDERS.keys()) + ["all"], default=None)
    p.add_argument("--list", action="store_true")
    p.add_argument("--reset", action="store_true")
    args = p.parse_args()

    if args.reset:
        cmd_reset()
        return 0
    if args.list:
        cmd_list()
        return 0
    if args.variant == "all":
        for v in VARIANT_BUILDERS:
            cmd_seed(v)
        return 0
    if args.variant:
        cmd_seed(args.variant)
        return 0
    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
