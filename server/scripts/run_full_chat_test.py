"""真跑 16 轮 chat 接口的端到端测试.

每轮通过 /api/chat/stream 发起请求, server 端会触发 IntentPlanner +
(可选) CorrectiveAgent + AnswerGenerator + (异步) SessionSummarizer 等 LLM 调用,
全部记录到 LLM_DEBUG_LOG_PATH 配置的 jsonl 文件.

Usage:
    cd server
    .venv/bin/python -m scripts.run_full_chat_test

Output:
    /tmp/full_chat_test.jsonl        每轮 LLM 调用 (server 写的, 这里只是拷贝过来)
    /tmp/full_chat_test_index.json   turn → jsonl line 范围映射
    /tmp/full_chat_test.html         渲染后浏览器视图
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

USER_ID = "plan_test_user"
SESSION_ID = "plan_test_full16"
SERVER = "http://localhost:8002"

# server 当前用的 LLM 日志路径 (启动时通过 env 设的)
SERVER_LLM_LOG = Path("/tmp/phase3_llm_calls.jsonl")

# 输出
OUT_JSONL = Path("/tmp/full_chat_test.jsonl")
OUT_INDEX = Path("/tmp/full_chat_test_index.json")
OUT_HTML = Path("/tmp/full_chat_test.html")

DATASET_DIR = Path(os.getenv("ORGANIZER_DATASET_DIR", "./ecommerce_agent_dataset"))

# 5 张图: 跑鞋A / 咖啡 / Macbook / 耳机 / 跑鞋B
IMAGE_PATHS = [
    DATASET_DIR / "3_服饰运动/images/p_clothes_007_live.jpg",   # T2: Nike Pegasus 41 跑步鞋
    DATASET_DIR / "4_食品生活/images/p_food_001_live.jpg",       # T5: 三顿半咖啡
    DATASET_DIR / "2_数码电子/images/p_digital_006_live.jpg",    # T9: MacBook Pro 14
    DATASET_DIR / "2_数码电子/images/p_digital_018_live.jpg",    # T13: AirPods Pro 3
    DATASET_DIR / "3_服饰运动/images/p_clothes_008_live.jpg",   # T15: Adidas Ultraboost 5
]


@dataclass
class TurnSpec:
    user: str
    image_idx: Optional[int] = None  # 索引 IMAGE_PATHS, None 表示纯文本


SCRIPT: list[TurnSpec] = [
    TurnSpec("想买双跑鞋"),
    TurnSpec("类似这种风格", image_idx=0),
    TurnSpec("再来一款防晒"),
    TurnSpec("敏感肌、不要酒精"),
    TurnSpec("", image_idx=1),  # T5: 只发图
    TurnSpec("推荐 3 款这种豆"),
    TurnSpec("你能记住我喝咖啡的口味吗"),
    TurnSpec("再聊聊手机"),
    TurnSpec("想买这种轻薄本", image_idx=2),
    TurnSpec("预算 8000"),
    TurnSpec("推荐一款降噪耳机"),
    TurnSpec("2000 以内"),
    TurnSpec("类似这个的", image_idx=3),
    TurnSpec("你叫什么"),
    TurnSpec("再看下这种", image_idx=4),
    TurnSpec("你能再推荐一双轻量的吗"),
]


def step(msg: str) -> None:
    print(f"\033[1;36m▶\033[0m {msg}", flush=True)


def upload_image(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as f:
        resp = requests.post(
            f"{SERVER}/api/images",
            files={"file": (path.name, f, "image/jpeg")},
            timeout=30,
        )
    resp.raise_for_status()
    return resp.json()["image_id"]


def send_chat(message: str, image_id: Optional[str]) -> tuple[str, list[str]]:
    """发起一轮 chat, 等到 SSE 完成. 返回 (assistant_text, product_ids)."""
    payload = {"user_id": USER_ID, "session_id": SESSION_ID, "message": message}
    if image_id:
        payload["image_id"] = image_id
    assistant_chunks: list[str] = []
    products: list[str] = []
    with requests.post(f"{SERVER}/api/chat/stream", json=payload, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace")
            if line.startswith("data:"):
                payload_str = line[5:].strip()
                try:
                    obj = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue
                if obj.get("type") == "token":
                    assistant_chunks.append(obj.get("content", ""))
                elif obj.get("type") == "product_cards":
                    products.extend([p.get("product_id", "") for p in obj.get("products", []) if isinstance(p, dict)])
                elif obj.get("type") == "done":
                    break
                elif obj.get("type") == "error":
                    print(f"  [server error] {obj.get('message')}")
    return ("".join(assistant_chunks)).strip(), products


def line_count(p: Path) -> int:
    if not p.exists():
        return 0
    return sum(1 for _ in p.open("r", encoding="utf-8"))


def reset_session() -> None:
    # 清掉 plan_test_full16 session 的所有记录 (跑 seed 的 reset 把 plan_test_user 全清了)
    subprocess.run(
        [sys.executable, "-m", "scripts.seed_attention_test", "--reset"],
        check=True,
        cwd=Path(__file__).resolve().parents[1],  # cwd = server/
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-reset", action="store_true", help="不要 reset, 在已有 history 后追加")
    parser.add_argument("--summarizer-wait", type=float, default=12.0, help="每轮后等多少秒让 SessionSummarizer 异步写完")
    args = parser.parse_args()

    if not args.skip_reset:
        step("Step 1: 重置 plan_test_user 的所有 conversation_turn / session_state / user_memory")
        reset_session()

    # 清空 server LLM 日志, 拿干净的本次记录
    step(f"Step 2: 清空 {SERVER_LLM_LOG}")
    SERVER_LLM_LOG.write_text("")

    step("Step 3: 上传 5 张图")
    image_ids: list[str] = []
    for i, p in enumerate(IMAGE_PATHS):
        image_id = upload_image(p)
        image_ids.append(image_id)
        print(f"  T{[2,5,9,13,15][i]} 用图 #{i}: {p.name} → image_id={image_id}")

    step(f"Step 4: 依次发 16 轮 query 到 {SERVER}/api/chat/stream")
    turn_index: list[dict] = []
    total_start = time.time()
    for i, spec in enumerate(SCRIPT):
        before = line_count(SERVER_LLM_LOG)
        t0 = time.time()
        img = image_ids[spec.image_idx] if spec.image_idx is not None else None
        try:
            assistant, products = send_chat(spec.user, img)
        except Exception as exc:  # noqa: BLE001
            print(f"  T{i+1} ❌ {exc}")
            continue
        # 等异步 SessionSummarizer
        time.sleep(args.summarizer_wait)
        after = line_count(SERVER_LLM_LOG)
        elapsed = time.time() - t0
        print(f"  T{i+1:>2}  {elapsed:>5.1f}s  jsonl[{before:>3}..{after:>3}]  {('图#'+str(spec.image_idx)) if spec.image_idx is not None else '纯文'} | user={spec.user[:30]!r} | answer={assistant[:50]!r}")
        turn_index.append({
            "turn": i + 1,
            "user_text": spec.user,
            "image_id": img,
            "image_path": str(IMAGE_PATHS[spec.image_idx]) if spec.image_idx is not None else None,
            "assistant_preview": assistant[:200],
            "product_ids_returned": products,
            "log_range": [before, after],
        })

    total = time.time() - total_start
    step(f"完成 16 轮, 总耗时 {total/60:.1f} 分钟")

    # 拷贝 jsonl 给后续 render 用
    shutil.copy(SERVER_LLM_LOG, OUT_JSONL)
    OUT_INDEX.write_text(json.dumps(turn_index, ensure_ascii=False, indent=2))
    print(f"  jsonl: {OUT_JSONL} ({line_count(OUT_JSONL)} 行)")
    print(f"  index: {OUT_INDEX}")

    # 渲染 HTML
    step("渲染 HTML")
    render_html(OUT_JSONL, OUT_INDEX, OUT_HTML)
    print(f"  HTML: {OUT_HTML}")
    print(f"\n打开浏览器看: open {OUT_HTML}")
    return 0


def render_html(jsonl: Path, index: Path, out: Path) -> None:
    import html as h
    rows = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    turns = json.loads(index.read_text())

    css = """
    <style>
      body { font-family: -apple-system, sans-serif; max-width: 1200px; margin: 24px auto; padding: 0 20px; color: #222; line-height: 1.5; }
      h1 { border-bottom: 2px solid #444; padding-bottom: 6px; }
      .turn { border: 2px solid #aaa; border-radius: 10px; margin: 22px 0; padding: 16px; background: #fafafa; }
      .turn h2 { margin: 0 0 8px 0; color: #1565c0; }
      .turn-meta { color: #555; font-size: 13px; margin-bottom: 10px; padding: 8px; background: #eef; border-radius: 4px; }
      .llm-call { border: 1px solid #ccc; border-radius: 6px; margin: 10px 0; padding: 10px; background: white; }
      .llm-call h3 { margin: 0 0 4px 0; font-size: 14px; }
      .meta { color: #666; font-size: 12px; margin-bottom: 6px; }
      details { margin: 6px 0; border-left: 3px solid #888; padding-left: 10px; }
      details summary { cursor: pointer; font-weight: 600; padding: 4px 0; user-select: none; font-size: 13px; }
      details[open] summary { color: #0066cc; }
      pre { background: #f5f5f5; padding: 10px; border-radius: 4px; overflow-x: auto; font-size: 11.5px; line-height: 1.4;
            white-space: pre-wrap; word-wrap: break-word; max-height: 500px; overflow-y: auto; }
      .system { border-left-color: #f9a825; }
      .user { border-left-color: #1976d2; }
      .response { border-left-color: #2e7d32; }
      .nav { position: sticky; top: 0; background: white; padding: 10px 0; border-bottom: 1px solid #ddd; margin-bottom: 12px; z-index: 10; }
      .nav a { display: inline-block; margin-right: 6px; padding: 4px 9px; background: #eee; border-radius: 4px; text-decoration: none; color: #333; font-size: 12px; }
      .nav a:hover { background: #1976d2; color: white; }
      .product-pill { display: inline-block; background: #ffe0b2; padding: 2px 6px; border-radius: 3px; font-size: 11px; margin: 2px 2px 2px 0; }
      .preview { color: #2e7d32; font-style: italic; }
    </style>
    """

    short_label = {
        "IntentPlanner": "Plan",
        "CorrectiveAgent": "Review",
        "AnswerGenerator": "Answer",
        "RepairAgent": "Repair",
        "SessionSummarizer": "Sum",
        "ImageAttributeExtractor": "Img",
    }

    nav = '<div class="nav">跳转到轮次: '
    for t in turns:
        nav += f'<a href="#turn{t["turn"]}">T{t["turn"]}</a> '
    nav += '</div>'

    sections = []
    for t in turns:
        lo, hi = t["log_range"]
        in_turn = rows[lo:hi]
        if not in_turn:
            llm_blocks_html = '<div style="color:#888">(此轮无新增 LLM 调用 — 可能后续 turn 一起记)</div>'
        else:
            llm_blocks_html = ""
            for r in in_turn:
                comp = r.get("component", "")
                op = r.get("operation", "")
                short = short_label.get(comp, comp[:8])
                dur = r.get("duration_ms", 0)
                err = r.get("error")
                sys_p = h.escape(r.get("system_prompt") or "")
                user_p = h.escape(r.get("user_prompt") or "")
                resp = h.escape(r.get("response") or "")
                llm_blocks_html += f"""
                <div class="llm-call">
                  <h3>🧠 {comp} <span style="color:#999">({short})</span></h3>
                  <div class="meta">operation: {h.escape(op)} &nbsp;·&nbsp; {dur:.0f} ms &nbsp;·&nbsp; sys {len(r.get('system_prompt') or '')}c · user {len(r.get('user_prompt') or '')}c · resp {len(r.get('response') or '')}c</div>
                  {f'<div style="color:#c00">ERROR: {h.escape(err)}</div>' if err else ''}
                  <details class="system">
                    <summary>📋 system_prompt</summary>
                    <pre>{sys_p}</pre>
                  </details>
                  <details class="user" open>
                    <summary>👤 user_prompt</summary>
                    <pre>{user_p}</pre>
                  </details>
                  <details class="response" open>
                    <summary>🤖 response</summary>
                    <pre>{resp}</pre>
                  </details>
                </div>
                """

        prods_html = " ".join(f'<span class="product-pill">{h.escape(p)}</span>' for p in t["product_ids_returned"])
        img_info = ""
        if t.get("image_path"):
            img_info = f'<br><b>带图</b>: {h.escape(Path(t["image_path"]).name)} → image_id={h.escape(t["image_id"] or "")}'
        sections.append(f"""
        <div class="turn" id="turn{t['turn']}">
          <h2>T{t['turn']}</h2>
          <div class="turn-meta">
            <b>用户消息</b>: {h.escape(t['user_text']) or '<i>(空, 只发图)</i>'}{img_info}<br>
            <b>助手回复 (前 200 字)</b>: <span class="preview">{h.escape(t['assistant_preview'])}</span><br>
            <b>返回的 product_ids</b>: {prods_html or '<i>无</i>'}<br>
            <b>本轮 LLM 调用数</b>: {hi - lo}
          </div>
          {llm_blocks_html}
        </div>
        """)

    html_text = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Phase3 16-Turn Full Chat Log</title>{css}</head><body>
<h1>Phase3 真跑 16 轮 chat — 每轮 LLM 输入/输出</h1>
<p style="color:#666">user_id: <code>{USER_ID}</code> &nbsp;·&nbsp; session_id: <code>{SESSION_ID}</code> &nbsp;·&nbsp; 共 {len(turns)} 轮 / {len(rows)} 次 LLM 调用</p>
{nav}
{"".join(sections)}
</body></html>
"""
    out.write_text(html_text, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
