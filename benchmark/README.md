# Benchmark Working Plan

本目录现在分成三层：

- 主数据集（`datasets/`）：当前主榜默认读取这里的六个数据文件。
- 旧数据归档（`legacy/`）：保留旧 benchmark 口径和历史报告，方便回查，不再作为主入口。
- 评测脚本（`eval_all.py`）：新版主入口，默认跑 65 条主榜计分样例；也可用 `--datasets` 手动跑子集。

## 默认主榜口径

当前最终冲刺第 1 块默认主榜共 65 个计分 turn：

- `retrieval_core.jsonl`：30 个明确商品推荐核心 case。
- `personalized_coarse.jsonl`：10 个粗品类 + 个性化 case。
- `cross_scenario.jsonl`：10 个场景/套装/多子类 case。
- `out_of_catalog.jsonl`：5 个库外商品拒绝 case。
- `route_boundary.jsonl`：5 个直接回答/澄清/无商品边界 case。
- `context_dialogues.jsonl`：5 组多轮上下文；每组会完整执行 setup turn 和追问 turn，但只把最后追问 turn 计入主报告。

## 数据集结构

| 数据集 | 数量 | 中文职责 | 主要指标 |
|---|---:|---|---|
| `retrieval_core.jsonl` | 30 | 明确商品推荐核心集 | `hit@5`, `recall@5`, `forbidden_clean@5` |
| `personalized_coarse.jsonl` | 10 | 粗品类 + 明确个性化 | `hit@5`, `recall@5`, `diverse_met@5`, `forbidden_clean@5`, `profile_used_ok` |
| `cross_scenario.jsonl` | 10 | 场景/套装/多子类推荐 | `hit@5`, `recall@5`, `diverse_met@5`, `forbidden_clean@5` |
| `out_of_catalog.jsonl` | 5 | 库外商品拒绝 | `route_ok`, `forbidden_clean@5` |
| `route_boundary.jsonl` | 5 | 直接回答/澄清/无商品边界 | `route_ok` |
| `context_dialogues.jsonl` | 5 组 / 5 计分 turn | 多轮上下文复用 | `route_ok`, `no_unwanted_retrieval`, `referenced_products_loaded` |

## 统一字段

单轮数据使用下面的最小结构：

```json
{
  "id": "retrieval_001",
  "query": "预算300以内，想要一支通勤和夏天户外都能用的清爽防晒。",
  "expected_route": "recommend",
  "relevant_product_ids": [],
  "acceptable_product_ids": [],
  "forbidden_product_ids": [],
  "min_diverse_subcategories": 0,
  "profile_fixture": null,
  "expected_tool_calls": ["product_search"],
  "forbidden_tool_calls": ["profile_lookup", "image_search"],
  "expected_internal_actions": [],
  "forbidden_internal_actions": ["repair_plan_generated"],
  "notes": ""
}
```

多轮数据使用 `turns` 数组，每轮同样只写最终期望和关键链路期望。

## 指标口径

- 商品命中（`hit@5`）：top-5 里至少命中一个相关或可接受商品。
- 商品召回（`recall@5`）：top-5 命中的相关商品数 / 标注相关商品总数。
- 多样性达标（`diverse_met@5`）：top-5 覆盖的子类数量达到 `min_diverse_subcategories`。
- 禁推清洁度（`forbidden_clean@5`）：top-5 没有出现 `forbidden_product_ids`。

主报告只展示这些最终指标和少量专项指标；工具调用、repair、cache/repository 读取只作为失败诊断，不堆到首页。

Repair 触发属于 Orchestrator 内部修复机制，不作为主 benchmark 数据集。需要验证 repair loop 时，应使用诊断测试或 mock 初检失败场景，而不是在端到端主榜强制要求 `repair_plan_generated`。

## 跑评测

新版主入口：

```bash
cd benchmark
python eval_all.py
```

也可以从项目根目录运行：

```bash
python benchmark/eval_all.py
```

Smoke run：

```bash
python benchmark/eval_all.py --limit 2
```

只跑指定数据集：

```bash
python benchmark/eval_all.py --datasets retrieval_core,route_boundary
```

评测入口会：

1. 读取 `datasets/*.jsonl`。
2. 每条 case 调 Orchestrator 完整链路。
3. 统一抽取 `final_route`、product cards、trace tool calls、internal actions。
4. 输出一个总报告：
   - `pass_rate`
   - `route_ok`
   - `hit@5`
   - `recall@5`
   - `diverse_met@5`
   - `forbidden_clean@5`
5. 写入：
   - `benchmark/report.md`
   - `benchmark/report.json`

详细失败样例写入 JSON；Markdown 主报告只放摘要和前 30 条失败。
