# 个性化电商 RAG Agent

本仓库实现了基于主办方商品数据集的个性化电商 RAG Agent。商品数据使用主办方提供的数据集，不使用临时编造商品。

```text
FastAPI -> Query Planner -> PostgreSQL 硬过滤 -> LlamaIndex + Milvus 召回
  -> 个性化重排 -> SSE 返回 token / 商品卡片 / 决策摘要
```

## Docker 一键启动

推荐先用 Docker 启动完整后端链路：

```bash
cp .env.example .env
docker compose up -d --build
curl http://localhost:8000/health
```

这会启动 PostgreSQL、Milvus、后端初始化任务和 FastAPI API。初始化任务会读取仓库内 `ecommerce_agent_dataset`，首次构建 `products` 表、文字向量索引和图片向量索引；后续启动会根据数据集、embedding 配置和 Milvus collection 状态跳过重复构建，避免重复消耗 embedding API。

图片向量索引默认使用远程图片 embedding。未配置 `DASHSCOPE_API_KEY` 或 `IMAGE_EMBEDDING_API_KEY` 时，Docker bootstrap 会跳过图片索引，不影响文字导购链路启动。

如需重建索引，可使用下文的 `backend-bootstrap` 命令；默认启动会根据商品数据、embedding 配置和 Milvus collection 状态跳过重复构建。

## 本地质量门禁

本项目的 benchmark 会真实调用 IntentPlanner / CorrectiveAgent / AnswerGenerator。为避免把模型 API Key 注入公共 CI 环境，仓库不默认启用 GitHub Actions 自动评测。提交前建议在本地或受控机器上启动 PostgreSQL 和 Milvus，执行商品与向量索引初始化，然后运行：

```text
pytest
benchmark/eval_all.py
benchmark/check_report.py
```

评测报告会写入 `benchmark/report.json` 和 `benchmark/report.md`。`check_report.py` 会检查 benchmark 总轮数、`pass_rate`、`route_ok`、`forbidden_clean@5` 和 `context_reuse_ok`，防止 Agent 主链路能力回退。

由于 benchmark 会真实调用模型，需要在本地 `.env` 中配置：

```text
LLM_API_KEY
LLM_BASE_URL
LLM_MODEL
```

`EMBEDDING_*` 和图片 embedding Key 可选；未配置时会使用现有降级路径或跳过图片索引。

## 本地开发启动

1. 复制环境变量：

```bash
cp .env.example .env
```

确认 `.env` 中的 `ORGANIZER_DATASET_DIR` 指向主办方数据目录，例如：

```text
./ecommerce_agent_dataset
```

2. 启动 PostgreSQL 和 Milvus：

```bash
docker compose up -d postgres etcd minio milvus
```

3. 安装后端依赖：

```bash
cd server
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
```

4. 初始化数据库和向量索引。脚本会清空并重建 `products` 表，确保只保留主办方商品：

```bash
python scripts/seed_products.py
```

5. 启动 API：

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

6. 测试 SSE：

```bash
curl -N -X POST http://localhost:8000/api/chat/stream ^
  -H "Content-Type: application/json" ^
  -d "{\"user_id\":\"u001\",\"session_id\":\"s001\",\"message\":\"我是油皮，预算150以内，推荐一款夏天用不闷的防晒\"}"
```

如果未配置 `LLM_API_KEY`，后端会使用模板化 Answer Generator，仍可完整返回检索商品、推荐理由和决策过程。商品卡片的 `image_url` 会指向 `/dataset/...`，由 FastAPI 挂载主办方图片目录提供访问。

## 模型配置

语言模型和向量模型都通过 `.env` 配置，使用 OpenAI-compatible API 路径：

```env
LLM_API_KEY=你的语言模型Key
LLM_BASE_URL=https://你的网关/v1
LLM_MODEL=deepseek-v4-flash
LLM_TIMEOUT_SECONDS=30
LLM_THINKING_TYPE=disabled
LLM_INPUT_PRICE_PER_1K=输入每千token价格
LLM_OUTPUT_PRICE_PER_1K=输出每千token价格
LLM_STREAM_INCLUDE_USAGE=true

EMBEDDING_API_KEY=你的向量模型Key
EMBEDDING_BASE_URL=https://你的网关/v1
EMBEDDING_MODEL=你的embedding模型名
EMBEDDING_DIM=模型输出维度
EMBEDDING_TIMEOUT_SECONDS=30
```

未配置 `LLM_*` 时，回答生成会自动降级为模板化推荐。未配置 `EMBEDDING_*` 时，向量检索会自动降级为本地 hash embedding，方便先把后端闭环跑起来。
`LLM_INPUT_PRICE_PER_1K` 和 `LLM_OUTPUT_PRICE_PER_1K` 可选，只影响 trace 里的 `estimated_cost` 估算；模型返回的 `usage` 和本地 `latency_ms` 会随 Langfuse metadata 一起记录。
`LLM_THINKING_TYPE=disabled` 用于 DeepSeek endpoint，关闭默认 thinking mode 以降低可见首字延迟；非 DeepSeek endpoint 不会发送该参数。

配置完成后可以检查模型连通性：

```bash
cd server
python scripts/check_models.py
```

如果商品数据、embedding 模型、`EMBEDDING_DIM` 或索引策略改变，需要重建文字索引：

```powershell
$env:BOOTSTRAP_FORCE_REINDEX="true"
docker compose run --rm backend-bootstrap
docker compose restart backend
Remove-Item Env:\BOOTSTRAP_FORCE_REINDEX
```

如果图片数据、图片 embedding 模型或 `IMAGE_EMBEDDING_DIM` 改变，需要重建图片索引：

```powershell
$env:BOOTSTRAP_FORCE_IMAGE_REINDEX="true"
docker compose run --rm backend-bootstrap
docker compose restart backend
Remove-Item Env:\BOOTSTRAP_FORCE_IMAGE_REINDEX
```

## 索引策略

当前文字索引采用“每商品一个聚合 chunk”，PostgreSQL `products` 表是 source of truth，Milvus collection 可随时重建；图片向量使用独立 collection，不和文字 embedding 混用。
