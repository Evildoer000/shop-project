# Shop Agent 阿里云部署手册

本文档用于把当前 Shop Agent 项目部署到阿里云 ECS。默认部署完整的 FastAPI 后端、React Web 页面和项目依赖的数据库与检索服务。

## 1. 当前部署结构

项目使用两个 Compose 文件：

- `docker-compose.yml`：后端及基础设施。
- `docker-compose.web.yml`：React Web 页面。

正常部署会启动：

| 服务 | 容器名 | 用途 |
| --- | --- | --- |
| PostgreSQL | `ecommerce-rag-postgres` | 用户、会话、记忆、商品和运行记录 |
| etcd | `ecommerce-rag-etcd` | Milvus 依赖 |
| MinIO | `ecommerce-rag-minio` | Milvus 对象存储 |
| Milvus | `ecommerce-rag-milvus` | 文本和图片向量检索 |
| Elasticsearch | `ecommerce-rag-elasticsearch` | IK 中文关键词检索 |
| 初始化任务 | `ecommerce-rag-backend-bootstrap` | 建表、导入和检查索引 |
| FastAPI | `ecommerce-rag-backend` | REST API 和 SSE 流式聊天 |
| React Web | `ecommerce-rag-web` | 浏览器端导购页面 |

`backend-bootstrap` 是一次性任务，成功后显示 `Exited (0)` 属于正常状态。

## 2. 服务器要求

推荐配置：

- Ubuntu 22.04 LTS 或 Ubuntu 24.04 LTS。
- 4 vCPU 起步。
- 8 GB 内存起步，推荐 16 GB。
- 100 GB 系统盘。
- x86_64 / amd64 架构。

PostgreSQL、Milvus、MinIO 和 Elasticsearch 会同时运行，不建议使用 2 核 4 GB 服务器。

## 3. 阿里云安全组

### 直接使用公网 IP

开放：

| 端口 | 来源 | 用途 |
| --- | --- | --- |
| `22` | 部署人员公网 IP | SSH |
| `3000` | 需要访问页面的 IP | React Web |
| `8000` | 需要访问接口的 IP | FastAPI |

### 使用域名和 HTTPS

开放：

- `22`：仅部署人员公网 IP。
- `80`：HTTP。
- `443`：HTTPS。

不要向公网开放：

- `5432`：PostgreSQL。
- `9200`：Elasticsearch。
- `9000`、`9001`：MinIO。
- `19530`、`9091`：Milvus。

## 4. 安装 Docker

登录服务器：

```bash
ssh root@服务器公网IP
```

安装基础工具：

```bash
apt-get update
apt-get install -y ca-certificates curl gnupg git vim unzip lsof htop
timedatectl set-timezone Asia/Shanghai
```

卸载可能冲突的旧包：

```bash
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do
  apt-get remove -y "$pkg" || true
done
```

添加 Docker 官方软件源：

```bash
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" \
  > /etc/apt/sources.list.d/docker.list

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin
```

启动并验证：

```bash
systemctl enable --now docker
docker --version
docker compose version
docker run --rm hello-world
```

本项目使用 `docker compose`，不是旧版 `docker-compose`。

## 5. 上传项目

推荐部署目录：

```bash
mkdir -p /opt/shop-agent
cd /opt/shop-agent
```

### 使用 Git

```bash
cd /opt
git clone 项目仓库地址 shop-agent
cd shop-agent
```

### 使用压缩包

在本地电脑执行：

```bash
scp shop-agent-deploy.tar.gz root@服务器公网IP:/opt/
```

在服务器执行：

```bash
cd /opt
mkdir -p shop-agent
tar -xzf shop-agent-deploy.tar.gz -C shop-agent
cd shop-agent
```

确认以下文件存在：

```text
docker-compose.yml
docker-compose.web.yml
.env.example
server/Dockerfile
web/Dockerfile
elasticsearch/Dockerfile
```

## 6. 配置 `.env`

```bash
cd /opt/shop-agent
cp .env.example .env
vim .env
```

不要把真实 `.env` 或 API Key 提交到 Git。

### 聊天模型

当前项目使用 OpenAI 兼容接口：

```env
LLM_API_KEY=填写聊天模型APIKey
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-v4-flash
LLM_TIMEOUT_SECONDS=30
LLM_MAX_RETRIES=2
LLM_RETRY_BACKOFF_SECONDS=0.8
LLM_THINKING_TYPE=disabled
```

如果使用其他模型服务，替换对应的 Base URL、模型名和 API Key。

### 文本 Embedding

与当前索引配置一致的示例：

```env
EMBEDDING_API_KEY=填写Embedding的APIKey
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIM=1024
EMBEDDING_TIMEOUT_SECONDS=30
```

注意：

- `EMBEDDING_DIM` 必须和模型输出维度一致。
- 当前 `text-embedding-v4` 配置使用 `1024` 维。
- 即使商品向量已经存在，用户查询转向量时仍需要有效的 Embedding Key。

### Elasticsearch

```env
ELASTICSEARCH_ENABLED=true
ELASTICSEARCH_URL=http://elasticsearch:9200
ELASTICSEARCH_INDEX=products
ELASTICSEARCH_TIMEOUT_SECONDS=10
```

项目的 Elasticsearch 镜像会自动安装 ES `8.15.5` 对应的 IK 中文分词插件，不需要手工安装。

### 图片检索，可选

```env
DASHSCOPE_API_KEY=填写DashScopeKey
IMAGE_EMBEDDING_BACKEND=dashscope
IMAGE_EMBEDDING_API_KEY=
IMAGE_EMBEDDING_MODEL=tongyi-embedding-vision-flash
IMAGE_EMBEDDING_DIM=768
IMAGE_RELEVANCE_THRESHOLD=0.20
```

不配置图片 Embedding Key 时，初始化任务会跳过图片索引，不影响文本导购。

### 图片理解，可选

```env
VLM_API_KEY=填写VLM的APIKey
VLM_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
VLM_MODEL=填写实际模型或Endpoint ID
VLM_TIMEOUT_SECONDS=10
VLM_MAX_RETRIES=0
```

### 重排序

当前默认不调用远程 reranker：

```env
RERANK_BACKEND=hybrid
RERANK_API_KEY=
RERANK_BASE_URL=
RERANK_MODEL=
RERANK_TIMEOUT_SECONDS=30
```

### Web 接口地址

使用公网 IP：

```env
WEB_API_BASE_URL=http://服务器公网IP:8000
WEB_PORT=3000
```

使用同一 HTTPS 域名：

```env
WEB_API_BASE_URL=https://你的域名
WEB_PORT=3000
```

不要保留：

```env
WEB_API_BASE_URL=http://localhost:8000
```

浏览器里的 `localhost` 指访问者自己的电脑，并不是阿里云服务器。

`WEB_API_BASE_URL` 是前端构建变量，修改后必须重新构建 Web 镜像。

### 初始化开关

首次部署保持：

```env
BOOTSTRAP_FORCE_REINDEX=false
BOOTSTRAP_FORCE_ES_REINDEX=false
BOOTSTRAP_FORCE_IMAGE_REINDEX=false
```

## 7. 启动服务

先校验 Compose：

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.web.yml \
  config
```

构建并启动：

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.web.yml \
  up -d --build
```

第一次构建需要下载多个 Docker 镜像和 IK 插件，时间取决于服务器网络。

查看状态：

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.web.yml \
  ps -a
```

观察初始化：

```bash
docker compose logs -f backend-bootstrap
```

初始化任务会依次：

1. 等待 PostgreSQL、Milvus 和 Elasticsearch。
2. 创建数据库表。
3. 初始化商品数据。
4. 检查或构建 Milvus 文本向量索引。
5. 检查或构建 Elasticsearch 中文关键词索引。
6. 在配置图片 Embedding 时检查图片向量索引。

FastAPI 后端只有在初始化成功后才会启动。

查看后端和 Web 日志：

```bash
docker compose logs -f backend
docker compose \
  -f docker-compose.yml \
  -f docker-compose.web.yml \
  logs -f web
```

## 8. 验证服务

### 健康检查

在服务器执行：

```bash
curl http://127.0.0.1:8000/health
```

预期：

```json
{"status":"ok"}
```

### Web 页面

公网 IP 模式访问：

```text
http://服务器公网IP:3000
```

当前登录规则：

- 手机号输入 5 到 20 位数字。
- 默认密码为 `88888`。
- 新手机号首次登录时自动创建用户。
- 不同用户的会话和记忆相互隔离。

该登录逻辑用于项目演示，不是正式生产账号系统。

### SSE 聊天接口

```bash
curl -N -X POST http://127.0.0.1:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "user_id": "deploy_test_user",
    "session_id": "deploy_test_session",
    "message": "我是油皮，预算150以内，推荐夏天通勤不闷的防晒霜",
    "image_id": null
  }'
```

正常情况下会持续返回 SSE 事件和最终回答。

### Elasticsearch

```bash
curl http://127.0.0.1:9200
curl http://127.0.0.1:9200/products/_count
```

### 期望容器状态

- `postgres`：healthy。
- `elasticsearch`：healthy。
- `backend-bootstrap`：Exited (0)。
- `backend`：healthy。
- `web`：Up。
- `milvus`、`etcd`、`minio`：Up。

## 9. 域名和 HTTPS

推荐使用宿主机 Nginx，把 Web 和 API 放在同一个域名下。

安装 Nginx：

```bash
apt-get install -y nginx
systemctl enable --now nginx
```

把 `.env` 修改为：

```env
WEB_API_BASE_URL=https://你的域名
```

重新构建 Web：

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.web.yml \
  build --no-cache web

docker compose \
  -f docker-compose.yml \
  -f docker-compose.web.yml \
  up -d web
```

创建配置：

```bash
vim /etc/nginx/sites-available/shop-agent.conf
```

写入：

```nginx
server {
    listen 80;
    server_name 你的域名;

    client_max_body_size 12m;

    location = /health {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 3600;
        proxy_send_timeout 3600;
    }

    location /uploads/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
    }

    location /dataset/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
    }

    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

启用：

```bash
ln -s /etc/nginx/sites-available/shop-agent.conf \
  /etc/nginx/sites-enabled/shop-agent.conf
nginx -t
systemctl reload nginx
```

申请 HTTPS 证书：

```bash
apt-get install -y certbot python3-certbot-nginx
certbot --nginx -d 你的域名
```

验证：

```bash
curl https://你的域名/health
```

HTTPS 正常后，阿里云安全组不再需要开放 `3000` 和 `8000`。

## 10. 常用命令

查看状态：

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.web.yml \
  ps -a
```

查看日志：

```bash
docker compose logs -f backend-bootstrap
docker compose logs -f backend
docker compose logs -f elasticsearch
docker compose \
  -f docker-compose.yml \
  -f docker-compose.web.yml \
  logs -f web
```

重启后端：

```bash
docker compose restart backend
```

修改前端 API 地址后重新构建：

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.web.yml \
  build --no-cache web

docker compose \
  -f docker-compose.yml \
  -f docker-compose.web.yml \
  up -d web
```

更新项目：

```bash
git pull
docker compose \
  -f docker-compose.yml \
  -f docker-compose.web.yml \
  up -d --build
```

停止服务并保留数据：

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.web.yml \
  down
```

不要随意执行：

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.web.yml \
  down -v
```

`-v` 会删除数据库和索引数据卷。

## 11. 常见问题

### 后端没有启动

检查：

```bash
docker compose ps -a
docker compose logs backend-bootstrap
```

如果 `backend-bootstrap` 是 `Exited (1)`，解决日志中的第一个异常后重新执行：

```bash
docker compose up -d --force-recreate backend-bootstrap
docker compose logs -f backend-bootstrap
docker compose up -d backend
```

### Elasticsearch 构建失败

构建阶段需要访问 Docker Registry 和 `get.infini.cloud` 下载 IK 插件。网络恢复后执行：

```bash
docker compose build --no-cache elasticsearch
docker compose up -d elasticsearch
docker compose up -d --force-recreate backend-bootstrap
```

### Embedding 维度错误

确认模型和维度一致：

```env
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIM=1024
```

更换模型或维度后不能继续使用旧向量索引。

### Web 打开但后端离线

先从访问 Web 的电脑打开：

```text
http://服务器公网IP:8000/health
```

检查：

- 安全组是否开放 `8000`。
- `WEB_API_BASE_URL` 是否使用公网 IP 或域名。
- 修改地址后是否重新构建 Web。
- HTTPS 页面是否调用了 HTTP API，导致 Mixed Content。

### SSE 中途断开

检查后端：

```bash
docker compose logs -f backend
```

使用 Nginx 时，确认 `/api/` 包含：

```nginx
proxy_buffering off;
proxy_cache off;
proxy_read_timeout 3600;
```

### 内存不足

```bash
free -h
docker stats
dmesg | grep -i -E "killed process|out of memory"
```

如果 Elasticsearch 或 Milvus 被 OOM Kill，优先升级服务器内存。

## 12. 最短部署流程

服务器已安装 Docker 时，按顺序执行：

```bash
cd /opt/shop-agent
cp .env.example .env
vim .env

docker compose \
  -f docker-compose.yml \
  -f docker-compose.web.yml \
  config

docker compose \
  -f docker-compose.yml \
  -f docker-compose.web.yml \
  up -d --build

docker compose logs -f backend-bootstrap

docker compose \
  -f docker-compose.yml \
  -f docker-compose.web.yml \
  ps -a

curl http://127.0.0.1:8000/health
```

公网 IP 模式访问：

```text
http://服务器公网IP:3000
```

默认登录密码：

```text
88888
```
