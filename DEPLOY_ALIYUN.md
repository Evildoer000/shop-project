# 阿里云 ECS 部署文档

本文档说明如何把本项目部署到一台新的阿里云 ECS 服务器上。推荐使用 Docker Compose 部署，因为项目已经提供了 `docker-compose.yml`，会一次性启动 PostgreSQL、Milvus、etcd、MinIO、后端初始化任务和 FastAPI 后端服务。

## 0. 部署目标

部署完成后，服务器上会运行这些容器：

- `ecommerce-rag-postgres`：PostgreSQL，保存商品、会话、用户记忆等结构化数据。
- `ecommerce-rag-etcd`：Milvus 依赖。
- `ecommerce-rag-minio`：Milvus 对象存储依赖。
- `ecommerce-rag-milvus`：向量数据库，保存文本向量和图片向量。
- `ecommerce-rag-backend-bootstrap`：一次性初始化任务，导入商品数据并构建向量索引。
- `ecommerce-rag-backend`：FastAPI 后端，默认监听 `8000` 端口。

## 0.1 已安装 Docker 后从这里开始

如果你的服务器已经能正常输出：

```bash
docker --version
docker compose version
```

并且 Docker 镜像加速器已经配置完成，可以跳过本文档第 4、5、6 节，直接按下面顺序做：

1. 在本地电脑项目根目录打包项目。
2. 从本地电脑用 `scp` 上传压缩包到服务器 `/opt/`。
3. 在服务器 `/opt` 解压项目。
4. 在服务器项目目录复制 `.env.example` 为 `.env`。
5. 修改 `.env` 中的模型和 embedding 配置。
6. 执行 `docker compose up -d --build`。
7. 用 `/health` 和 `/api/chat/stream` 验证。

注意：`scp` 上传命令是在你本地电脑执行，不是在服务器 SSH 里执行。看到类似下面的提示符，说明你已经在服务器里了：

```bash
root@your-server:~#
```

## 1. 推荐服务器配置

建议最低配置：

- 操作系统：Ubuntu 22.04 LTS 或 Ubuntu 24.04 LTS，64 位。
- CPU：4 vCPU 起步。
- 内存：8 GB 起步，推荐 16 GB。
- 系统盘：80 GB 起步，推荐 100 GB 以上。
- 架构：x86_64 / amd64。

说明：

- Milvus、PostgreSQL、MinIO 都会占用内存和磁盘。2 核 4 GB 机器可能能启动，但稳定性较差。
- 如果需要构建图片向量索引，需要配置 DashScope 或其他图片 embedding key，否则项目会跳过图片索引，不影响文本推荐链路。
- 如果服务器在国内，拉取 Docker Hub 镜像可能较慢，需要配置镜像加速。

## 2. 阿里云控制台准备

### 2.1 创建 ECS

建议选择：

- 地域：按你的用户位置选择，例如华东、华北、华南。
- 镜像：Ubuntu 22.04 LTS 或 Ubuntu 24.04 LTS。
- 实例规格：至少 4 vCPU / 8 GB。
- 登录方式：SSH 密钥优先，也可以使用密码。

### 2.2 配置安全组

入方向至少开放：

| 端口 | 协议 | 来源 | 用途 |
| --- | --- | --- | --- |
| 22 | TCP | 你的公网 IP | SSH 登录 |
| 8000 | TCP | 你的公网 IP 或 0.0.0.0/0 | 后端 API |

不建议直接对公网开放：

- `5432`：PostgreSQL
- `19530` / `9091`：Milvus
- `9000` / `9001`：MinIO

当前 `docker-compose.yml` 中这些内部服务有 `ports` 映射。如果安全组不开放这些端口，公网仍然访问不到。更严格的做法见本文档第 8 节。

## 3. 登录服务器

在本地终端执行：

```bash
ssh root@你的服务器公网IP
```

如果用密钥：

```bash
ssh -i /path/to/your-key.pem root@你的服务器公网IP
```

登录后先查看系统信息：

```bash
uname -a
cat /etc/os-release
free -h
df -h
```

## 4. 安装基础工具

Ubuntu 执行：

```bash
apt-get update
apt-get install -y ca-certificates curl gnupg git vim unzip lsof htop
```

建议设置时区：

```bash
timedatectl set-timezone Asia/Shanghai
timedatectl
```

## 5. 安装 Docker Engine 和 Compose 插件

推荐使用 Docker 官方 apt 源安装，而不是直接安装 Ubuntu 源里的 `docker.io`。Docker 官方文档也建议卸载冲突包后，从 Docker 官方仓库安装 `docker-ce`、`docker-ce-cli`、`containerd.io`、`docker-buildx-plugin` 和 `docker-compose-plugin`。

参考：

- Docker Ubuntu 官方安装文档：https://docs.docker.com/engine/install/ubuntu/
- Docker CentOS 官方安装文档：https://docs.docker.com/engine/install/centos/

### 5.1 卸载可能冲突的旧包

```bash
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do
  apt-get remove -y "$pkg" || true
done
```

### 5.2 添加 Docker 官方 GPG key 和 apt 源

```bash
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" \
  > /etc/apt/sources.list.d/docker.list

apt-get update
```

### 5.3 安装 Docker

```bash
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

### 5.4 启动 Docker 并设置开机自启

```bash
systemctl enable docker
systemctl start docker
systemctl status docker --no-pager
```

### 5.5 验证版本

```bash
docker --version
docker compose version
docker run --rm hello-world
```

注意：

- 本项目使用的是新版命令 `docker compose`，不是旧版 `docker-compose`。
- 如果 `docker compose version` 不存在，说明 Compose 插件没有安装成功。

## 6. 可选：配置 Docker 镜像加速

如果拉镜像很慢，可以配置 Docker registry mirrors。阿里云控制台通常会给每个账号提供专属镜像加速地址。

编辑：

```bash
mkdir -p /etc/docker
vim /etc/docker/daemon.json
```

示例：

```json
{
  "registry-mirrors": [
    "https://你的阿里云镜像加速地址"
  ],
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m",
    "max-file": "3"
  }
}
```

重启 Docker：

```bash
systemctl daemon-reload
systemctl restart docker
docker info | grep -A 5 "Registry Mirrors"
```

## 7. 上传或拉取项目代码

推荐把项目放在 `/opt` 下：

```bash
mkdir -p /opt
cd /opt
```

### 方式 A：从 Git 仓库拉取

如果项目已经推到 GitHub/Gitee：

```bash
git clone 你的仓库地址 ecommerce-rag-agent
cd ecommerce-rag-agent
```

### 方式 B：从本地打包上传

在你的本地电脑项目根目录执行。Windows PowerShell 示例：

```powershell
cd G:\ecommerce-rag-agent-codex-phase3-corrective-reflection-repair-worker
tar --exclude='.git' --exclude='client/android/.gradle' --exclude='client/android/build' -czf ecommerce-rag-agent.tar.gz .
scp .\ecommerce-rag-agent.tar.gz root@你的服务器公网IP:/opt/
```

如果使用 SSH 密钥登录，把上传命令改成：

```powershell
scp -i C:\路径\你的密钥.pem .\ecommerce-rag-agent.tar.gz root@你的服务器公网IP:/opt/
```

Linux/macOS 本地电脑示例：

```bash
cd /path/to/ecommerce-rag-agent-codex-phase3-corrective-reflection-repair-worker
tar --exclude='.git' --exclude='client/android/.gradle' --exclude='client/android/build' -czf ecommerce-rag-agent.tar.gz .
scp ecommerce-rag-agent.tar.gz root@你的服务器公网IP:/opt/
```

在服务器执行：

```bash
cd /opt
mkdir -p ecommerce-rag-agent
tar -xzf ecommerce-rag-agent.tar.gz -C ecommerce-rag-agent
cd ecommerce-rag-agent
```

确认关键文件存在：

```bash
ls -lah
ls -lah server
ls -lah ecommerce_agent_dataset
```

必须能看到：

- `docker-compose.yml`
- `.env.example`
- `server/Dockerfile`
- `server/requirements.txt`
- `ecommerce_agent_dataset`

## 8. 生产部署前建议修改 docker-compose 端口暴露

默认 `docker-compose.yml` 会把 PostgreSQL、MinIO、Milvus 端口映射到宿主机。如果安全组不放行这些端口，公网访问不到；但从安全角度，建议进一步改成只绑定本机 `127.0.0.1`。

编辑：

```bash
vim docker-compose.yml
```

把类似：

```yaml
ports:
  - "5432:5432"
```

改成：

```yaml
ports:
  - "127.0.0.1:5432:5432"
```

建议这样改：

```yaml
postgres:
  ports:
    - "127.0.0.1:5432:5432"

minio:
  ports:
    - "127.0.0.1:9000:9000"
    - "127.0.0.1:9001:9001"

milvus:
  ports:
    - "127.0.0.1:19530:19530"
    - "127.0.0.1:9091:9091"
```

后端 `8000` 如果要公网访问，可以保留：

```yaml
backend:
  ports:
    - "8000:8000"
```

如果后面要用 Nginx 反向代理，也可以改成：

```yaml
backend:
  ports:
    - "127.0.0.1:8000:8000"
```

## 9. 配置环境变量

复制环境变量文件：

```bash
cp .env.example .env
vim .env
```

非常重要：

- Docker 部署场景下，不需要手动修改 `.env` 里的 `DATABASE_URL`、`MILVUS_URI`、`ORGANIZER_DATASET_DIR`。
- `docker-compose.yml` 会把它们覆盖成容器内部地址：
  - `DATABASE_URL=postgresql+psycopg://rag:rag@postgres:5432/ecommerce_rag`
  - `MILVUS_URI=http://milvus:19530`
  - `ORGANIZER_DATASET_DIR=/data/ecommerce_agent_dataset`
- 你真正需要重点填写的是 `LLM_*`、`EMBEDDING_*`，以及可选的图片/VLM/Langfuse 配置。

重点配置这些项。

### 9.1 LLM 配置

必须配置，否则 Agent 主链路无法真实调用模型，部分逻辑会降级。

```env
LLM_API_KEY=你的大模型APIKey
LLM_BASE_URL=你的大模型BaseURL
LLM_MODEL=你的模型名
LLM_TIMEOUT_SECONDS=30
LLM_MAX_RETRIES=2
LLM_RETRY_BACKOFF_SECONDS=0.8
LLM_THINKING_TYPE=disabled
LLM_STREAM_INCLUDE_USAGE=true
```

项目 `docker-compose.yml` 默认模型名是：

```env
LLM_MODEL=deepseek-v4-flash
```

如果你用的是其他供应商，需要按供应商实际模型名填写。

### 9.2 文本 embedding 配置

完整体验文本导购、文本检索、商品卡和最终回答时，建议必须配置文本 embedding。项目文档里也明确说明，不建议把“未配置文本 embedding 的兜底路径”作为评审体验方式。

```env
EMBEDDING_API_KEY=你的EmbeddingKey
EMBEDDING_BASE_URL=你的EmbeddingBaseURL
EMBEDDING_MODEL=你的Embedding模型名
EMBEDDING_DIM=384
EMBEDDING_TIMEOUT_SECONDS=30
```

注意：

- `EMBEDDING_DIM` 必须和实际 embedding 模型输出维度一致。
- 如果维度填错，Milvus 建索引或检索时可能报错。
- 如果你更换了 embedding 模型或 `EMBEDDING_DIM`，需要重新跑 bootstrap 重建文本向量索引。

### 9.3 图片 embedding 配置

只体验文本导购时，可以先不配置图片 key；Docker bootstrap 会跳过图片索引或降级，不影响文本检索、商品卡和回答生成。

如果要体验“拍照找货”、图片理解和图片相似召回，再配置图片 embedding。默认走 DashScope：

```env
DASHSCOPE_API_KEY=你的DashScopeKey
IMAGE_EMBEDDING_BACKEND=dashscope
IMAGE_EMBEDDING_API_KEY=
IMAGE_EMBEDDING_MODEL=tongyi-embedding-vision-flash-2026-03-06
IMAGE_EMBEDDING_DIM=768
IMAGE_RELEVANCE_THRESHOLD=0.20
```

如果不配置 `DASHSCOPE_API_KEY` 或 `IMAGE_EMBEDDING_API_KEY`，Docker bootstrap 会跳过图片索引，不影响文本商品推荐。

### 9.4 VLM 图片属性提取配置

如果要支持上传图片后识别商品类型、颜色、风格等属性，配置：

```env
VLM_API_KEY=你的VLMKey
VLM_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
VLM_MODEL=你的VLM模型
VLM_TIMEOUT_SECONDS=10
VLM_MAX_RETRIES=0
```

如果只跑文本推荐，可以先不配置 VLM。

### 9.5 Langfuse 可观测配置，可选

如果有 Langfuse 服务：

```env
LANGFUSE_PUBLIC_KEY=你的public_key
LANGFUSE_SECRET_KEY=你的secret_key
LANGFUSE_HOST=https://你的langfuse域名
```

不配置也不影响主流程，代码里是 best-effort，观测失败不会让 RAG 流程失败。

### 9.6 Evidence Cache 配置

默认即可：

```env
EVIDENCE_CACHE_TTL_SECONDS=3600
EVIDENCE_CACHE_RECENT_TURNS=20
EVIDENCE_CACHE_MAX_CANDIDATES_PER_TURN=20
```

## 10. 首次启动

在项目根目录执行：

```bash
docker compose up -d --build
```

这个命令会：

1. 构建后端镜像。
2. 启动 PostgreSQL、etcd、MinIO、Milvus。
3. 等 PostgreSQL 健康检查通过。
4. 运行 `backend-bootstrap` 初始化商品表和向量索引。
5. bootstrap 成功后启动 `backend`。

查看容器：

```bash
docker compose ps
```

持续看日志：

```bash
docker compose logs -f
```

单独看初始化任务：

```bash
docker compose logs -f backend-bootstrap
```

第一次启动时，`backend` 会等待 `backend-bootstrap` 成功完成。bootstrap 负责：

- 建表和导入商品数据。
- 构建 PostgreSQL 商品事实数据。
- 构建 Milvus 文本向量索引。
- 如果配置了图片 embedding key，再构建 Milvus 图片向量索引。

如果你暂时没有图片 key，看到图片索引跳过相关日志通常不是致命问题；但如果文本 embedding 没配或维度不对，文本检索体验会受影响。

单独看后端：

```bash
docker compose logs -f backend
```

## 11. 验证服务

### 11.1 健康检查

在服务器上执行：

```bash
curl http://127.0.0.1:8000/health
```

从本地电脑执行：

```bash
curl http://你的服务器公网IP:8000/health
```

如果本地访问失败，检查：

- ECS 安全组是否开放 `8000`。
- 服务器防火墙是否放行。
- `docker compose ps` 中 backend 是否 healthy。
- `docker compose logs backend` 是否有报错。

### 11.2 测试流式聊天接口

在服务器执行：

```bash
curl -N -X POST http://127.0.0.1:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"user_id":"u001","session_id":"s001","message":"我是油皮，预算150以内，推荐一款夏天用不闷的防晒"}'
```

如果你在本地电脑测试：

```bash
curl -N -X POST http://你的服务器公网IP:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"user_id":"u001","session_id":"s001","message":"我是油皮，预算150以内，推荐一款夏天用不闷的防晒"}'
```

正常情况下会看到 SSE 流式输出，包括 trace、agent_update、商品卡片和最终 answer。

## 12. 常用运维命令

查看服务状态：

```bash
docker compose ps
```

查看所有日志：

```bash
docker compose logs -f
```

查看后端日志：

```bash
docker compose logs -f backend
```

重启后端：

```bash
docker compose restart backend
```

停止全部服务：

```bash
docker compose down
```

停止并删除 volume，慎用，会删除数据库和向量库数据：

```bash
docker compose down -v
```

更新代码后重新构建：

```bash
git pull
docker compose up -d --build
```

查看磁盘占用：

```bash
df -h
docker system df
```

清理无用镜像，谨慎执行：

```bash
docker image prune -f
```

## 13. 重新初始化商品和向量索引

如果改了商品数据、embedding 模型或 Milvus collection，需要重新跑 bootstrap。

### 13.1 普通重跑 bootstrap

```bash
docker compose run --rm backend-bootstrap
docker compose restart backend
```

### 13.2 强制重建文本索引

编辑 `.env`：

```env
BOOTSTRAP_FORCE_REINDEX=true
```

然后执行：

```bash
docker compose run --rm backend-bootstrap
docker compose restart backend
```

完成后建议把 `.env` 改回：

```env
BOOTSTRAP_FORCE_REINDEX=false
```

避免每次启动都重建索引。

### 13.3 强制重建图片索引

如果改了图片 embedding 模型、`IMAGE_EMBEDDING_DIM`，或重新配置了 DashScope/image embedding key，编辑 `.env`：

```env
BOOTSTRAP_FORCE_IMAGE_REINDEX=true
```

然后执行：

```bash
docker compose run --rm backend-bootstrap
docker compose restart backend
```

完成后改回：

```env
BOOTSTRAP_FORCE_IMAGE_REINDEX=false
```

## 14. 如果服务器是 Alibaba Cloud Linux / CentOS

如果你没有用 Ubuntu，而是 Alibaba Cloud Linux、CentOS Stream 9/10、Rocky Linux 等 RPM 系系统，Docker 官方文档建议使用 rpm repository 安装。

大致流程：

```bash
dnf remove -y docker docker-client docker-client-latest docker-common docker-latest docker-latest-logrotate docker-logrotate docker-engine || true
dnf install -y dnf-plugins-core
dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
dnf install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
systemctl enable docker
systemctl start docker
docker --version
docker compose version
```

注意：

- CentOS 7 已经过旧，不建议作为新部署系统。
- Docker 官方当前文档的 CentOS 支持重点是 CentOS Stream 9/10。
- 如果 `dnf` 不存在，可能是旧系统，建议直接换 Ubuntu 22.04/24.04，部署阻力更小。

## 15. 可选：使用 Nginx 反向代理

如果要绑定域名和 HTTPS，建议使用 Nginx 把公网 `80/443` 代理到本机 `8000`。

安装 Nginx：

```bash
apt-get install -y nginx
systemctl enable nginx
systemctl start nginx
```

新建配置：

```bash
vim /etc/nginx/sites-available/ecommerce-rag.conf
```

写入：

```nginx
server {
    listen 80;
    server_name 你的域名;

    client_max_body_size 20m;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 3600;
    }
}
```

启用配置：

```bash
ln -s /etc/nginx/sites-available/ecommerce-rag.conf /etc/nginx/sites-enabled/ecommerce-rag.conf
nginx -t
systemctl reload nginx
```

如果使用 Nginx，建议把 `docker-compose.yml` 中 backend 端口改成本机绑定：

```yaml
backend:
  ports:
    - "127.0.0.1:8000:8000"
```

安全组开放：

- `22` 给你的 IP
- `80` 给公网
- `443` 给公网

不再需要公网开放 `8000`。

HTTPS 可以用 Certbot：

```bash
apt-get install -y certbot python3-certbot-nginx
certbot --nginx -d 你的域名
```

## 16. 常见问题

### 16.1 `docker compose` 命令不存在

检查：

```bash
docker compose version
dpkg -l | grep docker-compose-plugin
```

修复：

```bash
apt-get update
apt-get install -y docker-compose-plugin
```

### 16.2 Milvus 启动失败

先看日志：

```bash
docker compose logs -f milvus
docker compose logs -f etcd
docker compose logs -f minio
```

常见原因：

- 内存不足。
- 磁盘不足。
- Docker 没有正常启动。

检查资源：

```bash
free -h
df -h
docker compose ps
```

### 16.3 backend-bootstrap 失败

查看日志：

```bash
docker compose logs backend-bootstrap
```

重点检查：

- `ecommerce_agent_dataset` 是否存在。
- `.env` 里的 embedding 配置是否正确。
- Milvus 是否已启动。
- PostgreSQL 是否健康。
- API key 是否正确。

如果只是图片索引失败，可以先不配置图片 embedding，确保文本链路先跑通。

### 16.4 接口能访问但回答很慢

可能原因：

- LLM API 慢。
- embedding API 慢。
- 服务器资源不足。
- 首次请求触发额外初始化。

查看后端日志：

```bash
docker compose logs -f backend
```

如果配置了 Langfuse，可以在 Langfuse 里看每个 span 和模型调用耗时。

### 16.5 外网访问不了 8000

检查四层：

```bash
docker compose ps
curl http://127.0.0.1:8000/health
ss -lntp | grep 8000
```

然后检查：

- 阿里云安全组是否开放 `8000`。
- 系统防火墙是否拦截。
- `docker-compose.yml` backend 是否绑定了 `127.0.0.1:8000:8000`。如果是，只能本机访问，公网不能直连。

Ubuntu 防火墙如果启用，可执行：

```bash
ufw status
ufw allow 8000/tcp
```

### 16.6 API key 泄露风险

不要把 `.env` 提交到 Git。

确认：

```bash
git status
cat .gitignore
```

`.gitignore` 中应该包含 `.env`。

## 17. 推荐上线检查清单

上线前确认：

- `docker compose ps` 中 backend 是 healthy。
- `curl http://127.0.0.1:8000/health` 成功。
- `/api/chat/stream` 可以返回 SSE。
- 阿里云安全组只开放必要端口。
- `.env` 没有提交到 Git。
- PostgreSQL、Milvus、MinIO 没有直接暴露给公网。
- 服务器磁盘剩余空间充足。
- 如果使用域名，Nginx 已开启 HTTPS。

## 18. 最短部署命令汇总

如果你已经有一台 Ubuntu 22.04/24.04 服务器，可以按下面顺序快速部署：

```bash
apt-get update
apt-get install -y ca-certificates curl gnupg git vim unzip lsof htop

for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do
  apt-get remove -y "$pkg" || true
done

install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" \
  > /etc/apt/sources.list.d/docker.list

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
systemctl enable docker
systemctl start docker

cd /opt
git clone 你的仓库地址 ecommerce-rag-agent
cd ecommerce-rag-agent
cp .env.example .env
vim .env

docker compose up -d --build
docker compose logs -f backend-bootstrap
docker compose ps
curl http://127.0.0.1:8000/health
```
