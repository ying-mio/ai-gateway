# Ubuntu 部署步骤

> 适用于本项目（FastAPI 网关）。以下命令均在 Ubuntu Server 上执行。

## 1) 安装系统依赖

```bash
sudo apt update
sudo apt install -y git docker.io docker-compose-plugin
sudo systemctl enable --now docker
```

## 2) 拉取代码

```bash
cd /opt
sudo git clone <你的仓库地址> ai-gateway
cd ai-gateway
```

## 3) 设置环境变量 `.env`

```bash
cp .env.example .env
nano .env
```

至少填写：

```env
GATEWAY_TOKEN=your_gateway_token
OPENROUTER_API_KEY=your_openrouter_key
DEFAULT_MODEL=openai/gpt-4o-mini
OPENROUTER_URL=https://openrouter.ai/api/v1/chat/completions
OPENROUTER_TIMEOUT=60
OPENROUTER_REFERER=http://localhost
OPENROUTER_TITLE=ai-gateway-minimal
```

## 4) 构建并启动

### 方案 A：Docker（推荐）

```bash
docker build -t ai-gateway:latest .
docker run -d --name ai-gateway --env-file .env -p 8000:8000 ai-gateway:latest
```

### 方案 B：Docker Compose（可选）

```bash
docker compose up -d --build
```

## 5) 验证服务

```bash
curl http://127.0.0.1:8000/health
```

预期返回：

```json
{"ok":true}
```

## 6) 常用运维命令

```bash
# 查看日志
docker logs -f ai-gateway

# 停止并删除容器（docker run 方案）
docker rm -f ai-gateway

# Compose 方案停止
docker compose down
```
