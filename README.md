# AI Gateway

一个自用 OpenAI 兼容网关，适合部署在 VPS（Docker Compose + 反向代理 + HTTPS）。

现在支持：**多个网关 Token 对应多个上游配置**（如 OpenRouter / 硅基流动 / Gemini / 其他 OpenAI 兼容中转）。

## 已实现接口
- `GET /`
- `GET /health`（不鉴权，方便探活）
- `GET /v1/models`（鉴权）
- `POST /v1/chat/completions`（OpenAI ChatCompletions 兼容，鉴权）
- `POST /chat`（简化接口，鉴权；内部复用同一核心逻辑）

## 鉴权
支持以下任意一种请求头：
- `X-API-Key: <gateway_token>`
- `Authorization: Bearer <gateway_token>`

鉴权失败返回 `401` JSON，不会返回 500。

## 环境变量
复制 `.env.example` 到 `.env` 后填写：

```bash
cp .env.example .env
```

### 推荐：多 token + 多上游

在 `.env` 中配置 `GATEWAY_ROUTES`（JSON）：

```json
[
  {
    "token": "gw_or_001",
    "provider": "openrouter",
    "base_url": "https://openrouter.ai/api/v1",
    "default_model": "openai/gpt-4o-mini",
    "api_keys": ["or-key-1", "or-key-2"],
    "timeout": 60,
    "extra_headers": {
      "HTTP-Referer": "https://your-domain.example",
      "X-Title": "ai-gateway"
    }
  },
  {
    "token": "gw_sf_001",
    "provider": "siliconflow",
    "base_url": "https://api.siliconflow.cn/v1",
    "default_model": "deepseek-ai/DeepSeek-V3",
    "api_keys": ["sf-key-1"]
  },
  {
    "token": "gw_gm_001",
    "provider": "gemini",
    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
    "default_model": "gemini-2.0-flash",
    "api_keys": ["gemini-key-1"]
  }
]
```

说明：
- 每个 `token` 是你给客户端发放的“网关密钥”。
- 每个 route 可绑定自己的 `provider / base_url / default_model / api_keys / timeout / extra_headers`。
- `chat_url`、`models_url` 可不填，默认由 `base_url` 自动拼成：
  - `chat_url = {base_url}/chat/completions`
  - `models_url = {base_url}/models`
- 同一路由下如果配置多个 `api_keys`，网关会按 `request_id` 稳定轮询。

### 兼容旧版：单 token

仍支持旧配置（`GATEWAY_TOKEN + PROVIDER + OPENROUTER_API_KEY` 等）。未配置 `GATEWAY_ROUTES` 时，会自动走旧逻辑。

## Docker 部署（VPS）
```bash
docker compose up -d --build
docker compose ps
docker compose logs --tail=80 ai-gateway
```

## 最短验证命令（本机 127.0.0.1）
```bash
export TOKEN='替换成你的 GATEWAY_TOKEN'

curl -i http://127.0.0.1:8000/health

curl -i http://127.0.0.1:8000/v1/models \
  -H "X-API-Key: ${TOKEN}"

curl -i http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${TOKEN}" \
  -d '{"model":"openai/gpt-4o-mini","messages":[{"role":"user","content":"hi"}]}'

curl -i http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${TOKEN}" \
  -d '{"message":"你好"}'
```

## 多 token 实战示例

假设你发了两个 token：
- `gw_or_001` → OpenRouter
- `gw_sf_001` → 硅基流动

那调用时只需要替换请求头里的 token：

```bash
# 走 OpenRouter
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: gw_or_001" \
  -d '{"messages":[{"role":"user","content":"hello"}]}'

# 走硅基流动
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: gw_sf_001" \
  -d '{"messages":[{"role":"user","content":"hello"}]}'
```

> 是否走哪个上游，完全由网关 token 决定，不需要客户端关心真实上游 key。

## 最短验证命令（线上域名 + HTTPS）
```bash
export TOKEN='替换成你的 GATEWAY_TOKEN'
export BASE_URL='https://your-domain.example'

curl -i "${BASE_URL}/health"

curl -i "${BASE_URL}/v1/models" \
  -H "Authorization: Bearer ${TOKEN}"

curl -i "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${TOKEN}" \
  -d '{"model":"openai/gpt-4o-mini","messages":[{"role":"user","content":"hello"}]}'
```

## 反向代理要点
- 反代到容器服务 `127.0.0.1:8000`
- 开启 HTTPS（Nginx/Caddy/Traefik 均可）
- 保留 `Authorization` 与 `X-API-Key` 请求头

## 安全建议（若历史曾泄露密钥）
1. 立刻在上游平台撤销并轮换泄露 key。
2. 将新 key 写入 `.env`，重启容器生效。
3. 确认仓库中不再包含 `.env`、私钥文件（已在 `.gitignore` 忽略）。
4. 如需进一步清理历史，可后续安排历史重写；生产上先“撤销+轮换”优先级更高。


## 常见问题排查（Internal Server Error）
如果前端发消息返回 500，请按顺序检查：

```bash
# 1) 先看容器日志（最关键）
docker compose logs -f --tail=200 ai-gateway

# 2) 确认 httpx 已安装（镜像更新后需要重建）
docker compose build --no-cache ai-gateway

docker compose up -d ai-gateway

# 3) 确认网关与上游配置
# 必填：GATEWAY_TOKEN + 至少一个上游 key
# 可选：UPSTREAM_BASE_URL（默认 https://openrouter.ai/api/v1）
# 可选：UPSTREAM_CHAT_URL（默认 ${UPSTREAM_BASE_URL}/chat/completions）
```

## 小白建议：如何减少 PR 冲突
- 不建议“默认 Accept incoming change”。
- 最稳妥流程：
  1. 在 GitHub 合并 PR 时选择 **Rebase and merge**（历史更线性，冲突更少）。
  2. 服务器上执行：
     ```bash
     git fetch origin
     git reset --hard origin/<你的分支名>
     ```
     这样会直接以远端最新代码覆盖本地，避免手动逐段解决冲突（注意会丢本地未提交改动）。
  3. 再执行 `docker compose up -d --build` 重新部署。
## /v1/models 上游透传验证
```bash
export TOKEN='替换成你的 GATEWAY_TOKEN'
export UPSTREAM_KEY='替换成你的上游 Key（OPENROUTER_API_KEY / UPSTREAM_API_KEY / OPENAI_API_KEY）'

# 网关：应返回与上游一致的 object=list, data=[...] 结构
curl -sS http://127.0.0.1:8000/v1/models \
  -H "X-API-Key: ${TOKEN}"

# 验证模型数量（若上游有多个模型，应 > 1）
curl -sS http://127.0.0.1:8000/v1/models \
  -H "X-API-Key: ${TOKEN}" | python -c 'import json,sys; print(len(json.load(sys.stdin).get("data", [])))'

# 直连上游对比响应形状
curl -sS https://openrouter.ai/api/v1/models \
  -H "Authorization: Bearer ${UPSTREAM_KEY}"
```
