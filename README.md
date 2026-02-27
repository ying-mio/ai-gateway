# AI Gateway

一个自用 OpenAI 兼容网关，适合部署在 VPS（Docker Compose + 反向代理 + HTTPS）。

## 已实现接口
- `GET /`
- `GET /health`（不鉴权，方便探活）
- `GET /v1/models`（鉴权）
- `POST /v1/chat/completions`（OpenAI ChatCompletions 兼容，鉴权）
- `POST /chat`（简化接口，鉴权；内部复用同一核心逻辑）

## 鉴权
支持以下任意一种请求头：
- `X-API-Key: <GATEWAY_TOKEN>`
- `Authorization: Bearer <GATEWAY_TOKEN>`

鉴权失败返回 `401` JSON，不会返回 500。

## 环境变量
复制 `.env.example` 到 `.env` 后填写：

```bash
cp .env.example .env
```

必须至少填写：
- `GATEWAY_TOKEN`
- `PROVIDER`（`openrouter` 或 `gemini`）
- 对应 provider 的 key（如 `OPENROUTER_API_KEY`）

> 支持多 key：例如 `OPENROUTER_API_KEYS=key1,key2`，服务会按 request_id 做稳定轮询。

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
