# AI Gateway

最小破坏式修复后的 FastAPI 网关：对外提供 OpenAI-compatible 接口，向上游（默认 OpenRouter）转发。

## 路由
- `GET /`
- `GET /health`（免鉴权）
- `GET /v1/models`（需网关鉴权，模型从上游真实拉取）
- `POST /v1/chat/completions`（OpenAI 兼容；支持 `stream=true`）
- `POST /chat`（简化接口，内部复用同一套聊天逻辑）

## 鉴权规则（网关层）
客户端调用网关时，二选一：
- `X-API-Key: <GATEWAY_TOKEN>`
- `Authorization: Bearer <GATEWAY_TOKEN>`

## 环境变量
复制并编辑：
```bash
cp .env.example .env
```

必填：
- `GATEWAY_TOKEN`：网关对外鉴权 token
- `UPSTREAM_BASE_URL`：上游基地址（如 `https://openrouter.ai/api/v1`、`https://api.openai.com/v1`）
- `UPSTREAM_API_KEY`：上游 API key（注意：不是 `GATEWAY_TOKEN`）

常用可选：
- `UPSTREAM_TIMEOUT`（默认 60）
- `DEFAULT_MODEL`（默认 `openai/gpt-4o-mini`）
- `UPSTREAM_PROVIDER`（默认 `openrouter`，用于附加 OpenRouter 推荐头）

## Docker Compose
```bash
docker compose up -d --build
docker compose ps
docker compose logs --tail=80 ai-gateway
```

## 验收命令（本地 127.0.0.1:8000）
```bash
export TOKEN='你的 GATEWAY_TOKEN'

curl -i http://127.0.0.1:8000/health

curl -i http://127.0.0.1:8000/v1/models \
  -H "X-API-Key: ${TOKEN}"

curl -i http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${TOKEN}" \
  -d '{"model":"openai/gpt-4o-mini","messages":[{"role":"user","content":"hi"}],"stream":false}'
```

## 验收命令（线上域名 foxrelay.xyz）
```bash
export TOKEN='你的 GATEWAY_TOKEN'

curl -i https://foxrelay.xyz/health

curl -i https://foxrelay.xyz/v1/models \
  -H "Authorization: Bearer ${TOKEN}"

curl -i https://foxrelay.xyz/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${TOKEN}" \
  -d '{"model":"openai/gpt-4o-mini","messages":[{"role":"user","content":"hello"}],"stream":false}'
```

## stream 调用示例
```bash
curl -N https://foxrelay.xyz/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${TOKEN}" \
  -d '{"model":"openai/gpt-4o-mini","messages":[{"role":"user","content":"写一句话"}],"stream":true}'
```

## 安全
- 不要提交 `.env` 或密钥文件。
- 若历史泄露过 key：优先在上游控制台撤销并轮换，再更新 `.env` 重启服务。
