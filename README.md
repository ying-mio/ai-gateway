# AI Gateway

最小可用 FastAPI 网关，当前提供：
- `GET /`
- `GET /health`
- `POST /chat`（转发 OpenRouter）

## 安装依赖（Windows PowerShell）
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 环境变量
复制 `.env.example` 后填写 `.env`：

```env
GATEWAY_TOKEN=your_gateway_token
OPENROUTER_API_KEY=your_openrouter_key
DEFAULT_MODEL=openai/gpt-4o-mini
OPENROUTER_URL=https://openrouter.ai/api/v1/chat/completions
OPENROUTER_TIMEOUT=60
OPENROUTER_REFERER=http://localhost
OPENROUTER_TITLE=ai-gateway-minimal
```

## 运行（Windows PowerShell）
```powershell
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```


## Windows PowerShell 一键测试命令
```powershell
# 测试 /health
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/health"

# 测试 /chat
$headers = @{ "Content-Type" = "application/json"; "X-API-Key" = "your_gateway_token" }; $body = '{"message":"你好"}'; Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/chat" -Headers $headers -Body $body
```

## 调用示例（必须带鉴权头）
```powershell
$headers = @{
  "Content-Type" = "application/json"
  "X-API-Key" = "your_gateway_token"
}
$body = '{"message":"你好"}'
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/chat" -Headers $headers -Body $body

# 可选字段示例（model / system / temperature / max_tokens）
$body2 = '{"message":"你好","model":"openai/gpt-4o-mini","system":"你是简洁助手","temperature":0.7,"max_tokens":256}'
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/chat" -Headers $headers -Body $body2
```

未提供或提供错误 `X-API-Key` 时，`/chat` 返回 `401 Unauthorized`。

`/chat` 响应包含：`reply`、`model`、`request_id`、`duration_ms`、`upstream_status`。
