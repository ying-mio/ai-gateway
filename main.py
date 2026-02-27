import logging
import os
import time
import uuid
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

load_dotenv()

# ---- logging ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("ai_gateway")

# ---- config ----
PROJECT_NAME = "ai-gateway"
GATEWAY_TOKEN = os.getenv("GATEWAY_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"))
OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_TIMEOUT = float(os.getenv("OPENROUTER_TIMEOUT", "60"))

if not OPENROUTER_API_KEY:
    raise RuntimeError("Missing OPENROUTER_API_KEY in .env")
if not GATEWAY_TOKEN:
    raise RuntimeError("Missing GATEWAY_TOKEN in .env")

app = FastAPI(title="AI Gateway (Minimal)")


class ChatIn(BaseModel):
    message: str
    model: str | None = None
    system: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


class ChatOut(BaseModel):
    reply: str
    model: str
    request_id: str
    duration_ms: float
    upstream_status: int


@app.get("/")
def root():
    return {
        "project": PROJECT_NAME,
        "time": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/chat", response_model=ChatOut)
def chat(payload: ChatIn, x_api_key: str | None = Header(default=None, alias="X-API-Key")):
     authorization: str | None = Header(default=None, alias="Authorization"),
):
    _check_gateway_key(x_api_key, authorization)

from fastapi import Request

@app.post("/v1/chat/completions", response_model=ChatOut)
async def chat_completions_alias(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None, alias="Authorization"),
):
    _check_gateway_key(x_api_key, authorization)

    body = await request.json()

    # OpenAI style: {"messages":[{"role":"user","content":"hi"}, ...]}
    if "messages" in body and isinstance(body["messages"], list):
        # 取最后一条 user 消息作为输入
        user_text = ""
        for m in reversed(body["messages"]):
            if m.get("role") == "user":
                user_text = m.get("content", "")
                break
        payload = ChatIn(message=user_text)
    else:
        # 兼容你原来的 {"message":"hi"}
        payload = ChatIn(**body)

    # 复用你现有的 chat 逻辑（如果你 chat() 是函数）
    return chat(payload, x_api_key)
from fastapi import Header, HTTPException
import os
import httpx

def _check_gateway_key(x_api_key: str | None, authorization: str | None):
    token = os.getenv("GATEWAY_TOKEN")
    bearer = None
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization.split(" ", 1)[1].strip()
    if (x_api_key != token) and (bearer != token):
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.get("/v1/models")
async def v1_models(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None, alias="Authorization"),
):
    _check_gateway_key(x_api_key, authorization)

    upstream_key = os.getenv("OPENROUTER_API_KEY")
    if not upstream_key:
        raise HTTPException(status_code=500, detail="Missing OPENROUTER_API_KEY")

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {upstream_key}"},
        )

    # OpenRouter 正常会给 JSON
    return r.json()
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        # OpenRouter 可选推荐字段
        "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "http://localhost"),
        "X-Title": os.getenv("OPENROUTER_TITLE", "ai-gateway-minimal"),
    }

    model = payload.model or DEFAULT_MODEL
    messages = [{"role": "user", "content": payload.message}]
    if payload.system:
        messages.insert(0, {"role": "system", "content": payload.system})

    data = {
        "model": model,
        "messages": messages,
    }
    if payload.temperature is not None:
        data["temperature"] = payload.temperature
    if payload.max_tokens is not None:
        data["max_tokens"] = payload.max_tokens

    upstream_status = None
    try:
        r = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=data,
            timeout=OPENROUTER_TIMEOUT,
        )
        upstream_status = r.status_code
    except requests.RequestException as exc:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "chat_request request_id=%s duration_ms=%s upstream_status=%s message_len=%s",
            request_id,
            duration_ms,
            upstream_status,
            len(payload.message),
        )
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {exc}") from exc

    if r.status_code >= 400:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "chat_request request_id=%s duration_ms=%s upstream_status=%s message_len=%s",
            request_id,
            duration_ms,
            upstream_status,
            len(payload.message),
        )
        raise HTTPException(status_code=502, detail=f"OpenRouter error {r.status_code}: {r.text}")

    try:
        j = r.json()
        reply = j["choices"][0]["message"]["content"]
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "chat_request request_id=%s duration_ms=%s upstream_status=%s message_len=%s",
            request_id,
            duration_ms,
            upstream_status,
            len(payload.message),
        )
        raise HTTPException(status_code=502, detail="Unexpected response format from upstream") from exc

    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        "chat_request request_id=%s duration_ms=%s upstream_status=%s message_len=%s",
        request_id,
        duration_ms,
        upstream_status,
        len(payload.message),
    )

    return ChatOut(
        reply=reply,
        model=model,
        request_id=request_id,
        duration_ms=duration_ms,
        upstream_status=upstream_status,
    )
