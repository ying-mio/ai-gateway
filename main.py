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
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_TIMEOUT = float(os.getenv("OPENROUTER_TIMEOUT", "60"))

if not OPENROUTER_API_KEY:
    raise RuntimeError("Missing OPENROUTER_API_KEY in .env")
if not GATEWAY_TOKEN:
    raise RuntimeError("Missing GATEWAY_TOKEN in .env")

app = FastAPI(title="AI Gateway (Minimal)")


class ChatIn(BaseModel):
    message: str


class ChatOut(BaseModel):
    reply: str
    model: str


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
    if x_api_key != GATEWAY_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        # OpenRouter 可选推荐字段
        "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "http://localhost"),
        "X-Title": os.getenv("OPENROUTER_TITLE", "ai-gateway-minimal"),
    }

    data = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": payload.message}],
    }

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

    return ChatOut(reply=reply, model=OPENROUTER_MODEL)
