import logging
import os
import time
import uuid
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import requests

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("ai_gateway")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
PROJECT_NAME = "ai-gateway"

if not OPENROUTER_API_KEY:
    raise RuntimeError("Missing OPENROUTER_API_KEY in .env")

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
def chat(payload: ChatIn):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    upstream_status = None

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        # 这两个是 OpenRouter 推荐的可选字段，填你的信息更规范
        # OpenRouter 推荐的可选字段
        "HTTP-Referer": "http://localhost",
        "X-Title": "ai-gateway-minimal",
    }
    data = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "user", "content": payload.message}
        ]
        "messages": [{"role": "user", "content": payload.message}],
    }

    try:
        r = requests.post(url, headers=headers, json=data, timeout=60)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {e}")
        upstream_status = r.status_code
    except requests.RequestException as exc:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "chat_request request_id=%s duration_ms=%s request_body=%s upstream_status=%s",
            request_id,
            duration_ms,
            payload.model_dump(),
            upstream_status,
        )
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {exc}") from exc

    if r.status_code >= 400:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "chat_request request_id=%s duration_ms=%s request_body=%s upstream_status=%s",
            request_id,
            duration_ms,
            payload.model_dump(),
            upstream_status,
        )
        raise HTTPException(status_code=502, detail=f"OpenRouter error {r.status_code}: {r.text}")

    j = r.json()
    try:
        reply = j["choices"][0]["message"]["content"]
    except Exception:
        raise HTTPException(status_code=502, detail=f"Unexpected response: {j}")
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "chat_request request_id=%s duration_ms=%s request_body=%s upstream_status=%s",
            request_id,
            duration_ms,
            payload.model_dump(),
            upstream_status,
        )
        raise HTTPException(status_code=502, detail=f"Unexpected response: {j}") from exc

    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        "chat_request request_id=%s duration_ms=%s request_body=%s upstream_status=%s",
        request_id,
        duration_ms,
        payload.model_dump(),
        upstream_status,
    )

    return ChatOut(reply=reply, model=OPENROUTER_MODEL)
    return ChatOut(reply=reply, model=OPENROUTER_MODEL)