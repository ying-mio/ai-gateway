import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

import requests
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ai_gateway")

PROJECT_NAME = "ai-gateway"

GATEWAY_TOKEN = os.getenv("GATEWAY_TOKEN", "").strip()
UPSTREAM_BASE_URL = os.getenv("UPSTREAM_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
UPSTREAM_API_KEY = os.getenv("UPSTREAM_API_KEY", "").strip()
UPSTREAM_TIMEOUT = float(os.getenv("UPSTREAM_TIMEOUT", "60"))
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "openai/gpt-4o-mini").strip()
UPSTREAM_PROVIDER = os.getenv("UPSTREAM_PROVIDER", "openrouter").strip().lower()
OPENROUTER_REFERER = os.getenv("OPENROUTER_REFERER", "https://foxrelay.xyz")
OPENROUTER_TITLE = os.getenv("OPENROUTER_TITLE", "ai-gateway")

if not GATEWAY_TOKEN:
    raise RuntimeError("Missing GATEWAY_TOKEN in environment")
if not UPSTREAM_API_KEY:
    raise RuntimeError("Missing UPSTREAM_API_KEY in environment")

app = FastAPI(title="AI Gateway")


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: Any


class OpenAIChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False


class ChatIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

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


def _auth_headers() -> dict[str, str]:
    headers = {"Authorization": f"Bearer {UPSTREAM_API_KEY}", "Content-Type": "application/json"}
    if UPSTREAM_PROVIDER == "openrouter":
        headers["HTTP-Referer"] = OPENROUTER_REFERER
        headers["X-Title"] = OPENROUTER_TITLE
    return headers


def _short(text: str, limit: int = 500) -> str:
    return text[:limit].replace("\n", " ")


def _check_gateway_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> None:
    bearer = None
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization.split(" ", 1)[1].strip()
    if x_api_key != GATEWAY_TOKEN and bearer != GATEWAY_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _extract_prompt(messages: list[ChatMessage]) -> str:
    for msg in reversed(messages):
        if msg.role == "user" and isinstance(msg.content, str) and msg.content.strip():
            return msg.content
    raise HTTPException(status_code=422, detail="messages must contain at least one user message")


def _chat_non_stream(payload: OpenAIChatRequest, request_id: str) -> tuple[dict[str, Any], int]:
    body = payload.model_dump(mode="json", exclude_none=True)
    body["model"] = body.get("model") or DEFAULT_MODEL
    url = f"{UPSTREAM_BASE_URL}/chat/completions"

    try:
        resp = requests.post(url, headers=_auth_headers(), json=body, timeout=UPSTREAM_TIMEOUT)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {exc.__class__.__name__}") from exc

    if resp.status_code >= 400:
        summary = _short(resp.text)
        logger.error("request_id=%s upstream_status=%s upstream_body=%s", request_id, resp.status_code, summary)
        raise HTTPException(status_code=502, detail=f"Upstream error {resp.status_code}: {summary}")

    try:
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail="Upstream returned non-JSON response") from exc

    if "choices" not in data:
        raise HTTPException(status_code=502, detail="Upstream response missing choices")

    return data, resp.status_code


async def _chat_stream(payload: OpenAIChatRequest, request: Request, request_id: str):
    body = payload.model_dump(mode="json", exclude_none=True)
    body["model"] = body.get("model") or DEFAULT_MODEL
    body["stream"] = True
    url = f"{UPSTREAM_BASE_URL}/chat/completions"

    def event_generator():
        upstream_status = 0
        start = time.perf_counter()
        try:
            with requests.post(url, headers=_auth_headers(), json=body, timeout=UPSTREAM_TIMEOUT, stream=True) as resp:
                upstream_status = resp.status_code
                if resp.status_code >= 400:
                    summary = _short(resp.text)
                    logger.error("request_id=%s upstream_status=%s upstream_body=%s", request_id, resp.status_code, summary)
                    err = {"error": {"message": f"Upstream error {resp.status_code}: {summary}", "type": "upstream_error"}}
                    yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                for line in resp.iter_lines(decode_unicode=True):
                    if line is None:
                        continue
                    if line:
                        yield f"{line}\n"
                    else:
                        yield "\n"
        except requests.RequestException as exc:
            err = {"error": {"message": f"Upstream request failed: {exc.__class__.__name__}", "type": "upstream_error"}}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.info(
                "request_id=%s path=%s duration_ms=%s upstream_status=%s model=%s stream=true",
                request_id,
                request.url.path,
                duration_ms,
                upstream_status,
                body.get("model"),
            )

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _process_chat(payload: OpenAIChatRequest, request: Request) -> tuple[dict[str, Any], int, float]:
    request_id = request.state.request_id
    start = time.perf_counter()
    prompt = _extract_prompt(payload.messages)

    data, upstream_status = _chat_non_stream(payload, request_id)

    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        "request_id=%s path=%s duration_ms=%s upstream_status=%s model=%s prompt_len=%s stream=false",
        request_id,
        request.url.path,
        duration_ms,
        upstream_status,
        data.get("model", payload.model or DEFAULT_MODEL),
        len(prompt),
    )
    return data, upstream_status, duration_ms


@app.middleware("http")
async def request_context(request: Request, call_next):
    request.state.request_id = str(uuid.uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


@app.get("/")
def root():
    return {"project": PROJECT_NAME, "time": datetime.now(timezone.utc).isoformat(), "upstream": UPSTREAM_BASE_URL}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/v1/models", dependencies=[Depends(_check_gateway_key)])
async def list_models(request: Request):
    url = f"{UPSTREAM_BASE_URL}/models"
    request_id = request.state.request_id
    try:
        resp = requests.get(url, headers=_auth_headers(), timeout=UPSTREAM_TIMEOUT)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {exc.__class__.__name__}") from exc

    if resp.status_code >= 400:
        summary = _short(resp.text)
        logger.error("request_id=%s upstream_status=%s upstream_body=%s", request_id, resp.status_code, summary)
        raise HTTPException(status_code=502, detail=f"Upstream error {resp.status_code}: {summary}")

    payload = resp.json()
    rows = payload.get("data", []) if isinstance(payload, dict) else []
    now = int(time.time())
    normalized = []
    for item in rows:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        normalized.append(
            {
                "id": item["id"],
                "object": "model",
                "created": item.get("created", now),
                "owned_by": item.get("owned_by") or item.get("name") or "upstream",
            }
        )

    return {"object": "list", "data": normalized}


@app.post("/chat", response_model=ChatOut, dependencies=[Depends(_check_gateway_key)])
async def chat(payload: ChatIn, request: Request):
    messages: list[ChatMessage] = []
    if payload.system:
        messages.append(ChatMessage(role="system", content=payload.system))
    messages.append(ChatMessage(role="user", content=payload.message))

    normalized = OpenAIChatRequest(
        model=payload.model,
        messages=messages,
        temperature=payload.temperature,
        max_tokens=payload.max_tokens,
        stream=False,
    )
    data, upstream_status, duration_ms = _process_chat(normalized, request)
    reply = data["choices"][0]["message"]["content"]

    return ChatOut(
        reply=reply,
        model=data.get("model", payload.model or DEFAULT_MODEL),
        request_id=request.state.request_id,
        duration_ms=duration_ms,
        upstream_status=upstream_status,
    )


@app.post("/v1/chat/completions", dependencies=[Depends(_check_gateway_key)])
async def chat_completions(payload: OpenAIChatRequest, request: Request):
    if payload.stream:
        return await _chat_stream(payload, request, request.state.request_id)

    data, _, _ = _process_chat(payload, request)
    return {
        "id": data.get("id", f"chatcmpl-{request.state.request_id}"),
        "object": data.get("object", "chat.completion"),
        "created": data.get("created", int(time.time())),
        "model": data.get("model", payload.model or DEFAULT_MODEL),
        "choices": data.get("choices", []),
        "usage": data.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
    }
