import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ai_gateway")

PROJECT_NAME = "ai-gateway"


def _csv_env(name: str) -> list[str]:
    raw = os.getenv(name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


PROVIDER = os.getenv("PROVIDER", "openrouter").strip().lower()
GATEWAY_TOKEN = os.getenv("GATEWAY_TOKEN", "").strip()
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "openai/gpt-4o-mini").strip()
AVAILABLE_MODELS = _csv_env("AVAILABLE_MODELS") or [DEFAULT_MODEL]
UPSTREAM_BASE_URL = os.getenv("UPSTREAM_BASE_URL", "https://openrouter.ai/api/v1").strip().rstrip("/")
UPSTREAM_TIMEOUT = float(os.getenv("UPSTREAM_TIMEOUT", os.getenv("OPENROUTER_TIMEOUT", "60")))

CHAT_COMPLETIONS_URL = os.getenv("UPSTREAM_CHAT_URL", os.getenv("OPENROUTER_URL", "")).strip()
if not CHAT_COMPLETIONS_URL:
    CHAT_COMPLETIONS_URL = f"{UPSTREAM_BASE_URL}/chat/completions"
OPENROUTER_TIMEOUT = float(os.getenv("OPENROUTER_TIMEOUT", "60"))
OPENROUTER_REFERER = os.getenv("OPENROUTER_REFERER", "http://localhost")
OPENROUTER_TITLE = os.getenv("OPENROUTER_TITLE", "ai-gateway")
OPENROUTER_API_KEYS = _csv_env("OPENROUTER_API_KEYS")
if not OPENROUTER_API_KEYS:
    one_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if one_key:
        OPENROUTER_API_KEYS = [one_key]

GEMINI_API_KEYS = _csv_env("GEMINI_API_KEYS")
if not GEMINI_API_KEYS:
    one_key = os.getenv("GEMINI_API_KEY", "").strip()
    if one_key:
        GEMINI_API_KEYS = [one_key]

if not GATEWAY_TOKEN:
    raise RuntimeError("Missing GATEWAY_TOKEN in environment")


def _resolve_upstream_api_key() -> str | None:
    for key_name in ("OPENROUTER_API_KEY", "UPSTREAM_API_KEY", "OPENAI_API_KEY"):
        value = os.getenv(key_name, "").strip()
        if value:
            return value
    return None

app = FastAPI(title="AI Gateway")


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: Any


class OpenAIChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None


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
    provider: str


def _mask_detail(text: str, keys: list[str]) -> str:
    masked = text
    for key in keys:
        if key:
            masked = masked.replace(key, "***")
    return masked


def _pick_key(keys: list[str], request_id: str) -> str | None:
    if not keys:
        return None
    idx = int(uuid.UUID(request_id)) % len(keys)
    return keys[idx]


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
        if msg.role != "user":
            continue
        content = msg.content
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") in {"text", None}:
                    part = item.get("text")
                    if isinstance(part, str):
                        text_parts.append(part)
            merged = "\n".join(p for p in text_parts if p.strip())
            if merged:
                return merged
    raise HTTPException(status_code=422, detail="messages must contain at least one user message with text content")


async def _call_upstream(payload: OpenAIChatRequest, request_id: str) -> tuple[str, int, str]:
    model = payload.model or DEFAULT_MODEL
    provider = PROVIDER

    if provider == "openrouter":
        api_key = _pick_key(OPENROUTER_API_KEYS, request_id) or _resolve_upstream_api_key()
        if not api_key:
            raise HTTPException(status_code=502, detail="Upstream openrouter not configured: missing API key")

        headers = _upstream_chat_headers(api_key)
        body: dict[str, Any] = {"model": model, "messages": [m.model_dump() for m in payload.messages]}
        if payload.temperature is not None:
            body["temperature"] = payload.temperature
        if payload.max_tokens is not None:
            body["max_tokens"] = payload.max_tokens

        try:
            async with httpx.AsyncClient(timeout=OPENROUTER_TIMEOUT) as client:
                resp = await client.post(CHAT_COMPLETIONS_URL, headers=headers, json=body)
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"Upstream request failed: {exc.__class__.__name__}") from exc

        if resp.status_code >= 400:
            detail = _mask_detail(resp.text[:500], OPENROUTER_API_KEYS)
            raise HTTPException(status_code=502, detail=f"Upstream openrouter error {resp.status_code}: {detail}")

        try:
            data = resp.json()
            reply = data["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail="Upstream openrouter returned invalid response format") from exc

        return reply, resp.status_code, model

    if provider == "gemini":
        # 占位实现：先保证接口与可扩展结构，后续可接 Gemini SDK/API。
        prompt = _extract_prompt(payload.messages)
        reply = f"[gemini-placeholder] {prompt}"
        return reply, 200, model

    raise HTTPException(status_code=502, detail=f"Unsupported provider: {provider}")


async def _process_chat(payload: OpenAIChatRequest, request: Request) -> dict[str, Any]:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    start = time.perf_counter()
    prompt = _extract_prompt(payload.messages)
    upstream_status = 0
    model = payload.model or DEFAULT_MODEL

    try:
        reply, upstream_status, model = await _call_upstream(payload, request_id)
    finally:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "request_id=%s path=%s duration_ms=%s provider=%s upstream_status=%s model=%s prompt_len=%s",
            request_id,
            request.url.path,
            duration_ms,
            PROVIDER,
            upstream_status,
            model,
            len(prompt),
        )

    return {
        "request_id": request_id,
        "reply": reply,
        "model": model,
        "duration_ms": duration_ms,
        "upstream_status": upstream_status,
        "provider": PROVIDER,
    }


@app.middleware("http")
async def request_context(request: Request, call_next):
    request.state.request_id = str(uuid.uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


@app.get("/")
def root():
    return {"project": PROJECT_NAME, "time": datetime.now(timezone.utc).isoformat(), "provider": PROVIDER}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/v1/models", dependencies=[Depends(_check_gateway_key)])
async def list_models():
    upstream_key = _resolve_upstream_api_key()
    if not upstream_key:
        raise HTTPException(status_code=502, detail="Upstream key is not configured")

    models_url = f"{UPSTREAM_BASE_URL}/models"
    headers = {"Authorization": f"Bearer {upstream_key}"}

    try:
        async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
            upstream_resp = await client.get(models_url, headers=headers)
    except Exception as exc:  # noqa: BLE001
        logger.warning("/v1/models upstream request failed: %s", exc.__class__.__name__)
        raise HTTPException(status_code=502, detail="Upstream error") from exc

    if upstream_resp.status_code != 200:
        content_type = upstream_resp.headers.get("content-type", "application/json")
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            media_type=content_type.split(";", 1)[0],
        )

    try:
        return upstream_resp.json()
    except json.JSONDecodeError as exc:
        logger.warning("/v1/models upstream returned non-JSON body")
        raise HTTPException(status_code=502, detail="Upstream error") from exc


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
    )
    result = await _process_chat(normalized, request)
    return ChatOut(**result)


@app.post("/v1/chat/completions", dependencies=[Depends(_check_gateway_key)])
async def chat_completions(request: Request):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    api_key = _pick_key(OPENROUTER_API_KEYS, request_id) or _resolve_upstream_api_key()
    if not api_key:
        raise HTTPException(status_code=502, detail="Upstream openrouter not configured: missing API key")

    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="Request body must be a JSON object")

    stream = bool(payload.get("stream"))
    headers = _upstream_chat_headers(api_key)

    if not stream:
        try:
            async with httpx.AsyncClient(timeout=OPENROUTER_TIMEOUT) as client:
                upstream_resp = await client.post(OPENROUTER_URL, headers=headers, json=payload)
        except httpx.RequestError as exc:
            logger.warning("/v1/chat/completions upstream request failed: %s", exc.__class__.__name__)
            raise HTTPException(status_code=502, detail="Upstream error") from exc

        if upstream_resp.status_code >= 400:
            content_type = upstream_resp.headers.get("content-type", "application/json")
            return Response(
                content=upstream_resp.content,
                status_code=upstream_resp.status_code,
                media_type=content_type.split(";", 1)[0],
            )

        content_type = upstream_resp.headers.get("content-type", "application/json")
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            media_type=content_type.split(";", 1)[0],
        )

    try:
        client = httpx.AsyncClient(timeout=OPENROUTER_TIMEOUT)
        req = client.build_request("POST", OPENROUTER_URL, headers=headers, json=payload)
        upstream_resp = await client.send(req, stream=True)
    except httpx.RequestError as exc:
        logger.warning("/v1/chat/completions upstream stream failed: %s", exc.__class__.__name__)
        raise HTTPException(status_code=502, detail="Upstream error") from exc

    if upstream_resp.status_code >= 400:
        body = await upstream_resp.aread()
        await upstream_resp.aclose()
        await client.aclose()
        content_type = upstream_resp.headers.get("content-type", "application/json")
        return Response(content=body, status_code=upstream_resp.status_code, media_type=content_type.split(";", 1)[0])

    content_type = upstream_resp.headers.get("content-type", "text/event-stream")
    return StreamingResponse(
        upstream_resp.aiter_raw(),
        status_code=upstream_resp.status_code,
        media_type=content_type.split(";", 1)[0],
        background=BackgroundTask(_aclose_upstream_response, upstream_resp, client),
    )
