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
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict
from starlette.background import BackgroundTask

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ai_gateway")

PROJECT_NAME = "ai-gateway"


def _csv_env(name: str) -> list[str]:
    raw = os.getenv(name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


LEGACY_PROVIDER = os.getenv("PROVIDER", "openrouter").strip().lower()
LEGACY_GATEWAY_TOKEN = os.getenv("GATEWAY_TOKEN", "").strip()
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "openai/gpt-4o-mini").strip()
AVAILABLE_MODELS = _csv_env("AVAILABLE_MODELS") or [DEFAULT_MODEL]
UPSTREAM_BASE_URL = os.getenv("UPSTREAM_BASE_URL", "https://openrouter.ai/api/v1").strip().rstrip("/")
CHAT_COMPLETIONS_URL = os.getenv("UPSTREAM_CHAT_URL", os.getenv("OPENROUTER_URL", "")).strip() or f"{UPSTREAM_BASE_URL}/chat/completions"
UPSTREAM_TIMEOUT = float(os.getenv("UPSTREAM_TIMEOUT", os.getenv("OPENROUTER_TIMEOUT", "60")))
OPENROUTER_TIMEOUT = float(os.getenv("OPENROUTER_TIMEOUT", str(UPSTREAM_TIMEOUT)))
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


class RouteConfig(BaseModel):
    token: str
    provider: str = "openrouter"
    default_model: str = DEFAULT_MODEL
    available_models: list[str] = [DEFAULT_MODEL]
    base_url: str = "https://openrouter.ai/api/v1"
    chat_url: str = "https://openrouter.ai/api/v1/chat/completions"
    models_url: str = "https://openrouter.ai/api/v1/models"
    timeout: float = UPSTREAM_TIMEOUT
    api_keys: list[str] = []
    extra_headers: dict[str, str] = {}


class TenantContext(BaseModel):
    tenant_id: str
    route: RouteConfig


def _resolve_upstream_api_key() -> str | None:
    for key_name in ("OPENROUTER_API_KEY", "UPSTREAM_API_KEY", "OPENAI_API_KEY"):
        value = os.getenv(key_name, "").strip()
        if value:
            return value
    return None


def _build_legacy_route() -> RouteConfig | None:
    if not LEGACY_GATEWAY_TOKEN:
        return None

    keys = OPENROUTER_API_KEYS
    if LEGACY_PROVIDER == "gemini":
        keys = GEMINI_API_KEYS

    fallback_key = _resolve_upstream_api_key()
    if fallback_key and fallback_key not in keys:
        keys = [*keys, fallback_key]

    extra_headers: dict[str, str] = {}
    if LEGACY_PROVIDER == "openrouter":
        extra_headers = {
            "HTTP-Referer": OPENROUTER_REFERER,
            "X-Title": OPENROUTER_TITLE,
        }

    return RouteConfig(
        token=LEGACY_GATEWAY_TOKEN,
        provider=LEGACY_PROVIDER,
        default_model=DEFAULT_MODEL,
        available_models=AVAILABLE_MODELS,
        base_url=UPSTREAM_BASE_URL,
        chat_url=CHAT_COMPLETIONS_URL,
        models_url=f"{UPSTREAM_BASE_URL}/models",
        timeout=OPENROUTER_TIMEOUT if LEGACY_PROVIDER == "openrouter" else UPSTREAM_TIMEOUT,
        api_keys=keys,
        extra_headers=extra_headers,
    )


def _parse_gateway_routes() -> dict[str, RouteConfig]:
    raw = os.getenv("GATEWAY_ROUTES", "").strip()
    parsed_routes: list[dict[str, Any]] = []

    if raw:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Invalid GATEWAY_ROUTES JSON") from exc

        if isinstance(data, dict):
            parsed_routes = list(data.values())
        elif isinstance(data, list):
            parsed_routes = data
        else:
            raise RuntimeError("GATEWAY_ROUTES must be a JSON list or object")

    routes: dict[str, RouteConfig] = {}

    for item in parsed_routes:
        if not isinstance(item, dict):
            continue

        token = str(item.get("token", "")).strip()
        if not token:
            continue

        provider = str(item.get("provider", "openrouter")).strip().lower()
        default_model = str(item.get("default_model", DEFAULT_MODEL)).strip() or DEFAULT_MODEL
        available_models = item.get("available_models")
        if not isinstance(available_models, list) or not available_models:
            available_models = [default_model]

        base_url = str(item.get("base_url", UPSTREAM_BASE_URL)).strip().rstrip("/")
        chat_url = str(item.get("chat_url", "")).strip() or f"{base_url}/chat/completions"
        models_url = str(item.get("models_url", "")).strip() or f"{base_url}/models"
        timeout = float(item.get("timeout", UPSTREAM_TIMEOUT))

        keys = item.get("api_keys")
        if not isinstance(keys, list):
            one_key = str(item.get("api_key", "")).strip()
            keys = [one_key] if one_key else []
        keys = [str(k).strip() for k in keys if str(k).strip()]

        headers = item.get("extra_headers", {})
        if not isinstance(headers, dict):
            headers = {}

        routes[token] = RouteConfig(
            token=token,
            provider=provider,
            default_model=default_model,
            available_models=[str(m).strip() for m in available_models if str(m).strip()] or [default_model],
            base_url=base_url,
            chat_url=chat_url,
            models_url=models_url,
            timeout=timeout,
            api_keys=keys,
            extra_headers={str(k): str(v) for k, v in headers.items()},
        )

    legacy_route = _build_legacy_route()
    if legacy_route and legacy_route.token not in routes:
        routes[legacy_route.token] = legacy_route

    if not routes:
        raise RuntimeError("Missing gateway auth config. Set GATEWAY_ROUTES or GATEWAY_TOKEN")

    return routes


ROUTES_BY_TOKEN = _parse_gateway_routes()
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
SUPABASE_TIMEOUT = float(os.getenv("SUPABASE_TIMEOUT", "5"))
MEMORY_MAX_ITEMS = int(os.getenv("MEMORY_MAX_ITEMS", "20"))


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


class PersonaUpdate(BaseModel):
    name: str | None = None
    system_prompt: str = ""


class MemoryCreate(BaseModel):
    title: str
    content: str
    tags: list[str] = []
    weight: int = 0


class MemoryUpdate(BaseModel):
    title: str | None = None
    content: str | None = None
    tags: list[str] | None = None
    weight: int | None = None


app = FastAPI(title="AI Gateway")


class SupabaseStore:
    def __init__(self, url: str, service_role_key: str, timeout: float = 5) -> None:
        self.enabled = bool(url and service_role_key)
        self.url = f"{url}/rest/v1" if self.enabled else ""
        self.timeout = timeout
        self._headers = {
            "apikey": service_role_key,
            "Authorization": f"Bearer {service_role_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }

    async def _request(
        self,
        method: str,
        table: str,
        params: dict[str, str] | None = None,
        json_body: Any | None = None,
        prefer: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self.enabled:
            return []

        headers = dict(self._headers)
        if prefer:
            headers["Prefer"] = prefer

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.request(
                    method,
                    f"{self.url}/{table}",
                    params=params,
                    json=json_body,
                    headers=headers,
                )
                resp.raise_for_status()
                if not resp.content:
                    return []
                data = resp.json()
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    return [data]
                return []
        except Exception as exc:  # noqa: BLE001
            logger.warning("supabase %s %s failed: %s", method, table, exc.__class__.__name__)
            return []

    async def get_persona(self, tenant_id: str) -> dict[str, Any] | None:
        rows = await self._request("GET", "personas", params={"tenant_id": f"eq.{tenant_id}", "limit": "1"})
        return rows[0] if rows else None

    async def upsert_persona(self, tenant_id: str, name: str | None, system_prompt: str) -> dict[str, Any]:
        payload = {
            "tenant_id": tenant_id,
            "name": name or tenant_id,
            "system_prompt": system_prompt,
        }
        rows = await self._request(
            "POST",
            "personas",
            json_body=payload,
            prefer="resolution=merge-duplicates,return=representation",
        )
        return rows[0] if rows else payload

    async def get_memories(self, tenant_id: str, limit: int) -> list[dict[str, Any]]:
        params = {
            "tenant_id": f"eq.{tenant_id}",
            "order": "weight.desc,created_at.desc",
            "limit": str(limit),
        }
        return await self._request("GET", "memories", params=params)

    async def create_memory(self, tenant_id: str, payload: MemoryCreate) -> dict[str, Any] | None:
        rows = await self._request(
            "POST",
            "memories",
            json_body={
                "tenant_id": tenant_id,
                "title": payload.title,
                "content": payload.content,
                "tags": payload.tags,
                "weight": payload.weight,
            },
        )
        return rows[0] if rows else None

    async def update_memory(self, tenant_id: str, memory_id: int, payload: MemoryUpdate) -> dict[str, Any] | None:
        data = payload.model_dump(exclude_none=True)
        if not data:
            return None
        rows = await self._request(
            "PATCH",
            "memories",
            params={"tenant_id": f"eq.{tenant_id}", "id": f"eq.{memory_id}"},
            json_body=data,
        )
        return rows[0] if rows else None

    async def delete_memory(self, tenant_id: str, memory_id: int) -> bool:
        rows = await self._request(
            "DELETE",
            "memories",
            params={"tenant_id": f"eq.{tenant_id}", "id": f"eq.{memory_id}"},
            prefer="return=representation",
        )
        return bool(rows)

    async def ensure_conversation(self, tenant_id: str, conversation_id: str | None) -> str:
        if conversation_id:
            rows = await self._request(
                "GET",
                "conversations",
                params={"tenant_id": f"eq.{tenant_id}", "id": f"eq.{conversation_id}", "limit": "1"},
            )
            if rows:
                return conversation_id

        conversation_id = str(uuid.uuid4())
        await self._request(
            "POST",
            "conversations",
            json_body={"id": conversation_id, "tenant_id": tenant_id},
        )
        return conversation_id

    async def insert_messages(self, messages: list[dict[str, Any]]) -> None:
        if not messages:
            return
        await self._request("POST", "messages", json_body=messages)

    async def list_conversations(self, tenant_id: str, limit: int = 50) -> list[dict[str, Any]]:
        return await self._request(
            "GET",
            "conversations",
            params={"tenant_id": f"eq.{tenant_id}", "order": "created_at.desc", "limit": str(limit)},
        )

    async def list_messages(self, tenant_id: str, conversation_id: str) -> list[dict[str, Any]]:
        return await self._request(
            "GET",
            "messages",
            params={
                "tenant_id": f"eq.{tenant_id}",
                "conversation_id": f"eq.{conversation_id}",
                "order": "created_at.asc",
            },
        )


store = SupabaseStore(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, timeout=SUPABASE_TIMEOUT)


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


def _upstream_headers(api_key: str, route: RouteConfig) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    headers.update(route.extra_headers)
    return headers


async def _aclose_upstream_response(resp: httpx.Response, client: httpx.AsyncClient) -> None:
    await resp.aclose()
    await client.aclose()


def _check_gateway_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> TenantContext:
    bearer = None
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization.split(" ", 1)[1].strip()

    token = x_api_key or bearer
    if not token:
        raise HTTPException(status_code=401, detail="Unauthorized")

    route = ROUTES_BY_TOKEN.get(token)
    if not route:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return TenantContext(tenant_id=token, route=route)


def _build_memory_system_prompt(system_prompt: str, memories: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    if system_prompt.strip():
        chunks.append(system_prompt.strip())

    if memories:
        chunks.append("以下是和当前租户相关的长期记忆，请优先参考：")
        for idx, memory in enumerate(memories, start=1):
            title = memory.get("title") or f"memory-{idx}"
            content = str(memory.get("content", "")).strip()
            tags = memory.get("tags") or []
            tag_text = f" [tags: {', '.join(str(t) for t in tags)}]" if tags else ""
            chunks.append(f"{idx}. {title}{tag_text}: {content}")

    return "\n\n".join([item for item in chunks if item])


def _inject_system_message(payload: dict[str, Any], system_text: str) -> dict[str, Any]:
    if not system_text.strip():
        return payload

    new_payload = dict(payload)
    messages = payload.get("messages", [])
    if not isinstance(messages, list):
        messages = []
    new_payload["messages"] = [{"role": "system", "content": system_text}, *messages]
    return new_payload


def _extract_last_user_content(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = [item.get("text", "") for item in content if isinstance(item, dict)]
            return "\n".join([str(t) for t in texts if str(t).strip()])
    return ""


def _extract_assistant_content(resp_json: dict[str, Any]) -> str:
    try:
        content = resp_json["choices"][0]["message"]["content"]
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return ""


def _extract_usage(resp_json: dict[str, Any]) -> dict[str, Any] | None:
    usage = resp_json.get("usage")
    return usage if isinstance(usage, dict) else None


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


async def _call_upstream(payload: OpenAIChatRequest, request_id: str, route: RouteConfig) -> tuple[str, int, str]:
    model = payload.model or route.default_model
    api_key = _pick_key(route.api_keys, request_id)
    if not api_key:
        raise HTTPException(status_code=502, detail=f"Upstream {route.provider} not configured: missing API key")

    body: dict[str, Any] = {"model": model, "messages": [m.model_dump() for m in payload.messages]}
    if payload.temperature is not None:
        body["temperature"] = payload.temperature
    if payload.max_tokens is not None:
        body["max_tokens"] = payload.max_tokens

    try:
        async with httpx.AsyncClient(timeout=route.timeout) as client:
            resp = await client.post(route.chat_url, headers=_upstream_headers(api_key, route), json=body)
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {exc.__class__.__name__}") from exc

    if resp.status_code >= 400:
        detail = _mask_detail(resp.text[:500], route.api_keys)
        raise HTTPException(status_code=502, detail=f"Upstream {route.provider} error {resp.status_code}: {detail}")

    try:
        data = resp.json()
        reply = data["choices"][0]["message"]["content"]
        if not isinstance(reply, str):
            reply = json.dumps(reply, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Upstream {route.provider} returned invalid response format") from exc

    return reply, resp.status_code, model


async def _process_chat(payload: OpenAIChatRequest, request: Request, route: RouteConfig) -> dict[str, Any]:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    start = time.perf_counter()
    prompt = _extract_prompt(payload.messages)
    upstream_status = 0
    model = payload.model or route.default_model

    try:
        reply, upstream_status, model = await _call_upstream(payload, request_id, route)
    finally:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "request_id=%s path=%s duration_ms=%s provider=%s upstream_status=%s model=%s prompt_len=%s",
            request_id,
            request.url.path,
            duration_ms,
            route.provider,
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
        "provider": route.provider,
    }


@app.middleware("http")
async def request_context(request: Request, call_next):
    request.state.request_id = str(uuid.uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


@app.get("/")
def root():
    providers = sorted({r.provider for r in ROUTES_BY_TOKEN.values()})
    return {"project": PROJECT_NAME, "time": datetime.now(timezone.utc).isoformat(), "providers": providers}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/v1/models")
async def list_models(ctx: TenantContext = Depends(_check_gateway_key)):
    route = ctx.route
    request_id = str(uuid.uuid4())
    upstream_key = _pick_key(route.api_keys, request_id)
    if not upstream_key:
        raise HTTPException(status_code=502, detail=f"Upstream {route.provider} key is not configured")

    headers = {"Authorization": f"Bearer {upstream_key}"}
    headers.update(route.extra_headers)

    try:
        async with httpx.AsyncClient(timeout=route.timeout) as client:
            upstream_resp = await client.get(route.models_url, headers=headers)
    except Exception as exc:  # noqa: BLE001
        logger.warning("/v1/models upstream request failed: %s", exc.__class__.__name__)
        raise HTTPException(status_code=502, detail="Upstream error") from exc

    content_type = upstream_resp.headers.get("content-type", "application/json")
    media_type = content_type.split(";", 1)[0]

    if upstream_resp.status_code != 200:
        return Response(content=upstream_resp.content, status_code=upstream_resp.status_code, media_type=media_type)

    try:
        return upstream_resp.json()
    except json.JSONDecodeError as exc:
        logger.warning("/v1/models upstream returned non-JSON body")
        raise HTTPException(status_code=502, detail="Upstream error") from exc


@app.post("/chat", response_model=ChatOut)
async def chat(payload: ChatIn, request: Request, ctx: TenantContext = Depends(_check_gateway_key)):
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
    result = await _process_chat(normalized, request, ctx.route)
    return ChatOut(**result)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, ctx: TenantContext = Depends(_check_gateway_key)):
    route = ctx.route
    tenant_id = ctx.tenant_id
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    api_key = _pick_key(route.api_keys, request_id)
    if not api_key:
        raise HTTPException(status_code=502, detail=f"Upstream {route.provider} not configured: missing API key")

    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="Request body must be a JSON object")

    conversation_id = payload.pop("conversation_id", None)
    user_message = _extract_last_user_content(payload.get("messages", []))

    persona = await store.get_persona(tenant_id)
    memories = await store.get_memories(tenant_id, MEMORY_MAX_ITEMS)
    injected_system = _build_memory_system_prompt((persona or {}).get("system_prompt", ""), memories)
    payload = _inject_system_message(payload, injected_system)

    headers = _upstream_headers(api_key, route)
    stream = bool(payload.get("stream"))

    if not stream:
        try:
            async with httpx.AsyncClient(timeout=route.timeout) as client:
                upstream_resp = await client.post(route.chat_url, headers=headers, json=payload)
        except httpx.RequestError as exc:
            logger.warning("/v1/chat/completions upstream request failed: %s", exc.__class__.__name__)
            raise HTTPException(status_code=502, detail="Upstream error") from exc

        content_type = upstream_resp.headers.get("content-type", "application/json")
        response = Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            media_type=content_type.split(";", 1)[0],
        )
        response.headers["X-Conversation-ID"] = conversation_id or ""

        if upstream_resp.status_code < 400:
            try:
                resp_json = upstream_resp.json()
                assistant_message = _extract_assistant_content(resp_json)
                usage = _extract_usage(resp_json)
                conv_id = await store.ensure_conversation(tenant_id, conversation_id)
                await store.insert_messages(
                    [
                        {
                            "conversation_id": conv_id,
                            "tenant_id": tenant_id,
                            "role": "user",
                            "content": user_message,
                            "model": payload.get("model") or route.default_model,
                        },
                        {
                            "conversation_id": conv_id,
                            "tenant_id": tenant_id,
                            "role": "assistant",
                            "content": assistant_message,
                            "model": payload.get("model") or route.default_model,
                            "token_usage": usage,
                        },
                    ]
                )
                response.headers["X-Conversation-ID"] = conv_id
            except Exception as exc:  # noqa: BLE001
                logger.warning("message archive failed: %s", exc.__class__.__name__)
        return response

    client: httpx.AsyncClient | None = None
    try:
        client = httpx.AsyncClient(timeout=route.timeout)
        req = client.build_request("POST", route.chat_url, headers=headers, json=payload)
        upstream_resp = await client.send(req, stream=True)
    except httpx.RequestError as exc:
        if client is not None:
            await client.aclose()
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


@app.get("/admin/persona")
async def get_persona(ctx: TenantContext = Depends(_check_gateway_key)):
    persona = await store.get_persona(ctx.tenant_id)
    return persona or {"tenant_id": ctx.tenant_id, "name": ctx.tenant_id, "system_prompt": ""}


@app.put("/admin/persona")
async def put_persona(payload: PersonaUpdate, ctx: TenantContext = Depends(_check_gateway_key)):
    return await store.upsert_persona(ctx.tenant_id, payload.name, payload.system_prompt)


@app.get("/admin/memories")
async def get_memories(limit: int = MEMORY_MAX_ITEMS, ctx: TenantContext = Depends(_check_gateway_key)):
    safe_limit = max(1, min(limit, 200))
    return await store.get_memories(ctx.tenant_id, safe_limit)


@app.post("/admin/memories")
async def post_memory(payload: MemoryCreate, ctx: TenantContext = Depends(_check_gateway_key)):
    memory = await store.create_memory(ctx.tenant_id, payload)
    if not memory:
        raise HTTPException(status_code=503, detail="memory storage unavailable")
    return memory


@app.put("/admin/memories/{memory_id}")
async def put_memory(memory_id: int, payload: MemoryUpdate, ctx: TenantContext = Depends(_check_gateway_key)):
    memory = await store.update_memory(ctx.tenant_id, memory_id, payload)
    if not memory:
        raise HTTPException(status_code=404, detail="memory not found")
    return memory


@app.delete("/admin/memories/{memory_id}")
async def delete_memory(memory_id: int, ctx: TenantContext = Depends(_check_gateway_key)):
    deleted = await store.delete_memory(ctx.tenant_id, memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="memory not found")
    return {"ok": True}


@app.get("/admin/conversations")
async def get_conversations(limit: int = 50, ctx: TenantContext = Depends(_check_gateway_key)):
    safe_limit = max(1, min(limit, 200))
    return await store.list_conversations(ctx.tenant_id, safe_limit)


@app.get("/admin/conversations/{conversation_id}/messages")
async def get_conversation_messages(conversation_id: str, ctx: TenantContext = Depends(_check_gateway_key)):
    return await store.list_messages(ctx.tenant_id, conversation_id)
