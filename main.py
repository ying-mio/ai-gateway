import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import requests

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

if not OPENROUTER_API_KEY:
    raise RuntimeError("Missing OPENROUTER_API_KEY in .env")

app = FastAPI(title="AI Gateway (Minimal)")

class ChatIn(BaseModel):
    message: str

class ChatOut(BaseModel):
    reply: str
    model: str

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat", response_model=ChatOut)
def chat(payload: ChatIn):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        # 这两个是 OpenRouter 推荐的可选字段，填你的信息更规范
        "HTTP-Referer": "http://localhost",
        "X-Title": "ai-gateway-minimal",
    }
    data = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "user", "content": payload.message}
        ]
    }

    try:
        r = requests.post(url, headers=headers, json=data, timeout=60)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {e}")

    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"OpenRouter error {r.status_code}: {r.text}")

    j = r.json()
    try:
        reply = j["choices"][0]["message"]["content"]
    except Exception:
        raise HTTPException(status_code=502, detail=f"Unexpected response: {j}")

    return ChatOut(reply=reply, model=OPENROUTER_MODEL)