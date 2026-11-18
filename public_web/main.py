import asyncio
import os
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
SYSTEM_PROMPT = """
You are JARVIS, a capable, calm, concise AI assistant.
You are operating through a public web chat. You cannot access the visitor's
computer, files, microphone, camera, accounts, or local network. Never claim
that you performed an action on their device. Explain this limitation when
relevant and offer instructions or code the visitor can use themselves.
Do not reveal private credentials, internal configuration, or personal
information about the person who created or deployed this service.
Be accurate. If you are unsure, say so. Keep answers clear and useful.
"""
app = FastAPI(title="JARVIS", docs_url=None, redoc_url=None)
_requests = defaultdict(deque)
MAX_PER_MINUTE = 8
MAX_PER_DAY = 100


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[dict[str, str]] = Field(default_factory=list, max_length=12)


def client_ip(request: Request) -> str:
    # Railway's edge proxy supplies this header. Use the first forwarded IP.
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip()[:64] if forwarded else (
        request.client.host if request.client else "unknown"
    )


def check_rate_limit(ip: str) -> None:
    now = time.time()
    bucket = _requests[ip]
    while bucket and now - bucket[0] > 86400:
        bucket.popleft()
    minute_count = sum(1 for stamp in bucket if now - stamp < 60)
    if minute_count >= MAX_PER_MINUTE or len(bucket) >= MAX_PER_DAY:
        raise HTTPException(
            status_code=429,
            detail="Usage limit reached. Please wait before sending another message.",
        )
    bucket.append(now)


@app.get("/")
async def home():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "jarvis-web"}


@app.post("/api/chat")
async def chat(payload: ChatRequest, request: Request):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="JARVIS is not configured yet. The service owner must add its Gemini API key.",
        )

    check_rate_limit(client_ip(request))
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    contents = []
    total_chars = 0
    for item in payload.history[-12:]:
        role = item.get("role", "")
        text = item.get("text", "")
        if role not in ("user", "model") or not isinstance(text, str):
            continue
        text = text[:4000]
        total_chars += len(text)
        if total_chars > 12000:
            break
        contents.append(types.Content(
            role=role,
            parts=[types.Part(text=text)]
        ))
    contents.append(types.Content(
        role="user",
        parts=[types.Part(text=message)]
    ))

    try:
        client = genai.Client(api_key=api_key)
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=900,
            ),
        )
        answer = (response.text or "").strip()
        if not answer:
            raise RuntimeError("The model returned an empty response.")
        return {"reply": answer, "model": MODEL}
    except Exception:
        # Do not send provider error details or credentials to public clients.
        raise HTTPException(
            status_code=502,
            detail="JARVIS could not get a response right now. Please try again shortly.",
        )


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
