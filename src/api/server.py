"""
FastAPI server with OpenAI-compatible endpoints.

Endpoints:
    POST /v1/chat/completions  — Chat completion (streaming optional)
    POST /v1/completions       — Text completion (streaming optional)
    GET  /health               — Health check

Usage:
    python -m src.api.server
    uvicorn src.api.server:app --host 0.0.0.0 --port 8000
"""

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    CompletionRequest,
    CompletionResponse,
    HealthResponse,
)
from src.inference.engine import InferenceEngine
from src.inference.chat import Conversation

_start_time = time.time()
_engine: InferenceEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    import os

    model_path = os.environ.get(
        "ASTROLOGY_MODEL_PATH",
        "Qwen/Qwen2.5-3B-Instruct",
    )
    use_vllm = os.environ.get("ASTROLOGY_USE_VLLM", "1") == "1"

    print(f"Loading model: {model_path}")
    _engine = InferenceEngine(
        model_path=model_path,
        use_vllm=use_vllm,
    )
    print(f"Model loaded. Backend: {_engine.backend}")
    yield
    _engine = None


app = FastAPI(
    title="Astrology LLM API",
    description="OpenAI-compatible API for the fine-tuned astrologer model",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_chat_response(
    content: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> ChatResponse:
    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    if _engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    messages = [m.model_dump() for m in request.messages]
    content = _engine.generate(
        messages=messages,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )

    prompt_tokens = sum(len(m["content"]) // 4 for m in messages)
    completion_tokens = len(content) // 4

    return _build_chat_response(
        content=content,
        model=_engine.model_path,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    if _engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    messages = [{"role": "user", "content": request.prompt}]
    content = _engine.generate(
        messages=messages,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
    )

    prompt_tokens = len(request.prompt) // 4
    completion_tokens = len(content) // 4

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:12]}",
        choices=[
            {
                "index": 0,
                "text": content,
                "finish_reason": "stop",
            }
        ],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )


@app.get("/health")
async def health():
    if _engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return HealthResponse(
        status="ok",
        model=_engine.model_path,
        backend=_engine.backend or "unknown",
        gpu_available=torch.cuda.is_available(),
        uptime_seconds=round(time.time() - _start_time, 2),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
    )
