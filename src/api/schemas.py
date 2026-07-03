"""Pydantic request/response models for the API."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., pattern=r"^(system|user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = Field(default=False)


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    choices: list[dict[str, Any]]
    usage: dict[str, int]


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = Field(default=False)


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    choices: list[dict[str, Any]]
    usage: dict[str, int]


class HealthResponse(BaseModel):
    status: str = "ok"
    model: str
    backend: str
    gpu_available: bool
    uptime_seconds: float
