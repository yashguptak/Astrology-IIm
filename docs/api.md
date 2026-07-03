# API Documentation

## Base URL

```
http://localhost:8000
https://your-domain.com
```

## Authentication

No authentication required by default. For production, add API key via Nginx or application middleware.

## Endpoints

### `GET /health`

Returns server and model status.

**Response:**
```json
{
    "status": "ok",
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "backend": "vllm",
    "gpu_available": true,
    "uptime_seconds": 3600.0
}
```

### `POST /v1/chat/completions`

OpenAI-compatible chat completion. Supports streaming via SSE.

**Request:**
```json
{
    "messages": [
        {"role": "system", "content": "You are Vedaz AI astrologer."},
        {"role": "user", "content": "Mera naam Rahul hai. DOB 15 March 1995."}
    ],
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "stream": false
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `messages` | array | required | List of message objects with `role` and `content` |
| `max_tokens` | int | 512 | Maximum tokens in response (1-4096) |
| `temperature` | float | 0.7 | Sampling temperature (0.0-2.0) |
| `top_p` | float | 0.9 | Nucleus sampling threshold (0.0-1.0) |
| `stream` | bool | false | Enable SSE streaming |

**Response (non-streaming):**
```json
{
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Namaste Rahul! Aapki kundli mein..."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 25,
        "completion_tokens": 150,
        "total_tokens": 175
    }
}
```

**Response (streaming):**
```
data: {"id":"chatcmpl-abc123","choices":[{"delta":{"role":"assistant","content":"Namaste"}}]}
data: {"id":"chatcmpl-abc123","choices":[{"delta":{"content":" Rahul"}}]}
data: {"id":"chatcmpl-abc123","choices":[{"delta":{"content":"!"}}]}
data: [DONE]
```

### `POST /v1/completions`

Simple text completion (non-chat).

**Request:**
```json
{
    "prompt": "Tell me about my career",
    "max_tokens": 256,
    "temperature": 0.7,
    "stream": false
}
```

## Python Client Examples

### Using OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)

# Chat
response = client.chat.completions.create(
    model="qwen2.5-3b-astrologer",
    messages=[
        {"role": "system", "content": "You are Vedaz AI astrologer."},
        {"role": "user", "content": "Mera naam Rahul hai. DOB 15 March 1995, 8:30 AM, Delhi."}
    ],
    max_tokens=512,
    temperature=0.7,
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="qwen2.5-3b-astrologer",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Using httpx

```python
import httpx

response = httpx.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 256,
    },
    timeout=60,
)
print(response.json()["choices"][0]["message"]["content"])
```

### Using curl

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 256
    }'

# Streaming
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "Hello"}], "stream": true}'
```

## Conversation Format for Astrologer

The model expects messages in ChatML format. The first message can optionally be a system prompt:

```json
{
    "messages": [
        {
            "role": "system",
            "content": "You are Vedaz's AI Vedic astrologer. You give compassionate, balanced guidance."
        },
        {
            "role": "user",
            "content": "Mera naam Rahul hai. DOB 15 March 1995, 8:30 AM, Delhi. Mera career kaisa rahega?"
        }
    ]
}
```

The system prompt is optional — the model will work without it using its default behavior.
