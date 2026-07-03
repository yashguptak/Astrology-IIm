# vLLM Model Serving

## Why vLLM

| Feature | Benefit |
|---------|---------|
| **PagedAttention** | Manages KV cache in non-contiguous memory blocks, reducing fragmentation and increasing batch size by 2-4x |
| **Continuous Batching** | Dynamically adds/removes requests from the running batch, maximizing GPU utilization |
| **Tensor Parallelism** | Distributes model across multiple GPUs automatically |
| **OpenAI-compatible API** | Drop-in replacement for any OpenAI client — no code changes needed |
| **FP16/BF16/FP8** | Supports multiple precisions for memory/quality trade-offs |

## Architecture

```
Request → HTTP Server → Scheduler → Block Manager → GPU Kernels → Response
                               ↓
                          KV Cache
                     (PagedAttention)
```

## Serving Commands

### Basic vLLM Server (merged model)

```bash
python -m vllm.entrypoints.openai.api_server \
    --model training/merged/qwen2.5-3b-astrologer \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16
```

### vLLM Server with LoRA adapters (without merging)

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct \
    --enable-lora \
    --lora-modules astrologer=training/checkpoints/final \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16
```

### vLLM with tensor parallelism (multi-GPU)

```bash
python -m vllm.entrypoints.openai.api_server \
    --model training/merged/qwen2.5-3b-astrologer \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 \
    --port 8000
```

## Parameter Reference

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `--model` | Model path or HuggingFace ID | `Qwen/Qwen2.5-3B-Instruct` |
| `--host` | Bind address | `0.0.0.0` |
| `--port` | Server port | `8000` |
| `--max-model-len` | Maximum context length | `4096` |
| `--gpu-memory-utilization` | Fraction of GPU memory to use (0.0-1.0) | `0.85` |
| `--dtype` | Model precision | `bfloat16` |
| `--trust-remote-code` | Allow custom model code | (required for Qwen) |
| `--tensor-parallel-size` | Number of GPUs for TP | `1` (or GPU count) |
| `--pipeline-parallel-size` | Number of GPUs for PP | `1` |
| `--max-num-seqs` | Max sequences in batch | `256` |
| `--max-num-batched-tokens` | Max tokens per batch | `8192` |
| `--enable-lora` | Enable LoRA adapter serving | (if not merged) |
| `--lora-modules` | LoRA module mappings | `name=path` |

## Testing the vLLM API

```bash
# Health check
curl http://localhost:8000/health

# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen2.5-3b-astrologer",
        "messages": [
            {"role": "user", "content": "Mera naam Rahul hai. DOB 15 March 1995, 8:30 AM, Delhi."}
        ],
        "max_tokens": 256,
        "temperature": 0.7
    }'

# Streaming
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen2.5-3b-astrologer",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": true
    }'
```

## OpenAI Client Integration

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # vLLM doesn't require auth
)

response = client.chat.completions.create(
    model="qwen2.5-3b-astrologer",
    messages=[
        {"role": "user", "content": "Mera naam Rahul hai. DOB 15 March 1995."}
    ],
    max_tokens=256,
    temperature=0.7,
)
print(response.choices[0].message.content)
```

## GPU Memory Optimization

| Setting | VRAM Saved | Quality Impact |
|---------|-----------|---------------|
| `--gpu-memory-utilization 0.85` | 15% | None |
| `--max-model-len 4096` | Controls KV cache | Limits context |
| `--dtype bfloat16` | 50% vs fp32 | Minimal |
| `--max-num-seqs 128` | Controls batch RAM | Lower throughput |

## Troubleshooting

| Error | Solution |
|-------|----------|
| "CUDA out of memory" | Reduce `gpu-memory-utilization` or `max-model-len` |
| "Model not supported" | Use `--trust-remote-code` |
| "Tokenizer not found" | Use full HuggingFace model ID |
| "LoRA loading failed" | Merge LoRA first with `src/training/merge.py` |
| "Slow first request" | Normal — model loads on first request. Pre-warm with a health check. |
