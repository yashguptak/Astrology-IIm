# Model Selection: Qwen for Astrologer Chatbot

## Qwen2.5 vs Qwen3

| Feature | Qwen2.5 | Qwen3 |
|---------|---------|-------|
| Release | Sep 2024 | Apr 2025 |
| Community maturity | Extensive fine-tuning guides, TRL examples | Smaller community feedback |
| SFT documentation | Well-documented LoRA/QLoRA recipes | Fewer published SFT recipes |
| vLLM support | Production-proven, fully supported | Supported but less battle-tested |
| Thinker mode | Not available | Available (adds reasoning tokens) |
| Context length | 32K tokens | 128K tokens |
| Multilingual (Hindi) | Strong — 29 languages | Improved but less tested in production |

**Recommendation: Qwen2.5** — for SFT, the maturity of fine-tuning guides, TRL compatibility, and production-proven vLLM serving outweigh Qwen3's newer features. Thinker mode in Qwen3 adds latency without benefit for a conversational astrologer.

## Size Variants

| Variant | Params | Training VRAM (QLoRA) | Inference VRAM | Cost/hr (T4) | Quality | Verdict |
|---------|--------|----------------------|----------------|-------------|---------|---------|
| Qwen2.5-0.5B | 0.5B | 3 GB | 2 GB | $0.11 | Too small | ✗ |
| Qwen2.5-1.5B | 1.5B | 5 GB | 4 GB | $0.11 | Basic | ✗ |
| **Qwen2.5-3B-Instruct** | **3B** | **8 GB** | **6 GB** | **$0.11** | **Good** | **✅ Winner** |
| Qwen3-4B | 4B | 10 GB | 8 GB | $0.11 | Better | ⚠️ Alternative |
| Qwen2.5-7B-Instruct | 7B | 14 GB | 14 GB | $0.34 | Excellent | Budget concern |
| Qwen3-8B | 8B | 18 GB | 16 GB | $0.34 | Excellent | Overkill |

## Why Qwen2.5-3B-Instruct

| Factor | Analysis |
|--------|----------|
| **Training cost** | Trains on the cheapest cloud GPU (T4 16GB at ~$0.11/hr). Total: ~$0.33 for 3 hours. |
| **Inference cost** | Runs on a $10-15/month VPS with 6GB VRAM or even CPU (2-3 tok/s). |
| **Quality** | 3B params is sufficient for a narrow-domain astrologer chatbot. The model's existing Hindi-English Hinglish capability from pre-training is excellent. |
| **ChatML native** | Qwen2.5 was trained with ChatML format — identical to our dataset. Zero format mismatch. |
| **Upgrade path** | If quality is insufficient, switching to 7B requires changing only the model name in config. |

## VRAM Breakdown (Qwen2.5-3B, QLoRA 4-bit)

| Component | VRAM |
|-----------|------|
| Model weights (4-bit) | ~2.0 GB |
| LoRA adapters (r=16) | ~0.3 GB |
| Gradients | ~1.0 GB |
| Optimizer (8-bit AdamW) | ~1.0 GB |
| Activations (batch=2, seq=2048) | ~3.7 GB |
| **Total training** | **~8 GB** |
| **Inference (merged, bf16)** | **~6 GB** |

## How QLoRA Makes 3B Viable

QLoRA (4-bit NormalFloat) reduces memory ~4x vs full fine-tuning:
- Full 3B fine-tuning: ~24 GB
- QLoRA 3B: ~8 GB
- Gradient checkpointing reduces activations by ~30%

This enables training on a T4 (16 GB) with comfortable headroom.
