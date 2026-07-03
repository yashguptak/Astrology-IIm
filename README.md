# Vedaz Astrology LLM

Qwen2.5-3B-Instruct fine-tuned as a compassionate Vedic astrologer chatbot with QLoRA SFT. Supports Hindi, English, and Hinglish conversations.

## Project Status

| Dataset | Entries |
|---------|---------|
| Raw conversations | 60 |
| Unique after dedup | 55 |
| Training split | 49 (89%) |
| Validation split | 6 (11%) |
| Total messages | 420 |
| Avg conversation length | 7.6 turns |
| Avg synthetic length | 46 turns |

## Key Features

- **ChatML format** — Qwen2.5 native format with system/user/assistant roles
- **Hindi-English code-mixing** — natural Hinglish astrology dialogues
- **Ethical guardrails** — crisis helpline referrals, no predictions of death/illness, no third-party chart analysis
- **QLoRA SFT** — trains on T4 16GB ($0.11/hr on Colab/RunPod)
- **vLLM inference** — OpenAI-compatible `/v1/chat/completions` API
- **Production-ready** — FastAPI, Docker, Nginx, systemd service included

## Quick Start

```bash
# Install
pip install -r requirements-local.txt

# Run data pipeline
python run_pipeline.py

# Train (auto-detects GPU; exits gracefully on CPU)
python src/training/train.py

# Chat interactively
python src/inference/interactive_chat.py

# Launch API server
python src/api/server.py

# Launch Streamlit UI
streamlit run streamlit_app.py

# Evaluate
python src/evaluation/run_evaluation.py
```

## Training (GPU Required)

| Platform | Command |
|----------|---------|
| Colab (recommended) | Upload `scripts/train_colab.ipynb` |
| Ubuntu/RunPod | `bash scripts/train_linux.sh` |
| Windows | `.\scripts\train_windows.ps1` |
| Docker | `bash scripts/train_docker.sh` |

Training produces a QLoRA adapter in `training/checkpoints/`, then merges to `training/merged/`.

## Inference

```python
from src.inference import AstrologyEngine
engine = AstrologyEngine(model_name="path/to/merged")
response = engine.chat("Namaste ji, mera breakup ho gaya hai.")
```

For production: build the Docker API image and deploy with docker-compose.

## Deployment

```bash
# On Ubuntu 22.04 VPS
bash deployment/setup_vps.sh

# Or with Docker
docker compose up -d
```

Serves at `http://localhost:8000` with `/v1/chat/completions`, `/v1/completions`, and `/health` endpoints.

## Evaluation Metrics

- BLEU, ROUGE-L, perplexity
- Latency (p50/p95/p99) and throughput (req/s)
- Hallucination detection (red-flag patterns: death predictions, guarantee claims, medical advice)

## Project Structure

```
astrology-llm/
├── data/
│   ├── raw/chat_data.json        # 55 deduplicated conversations
│   ├── processed/                # Pipeline outputs (train.json, val.json)
│   └── synthetic/                # Generated conversations
├── src/
│   ├── data_pipeline/            # validator, cleaner, deduplicator, formatter, splitter
│   ├── training/                 # config, dataset, train, merge, callbacks
│   ├── inference/                # engine, history manager, interactive chat
│   ├── api/                      # FastAPI server, schemas
│   └── evaluation/               # metrics, benchmark, hallucination detection
├── configs/                      # YAML configs (training, model, dataset)
├── scripts/                      # train_windows.ps1, train_linux.sh, train_colab.ipynb
├── deployment/                   # Dockerfile, nginx.conf, systemd service, setup script
├── docs/                         # model_selection.md, api.md, vllm_serving.md
└── tests/                        # pytest suite (33 tests)
```

## Tests

```bash
python -m pytest -v
```

All 33 tests pass (validator, cleaner, deduplicator, formatter, splitter, conversation generation).

## Model Selection

See `docs/model_selection.md` for the full analysis. TL;DR: Qwen2.5-3B-Instruct is the best fit for cost, quality, and training accessibility.

## License

MIT
