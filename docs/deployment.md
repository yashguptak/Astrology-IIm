# Deployment Guide

## Architecture Overview

```
User → HTTPS (443) → Nginx → FastAPI (8000) → vLLM (Python) → GPU
                              ↑
                          Systemd (auto-restart)
```

## Prerequisites

| Resource | Requirement |
|----------|------------|
| VPS | Ubuntu 22.04 LTS |
| RAM | 16 GB minimum (32 GB recommended) |
| GPU | NVIDIA GPU with 8+ GB VRAM (T4, RTX 3060, etc.) |
| Disk | 50 GB SSD free |
| Domain | (Optional) For HTTPS |

## Quick Start (Automated)

```bash
# SSH into VPS
ssh root@your-server-ip

# Download and run setup script
wget -O setup_vps.sh https://raw.githubusercontent.com/YOUR_USERNAME/astrology-llm/main/deployment/setup_vps.sh
chmod +x setup_vps.sh
sudo ./setup_vps.sh
```

## Manual Step-by-Step Deployment

### 1. SSH Login

```bash
ssh root@192.168.1.100
# Expected: Welcome to Ubuntu 22.04 LTS
```

### 2. System Update

```bash
sudo apt update && sudo apt upgrade -y
# Expected: Reading package lists... Done
```

### 3. Install NVIDIA Driver

```bash
# Check GPU
lspci | grep -i nvidia
# Expected: NVIDIA Corporation GA106 [GeForce RTX 3060]

# Install driver
sudo apt install -y nvidia-driver-550
sudo reboot
# After reboot:
nvidia-smi
# Expected: shows GPU name, driver version, memory
```

### 4. Install CUDA 12.4

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-4
nvcc --version
# Expected: release 12.4, V12.4.131
```

### 5. Install Python + Git

```bash
sudo apt install -y python3.10 python3.10-venv python3-pip git
python3 --version
# Expected: Python 3.10.x
```

### 6. Clone Repository

```bash
cd /opt
sudo git clone https://github.com/YOUR_USERNAME/astrology-llm.git
sudo chown -R $USER:$USER astrology-llm
cd astrology-llm
```

### 7. Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-gpu.txt
pip install -r requirements-local.txt
```

### 8. Download Model

```bash
# Pre-download model to avoid first-request delay
python scripts/download_model.py --model Qwen/Qwen2.5-3B-Instruct
# Expected: Downloading... Model cached at /root/.cache/huggingface
```

### 9. Run vLLM Server

```bash
# Start vLLM with merged model
python -m vllm.entrypoints.openai.api_server \
    --model training/merged/qwen2.5-3b-astrologer \
    --host 0.0.0.0 \
    --port 8001 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16

# Expected: INFO: Started server process [1234]
#           INFO: Uvicorn running on http://0.0.0.0:8001
```

### 10. Test Endpoints

```bash
# Health check — in a new terminal
curl http://localhost:8001/health
# Expected: {"status": "ok", "model": "...", "gpu_available": true}

# Chat completion
curl -X POST http://localhost:8001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 100
    }'
# Expected: {"id": "chatcmpl-...", "choices": [...], "usage": {...}}
```

### 11. Configure Nginx

```bash
sudo apt install -y nginx
sudo nano /etc/nginx/sites-available/astrology-api
# Paste the config from deployment/nginx.conf
sudo ln -s /etc/nginx/sites-available/astrology-api /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default
sudo nginx -t
# Expected: syntax is ok — test is successful
sudo systemctl restart nginx
```

### 12. Set Up HTTPS (with certbot)

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d astrology.yourdomain.com
# Expected: Congratulations! Your certificate has been installed.
```

### 13. Set Up Systemd Service

```bash
sudo cp deployment/astrology-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable astrology-api
sudo systemctl start astrology-api
sudo systemctl status astrology-api
# Expected: active (running)
```

### 14. Configure Firewall

```bash
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw --force enable
sudo ufw status
# Expected: Status: active — 22, 80, 443 ALLOW
```

### 15. Monitor

```bash
# View logs
sudo journalctl -u astrology-api -f

# View Docker logs (if using Docker)
docker compose -f /opt/astrology-llm/deployment/docker-compose.yml logs -f

# GPU monitoring
watch -n 1 nvidia-smi

# Resource usage
htop
```

## Post-Deployment Checklist

- [ ] `curl http://localhost:8001/health` returns `{"status": "ok"}`
- [ ] `curl -X POST http://localhost:8001/v1/chat/completions ...` returns valid response
- [ ] Model loads without CUDA OOM
- [ ] Nginx reverse proxy working on port 80/443
- [ ] HTTPS certificate valid
- [ ] Firewall only exposes ports 22, 80, 443
- [ ] Systemd auto-restart configured
- [ ] Log rotation configured
- [ ] API rate limiting active

## Troubleshooting

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| `nvidia-smi` not found | Driver not installed | `sudo apt install nvidia-driver-550` |
| CUDA OOM on start | GPU memory insufficient | Reduce `gpu-memory-utilization` to 0.7 |
| First request very slow | Model loading from disk | Pre-download model, or use container |
| 502 Bad Gateway from Nginx | Backend not running | `systemctl restart astrology-api` |
| Port already in use | Conflicting service | `kill $(lsof -t -i:8001)` |
| SSL cert expired | Certbot not renewing | `sudo certbot renew` |
| Model returns gibberish | Wrong tokenizer/dtype | Check `--dtype` matches model |

## Memory Optimization

```bash
# vLLM already optimized — set these env vars for extra safety
export VLLM_USE_V1=0            # Use stable vLLM scheduler
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Start with conservative settings
python -m vllm.entrypoints.openai.api_server \
    --model training/merged/qwen2.5-3b-astrologer \
    --gpu-memory-utilization 0.8 \
    --max-model-len 2048 \
    --swap-space 4 \
    --dtype float16
```

## Monitoring with Prometheus + Grafana

```yaml
# prometheus.yml — add to deployment/monit/
scrape_configs:
  - job_name: 'astrology-api'
    static_configs:
      - targets: ['localhost:8001']
```
