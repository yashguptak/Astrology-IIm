#!/usr/bin/env bash
#
# Complete VPS setup script for Astrology LLM API
#
# Tested on: Ubuntu 22.04 LTS
# Usage: chmod +x setup_vps.sh && sudo ./setup_vps.sh
#
# What it does:
#   1. System update
#   2. Install Python 3.10, Git, build tools
#   3. Install NVIDIA driver 550 + CUDA 12.4
#   4. Install Docker + NVIDIA Container Toolkit
#   5. Clone repo and build deployment container
#   6. Configure Nginx reverse proxy
#   7. Set up Systemd service
#   8. Configure UFW firewall
#   9. Start the API server
#
# Expected output: Running API at http://<SERVER_IP>:8000
#                  Health check: curl http://<SERVER_IP>:8000/health
#

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}  ASTROLOGY LLM — VPS Setup Script${NC}"
echo -e "${CYAN}  Ubuntu 22.04${NC}"
echo -e "${CYAN}============================================================${NC}"

# Config
REPO_URL="${REPO_URL:-https://github.com/YOUR_USERNAME/astrology-llm.git}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
DOMAIN="${DOMAIN:-}"
EMAIL="${EMAIL:-}"

# --- Step 1: System update ---
echo ""
echo -e "${YELLOW}[1/9] System update...${NC}"
apt-get update && apt-get upgrade -y
apt-get install -y curl wget git build-essential software-properties-common

# --- Step 2: Install Python ---
echo ""
echo -e "${YELLOW}[2/9] Installing Python 3.10...${NC}"
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update
apt-get install -y python3.10 python3.10-venv python3.10-dev python3-pip
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
python3 --version

# --- Step 3: Install NVIDIA Driver + CUDA ---
echo ""
echo -e "${YELLOW}[3/9] Installing NVIDIA driver + CUDA 12.4...${NC}"

# Check if NVIDIA GPU present
if lspci | grep -i nvidia > /dev/null 2>&1; then
    echo "NVIDIA GPU detected. Installing driver and CUDA..."

    # Add NVIDIA repository
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    apt-get update

    # Install driver 550 + CUDA 12.4
    apt-get install -y nvidia-driver-550 cuda-toolkit-12-4

    # Clean up
    rm -f cuda-keyring_1.1-1_all.deb

    echo ""
    echo -e "${YELLOW}  NVIDIA driver installed. REBOOT REQUIRED to load the driver.${NC}"
    echo -e "${YELLOW}  After reboot, verify with: nvidia-smi${NC}"
else
    echo -e "${RED}  No NVIDIA GPU detected.${NC}"
    echo "  vLLM requires a GPU. You can still use CPU inference (very slow)."
    echo "  Skipping driver installation."
fi

# --- Step 4: Install Docker ---
echo ""
echo -e "${YELLOW}[4/9] Installing Docker...${NC}"
apt-get install -y docker.io docker-compose-v2
systemctl enable docker
systemctl start docker

# NVIDIA Container Toolkit
if command -v nvidia-smi &> /dev/null; then
    echo "Installing NVIDIA Container Toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update
    apt-get install -y nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
fi

# --- Step 5: Clone repo ---
echo ""
echo -e "${YELLOW}[5/9] Cloning repository...${NC}"
cd /opt
if [ -d astrology-llm ]; then
    echo "Updating existing repo..."
    cd astrology-llm && git pull
else
    git clone "$REPO_URL"
    cd astrology-llm
fi

# --- Step 6: Build and start Docker container ---
echo ""
echo -e "${YELLOW}[6/9] Building Docker image and starting service...${NC}"

cat > /opt/astrology-llm/deployment/docker-compose.yml << 'DOCKERCOMPOSE'
version: '3.8'

services:
  astrology-api:
    build:
      context: .
      dockerfile: deployment/Dockerfile.api
    ports:
      - "8001:8000"
    environment:
      - ASTROLOGY_MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}
      - ASTROLOGY_USE_VLLM=1
    volumes:
      - huggingface_cache:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped
    logging:
      driver: "json-file"
      options:
        max-size: "50m"
        max-file: "3"

volumes:
  huggingface_cache:
DOCKERCOMPOSE

cd /opt/astrology-llm
docker compose -f deployment/docker-compose.yml build
docker compose -f deployment/docker-compose.yml up -d

echo ""
echo "Container status:"
docker ps | grep astrology-api

# --- Step 7: Nginx reverse proxy ---
echo ""
echo -e "${YELLOW}[7/9] Configuring Nginx...${NC}"
apt-get install -y nginx certbot python3-certbot-nginx

if [ -n "$DOMAIN" ]; then
    cat > /etc/nginx/sites-available/astrology-api << 'NGINX'
server {
    listen 80;
    server_name DOMAIN;

    location / {
        proxy_pass http://127.0.0.1:8001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Long timeout for streaming
        proxy_read_timeout 120s;
        proxy_send_timeout 120s;
    }

    # Increase body size for long prompts
    client_max_body_size 10M;
}
NGINX

    sed -i "s/DOMAIN/$DOMAIN/g" /etc/nginx/sites-available/astrology-api

    # Enable site
    ln -sf /etc/nginx/sites-available/astrology-api /etc/nginx/sites-enabled/
    rm -f /etc/nginx/sites-enabled/default
    nginx -t
    systemctl restart nginx

    # SSL cert
    certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos -m "$EMAIL" || true
else
    echo "No domain provided. Skipping Nginx HTTPS setup."
    echo "API will be available on port 8001."
fi

# --- Step 8: Firewall ---
echo ""
echo -e "${YELLOW}[8/9] Configuring firewall...${NC}"
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable
ufw status

# --- Step 9: Verify ---
echo ""
echo -e "${YELLOW}[9/9] Verifying deployment...${NC}"
sleep 5
curl -s http://localhost:8001/health | python3 -m json.tool || echo "Service may still be starting. Check: docker logs astrology-api"

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  DEPLOYMENT COMPLETE${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "  API: http://<SERVER_IP>:8001"
echo "  Health: curl http://<SERVER_IP>:8001/health"
echo ""
echo "  To view logs:"
echo "    docker logs -f astrology-api"
echo ""
echo "  To restart:"
echo "    docker compose -f /opt/astrology-llm/deployment/docker-compose.yml restart"
echo ""
echo "  If you added a domain, HTTPS is at:"
echo "    https://$DOMAIN"
echo ""
echo -e "${YELLOW}  IMPORTANT: If this is the first run, the model will download${NC}"
echo -e "${YELLOW}  (~6 GB). This may take 5-10 minutes on first startup.${NC}"
echo ""
