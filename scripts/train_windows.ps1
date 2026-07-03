<#
.SYNOPSIS
    QLoRA SFT Training Launcher for Windows
.DESCRIPTION
    Detects GPU. If none found, prints Colab instructions.
    If GPU found, runs training with configurable parameters.
.PARAMETER Config
    Path to training config YAML (default: configs/training.yaml)
.PARAMETER ResumeFrom
    Resume from a checkpoint path
.PARAMETER MergeOnly
    Skip training, only merge LoRA adapters
.EXAMPLE
    .\scripts\train_windows.ps1
    .\scripts\train_windows.ps1 -Config configs\training.yaml
#>

param(
    [string]$Config = "configs/training.yaml",
    [string]$ResumeFrom = "",
    [switch]$MergeOnly = $false
)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  ASTROLOGY LLM — Training Launcher (Windows)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found. Please install Python 3.10+." -ForegroundColor Red
    exit 1
}
Write-Host "Python: $pythonVersion"

# Check pip and install deps
Write-Host ""
Write-Host "Step 1: Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements-local.txt 2>&1 | Out-Null

# Check GPU
$hasGpu = $false
$nvidiaCheck = nvidia-smi 2>&1
if ($LASTEXITCODE -eq 0) {
    $hasGpu = $true
    Write-Host "GPU detected:" -ForegroundColor Green
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
}
else {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Red
    Write-Host "  NO GPU DETECTED" -ForegroundColor Red
    Write-Host "============================================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "  This machine does not have a CUDA-capable GPU."
    Write-Host "  Training must run on a cloud GPU instance."
    Write-Host ""
    Write-Host "  Option 1: Google Colab (free T4 GPU)"
    Write-Host "    1. Upload project to Google Drive"
    Write-Host "    2. Open: scripts/train_colab.ipynb"
    Write-Host "    3. Run all cells"
    Write-Host ""
    Write-Host "  Option 2: RunPod / Vast.ai / Lambda Labs"
    Write-Host "    1. Create T4 (16GB) instance"
    Write-Host "    2. Clone this repo"
    Write-Host "    3. Run: bash scripts/train_linux.sh"
    Write-Host ""
    Write-Host "  Option 3: WSL2 with GPU passthrough"
    Write-Host "    1. Install WSL2 + CUDA on Windows"
    Write-Host "    2. Run: bash scripts/train_linux.sh"
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Red
}

if (-not $hasGpu) {
    exit 0
}

# Run training
Write-Host ""
Write-Host "Step 2: Starting training..." -ForegroundColor Yellow
Write-Host "Config: $Config"
if ($ResumeFrom) {
    Write-Host "Resume: $ResumeFrom"
}

$pythonArgs = @("-m", "src.training.train", "--config", $Config)
if ($ResumeFrom) {
    $pythonArgs += @("--resume_from_checkpoint", $ResumeFrom)
}

python $pythonArgs

Write-Host ""
Write-Host "Training complete!" -ForegroundColor Green
