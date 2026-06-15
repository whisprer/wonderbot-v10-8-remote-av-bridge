#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV_DIR="${VENV_DIR:-/srv/wonderbot/.venv}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-/srv/weights/wonderbot}"
PROFILE="${PROFILE:-configs/profiles/dual-p40-server-qwen14b.toml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

sudo mkdir -p "$WEIGHTS_ROOT" "$WEIGHTS_ROOT/state" "$WEIGHTS_ROOT/offload" "$WEIGHTS_ROOT/hf-cache" /srv/wonderbot
sudo chown -R "$USER":"$USER" "$WEIGHTS_ROOT" /srv/wonderbot

sudo apt-get update
sudo apt-get install -y \
  git curl wget unzip tmux htop nvtop pciutils jq \
  python3 python3-venv python3-dev build-essential \
  ffmpeg libsndfile1 libgl1 ca-certificates

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[warn] nvidia-smi not found. Install the NVIDIA driver before using this profile." >&2
fi

$PYTHON_BIN -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# Pascal GPUs (Tesla P40) should use the cu126 wheel path with modern PyTorch 2.12+.
pip install --index-url https://download.pytorch.org/whl/cu126 torch torchaudio
pip install -e "$REPO_ROOT[server-hf,live-full,hf-voice,voice,dev]"

export HF_HOME="$WEIGHTS_ROOT/hf-cache"
export TRANSFORMERS_CACHE="$WEIGHTS_ROOT/hf-cache"
export BITSANDBYTES_NOWELCOME=1

python "$REPO_ROOT/scripts/server/verify_dual_p40.py"

echo
echo "Bootstrap complete."
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Run with: wonderbot --config $REPO_ROOT/$PROFILE --diagnostics"
