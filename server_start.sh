#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PORT="${PORT:-7860}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi was not found. Install the NVIDIA driver first." >&2
  exit 1
fi

echo "Detected NVIDIA GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if [ ! -d ".venv" ]; then
  echo "Creating Python virtual environment..."
  "$PYTHON_BIN" -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
else
  # shellcheck disable=SC1091
  source .venv/bin/activate
  if [ "${LTX_UPDATE_DEPS:-0}" = "1" ]; then
    python -m pip install --upgrade -r requirements.txt
  fi
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Leave LTX_MAX_GPU_WORKERS unset to use every visible CUDA GPU automatically.
# Example to limit a 2-GPU server to one worker:
#   LTX_MAX_GPU_WORKERS=1 ./server_start.sh

echo "Starting LTX server on 0.0.0.0:${PORT}"
exec python run.py --server --port "$PORT"
