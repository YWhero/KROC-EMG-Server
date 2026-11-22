#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
export HF_HOME="$ROOT/.cache/huggingface" TORCH_HOME="$ROOT/.cache/torch"
export XDG_CACHE_HOME="$ROOT/.cache" TMPDIR="$ROOT/.cache/tmp"
export PYTHONPYCACHEPREFIX="$ROOT/.cache/pycache"
export CUDA_CACHE_PATH="$ROOT/.cache/cuda"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
mkdir -p "$TMPDIR"
cd "$ROOT"
exec "$ROOT/.venv/bin/python" -m kroc_act.train "$@"
