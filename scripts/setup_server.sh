#!/usr/bin/env bash
# Run manually on the server AFTER reviewing this project.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
export UV_CACHE_DIR="$ROOT/.cache/uv" UV_NO_CACHE=false UV_LINK_MODE=hardlink
export HF_HOME="$ROOT/.cache/huggingface" TORCH_HOME="$ROOT/.cache/torch"
export TMPDIR="$ROOT/.cache/tmp"
export XDG_CACHE_HOME="$ROOT/.cache" PYTHONPYCACHEPREFIX="$ROOT/.cache/pycache"
export UV_PYTHON_INSTALL_DIR="$ROOT/.cache/uv-python"
mkdir -p "$TMPDIR"
test "$(git -C "$ROOT/vendor/lerobot" rev-parse HEAD)" = "240b4a0314ae0879cdd928c7f4bdc1eee9a01b3b"
uv venv "$ROOT/.venv" --python python3.12
uv pip install --python "$ROOT/.venv/bin/python" \
  --constraint "$ROOT/requirements-lock.txt" \
  --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match \
  -e "$ROOT/vendor/lerobot[training]" -e "$ROOT" \
  'torch==2.11.0' 'torchvision==0.26.0' 'torchcodec==0.11.0'
uv pip check --python "$ROOT/.venv/bin/python"
"$ROOT/.venv/bin/python" -m unittest discover -s "$ROOT/tests" -v
