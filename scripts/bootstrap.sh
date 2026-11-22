#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT"
for entry in vendor vendor/lerobot data .cache .venv; do
  [[ ! -L "$ROOT/$entry" ]] || { echo "Refusing symlink: $entry"; exit 1; }
done
export UV_CACHE_DIR="$ROOT/.cache/uv" HF_HOME="$ROOT/.cache/huggingface"
export TORCH_HOME="$ROOT/.cache/torch" XDG_CACHE_HOME="$ROOT/.cache"
export TMPDIR="$ROOT/.cache/tmp" PYTHONPYCACHEPREFIX="$ROOT/.cache/pycache"
export CUDA_CACHE_PATH="$ROOT/.cache/cuda"
export UV_PYTHON_INSTALL_DIR="$ROOT/.cache/uv-python"
mkdir -p "$TMPDIR" "$ROOT/vendor"
command -v git >/dev/null
command -v ffmpeg >/dev/null || { echo 'Install ffmpeg using your server image/package manager first.'; exit 1; }
if ! command -v uv >/dev/null; then
  python3 -m venv "$ROOT/.bootstrap"
  "$ROOT/.bootstrap/bin/python" -m pip install 'uv==0.9.8'
  export PATH="$ROOT/.bootstrap/bin:$PATH"
fi
PIN=240b4a0314ae0879cdd928c7f4bdc1eee9a01b3b
if [[ ! -d "$ROOT/vendor/lerobot" ]]; then
  git init "$ROOT/vendor/lerobot"
  git -C "$ROOT/vendor/lerobot" remote add origin https://github.com/ROBOTIS-GIT/lerobot-cyclo.git
  git -C "$ROOT/vendor/lerobot" fetch --depth 1 origin "$PIN"
  git -C "$ROOT/vendor/lerobot" checkout --detach "$PIN"
fi
test "$(git -C "$ROOT/vendor/lerobot" rev-parse HEAD)" = "$PIN"
bash scripts/setup_server.sh
if [[ "${1:-}" != --env-only ]]; then
  "$ROOT/.venv/bin/python" scripts/reproduce_data.py
fi
"$ROOT/.venv/bin/python" scripts/doctor.py
echo 'Setup complete. See docs/TRAINING.md and docs/INFERENCE.md for the GPU services.'
