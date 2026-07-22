#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "[SCRIPT] start"
echo "[SCRIPT] pwd=$PWD"
echo "[SCRIPT] shell=$SHELL"

if command -v conda >/dev/null 2>&1; then
  if conda env list 2>/dev/null | grep -qE '^\s*BraTS\s'; then
    PYTHON_CMD=(conda run -n BraTS python)
  elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD=(python)
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD=(python3)
  else
    echo "No Python interpreter found." >&2
    exit 1
  fi
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD=(python)
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD=(python3)
else
  echo "No Python interpreter found." >&2
  exit 1
fi

config="configs/train_diffusion_inpaint_mni.json"

echo "==> Training unified MNI inpainting model with $config"
echo "[RUN] ${PYTHON_CMD[*]} -m monai.bundle run --config_file $config"
"${PYTHON_CMD[@]}" -m monai.bundle run --config_file "$config"
status=$?
echo "[DONE] unified training status=$status"
if [ $status -ne 0 ]; then
  echo "Training failed" >&2
  exit $status
fi
