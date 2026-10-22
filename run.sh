#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$ROOT/venv/bin/python"
KERNEL_NAME="signal-processing-notebooks"
KERNEL_DISPLAY="Python (Signal Processing Notebooks)"

if [[ ! -x "$PY" ]]; then
  echo "[ERROR] Python virtual environment not found at '$PY'."
  exit 1
fi

echo "[1/2] Installing project and dependencies..."
"$PY" -m pip install -e "$ROOT"

echo "[2/2] Registering notebook kernel '$KERNEL_NAME'..."
"$PY" -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY"

echo "Starting Jupyter Lab in notebooks directory..."
exec "$PY" -m jupyter lab --notebook-dir "$ROOT/notebooks"
