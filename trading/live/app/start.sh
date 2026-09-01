#!/usr/bin/env bash
set -euo pipefail

# resolve repo root regardless of where this script is invoked from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${SCRIPT_DIR}"                        # trading/live/app
REPO_ROOT="$(cd "${APP_DIR}/../../.." && pwd)" # qtech repo root
VENV="${REPO_ROOT}/.venv"

# auto-activate the qtech venv
if [ -f "${VENV}/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
else
  echo "[start.sh] venv not found at ${VENV}; falling back to system python" >&2
fi

# build the frontend if it hasn't been built yet
if [ ! -d "${APP_DIR}/frontend/dist" ]; then
  echo "[start.sh] building frontend..."
  (cd "${APP_DIR}/frontend" && npm run build)
fi

echo "[start.sh] serving webapp at http://127.0.0.1:8000"
python -m trading.live.app.backend.server
