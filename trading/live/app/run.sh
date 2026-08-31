#!/usr/bin/env bash
# Build the frontend (if needed) and start the live trading app.
# Run from the repo root; auto-activates the .venv.
# Open http://127.0.0.1:8000 when ready.

set -e

# ensure we're at repo root so `trading` imports and .venv is found
# script lives at trading/live/app/run.sh -> go up 3 levels
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
if [ "$(pwd)" != "$ROOT" ]; then
  echo "[run] changing to repo root: $ROOT"
  cd "$ROOT"
fi

# activate venv if present (prefer repo-local, else any active/default python)
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  PY=".venv/bin/python"
elif [ -n "$VIRTUAL_ENV" ]; then
  PY="python"
else
  PY="python"
fi

APPDIR="$ROOT/trading/live/app"
FRONTEND="$APPDIR/frontend"
DIST="$FRONTEND/dist"

# build frontend if dist is missing
if [ ! -f "$DIST/index.html" ]; then
  echo "[run] building frontend..."
  (cd "$FRONTEND" && npm install && npm run build)
fi

echo "[run] using python: $PY"
echo "[run] starting backend at http://127.0.0.1:8000"
echo "[run] press Ctrl+C to stop"
$PY -m trading.live.app.backend.server
