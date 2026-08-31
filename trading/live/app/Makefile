.DEFAULT_GOAL := run
FRONTEND := frontend
VENV := $(shell bash -c 'cd "$$(dirname "$(CURDIR)")/../.." && pwd')/.venv/bin/python
PY := $(shell command -v $(VENV) 2>/dev/null || command -v python)

.PHONY: dev run install build clean

# single-command dev: backend + vite hot-reload together
dev:
	@echo "[dev] starting backend (port 8000) + vite (port 5173)"
	@trap 'kill 0' EXIT; \
	$(PY) -m trading.live.app.backend.server & \
	(cd $(FRONTEND) && npm run dev)

# production-ish: build dist and serve from the backend
run: build
	@echo "[run] serving built app at http://127.0.0.1:8000"
	$(PY) -m trading.live.app.backend.server

install:
	pip install -e .
	(cd $(FRONTEND) && npm install)

build:
	(cd $(FRONTEND) && npm run build)

clean:
	rm -rf $(FRONTEND)/dist
