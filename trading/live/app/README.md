# Live Trading App

Realtime two-row chart: candlesticks on top, the algorithm's decision (`-1 / 0 / 1`) on the bottom.

## Structure

- `backend/` — aiohttp server (`server.py`) + strategies (`strategies.py`)
- `frontend/` — React + Vite + lightweight-charts app

## Build

Backend (Python 3.11, from repo root):

```bash
pip install -e .
```

Frontend:

```bash
cd trading/live/app/frontend
npm install
npm run build     # outputs to frontend/dist, served by the backend
```

(Requires the backend running first.)

## Run

```bash
# terminal 1
python -m trading.live.app.backend.server

# terminal 2 (dev, hot reload)
cd trading/live/app/frontend && npm run dev
```

Then open `http://localhost:5173`. Or serve the already-built app directly from the backend at `http://127.0.0.1:8000`.