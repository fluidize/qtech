"""Live trading app backend.

A thin aiohttp server that:
- exposes the registered strategies over REST (``/api/strategies``)
- runs a strategy once over historical data (``/api/backtest``)
- streams candle closes + live algorithm decisions over WebSocket (``/ws/stream``)
- serves the built frontend statically (``/api/*`` routes take priority)

The algorithm side follows the backtesting contract exactly: each strategy is a
pure ``data -> signal_series`` function, and signals are discretized to
``{-1, 0, 1}`` for display.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict

import aiohttp
import numpy as np
import pandas as pd
from aiohttp import web

import trading.model_tools as mt
from .strategies import STRATEGIES, normalize_signals

# static frontend build output
FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"

# default feed settings
DEFAULT_SYMBOL = "SOL-USDT"
DEFAULT_INTERVAL = "1h"
DEFAULT_DAYS = 60
DEFAULT_TICK_SECONDS = 1.0
BUFFER_SIZE = 500


def _synthetic_data(days: int, interval: str) -> pd.DataFrame:
    """Generate random-walk OHLCV so the app runs offline when a feed is unavailable."""
    bar_minutes = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440}.get(interval, 60)
    n = max(days * (24 * 60 // bar_minutes), 100)
    rng = np.random.default_rng(0)
    drift = rng.normal(0, 0.002, n).cumsum()
    close = 150.0 * np.exp(drift)
    open_ = close * (1 + rng.normal(0, 0.001, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.002, n)))
    t0 = pd.Timestamp("2024-01-01")
    idx = [t0 + pd.Timedelta(minutes=bar_minutes * i) for i in range(n)]
    return pd.DataFrame(
        {
            "Datetime": idx,
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": rng.uniform(1e3, 1e5, n),
        }
    )


async def _fetch_feed(symbol, days, interval) -> pd.DataFrame:
    """Try the real feed, fall back to synthetic data so the app always has bars."""
    data = await asyncio.to_thread(mt.fetch_data, [symbol], days, interval, 0, "binance", verbose=False)
    if data.empty:
        data = _synthetic_data(days, interval)
    return data


def _candle_to_json(ts, candle: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "time": int(candle["Datetime"].timestamp()),
        "open": float(candle["Open"]),
        "high": float(candle["High"]),
        "low": float(candle["Low"]),
        "close": float(candle["Close"]),
        "volume": float(candle.get("Volume", 0.0)),
    }


def _asdict(row: Any) -> Dict[str, Any]:
    return {k: row[k] for k in ("Datetime", "Open", "High", "Low", "Close", "Volume")}


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def handle_strategies(request: web.Request) -> web.Response:
    payload = {name: {"params": meta["params"]} for name, meta in STRATEGIES.items()}
    return web.json_response(payload)


async def handle_backtest(request: web.Request) -> web.Response:
    """Run a strategy once over historical data, return candles + decisions."""
    query = request.query
    symbol = query.get("symbol", DEFAULT_SYMBOL)
    interval = query.get("interval", DEFAULT_INTERVAL)
    days = int(query.get("days", DEFAULT_DAYS))
    strategy_name = query.get("strategy", "ema_cross")
    params = json.loads(query.get("params", "{}"))

    meta = STRATEGIES.get(strategy_name)
    if meta is None:
        return web.json_response({"error": f"unknown strategy: {strategy_name}"}, status=404)

    data = await _fetch_feed(symbol, days, interval)
    signals = normalize_signals(meta["func"](data, **params))

    candles = [_candle_to_json(i, _asdict(data.iloc[i])) for i in range(len(data))]
    decisions = [
        {"time": int(ts.timestamp()), "value": float(v)}
        for ts, v in zip(data["Datetime"], signals)
    ]

    return web.json_response(
        {"symbol": symbol, "interval": interval, "candles": candles, "decisions": decisions}
    )


async def handle_stream(request: web.Request) -> web.WebSocketResponse:
    """Replay historical candles one-by-one, emitting the live algorithm decision each close."""
    query = request.query
    symbol = query.get("symbol", DEFAULT_SYMBOL)
    interval = query.get("interval", DEFAULT_INTERVAL)
    days = int(query.get("days", DEFAULT_DAYS))
    strategy_name = query.get("strategy", "ema_cross")
    params = json.loads(query.get("params", "{}"))
    tick_seconds = float(query.get("tick", DEFAULT_TICK_SECONDS))

    meta = STRATEGIES.get(strategy_name)
    if meta is None:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({"type": "error", "message": f"unknown strategy: {strategy_name}"})
        return ws

    ws = web.WebSocketResponse()
    await ws.prepare(request)

    data = await _fetch_feed(symbol, days, interval)

    buffer = data.iloc[:0].copy()
    for i in range(len(data)):
        row = _asdict(data.iloc[i])
        buffer = pd.concat([buffer, pd.DataFrame([row])], ignore_index=True)
        buffer = buffer.tail(BUFFER_SIZE)

        signals = normalize_signals(meta["func"](buffer, **params))
        decision = float(signals.iloc[-1])

        await ws.send_json(
            {"type": "candle", "candle": _candle_to_json(i, row), "decision": decision}
        )
        await asyncio.sleep(tick_seconds)

    await ws.send_json({"type": "done"})
    return ws


async def handle_index(request: web.Request) -> web.Response:
    index = FRONTEND_DIST / "index.html"
    if not index.exists():
        return web.json_response(
            {"error": "frontend not built. Run: npm run build (see README)"}, status=503
        )
    return web.FileResponse(index)


async def handle_static(request: web.Request) -> web.Response:
    return web.FileResponse(FRONTEND_DIST / request.match_info["path"])


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/api/health", handle_health)
    app.router.add_get("/api/strategies", handle_strategies)
    app.router.add_get("/api/backtest", handle_backtest)
    app.router.add_get("/ws/stream", handle_stream)
    app.router.add_get("/", handle_index)
    app.router.add_get("/{path:.+}", handle_static)
    return app


if __name__ == "__main__":
    web.run_app(create_app(), host="127.0.0.1", port=8000)
