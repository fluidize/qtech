import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import aiohttp
import numpy as np
import pandas as pd
from aiohttp import web


class LogBroker:
    def __init__(self) -> None:
        self._loops: Dict[int, asyncio.AbstractEventLoop] = {}
        self._subs: List[asyncio.Queue] = []

    def subscribe(self, loop: asyncio.AbstractEventLoop) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._subs.append(q)
        self._loops[id(q)] = loop
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self._subs:
            self._subs.remove(q)
        self._loops.pop(id(q), None)

    def publish(self, message: str) -> None:
        for q in list(self._subs):
            loop = self._loops.get(id(q))
            if loop is None or loop.is_closed():
                continue

            async def _push(q=q):
                if q.full():
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                q.put_nowait(message)

            loop.call_soon_threadsafe(lambda: asyncio.ensure_future(_push()))


LOG_BROKER = LogBroker()
LOGGER = logging.getLogger("trading.live")


class WsLogHandler(logging.Handler):
    """A logging.Handler that forwards every record to the live log subscribers."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            LOG_BROKER.publish(self.format(record))
        except Exception:
            pass


def log(*args, **kwargs) -> None:
    LOGGER.info(*args, **kwargs)


def install_ws_log_handler() -> None:
    LOGGER.setLevel(logging.INFO)
    if not any(isinstance(h, WsLogHandler) for h in LOGGER.handlers):
        handler = WsLogHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s %(levelname)s] %(message)s", "%H:%M:%S")
        )
        LOGGER.addHandler(handler)
    # shut up noisy library loggers unless something real happens
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)


import trading.model_tools as mt
from .registry import get_strategies, get_strategy

# static frontend build output
FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"

# default feed settings
DEFAULT_SYMBOL = "SOL-USDT"
DEFAULT_INTERVAL = "1h"
DEFAULT_DAYS = 60
DEFAULT_TICK_SECONDS = 1.0
BUFFER_SIZE = 500

# Binance endpoints (only the historical REST + live websocket are used for the stream)
BINANCE_REST = "https://api.binance.com/api/v3/klines"
BINANCE_WS_BASE = "wss://stream.binance.com:9443/ws/"


def _synthetic_data(days: int, interval: str) -> pd.DataFrame:
    """Generate random-walk OHLCV so the app runs offline when a feed is unavailable."""
    bar_minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}.get(
        interval, 60
    )
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
    log(f"fetching feed symbol={symbol} interval={interval} days={days}")
    data = await asyncio.to_thread(
        mt.fetch_data, [symbol], days, interval, 0, "binance", verbose=False
    )
    if data.empty:
        log(f"feed empty for {symbol} - falling back to synthetic data")
        data = _synthetic_data(days, interval)
    else:
        log(f"fetched {len(data)} bars for {symbol}")
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
    payload = {
        name: {"params": meta["params"]} for name, meta in get_strategies().items()
    }
    return web.json_response(payload)


async def handle_backtest(request: web.Request) -> web.Response:
    """Run a strategy once over historical data, return candles + decisions."""
    query = request.query
    symbol = query.get("symbol", DEFAULT_SYMBOL)
    interval = query.get("interval", DEFAULT_INTERVAL)
    days = int(query.get("days", DEFAULT_DAYS))
    strategy_name = query.get("strategy", "ema_cross")
    params = json.loads(query.get("params", "{}"))

    meta = get_strategy(strategy_name)
    if meta is None:
        return web.json_response(
            {"error": f"unknown strategy: {strategy_name}"}, status=404
        )

    data = await _fetch_feed(symbol, days, interval)
    log(f"running backtest strategy={strategy_name} bars={len(data)} params={params}")
    signals = meta["func"](data, **params)

    candles = [_candle_to_json(i, _asdict(data.iloc[i])) for i in range(len(data))]
    decisions = [
        {"time": int(ts.timestamp()), "value": float(v)}
        for ts, v in zip(data["Datetime"], signals)
    ]

    return web.json_response(
        {
            "symbol": symbol,
            "interval": interval,
            "candles": candles,
            "decisions": decisions,
        }
    )


def _kline_to_json(kline: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a raw Binance kline payload (``k``) to our candle JSON."""
    return {
        "time": int(kline["t"] // 1000),
        "open": float(kline["o"]),
        "high": float(kline["h"]),
        "low": float(kline["l"]),
        "close": float(kline["c"]),
        "volume": float(kline["v"]),
    }


def _binance_symbol(symbol: str) -> str:
    """'SOL-USDT' -> 'SOLUSDT' (strip the quote-currency separator)."""
    return symbol.replace("-", "").replace("/", "").upper()


async def _fetch_recent_klines(symbol: str, interval: str, limit: int) -> List[dict]:
    """Pull the most recent closed candles from Binance REST to warm up the strategy."""
    url = f"{BINANCE_REST}?symbol={_binance_symbol(symbol)}&interval={interval}&limit={limit}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            resp.raise_for_status()
            raw = await resp.json()
    return [
        {
            "time": int(row[0] // 1000),  # open time is ms
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
        }
        for row in raw
    ]


def _decide(func, buffer: pd.DataFrame, **params) -> float:
    """Run a strategy over the buffer and return the latest discretized decision."""
    signals = func(buffer, **params)
    return float(signals.iloc[-1])


async def handle_stream(request: web.Request) -> web.WebSocketResponse:
    """Live Binance kline feed.

    Seeds the strategy with recent history (so EMAs have warmup), then forwards
    every new closed candle plus the algorithm's decision over the WebSocket.
    """
    query = request.query
    symbol = query.get("symbol", DEFAULT_SYMBOL)
    interval = query.get("interval", DEFAULT_INTERVAL)
    strategy_name = query.get("strategy", "ema_cross")
    params = json.loads(query.get("params", "{}"))

    meta = get_strategy(strategy_name)
    if meta is None:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json(
            {"type": "error", "message": f"unknown strategy: {strategy_name}"}
        )
        return ws

    ws = web.WebSocketResponse()
    await ws.prepare(request)

    try:
        seed = await _fetch_recent_klines(symbol, interval, BUFFER_SIZE)
    except Exception as exc:
        log(f"failed to seed history for {symbol}: {exc}")
        await ws.send_json(
            {"type": "error", "message": f"failed to seed history: {exc}"}
        )
        return ws
    log(f"stream seeding {len(seed)} historical candles for {symbol} {interval}")

    func = meta["func"]

    def to_df(candles: List[dict]) -> pd.DataFrame:
        """Build a strategy-friendly OHLCV frame (uppercase cols) from candle dicts."""
        return pd.DataFrame(
            {
                "Datetime": pd.to_datetime([c["time"] for c in candles], unit="s"),
                "Open": [c["open"] for c in candles],
                "High": [c["high"] for c in candles],
                "Low": [c["low"] for c in candles],
                "Close": [c["close"] for c in candles],
                "Volume": [c["volume"] for c in candles],
            }
        )

    buffer: List[dict] = []
    seed_df = to_df(seed)
    seed_signals = func(seed_df, **params)
    # replay the seed so the chart has context and the strategy is warmed up
    for i, row in enumerate(seed):
        buffer.append(row)
        decision = float(seed_signals.iloc[i]) if i < len(seed_signals) else 0.0
        await ws.send_json({"type": "candle", "candle": row, "decision": decision})

    stream_url = f"{BINANCE_WS_BASE}{_binance_symbol(symbol).lower()}@kline_{interval}"
    log(f"connecting to Binance websocket {stream_url}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(stream_url, heartbeat=30) as bws:
                log(f"websocket connected for {symbol} {interval}")
                last_log = 0.0
                async for msg in bws:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    payload = json.loads(msg.data)
                    k = payload.get("k")
                    if not k:
                        continue
                    candle = _kline_to_json(k)
                    buffer.append(candle)
                    buffer = buffer[-BUFFER_SIZE:]
                    decision = _decide(func, to_df(buffer), **params)
                    await ws.send_json(
                        {"type": "candle", "candle": candle, "decision": decision}
                    )
                    now = time.monotonic()
                    if now - last_log >= 2.0:
                        last_log = now
                        log(
                            f"live {symbol} {interval} @ {candle['time']} close={candle['close']:.6f} decision={decision:+.0f}"
                        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log(f"live stream error for {symbol}: {exc}")
        if not ws.closed:
            try:
                await ws.send_json(
                    {"type": "error", "message": f"live stream error: {exc}"}
                )
            except (ConnectionResetError, aiohttp.ClientConnectionResetError):
                pass
    return ws


async def handle_logs(request: web.Request) -> web.WebSocketResponse:
    """Stream the Python backend's log output to the frontend."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    q = LOG_BROKER.subscribe(request.app.loop)
    try:
        await ws.send_json({"type": "log", "line": "[backend] log stream connected"})
        while not ws.closed:
            try:
                line = await asyncio.wait_for(q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            try:
                await ws.send_json({"type": "log", "line": line})
            except (ConnectionResetError, aiohttp.ClientConnectionResetError):
                break
    except asyncio.CancelledError:
        raise
    except Exception:
        pass
    finally:
        LOG_BROKER.unsubscribe(q)
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
    install_ws_log_handler()
    app.router.add_get("/api/health", handle_health)
    app.router.add_get("/api/strategies", handle_strategies)
    app.router.add_get("/api/backtest", handle_backtest)
    app.router.add_get("/ws/stream", handle_stream)
    app.router.add_get("/ws/logs", handle_logs)
    app.router.add_get("/", handle_index)
    app.router.add_get("/{path:.+}", handle_static)
    return app


if __name__ == "__main__":
    web.run_app(create_app(), host="127.0.0.1", port=8000)
