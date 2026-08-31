import React, { useEffect, useRef, useState } from "react";
import LiveChart from "./LiveChart.jsx";

const DEFAULTS = {
  symbol: "SOL-USDT",
  interval: "1h",
  days: 60,
  strategy: "ema_cross",
};

export default function App() {
  const [symbol, setSymbol] = useState(DEFAULTS.symbol);
  const [interval, setInterval] = useState(DEFAULTS.interval);
  const [strategy, setStrategy] = useState(DEFAULTS.strategy);
  const [days, setDays] = useState(DEFAULTS.days);
  const [params, setParams] = useState({});
  const [strategies, setStrategies] = useState({});
  const [candles, setCandles] = useState([]);
  const [decisions, setDecisions] = useState([]);
  const [status, setStatus] = useState("idle");
  const wsRef = useRef(null);

  const streamUrl = () => {
    const q = new URLSearchParams({
      symbol,
      interval,
      strategy,
      params: JSON.stringify(params),
    });
    return `${location.origin === "http://localhost:5173" ? "ws://localhost:5173" : "ws://localhost:8000"}/ws/stream?${q}`;
  };

  const fetchStrategies = () => {
    fetch("/api/strategies").then((r) => r.json()).then((data) => {
      setStrategies(data);
      if (data[strategy]) setParams(data[strategy].params);
    });
  };

  const start = () => {
    if (wsRef.current) wsRef.current.close();
    setCandles([]);
    setDecisions([]);
    setStatus("connecting");
    const ws = new WebSocket(streamUrl());
    wsRef.current = ws;
    ws.onopen = () => setStatus("streaming");
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "candle") {
        setCandles((prev) => {
          const next = prev.filter((c) => c.time !== msg.candle.time);
          next.push(msg.candle);
          return next;
        });
        setDecisions((prev) => {
          const next = prev.filter((d) => d.time !== msg.candle.time);
          next.push({ time: msg.candle.time, value: msg.decision });
          return next;
        });
      } else if (msg.type === "error") {
        setStatus("error: " + msg.message);
      } else if (msg.type === "done") {
        setStatus("done");
      }
    };
    ws.onerror = () => setStatus("error");
    ws.onclose = () => setStatus((s) => (s === "streaming" ? "closed" : s));
  };

  const stop = () => {
    if (wsRef.current) wsRef.current.close();
    wsRef.current = null;
    setStatus("stopped");
  };

  const loadBacktest = () => {
    const q = new URLSearchParams({
      symbol,
      interval,
      days: String(days),
      strategy,
      params: JSON.stringify(params),
    });
    fetch(`/api/backtest?${q}`).then((r) => r.json()).then((data) => {
      if (data.error) return setStatus("error: " + data.error);
      setCandles(data.candles);
      setDecisions(data.decisions);
      setStatus("backtest loaded");
    });
  };

  useEffect(fetchStrategies, []);

  const onStrategyChange = (name) => {
    setStrategy(name);
    if (strategies[name]) setParams(strategies[name].params);
  };

  const statusColor = {
    idle: "#888",
    connecting: "#eab308",
    streaming: "#22c55e",
    done: "#22c55e",
    closed: "#888",
    stopped: "#888",
    "backtest loaded": "#38bdf8",
  }[status] || "#888";

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", background: "#0d1117", color: "#e6edf3" }}>
      <div style={{ padding: "14px 18px", background: "#161b22", borderBottom: "1px solid #21262d" }}>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "flex-end" }}>
          <label style={{ display: "flex", flexDirection: "column", gap: 5 }}>
            <span style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: "0.05em", color: "#8b949e" }}>Symbol</span>
            <input
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              style={{ padding: "7px 10px", background: "#0d1117", border: "1px solid #30363d", borderRadius: 6, color: "#e6edf3", minWidth: 130 }}
            />
          </label>
          <label style={{ display: "flex", flexDirection: "column", gap: 5 }}>
            <span style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: "0.05em", color: "#8b949e" }}>Interval</span>
            <select
              value={interval}
              onChange={(e) => setInterval(e.target.value)}
              style={{ padding: "7px 10px", background: "#0d1117", border: "1px solid #30363d", borderRadius: 6, color: "#e6edf3" }}
            >
              {["1m", "5m", "15m", "1h", "4h", "1d"].map((i) => <option key={i}>{i}</option>)}
            </select>
          </label>
          <label style={{ display: "flex", flexDirection: "column", gap: 5 }}>
            <span style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: "0.05em", color: "#8b949e" }}>Strategy</span>
            <select
              value={strategy}
              onChange={(e) => onStrategyChange(e.target.value)}
              style={{ padding: "7px 10px", background: "#0d1117", border: "1px solid #30363d", borderRadius: 6, color: "#e6edf3" }}
            >
              {Object.keys(strategies).map((s) => <option key={s}>{s}</option>)}
            </select>
          </label>
          <label style={{ display: "flex", flexDirection: "column", gap: 5 }}>
            <span style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: "0.05em", color: "#8b949e" }}>Days</span>
            <input
              type="number"
              min="1"
              value={days}
              onChange={(e) => setDays(Number(e.target.value))}
              style={{ padding: "7px 10px", background: "#0d1117", border: "1px solid #30363d", borderRadius: 6, color: "#e6edf3", width: 80 }}
            />
          </label>
          <div style={{ display: "flex", gap: 8 }}>
            <button
              onClick={start}
              style={{ padding: "8px 16px", background: "#238636", border: "1px solid #2ea043", borderRadius: 6, color: "#fff", cursor: "pointer", fontWeight: 600 }}
            >
              ▶ Start Stream
            </button>
            <button
              onClick={loadBacktest}
              style={{ padding: "8px 16px", background: "#1f6feb", border: "1px solid #388bfd", borderRadius: 6, color: "#fff", cursor: "pointer", fontWeight: 600 }}
            >
              ⤓ Backtest
            </button>
            <button
              onClick={stop}
              style={{ padding: "8px 16px", background: "#da3633", border: "1px solid #f85149", borderRadius: 6, color: "#fff", cursor: "pointer", fontWeight: 600 }}
            >
              ■ Stop
            </button>
          </div>
        </div>
        <div style={{ marginTop: 10, fontSize: 12, color: "#8b949e", display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%", background: statusColor, display: "inline-block" }} />
          {status}
        </div>
      </div>
      <div style={{ flex: 1, minHeight: 0 }}>
        <LiveChart candles={candles} decisions={decisions} symbol={symbol} interval={interval} />
      </div>
    </div>
  );
}
