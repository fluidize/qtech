import React, { useEffect, useRef, useState } from "react";
import LiveChart from "./LiveChart.jsx";

const DEFAULTS = {
  symbol: "SOL-USDT",
  interval: "1h",
  days: 60,
  strategy: "ema_cross",
  tick: 0.05,
};

export default function App() {
  const [symbol, setSymbol] = useState(DEFAULTS.symbol);
  const [interval, setInterval] = useState(DEFAULTS.interval);
  const [strategy, setStrategy] = useState(DEFAULTS.strategy);
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
      days: String(DEFAULTS.days),
      strategy,
      params: JSON.stringify(params),
      tick: String(DEFAULTS.tick),
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
      days: String(DEFAULTS.days),
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

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", padding: 8, gap: 8 }}>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
        <label>
          Symbol
          <input value={symbol} onChange={(e) => setSymbol(e.target.value)} />
        </label>
        <label>
          Interval
          <select value={interval} onChange={(e) => setInterval(e.target.value)}>
            {["15m", "1h", "4h", "1d"].map((i) => <option key={i}>{i}</option>)}
          </select>
        </label>
        <label>
          Strategy
          <select value={strategy} onChange={(e) => onStrategyChange(e.target.value)}>
            {Object.keys(strategies).map((s) => <option key={s}>{s}</option>)}
          </select>
        </label>
        <button onClick={start}>Start Stream</button>
        <button onClick={loadBacktest}>Load Backtest</button>
        <button onClick={stop}>Stop</button>
        <span style={{ color: "grey" }}>{status}</span>
      </div>
      <div style={{ flex: 1, minHeight: 0 }}>
        <LiveChart candles={candles} decisions={decisions} symbol={symbol} interval={interval} />
      </div>
    </div>
  );
}
