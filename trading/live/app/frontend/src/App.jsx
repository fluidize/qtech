import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import LiveChart from "./LiveChart.jsx";
import LogPanel from "./LogPanel.jsx";

const DEFAULTS = {
  symbol: "SOL-USDT",
  interval: "1h",
  days: 60,
  strategy: "ema_cross",
};

// Portfolio equity curve: start holding 1 unit of the base currency (worth its
// t0 close), then apply the strategy's position (from the previous bar) to each
// bar's return. Produces values in the same price units as the chart.
function computeEquity(candles, decisions) {
  if (!candles.length) return [];
  const closes = candles.map((c) => c.close);
  let val = closes[0]; // 1 unit bought at the first bar's price
  const out = [{ time: candles[0].time, value: val }];
  for (let i = 1; i < closes.length; i++) {
    const pos = decisions[i - 1]?.value ?? 0; // position from previous bar (no lookahead)
    val *= 1 + pos * (closes[i] / closes[i - 1] - 1);
    out.push({ time: candles[i].time, value: val });
  }
  return out;
}

const PERIODS_PER_YEAR = {
  "1m": 252 * 24 * 60,
  "5m": 252 * 24 * 12,
  "15m": 252 * 24 * 4,
  "1h": 252 * 24,
  "4h": 252 * 6,
  "1d": 252,
};

function erf(x) {
  const p = 0.3275911;
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429;
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const t = 1 / (1 + p * x);
  const y = 1 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * Math.exp(-x * x);
  return sign * y;
}
// two-sided p-value of the Sharpe ratio (t = sharpePerPeriod * sqrt(n)) via normal approx
function sharpePval(sharpeAnnual, n, py) {
  if (!sharpeAnnual || n < 2 || !py) return null;
  const t = (sharpeAnnual / Math.sqrt(py)) * Math.sqrt(n);
  return 1 - erf(Math.abs(t) / Math.SQRT2);
}

function computeMetrics(candles, decisions, interval) {
  if (!candles.length || candles.length < 2) return null;
  const closes = candles.map((c) => c.close);
  const sig = decisions.map((d) => d.value);

  const n = closes.length - 1;
  const rets = [];
  for (let i = 1; i < closes.length; i++) {
    const pos = sig[i - 1] || 0; // position from previous bar (no lookahead)
    rets.push(pos * (closes[i] / closes[i - 1] - 1));
  }

  // equity + drawdown
  let equity = 1;
  let peak = 1;
  let maxDD = 0;
  for (const r of rets) {
    equity *= 1 + r;
    peak = Math.max(peak, equity);
    maxDD = Math.max(maxDD, (peak - equity) / peak);
  }

  const avg = rets.reduce((a, b) => a + b, 0) / n;
  const std = Math.sqrt(rets.reduce((a, b) => a + (b - avg) ** 2, 0) / n);
  const downside = Math.sqrt(
    rets.reduce((a, b) => a + Math.min(b, 0) ** 2, 0) / n
  );
  const py = PERIODS_PER_YEAR[interval] || 252;
  const annualVol = std * Math.sqrt(py);
  const sharpe = std ? (avg / std) * Math.sqrt(py) : 0;
  const sortino = downside ? (avg / downside) * Math.sqrt(py) : 0;
  const totalReturn = equity - 1;
  const buyHold = closes[closes.length - 1] / closes[0] - 1;
  const calmar = maxDD > 0 ? totalReturn / maxDD : 0;

  // reconstruct trades (contiguous runs of nonzero signal)
  const runs = [];
  let dir = 0;
  let start = 0;
  for (let i = 0; i < sig.length; i++) {
    const s = sig[i] || 0;
    if (s !== dir) {
      if (dir !== 0) runs.push({ dir, start, end: i });
      dir = s;
      start = i;
    }
  }
  if (dir !== 0) runs.push({ dir, start, end: sig.length });

  const pnls = runs.map((t) => {
    const entry = closes[t.start];
    const exit = closes[Math.min(t.end, closes.length - 1)];
    return t.dir * (exit / entry - 1);
  });
  const trades = pnls.length;
  const wins = pnls.filter((p) => p > 0);
  const losses = pnls.filter((p) => p < 0);
  const grossProfit = wins.reduce((a, b) => a + b, 0);
  const grossLoss = Math.abs(losses.reduce((a, b) => a + b, 0));
  const winRate = trades ? wins.length / trades : 0;
  const avgWin = wins.length ? grossProfit / wins.length : 0;
  const avgLoss = losses.length ? grossLoss / losses.length : 0;
  const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : Infinity;
  const rr = avgLoss > 0 ? avgWin / avgLoss : 0;
  const breakeven = rr > 0 ? 1 / (rr + 1) : 0;

  // alpha / beta / information ratio vs buy-&-hold benchmark
  const bench = [];
  for (let i = 1; i < closes.length; i++) bench.push(closes[i] / closes[i - 1] - 1);
  let alpha = 0;
  let beta = 0;
  let info = 0;
  const bAvg = bench.reduce((a, b) => a + b, 0) / bench.length;
  const bStd = Math.sqrt(bench.reduce((a, b) => a + (b - bAvg) ** 2, 0) / bench.length);
  const cov = rets.reduce((a, b, i) => a + (b - avg) * (bench[i] - bAvg), 0) / n;
  if (bStd > 0) {
    beta = cov / (bStd * bStd);
    alpha = (avg - beta * bAvg) * py;
  }
  const active = rets.map((r, i) => r - bench[i]);
  const aAvg = active.reduce((a, b) => a + b, 0) / active.length;
  const aStd = Math.sqrt(active.reduce((a, b) => a + (b - aAvg) ** 2, 0) / active.length);
  if (aStd > 0) info = (aAvg / aStd) * Math.sqrt(py);

  // R2 of equity curve against bar index
  let r2 = 0;
  let cum = 1;
  const eq = [];
  for (const r of rets) {
    cum *= 1 + r;
    eq.push(cum);
  }
  const sx = eq.reduce((a, b) => a + b, 0) / eq.length;
  let sy = 0;
  for (let i = 0; i < eq.length; i++) sy += i + 1;
  sy /= eq.length;
  let num = 0;
  let dx = 0;
  let dy = 0;
  for (let i = 0; i < eq.length; i++) {
    num += (eq[i] - sx) * (i + 1 - sy);
    dx += (eq[i] - sx) ** 2;
    dy += (i + 1 - sy) ** 2;
  }
  if (dx > 0 && dy > 0) r2 = (num * num) / (dx * dy);

  return {
    totalReturn,
    buyHold,
    maxDrawdown: maxDD,
    sharpe,
    sharpePval: sharpePval(sharpe, n, py),
    sortino,
    annualVol,
    calmar,
    trades,
    winRate,
    profitFactor,
    rr,
    breakeven,
    alpha,
    beta,
    informationRatio: info,
    r2,
    bars: n,
  };
}

const inputStyle = { padding: "7px 10px", background: "#232834", border: "1px solid #3d424d", borderRadius: 0, color: "#cbccc6" };
const fieldStyle = { display: "flex", flexDirection: "column", gap: 5 };
const labelStyle = { fontSize: 12, textTransform: "uppercase", letterSpacing: "0.05em", color: "#707a8c" };

function Field({ label, children }) {
  return (
    <label style={fieldStyle}>
      <span style={labelStyle}>{label}</span>
      {children}
    </label>
  );
}

export default function App() {
  const [btSymbol, setBtSymbol] = useState(DEFAULTS.symbol);
  const [btInterval, setBtInterval] = useState(DEFAULTS.interval);
  const [btDays, setBtDays] = useState(DEFAULTS.days);
  const [btStrategy, setBtStrategy] = useState(DEFAULTS.strategy);
  const [btDataSource, setBtDataSource] = useState(DEFAULTS.dataSource ?? "binance");

  const [liveSymbol, setLiveSymbol] = useState(DEFAULTS.symbol);
  const [liveInterval, setLiveInterval] = useState(DEFAULTS.interval);
  const [liveStrategy, setLiveStrategy] = useState(DEFAULTS.strategy);

  const [strategies, setStrategies] = useState({});
  const [dataSources, setDataSources] = useState([]);
  const [candles, setCandles] = useState([]);
  const [decisions, setDecisions] = useState([]);
  const [status, setStatus] = useState("idle");
  const [isLive, setIsLive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [showPrice, setShowPrice] = useState(true);
  const [showStrategy, setShowStrategy] = useState(true);
  const [mode, setMode] = useState("backtest"); // "backtest" | "live" (which one is displayed)
  const [loaded, setLoaded] = useState(null); // params of data currently on chart {symbol, interval, days, source, live}
  const wsRef = useRef(null);
  const modeRef = useRef("backtest"); // "backtest" | "live"
  const chartRef = useRef(null);

  const btParams = useMemo(() => strategies[btStrategy]?.params ?? {}, [strategies, btStrategy]);
  const liveParams = useMemo(() => strategies[liveStrategy]?.params ?? {}, [strategies, liveStrategy]);

  const metrics = useMemo(() => computeMetrics(candles, decisions, btInterval), [candles, decisions, btInterval]);
  const equity = useMemo(() => computeEquity(candles, decisions), [candles, decisions]);

  const fetchStrategies = () => {
    fetch("/api/strategies").then((r) => r.json()).then((data) => {
      setStrategies(data);
    });
    fetch("/api/datasources").then((r) => r.json()).then((data) => {
      setDataSources(data.sources ?? []);
      setBtDataSource((v) => v || data.default || "binance");
    });
  };

  const loadBacktest = useCallback(() => {
    modeRef.current = "backtest";
    setMode("backtest");
    if (wsRef.current) wsRef.current.close();
    wsRef.current = null;
    setIsLive(false);
    setLoading(true);
    const q = new URLSearchParams({
      symbol: btSymbol,
      interval: btInterval,
      days: String(btDays),
      strategy: btStrategy,
      data_source: btDataSource,
      params: JSON.stringify(btParams),
    });
    fetch(`/api/backtest?${q}`)
      .then((r) => r.json())
      .then((data) => {
        if (data.error) return setStatus("error: " + data.error);
        setLoaded({ symbol: data.symbol ?? btSymbol, interval: data.interval ?? btInterval, days: data.days ?? btDays, source: data.data_source ?? btDataSource, live: false });
        setCandles(data.candles);
        setDecisions(data.decisions);
        setStatus("backtest loaded");
      })
      .catch(() => setStatus("error: request failed"))
      .finally(() => setLoading(false));
  }, [btSymbol, btInterval, btDays, btStrategy, btParams, btDataSource]);

  const start = useCallback(() => {
    modeRef.current = "live";
    setMode("live");
    if (wsRef.current) wsRef.current.close();
    setCandles([]);
    setDecisions([]);
    setIsLive(true);
    setStatus("connecting");
    setLoaded({ symbol: liveSymbol, interval: liveInterval, days: null, source: "live", live: true });
    const q = new URLSearchParams({
      symbol: liveSymbol,
      interval: liveInterval,
      strategy: liveStrategy,
      params: JSON.stringify(liveParams),
    });
    const base = location.origin === "http://localhost:5173" ? "ws://localhost:5173" : "ws://localhost:8000";
    const ws = new WebSocket(`${base}/ws/stream?${q}`);
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
        setIsLive(false);
      } else if (msg.type === "done") {
        setStatus("done");
      }
    };
    ws.onerror = () => { setStatus("error"); setIsLive(false); };
    ws.onclose = () => { if (wsRef.current === ws) setIsLive(false); setStatus((s) => (s === "streaming" ? "closed" : s)); };
  }, [liveSymbol, liveInterval, liveStrategy, liveParams]);

  const stop = () => {
    if (wsRef.current) wsRef.current.close();
    wsRef.current = null;
    setIsLive(false);
    setStatus("stopped");
  };

  useEffect(fetchStrategies, []);

  // Run the initial backtest once strategies are loaded. Subsequent runs are
  // manual only (Backtest / Start Stream buttons) — field edits don't rerun.
  const initialRunRef = useRef(false);
  useEffect(() => {
    if (initialRunRef.current) return;
    if (!Object.keys(strategies).length) return;
    initialRunRef.current = true;
    loadBacktest();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [strategies]);

  const statusColor = {
    idle: "#707a8c",
    connecting: "#ffd173",
    streaming: "#a6cc70",
    done: "#a6cc70",
    closed: "#707a8c",
    stopped: "#707a8c",
    "backtest loaded": "#5ccfe6",
  }[status] || "#707a8c";

  const glow = (active, color) =>
    active
      ? { border: `1px solid ${color}`, boxShadow: `0 0 0 1px ${color}55, 0 0 14px ${color}44` }
      : { border: "1px solid #2a3040" };

  return (
    <div style={{ display: "flex", height: "100vh", background: "#1f2430", color: "#cbccc6" }}>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column" }}>
      <div style={{ padding: "14px 18px", background: "#232834", borderBottom: "1px solid #2a3040", display: "flex", flexDirection: "column", gap: 10 }}>
        <div style={{ display: "flex", gap: 14, flexWrap: "wrap" }}>
            <div style={{ flex: 1, minWidth: 340, background: "#1f2430", borderRadius: 0, padding: "12px 14px", display: "flex", flexDirection: "column", gap: 8, border: "1px solid " + (mode === "backtest" ? "#5ccfe6" : "#2a3040") }}>
            <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "flex-end" }}>
              <Field label="Symbol">
                <input value={btSymbol} onChange={(e) => setBtSymbol(e.target.value)} style={{ ...inputStyle, minWidth: 130 }} />
              </Field>
              <Field label="Interval">
                <select value={btInterval} onChange={(e) => setBtInterval(e.target.value)} style={inputStyle}>
                  {["1m", "5m", "15m", "1h", "4h", "1d"].map((i) => <option key={i}>{i}</option>)}
                </select>
              </Field>
              <Field label="Days">
                <input type="number" min="1" value={btDays} onChange={(e) => setBtDays(Number(e.target.value))} style={{ ...inputStyle, width: 80 }} />
              </Field>
              <Field label="Strategy">
                <select value={btStrategy} onChange={(e) => setBtStrategy(e.target.value)} style={inputStyle}>
                  {Object.keys(strategies).map((s) => <option key={s}>{s}</option>)}
                </select>
              </Field>
              <Field label="Data Source">
                <select value={btDataSource} onChange={(e) => setBtDataSource(e.target.value)} style={inputStyle}>
                  {dataSources.map((s) => <option key={s}>{s}</option>)}
                </select>
              </Field>
            </div>
            <div style={{ display: "flex", gap: 10, alignItems: "flex-end" }}>
              <button
                onClick={loadBacktest}
                disabled={loading}
                style={{ padding: "8px 16px", height: 35, width: "100%", background: "#5ccfe6", border: "1px solid #5ccfe6", borderRadius: 0, color: "#fff", cursor: loading ? "default" : "pointer", fontWeight: 600, opacity: loading ? 0.7 : 1 }}
              >
                {loading ? "⏳ Backtesting…" : "⤓ Backtest"}
              </button>
            </div>
          </div>

            <div style={{ flex: 1, minWidth: 340, background: "#1f2430", borderRadius: 0, padding: "12px 14px", display: "flex", flexDirection: "column", gap: 8, border: "1px solid " + (mode === "live" ? "#a6cc70" : "#2a3040") }}>
            <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "flex-end" }}>
              <Field label="Symbol">
                <input value={liveSymbol} onChange={(e) => setLiveSymbol(e.target.value)} style={{ ...inputStyle, minWidth: 130 }} />
              </Field>
              <Field label="Interval">
                <select value={liveInterval} onChange={(e) => setLiveInterval(e.target.value)} style={inputStyle}>
                  {["1m", "5m", "15m", "1h", "4h", "1d"].map((i) => <option key={i}>{i}</option>)}
                </select>
              </Field>
              <Field label="Strategy">
                <select value={liveStrategy} onChange={(e) => setLiveStrategy(e.target.value)} style={inputStyle}>
                  {Object.keys(strategies).map((s) => <option key={s}>{s}</option>)}
                </select>
              </Field>
            </div>
            <div style={{ display: "flex", gap: 10, alignItems: "flex-end" }}>
              <button
                onClick={() => (isLive ? stop() : start())}
                style={{
                  padding: "8px 16px",
                  width: "100%",
                  background: isLive ? "#f28779" : "#a6cc70",
                  border: isLive ? "1px solid #f28779" : "1px solid #a6cc70",
                  borderRadius: 0,
                  color: "#fff",
                  cursor: "pointer",
                  fontWeight: 600,
                  height: 35,
                  display: "inline-flex",
                  alignItems: "center",
                  justifyContent: "center",
                  lineHeight: 1,
                }}
              >
                {isLive ? "■ Stop Stream" : "▶ Start Stream"}
              </button>
            </div>
          </div>
        </div>

        <div style={{ fontSize: 12, color: "#707a8c", display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%", background: statusColor, display: "inline-block" }} />
          {status}
          <span style={{ marginLeft: 8, display: "flex", gap: 6 }}>
            <button
              onClick={() => setShowPrice((v) => !v)}
              style={{ fontSize: 12, padding: "1px 8px", background: showPrice ? "#a6cc70" : "#2a3040", border: "1px solid #3d424d", borderRadius: 0, color: "#ffffff", cursor: "pointer", lineHeight: 1.6 }}
            >
              {showPrice ? "✓ Price" : "Price"}
            </button>
            <button
              onClick={() => setShowStrategy((v) => !v)}
              style={{ fontSize: 12, padding: "1px 8px", background: showStrategy ? "#a6cc70" : "#2a3040", border: "1px solid #3d424d", borderRadius: 0, color: "#ffffff", cursor: "pointer", lineHeight: 1.6 }}
            >
              {showStrategy ? "✓ Strategy" : "Strategy"}
            </button>
            <button
              onClick={() => chartRef.current?.fit()}
              title="Fit chart to screen"
              style={{ fontSize: 12, padding: "1px 8px", background: "#2a3040", border: "1px solid #3d424d", borderRadius: 0, color: "#ffffff", cursor: "pointer", lineHeight: 1.6 }}
            >
              Fit
            </button>
          </span>
        </div>

        {metrics && (
          <div style={{ display: "flex", flexWrap: "wrap", gap: 18, fontSize: 12 }}>
            {[
              ["Total Return", metrics.totalReturn, (v) => `${(v * 100).toFixed(2)}%`, metrics.totalReturn >= 0 ? "#a6cc70" : "#f28779"],
              ["Buy & Hold", metrics.buyHold, (v) => `${(v * 100).toFixed(2)}%`, metrics.buyHold >= 0 ? "#a6cc70" : "#f28779"],
              ["Max Drawdown", metrics.maxDrawdown, (v) => `${(v * 100).toFixed(2)}%`, "#f28779"],
              ["Sharpe", metrics.sharpe, (v) => v.toFixed(2), "#cbccc6"],
              ["Sharpe P", metrics.sharpePval, (v) => (v == null ? "—" : v.toFixed(3)), metrics.sharpePval != null && metrics.sharpePval < 0.05 ? "#a6cc70" : "#707a8c"],
              ["Sortino", metrics.sortino, (v) => v.toFixed(2), "#cbccc6"],
              ["Volatility", metrics.annualVol, (v) => `${(v * 100).toFixed(2)}%`, "#cbccc6"],
              ["Calmar", metrics.calmar, (v) => v.toFixed(2), "#cbccc6"],
              ["Trades", metrics.trades, (v) => String(v), "#cbccc6"],
              ["Win Rate", metrics.winRate, (v) => `${(v * 100).toFixed(1)}%`, "#cbccc6"],
              ["Profit Factor", metrics.profitFactor, (v) => (v === Infinity ? "∞" : v.toFixed(2)), "#cbccc6"],
              ["Risk/Reward", metrics.rr, (v) => v.toFixed(2), "#cbccc6"],
              ["Breakeven", metrics.breakeven, (v) => `${(v * 100).toFixed(1)}%`, "#cbccc6"],
              ["Alpha", metrics.alpha, (v) => v.toFixed(2), "#cbccc6"],
              ["Beta", metrics.beta, (v) => v.toFixed(2), "#cbccc6"],
              ["Info Ratio", metrics.informationRatio, (v) => v.toFixed(2), "#cbccc6"],
              ["R²", metrics.r2, (v) => v.toFixed(3), "#707a8c"],
              ["Bars", metrics.bars, (v) => String(v), "#707a8c"],
            ].map(([label, value, fmt, color]) => (
              <div key={label} style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                <span style={{ fontSize: 12, textTransform: "uppercase", letterSpacing: "0.05em", color: "#707a8c" }}>{label}</span>
                <span style={{ fontWeight: 600, color, fontSize: 14 }}>{fmt(value)}</span>
              </div>
            ))}
          </div>
        )}
      </div>
      <div style={{ flex: 1, minHeight: 0, padding: 12 }}>
        <div style={{ height: "100%", borderRadius: 16, overflow: "hidden", border: "1px solid #3d424d", boxShadow: "0 0 0 1px rgba(92, 207, 230, 0.12), 0 4px 16px rgba(0,0,0,0.06)", background: "#1f2430", position: "relative" }}>
          <LiveChart ref={chartRef} candles={candles} decisions={decisions} equity={equity} showPrice={showPrice} showStrategy={showStrategy} symbol={loaded?.symbol ?? ""} interval={loaded?.interval ?? ""} days={loaded?.days ?? null} live={loaded?.live ?? false} />
          {loading && (
            <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(31,36,48,0.7)", zIndex: 10 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10, color: "#cbccc6", fontSize: 14 }}>
                <span style={{ width: 16, height: 16, borderRadius: "50%", border: "2px solid #3d424d", borderTopColor: "#a6cc70", animation: "spin 0.8s linear infinite" }} />
                loading backtest data…
              </div>
            </div>
          )}
        </div>
      </div>
      </div>
      <LogPanel />
    </div>
  );
}
