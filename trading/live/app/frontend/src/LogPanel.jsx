import { useEffect, useRef, useState } from "react";

function logsUrl() {
  if (location.origin === "http://localhost:5173") return "ws://localhost:5173/ws/logs";
  return "ws://localhost:8000/ws/logs";
}

const MAX_LINES = 500;

export default function LogPanel() {
  const [lines, setLines] = useState([]);
  const [status, setStatus] = useState("idle");
  const scrollRef = useRef(null);
  const wsRef = useRef(null);

  useEffect(() => {
    let ws;
    try {
      ws = new WebSocket(logsUrl());
    } catch (e) {
      setStatus("error: " + e.message);
      return;
    }
    wsRef.current = ws;
    ws.onopen = () => setStatus("connected");
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "log") {
        setLines((prev) => {
          const next = [...prev, msg.line];
          return next.length > MAX_LINES ? next.slice(next.length - MAX_LINES) : next;
        });
      }
    };
    ws.onerror = () => setStatus("error");
    ws.onclose = () => setStatus((s) => (s === "connected" ? "closed" : s));
    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, []);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [lines]);

  const clear = () => setLines([]);

  const statusColor = { connected: "#a6cc70", closed: "#707a8c", error: "#f28779" }[status] || "#707a8c";

  return (
    <div
      style={{
        width: "15%",
        minWidth: 180,
        maxWidth: 420,
        display: "flex",
        flexDirection: "column",
        background: "#232834",
        borderLeft: "1px solid #2a3040",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "6px 12px", borderBottom: "1px solid #2a3040", gap: 8 }}>
        <span style={{ fontSize: 12, textTransform: "uppercase", letterSpacing: "0.05em", color: "#707a8c", whiteSpace: "nowrap", display: "flex", alignItems: "center" }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%", background: statusColor, display: "inline-block", marginRight: 6 }} />
          Python Output
        </span>
        <div style={{ display: "flex", gap: 4 }}>
          <button onClick={clear} style={{ fontSize: 12, padding: "2px 8px", background: "#2a3040", border: "1px solid #3d424d", borderRadius: 0, color: "#cbccc6", cursor: "pointer" }}>
            C
          </button>
        </div>
      </div>
      <style>{`*::-webkit-scrollbar { width: 0; height: 0; }`}</style>
      <div ref={scrollRef} style={{ flex: 1, overflowY: "auto", overflowX: "hidden", padding: "8px 12px", fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace", fontSize: 12, lineHeight: 1.5, color: "#cbccc6", whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
        {lines.length === 0 ? (
          <span style={{ color: "#5c6773" }}>waiting for backend output…</span>
        ) : (
          lines.map((l, i) => <div key={i}>{l}</div>)
        )}
      </div>
    </div>
  );
}
