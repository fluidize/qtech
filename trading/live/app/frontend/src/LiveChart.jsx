import { useEffect, useRef } from "react";
import { createChart, ColorType, CandlestickSeries, LineSeries } from "lightweight-charts";

export default function LiveChart({ candles, decisions, symbol, interval }) {
  const containerRef = useRef(null);
  const chartRef = useRef(null);
  const priceSeriesRef = useRef(null);
  const signalSeriesRef = useRef(null);

  useEffect(() => {
    const el = containerRef.current;
    const chart = createChart(el, {
      layout: { background: { type: ColorType.Solid, color: "#000" }, textColor: "#ccc" },
      grid: { vertLines: { color: "#1a1a1a" }, horzLines: { color: "#1a1a1a" } },
      width: el.clientWidth,
      height: el.clientHeight,
      timeScale: { borderColor: "#333", rightOffset: 5 },
    });

    const priceSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#26a69a",
      downColor: "#ef5350",
      borderVisible: false,
      wickUpColor: "#26a69a",
      wickDownColor: "#ef5350",
    });

    const signalSeries = chart.addSeries(LineSeries, {
      color: "#3b82f6",
      lineWidth: 2,
    });
    signalSeries.moveToPane(1);

    priceSeriesRef.current = priceSeries;
    signalSeriesRef.current = signalSeries;
    chartRef.current = chart;

    const onResize = () => {
      chart.applyOptions({ width: el.clientWidth, height: el.clientHeight });
    };
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      chart.remove();
    };
  }, []);

  useEffect(() => {
    if (!candles.length) return;
    priceSeriesRef.current.setData(
      candles.map((c) => ({ time: c.time, open: c.open, high: c.high, low: c.low, close: c.close }))
    );
    const withSignal = candles.map((c, i) => ({ time: c.time, value: decisions[i]?.value ?? 0 }));
    signalSeriesRef.current.setData(withSignal);
    chartRef.current.timeScale().fitContent();
  }, [candles, decisions]);

  return (
    <div style={{ height: "100%", width: "100%" }} ref={containerRef} />
  );
}
