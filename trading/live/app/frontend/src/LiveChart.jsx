import { useEffect, useRef } from "react";
import { createChart, ColorType } from "lightweight-charts";

export default function LiveChart({ candles, decisions, symbol, interval }) {
  const priceRef = useRef(null);
  const signalRef = useRef(null);
  const priceChartRef = useRef(null);
  const signalChartRef = useRef(null);
  const priceSeriesRef = useRef(null);
  const signalSeriesRef = useRef(null);

  useEffect(() => {
    const price = createChart(priceRef.current, {
      layout: { background: { type: ColorType.Solid, color: "#000" }, textColor: "#ccc" },
      grid: { vertLines: { color: "#1a1a1a" }, horzLines: { color: "#1a1a1a" } },
      width: priceRef.current.clientWidth,
      height: priceRef.current.clientHeight,
      timeScale: { visible: false },
      handleScroll: false,
      handleScale: false,
    });
    const signal = createChart(signalRef.current, {
      layout: { background: { type: ColorType.Solid, color: "#000" }, textColor: "#ccc" },
      grid: { vertLines: { color: "#1a1a1a" }, horzLines: { color: "#1a1a1a" } },
      width: signalRef.current.clientWidth,
      height: signalRef.current.clientHeight,
      timeScale: { borderColor: "#333" },
    });

    const priceSeries = price.addCandlestickSeries({
      upColor: "#26a69a",
      downColor: "#ef5350",
      borderVisible: false,
      wickUpColor: "#26a69a",
      wickDownColor: "#ef5350",
    });
    const signalSeries = signal.addLineSeries({
      color: "#3b82f6",
      lineWidth: 2,
    });

    priceSeriesRef.current = priceSeries;
    signalSeriesRef.current = signalSeries;
    priceChartRef.current = price;
    signalChartRef.current = signal;

    const onResize = () => {
      const w = priceRef.current.clientWidth;
      price.applyOptions({ width: w, height: priceRef.current.clientHeight });
      signal.applyOptions({ width: w, height: signalRef.current.clientHeight });
    };
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      price.remove();
      signal.remove();
    };
  }, []);

  useEffect(() => {
    if (!candles.length) return;
    priceSeriesRef.current.setData(
      candles.map((c) => ({ time: c.time, open: c.open, high: c.high, low: c.low, close: c.close }))
    );
    signalSeriesRef.current.setData(candles.map((c) => ({ time: c.time, value: 0 })));
    priceChartRef.current.timeScale().fitContent();
    signalChartRef.current.timeScale().fitContent();
  }, [symbol, interval]);

  useEffect(() => {
    if (!candles.length) return;
    const withSignal = candles.map((c, i) => ({ time: c.time, value: decisions[i] ?? 0 }));
    signalSeriesRef.current.setData(withSignal);
    signalChartRef.current.timeScale().fitContent();
  }, [candles, decisions]);

  return (
    <div className="charts">
      <div style={{ height: "60%" }} ref={priceRef} />
      <div style={{ height: "40%" }} ref={signalRef} />
    </div>
  );
}
