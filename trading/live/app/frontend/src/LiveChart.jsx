import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";
import { createChart, ColorType, CandlestickSeries, LineSeries, createTextWatermark } from "lightweight-charts";

function signalColor(v) {
  const a = 0.5 * Math.abs(v);
  if (v > 0) return `rgba(126, 211, 79, ${a})`;
  if (v < 0) return `rgba(255, 111, 97, ${a})`;
  return "rgba(0, 0, 0, 0)";
}

// fullBarWidth: draws a band that completely fills the bar's horizontal space
// so neighbouring bars have no gaps between them.
function fullBarWidth(xMedia, halfBarSpacingMedia, horizontalPixelRatio) {
  const fullWidthLeftBitmap = Math.round((xMedia - halfBarSpacingMedia) * horizontalPixelRatio);
  const fullWidthRightBitmap = Math.round((xMedia + halfBarSpacingMedia) * horizontalPixelRatio);
  return {
    position: fullWidthLeftBitmap,
    length: fullWidthRightBitmap - fullWidthLeftBitmap,
  };
}

function BackgroundShadeRenderer() {
  this._data = null;
}
BackgroundShadeRenderer.prototype = {
  draw(target) {
    target.useBitmapCoordinateSpace((scope) => this._drawImpl(scope));
  },
  update(data) {
    this._data = data;
  },
  _drawImpl(scope) {
    const data = this._data;
    if (data === null || data.bars.length === 0 || data.visibleRange === null) return;
    const ctx = scope.context;
    const halfWidth = data.barSpacing / 2;
    for (let i = data.visibleRange.from; i < data.visibleRange.to; i++) {
      const bar = data.bars[i];
      if (!bar) continue;
      const fill = fullBarWidth(bar.x, halfWidth, scope.horizontalPixelRatio);
      ctx.fillStyle = signalColor(bar.originalData.value ?? 0);
      ctx.fillRect(fill.position, 0, fill.length, scope.bitmapSize.height);
    }
  },
};

function BackgroundShadeSeries() {
  this._renderer = new BackgroundShadeRenderer();
}
BackgroundShadeSeries.prototype = {
  priceValueBuilder() {
    // NaN prevents this series from affecting the price scale scaling,
    // and from showing a crosshair or price line.
    return [NaN];
  },
  isWhitespace(data) {
    return data.value === undefined;
  },
  renderer() {
    return this._renderer;
  },
  update(data) {
    this._renderer.update(data);
  },
  defaultOptions() {
    return {};
  },
};

export default forwardRef(function LiveChart({ candles, decisions, equity, showPrice, showStrategy, symbol, interval, days, live }, ref) {
  const containerRef = useRef(null);
  const chartRef = useRef(null);
  const priceSeriesRef = useRef(null);
  const tintSeriesRef = useRef(null);
  const equitySeriesRef = useRef(null);
  const watermarkRef = useRef(null);

  useEffect(() => {
    const el = containerRef.current;
    const chart = createChart(el, {
      autoSize: true,
      hoveredSeriesOnTop: false,
      layout: { background: { type: ColorType.Solid, color: "#232834" }, textColor: "#cbccc6", fontFamily: "Departure Mono" },
      grid: { vertLines: { visible: false }, horzLines: { color: "#2a3040" } },
      timeScale: { borderColor: "#3d424d", rightOffset: 5, shiftVisibleRangeOnNewBar: true, minBarSpacing: 0.01 },
    });

    const tintSeries = chart.addCustomSeries(new BackgroundShadeSeries(), {});
    const priceSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#7ed34f",
      downColor: "#ff6f61",
      borderVisible: false,
      wickUpColor: "#7ed34f",
      wickDownColor: "#ff6f61",
    });
    const equitySeries = chart.addSeries(LineSeries, {
      color: "#5ccfe6",
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    });

    tintSeriesRef.current = tintSeries;
    priceSeriesRef.current = priceSeries;
    equitySeriesRef.current = equitySeries;
    chartRef.current = chart;

    const watermark = createTextWatermark(chart.panes()[0], {
      horzAlign: "center",
      vertAlign: "center",
      lines: [
        { text: "", color: "rgba(112, 122, 140, 0.25)", fontSize: 48, fontFamily: "Departure Mono" },
      ],
    });
    watermarkRef.current = watermark;

    return () => {
      chart.remove();
    };
  }, []);

  useEffect(() => {
    const suffix = live ? " · live" : ` · ${days || ""}d`;
    watermarkRef.current?.applyOptions({
      lines: [
        { text: `${symbol} · ${interval}${suffix}`, color: "rgba(112, 122, 140, 0.25)", fontSize: 48, fontFamily: "Departure Mono" },
      ],
    });
  }, [symbol, interval, days, live]);

  const firstTimeRef = useRef(null);

  useImperativeHandle(ref, () => ({
    fit: () => chartRef.current?.timeScale().fitContent(),
  }));

  useEffect(() => {
    if (!candles.length) return;
    const firstTime = candles[0].time;

    priceSeriesRef.current.setData(
      showPrice
        ? candles.map((c) => ({ time: c.time, open: c.open, high: c.high, low: c.low, close: c.close }))
        : []
    );
    equitySeriesRef.current.setData(showStrategy ? equity : []);
    tintSeriesRef.current.setData(
      candles.map((c, i) => ({ time: c.time, value: decisions[i]?.value ?? 0 }))
    );

    if (firstTimeRef.current === null || firstTime !== firstTimeRef.current) {
      if (live) {
        chartRef.current.timeScale().resetTimeScale();
      } else {
        chartRef.current.timeScale().fitContent();
      }
      firstTimeRef.current = firstTime;
    }
  }, [candles, decisions, equity, showPrice, showStrategy]);

  return <div style={{ height: "100%", width: "100%" }} ref={containerRef} />;
});
