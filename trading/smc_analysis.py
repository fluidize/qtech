import pandas as pd
import numpy as np
import plotly.graph_objects as go
import model_tools as mt

def pivot_points(open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 10) -> np.ndarray:
    """
    Calculate pivot points from high and low prices.
    """
    pivot_points = pd.DataFrame(columns=["Type", "Price_Index", "Price"], index=open.index)

    open = open.values
    high = high.values
    low = low.values
    close = close.values
    for i in range(window, len(open) - window):
        _high = high[i]
        _low = low[i]

        is_swing_high = all(_high >= open[i - j] and _high >= open[i + j] for j in range(1, window + 1))
        is_swing_low = all(_low <= open[i - j] and _low <= open[i + j] for j in range(1, window + 1))

        if is_swing_high:
            pivot_points.loc[i] = ('Swing_High', i, _high)
        elif is_swing_low:
            pivot_points.loc[i] = ('Swing_Low', i, _low)
        else:
            pivot_points.loc[i] = ('None', i, np.nan)

    return pivot_points

def support_resistance_levels(open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 10) -> tuple[pd.Series, pd.Series]:
    """
    Calculate support and resistance levels from pivot points and forward fill them.
    
    Args:
        open: pd.Series - Open prices
        high: pd.Series - High prices
        low: pd.Series - Low prices
        close: pd.Series - Close prices
        window: int - Window size for finding pivot points
        
    Returns:
        support: pd.Series - Support levels
        resistance: pd.Series - Resistance levels
    """

    support = low.rolling(window=window).min()
    resistance = high.rolling(window=window).max()
    
    return support, resistance

def fvg(open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, pct_threshold: float = 0.005) -> pd.DataFrame:
    """
    Calculate the Fair Value Gap from open, high, low, and close prices.
    """
    pct_threshold = pct_threshold / 100
    fvg = pd.DataFrame(index=open.index, columns=["Lower_Range", "Upper_Range", "Direction"])
    for i in range(2, len(high)):  # need at least 3 candles
        # Bullish FVG: Candle3_low > Candle1_high
        if low[i] > high[i-2]:
            gap_size = low[i] - high[i-2]
            if gap_size / high[i-2] >= pct_threshold:
                fvg.loc[i, "Lower_Range"] = high[i-2]
                fvg.loc[i, "Upper_Range"] = low[i]
                fvg.loc[i, "Direction"] = "Bullish"

        # Bearish FVG: Candle3_high < Candle1_low
        if high[i] < low[i-2]:
            gap_size = low[i-2] - high[i]
            if gap_size / low[i-2] >= pct_threshold:
                fvg.loc[i, "Lower_Range"] = high[i]
                fvg.loc[i, "Upper_Range"] = low[i-2]
                fvg.loc[i, "Direction"] = "Bearish"
    return fvg

if __name__ == "__main__":
    data = mt.fetch_data("BTC-USDT", 1, "5m", 0, data_source="kucoin", use_cache=True)
    pivots = pivot_points(data["Open"], data["High"], data["Low"], data["Close"])
    support, resistance = support_resistance_levels(data["Open"], data["High"], data["Low"], data["Close"])
    fvg = fvg(data["Open"], data["High"], data["Low"], data["Close"])

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"], name="Close"))
    
    # Add support and resistance lines
    fig.add_trace(go.Scatter(x=data.index, y=support, name="Support", line=dict(color="green", width=1)))
    fig.add_trace(go.Scatter(x=data.index, y=resistance, name="Resistance", line=dict(color="red", width=1)))

    fig.show()