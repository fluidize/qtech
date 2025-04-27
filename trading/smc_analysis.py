import pandas as pd
import numpy as np
import plotly.graph_objects as go
import model_tools as mt

def pivot_points(open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 10) -> np.ndarray:
    """
    Calculate pivot points from high and low prices.
    """
    pivot_points = pd.DataFrame(columns=["Type", "Price_Index", "Price"])

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
        forward_fill_periods: int - Number of periods to forward fill the levels
        
    Returns:
        support: pd.Series - Support levels
        resistance: pd.Series - Resistance levels
    """
    pivots = pivot_points(open, high, low, close, window)
    
    support = pd.Series(index=open.index, dtype=float)
    resistance = pd.Series(index=open.index, dtype=float)
    
    # Fill support and resistance from pivot points
    for idx, row in pivots.iterrows():
        if row['Type'] == 'Swing_Low':
            support.loc[idx] = row['Price']
        elif row['Type'] == 'Swing_High':
            resistance.loc[idx] = row['Price']
    
    # Forward fill the levels for specified number of periods
    support = support.ffill()
    resistance = resistance.ffill()
    
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
    data = mt.fetch_data("BTC-USDT", 1, "5min", 0, kucoin=True, use_cache=True)
    pivots = pivot_points(data["Open"], data["High"], data["Low"], data["Close"])
    support, resistance = support_resistance_levels(data["Open"], data["High"], data["Low"], data["Close"])
    fvg = fvg(data["Open"], data["High"], data["Low"], data["Close"])

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"], name="Close"))
    print(fvg)
    # # Add FVG boxes
    for idx, row in fvg.iterrows():
        if pd.notna(row["Lower_Range"]):
            if row["Direction"] == "Bullish":
                fig.add_shape(
                    type="rect",
                    x0=idx,
                    x1=idx+5,
                    y0=row["Lower_Range"],
                    y1=row["Upper_Range"],
                    fillcolor="blue",
                    opacity=0.2,
                    line=dict(width=0),
                    layer="below"
                )
            elif row["Direction"] == "Bearish":
                fig.add_shape(
                    type="rect",
                    x0=idx,
                    x1=idx+5,
                    y0=row["Upper_Range"],
                    y1=row["Lower_Range"],
                    fillcolor="red",
                    opacity=0.2,
                    line=dict(width=0),
                    layer="below"
                )
    fig.show()
