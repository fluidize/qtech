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

if __name__ == "__main__":
    data = mt.fetch_data("BTC-USDT", 1, "1min", 0, kucoin=True, use_cache=True)
    pivots = pivot_points(data["Open"], data["High"], data["Low"], data["Close"])
    support, resistance = support_resistance_levels(data["Open"], data["High"], data["Low"], data["Close"])
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"], name="Close"))
    # fig.add_trace(go.Scatter(x=pivot_points["Price_Index"][pivot_points["Type"] == "Swing_High"], y=pivot_points["Price"][pivot_points["Type"] == "Swing_High"], mode="markers", name="Swing High", marker=dict(color="red", size=10)))
    # fig.add_trace(go.Scatter(x=pivot_points["Price_Index"][pivot_points["Type"] == "Swing_Low"], y=pivot_points["Price"][pivot_points["Type"] == "Swing_Low"], mode="markers", name="Swing Low", marker=dict(color="blue", size=10)))
    fig.add_trace(go.Scatter(x=data.index, y=support, mode="lines", name="Support", line=dict(color="green", dash="dash")))
    fig.add_trace(go.Scatter(x=data.index, y=resistance, mode="lines", name="Resistance", line=dict(color="red", dash="dash")))
    fig.show()
