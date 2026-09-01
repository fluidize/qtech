import pandas as pd

import trading.technical_analysis as ta


def ema_cross(
    data: pd.DataFrame, fast_period: int = 20, slow_period: int = 50
) -> pd.Series:
    close = data["Close"]
    fast = ta.ema(close, timeperiod=fast_period)
    slow = ta.ema(close, timeperiod=slow_period)

    signals = pd.Series(0, index=data.index, dtype=float)
    signals[fast > slow] = 1.0
    signals[fast < slow] = -1.0
    return signals


def mean_reversion(
    data: pd.DataFrame, window: int = 20, z_threshold: float = 1.5
) -> pd.Series:
    close = data["Close"]
    mean = close.rolling(window).mean()
    std = close.rolling(window).std()
    z = (close - mean) / std

    signals = pd.Series(0, index=data.index, dtype=float)
    signals[z > z_threshold] = -1.0
    signals[z < -z_threshold] = 1.0
    return signals


def supertrend(
    data: pd.DataFrame, period: int = 10, multiplier: int = 3
) -> pd.Series:
    direction = ta.supertrend_direction(
        data["High"], data["Low"], data["Close"], period=period, multiplier=multiplier
    )
    return direction.fillna(0.0)
