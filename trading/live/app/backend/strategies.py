"""Strategy functions that follow the backtesting contract.

A strategy is a pure function of the form::

    def strategy(data: pd.DataFrame, **params) -> pd.Series

where ``data`` has OHLCV columns and it returns a signal series with values
interpreted as the algorithm's decision (typically in ``{-1, 0, 1}``).

These are the same functions used by ``trading/backtesting``.
"""

import numpy as np
import pandas as pd

import trading.technical_analysis as ta


def ema_cross(data: pd.DataFrame, fast_period: int = 20, slow_period: int = 50) -> pd.Series:
    """EMA crossover. +1 when fast > slow (long), -1 when fast < slow (short), else 0."""
    close = data["Close"]
    fast = ta.ema(close, timeperiod=fast_period)
    slow = ta.ema(close, timeperiod=slow_period)

    signals = pd.Series(0, index=data.index, dtype=float)
    signals[fast > slow] = 1.0
    signals[fast < slow] = -1.0
    return signals


def mean_reversion(data: pd.DataFrame, window: int = 20, z_threshold: float = 1.5) -> pd.Series:
    """Z-score mean reversion. Short when price is far above its rolling mean, long when far below."""
    close = data["Close"]
    mean = close.rolling(window).mean()
    std = close.rolling(window).std()
    z = (close - mean) / std

    signals = pd.Series(0, index=data.index, dtype=float)
    signals[z > z_threshold] = -1.0
    signals[z < -z_threshold] = 1.0
    return signals


STRATEGIES = {
    "ema_cross": {
        "func": ema_cross,
        "params": {"fast_period": 20, "slow_period": 50},
    },
    "mean_reversion": {
        "func": mean_reversion,
        "params": {"window": 20, "z_threshold": 1.5},
    },
}


def normalize_signals(signals: pd.Series) -> pd.Series:
    """Clip a continuous signal series into the discrete decision set {-1, 0, 1}."""
    return pd.Series(np.sign(np.clip(signals, -1, 1)), index=signals.index).fillna(0)
