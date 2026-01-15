from rich import print
import pandas as pd
import numpy as np

from trading.backtesting.backtesting import VectorizedBacktesting
from trading.backtesting.algorithm_optim import QuantitativeScreener
import trading.backtesting.testing.cstrats as cs
import trading.backtesting.basic_strategies as bs
import trading.technical_analysis as ta

def strategy(data, ma_window):
    signals = pd.Series(0, index=data.index)
    ma = ta.ema(data['Close'], timeperiod=ma_window)
    gap = ((data['Close'] - ma) / data['Close']) * 50
    signals = -gap
    return np.tanh(signals)

#training
qs = QuantitativeScreener(
    symbols=["JUP-USDT"],
    days=365,
    intervals=["15m"],
    age_days=0,
    data_source="binance",
    initial_capital=10000,
    slippage_pct=0.01,
    commission_fixed=0.00,
    cache_expiry_hours=48
)

qs.optimize(
    strategy_func=strategy,
    param_space={"ma_window": (2, 200)},
    metric="Alpha * np.clip((1 + Max_Drawdown), 0, None)",
    n_trials=100,
    direction="maximize",
    save_params=False
)
qs.plot_best_performance(mode="tradingview")
print(qs.get_best_metrics())
best_set = qs.get_best()
print(best_set)