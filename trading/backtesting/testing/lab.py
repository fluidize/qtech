from rich import print
import pandas as pd

import sys
sys.path.append("")
from trading.backtesting.backtesting import VectorizedBacktesting
from trading.backtesting.algorithm_optim import QuantitativeScreener
import trading.backtesting.testing.cstrats as cs
import trading.backtesting.basic_strategies as bs
import trading.technical_analysis as ta

def strategy(data, fast_ma_window, slow_ma_window):
    signals = pd.Series(0, index=data.index)
    fast_ma = ta.sma(data['Close'], timeperiod=fast_ma_window)
    slow_ma = ta.sma(data['Close'], timeperiod=slow_ma_window)
    signals[fast_ma > slow_ma] = 1
    signals[fast_ma < slow_ma] = 0
    return signals

#training
qs = QuantitativeScreener(
    symbols=["SOL-USDT"],
    days=365,
    intervals=["30m"],
    age_days=0,
    data_source="binance",
    initial_capital=10000,
    slippage_pct=0.03,
    commission_fixed=0.00,
    cache_expiry_hours=48
)

qs.optimize(
    strategy_func=strategy,
    param_space={"fast_ma_window": (2, 200), "slow_ma_window": (2, 200)},
    metric="Alpha * np.clip(1 + Max_Drawdown, 0, None)",
    n_trials=100,
    direction="maximize",
    save_params=False
)
qs.plot_best_performance(mode="standard")
print(qs.get_best_metrics())
best_set = qs.get_best()
print(best_set)