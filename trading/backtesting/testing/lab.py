from rich import print
import pandas as pd

import sys
sys.path.append("")
from trading.backtesting.backtesting import VectorizedBacktesting
from trading.backtesting.algorithm_optim import QuantitativeScreener
import trading.backtesting.testing.cstrats as cs
import trading.backtesting.basic_strategies as bs
import trading.technical_analysis as ta

def strategy(data, macd_38492_fastperiod, macd_38492_slowperiod, macd_38492_signalperiod, mass_index_38495_timeperiod, mass_index_38495_ema_period):
    signals = pd.Series(0, index=data.index)
    macd_38492, macd_38493, macd_38494 = ta.macd(series=data['Close'], fastperiod=macd_38492_fastperiod, slowperiod=macd_38492_slowperiod, signalperiod=macd_38492_signalperiod)
    mass_index_38495 = ta.mass_index(high=data['High'], low=data['Low'], timeperiod=mass_index_38495_timeperiod, ema_period=mass_index_38495_ema_period)
    LOGIC_macd_38494_Lt_macd_38493 = macd_38494 < macd_38493
    LOGIC_mass_index_38495_Gt_High = mass_index_38495 > data['High']
    signals[LOGIC_mass_index_38495_Gt_High] = 3
    signals[LOGIC_macd_38494_Lt_macd_38493] = 2
    return signals

#training
qs = QuantitativeScreener(
    symbols=["SOL-USDT"],
    days=1095,
    intervals=["30m"],
    age_days=0,
    data_source="binance",
    initial_capital=10000,
    slippage_pct=0.01,
    commission_fixed=0.00,
    cache_expiry_hours=48
)

qs.optimize(
    strategy_func=strategy,
    param_space={"macd_38492_fastperiod": (2, 200), "macd_38492_slowperiod": (2, 200), "macd_38492_signalperiod": (2, 200), "mass_index_38495_timeperiod": (2, 200), "mass_index_38495_ema_period": (2, 200)},
    metric="Alpha * np.clip(1 + Max_Drawdown, 0, None)",
    n_trials=100,
    direction="maximize",
    save_params=False
)
qs.plot_best_performance(mode="standard")
best_set = qs.get_best()
print(best_set)