from rich import print
import pandas as pd
import numpy as np

from trading.backtesting.backtesting import VectorizedBacktesting
from trading.backtesting.algorithm_optim import QuantitativeScreener
import trading.backtesting.testing.cstrats as cs
import trading.technical_analysis as ta

def strategy(data, vwap_bands156857_timeperiod, vwap_bands156857_stdev_multiplier, LOGIC_vwap_bands156859_Lt_const_156878_constant, LOGIC_vwap_bands156859_Lt_const_156882_constant):
    signals = pd.Series(0, index=data.index)
    vwap_bands156857, vwap_bands156858, vwap_bands156859 = ta.vwap_bands(high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume'], timeperiod=vwap_bands156857_timeperiod, stdev_multiplier=vwap_bands156857_stdev_multiplier)
    LOGIC_vwap_bands156858_Gt_High = vwap_bands156858 > data['High']
    LOGIC_vwap_bands156859_Lt_const_156878 = vwap_bands156859 < LOGIC_vwap_bands156859_Lt_const_156878_constant
    LOGIC_COMPOSITE_156879 = LOGIC_vwap_bands156859_Lt_const_156878 | LOGIC_vwap_bands156858_Gt_High
    LOGIC_vwap_bands156859_Lt_const_156882 = vwap_bands156859 < LOGIC_vwap_bands156859_Lt_const_156882_constant
    signals[LOGIC_COMPOSITE_156879] = 1
    signals[LOGIC_vwap_bands156859_Lt_const_156882] = 0
    return signals
params={'vwap_bands156857_timeperiod': 91, 'vwap_bands156857_stdev_multiplier': 4.102392063082672, 'LOGIC_vwap_bands156859_Lt_const_156878_constant': 50, 'LOGIC_vwap_bands156859_Lt_const_156882_constant': 12}
search_space={'vwap_bands156857_timeperiod': (2, 200), 'vwap_bands156857_stdev_multiplier': (1.0, 5.0), 'LOGIC_vwap_bands156859_Lt_const_156878_constant': (-100, 100), 'LOGIC_vwap_bands156859_Lt_const_156882_constant': (-100, 100)}


#training
qs = QuantitativeScreener(
    symbols=["SOL-USDT"],
    days=800,
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
    param_space=search_space,
    metric="Sharpe_Ratio * np.clip((1 + Max_Drawdown), 0, None)",
    n_trials=100,
    direction="maximize",
    save_params=False
)
qs.plot_best_performance(mode="standard")
print(qs.get_best_metrics())
best_set = qs.get_best()
print(best_set)