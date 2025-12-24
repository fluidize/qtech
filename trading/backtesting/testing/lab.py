import pandas as pd
import sys
sys.path.append("")

from trading.backtesting.algorithm_optim import QuantitativeScreener
import trading.backtesting.testing.cstrats as cs
import trading.backtesting.basic_strategies as bs
import trading.technical_analysis as ta

qs = QuantitativeScreener(
    symbols=["SOL-USDT"],
    days=365,
    intervals=["1h"],
    age_days=0,
    data_source="binance",
    initial_capital=10000,
    slippage_pct=0.02/100,
    commission_fixed=0.00,
    cache_expiry_hours=999
)

import numpy as np

def strategy(data, process_noise=0.0001, measurement_noise=1):
    signals = pd.Series(0, index=data.index)
    
    kalman = ta.kalman_filter(data["Close"], process_noise=process_noise, measurement_noise=measurement_noise)
    signals[kalman > kalman.shift(1)] = 3
    signals[kalman < kalman.shift(1)] = 2

    return signals, (kalman, True)

qs.optimize(
    strategy_func=cs.trend_strength_strategy,
    param_space=cs.trend_strength_strategy.param_space,
    metric="Sharpe_Ratio * np.clip((1 + Max_Drawdown), 0, None)",
    n_trials=1000,
    direction="maximize",
    save_params=False
)

qs.plot_best_performance(mode="standard")

import matplotlib.pyplot as plt
from optuna.visualization import matplotlib as ov

study = qs.get_best_study()
fig = ov.plot_optimization_history(study)
plt.show()