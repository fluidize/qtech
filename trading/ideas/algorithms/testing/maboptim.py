import sys
sys.path.append("")

import trading.backtesting.algorithm_optim as ao
import trading.backtesting.testing.cstrats as cs
import trading.backtesting.testing.basic_strategies as bs
import pandas as pd

mabo = ao.MultiAssetBayesianOptimizer(
    symbols=["XLK", "XLF", "XLY", "XLC", "XLE", "XLI", "XLB", "XLV", "XLF", "XLK", "XLC", "XLE", "XLI", "XLB", "XLV"],
    initial_capitals=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    days=728,
    intervals=["1d"],
    age_days=0,
    data_source="yfinance",
    cache_expiry_hours=9999
)

mabo.optimize(
    strategy_func=bs.supertrend_strategy,
    param_space={
        "supertrend_window": (2, 100),
        "supertrend_multiplier": (1, 10)
    },
    metric="Sharpe_Ratio * (1 + Max_Drawdown)",
    n_trials=1000,
    direction="maximize",
    save_params=False,
    fixed_exceptions=[],
    float_exceptions=[]
)

mabo.plot_best_performance()
print(mabo.get_best_metrics())