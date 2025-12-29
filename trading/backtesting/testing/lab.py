from rich import print
import sys
sys.path.append("")

from trading.backtesting.algorithm_optim import QuantitativeScreener
import trading.backtesting.testing.cstrats as cs
import trading.backtesting.basic_strategies as bs
import trading.technical_analysis as ta

qs = QuantitativeScreener(
    symbols=["SOL-USDT"],
    days=1095,
    intervals=["30m"],
    age_days=0,
    data_source="binance",
    initial_capital=10000,
    slippage_pct=0.02/100,
    commission_fixed=0.00,
    cache_expiry_hours=999
)

qs.optimize(
    strategy_func=cs.trend_strength_strategy,
    param_space=cs.trend_strength_strategy.param_space,
    metric="Alpha * abs(Sharpe_T_Stat)",
    n_trials=100,
    direction="maximize",
    save_params=False
)

qs.plot_best_performance(mode="standard")
print(qs.get_best_metrics())

import matplotlib.pyplot as plt
from optuna.visualization import matplotlib as ov

study = qs.get_best_study()
fig = ov.plot_optimization_history(study)
plt.show()