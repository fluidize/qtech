import sys
sys.path.append("")

from trading.backtesting.backtesting import VectorizedBacktesting
from trading.backtesting.algorithm_optim import QuantitativeScreener
import trading.backtesting.testing.cstrats as cs

qs = QuantitativeScreener(
    symbols=["SOL-USDT"],
    days=365,
    intervals=["15m", "30m", "1h", "4h"],
    age_days=0,
    data_source="binance",
    initial_capital=10000,
    slippage_pct=0.00,
    commission_fixed=0.00,
    cache_expiry_hours=9999
)

qs.optimize(
    strategy_func=cs.trend_oscillation_strategy,
    param_space={
        "slow_ma_period": (2, 50),
        "fast_ma_period": (2, 50)
    },
    metric="Sharpe_Ratio * Sortino_Ratio",
    n_trials=100,
    direction="maximize",
    save_params=True
)

qs.plot_best_performance(mode="standard")
print(qs.get_best_metrics())









