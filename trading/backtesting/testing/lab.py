import sys
sys.path.append("")

from trading.backtesting.algorithm_optim import QuantitativeScreener
import trading.backtesting.testing.cstrats as cs

qs = QuantitativeScreener(
    symbols=["SOL-USDT"],
    days=365,
    intervals=["1h", "4h"],
    age_days=0,
    data_source="binance",
    initial_capital=10000,
    slippage_pct=0.00,
    commission_fixed=0.00,
    cache_expiry_hours=9999
)

qs.optimize(
    strategy_func=cs.trend_reversal_strategy_v2,
    param_space={
        "supertrend_window": (2, 50),
        "supertrend_multiplier": (1, 5),
        "bbdev": (1, 10),
        "bb_window": (1, 100),
        "bbw_ma_window": (1, 100)
    },
    metric="Sharpe_Ratio * (1 + Max_Drawdown)",
    n_trials=100,
    direction="maximize",
    save_params=True
)

qs.plot_best_performance(mode="standard")
print(qs.get_best_metrics())









