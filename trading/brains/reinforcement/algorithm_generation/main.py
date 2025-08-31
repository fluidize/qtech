import sys
sys.path.append("")

from trading.backtesting import backtesting as bt
from skeleton import Skeleton

backtest = bt.VectorizedBacktesting(
    instance_name="default",
    initial_capital=10000,
    slippage_pct=0.001,
    commission_fixed=0.0,
    reinvest=False,
    leverage=1.0
)

backtest.fetch_data(
    symbol="SOL-USDT",
    chunks=20,
    interval="5m",
    age_days=0,
    data_source="binance",
    use_cache=True
)

strategy = Skeleton("SMA", [0, 1, 2, 3])
backtest.run_strategy(strategy, verbose=True)
print(backtest.get_performance_metrics())
backtest.plot_performance(extended=False)