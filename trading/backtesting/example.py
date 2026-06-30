import trading.backtesting.backtesting as bt
import trading.technical_analysis as ta
import pandas as pd

vb = bt.VectorizedBacktest(
    instance_name="Example",
    initial_capital=10000.0,
    slippage_pct=0.001,
    commission_fixed=0.0,
    leverage=1.0,
)

vb.fetch_data(
    symbols=["SOL-USDT"],
    days=365,
    interval="15m",
    age_days=0,
    data_source="binance",
    cache_expiry_hours=-1,
    retry_limit=3,
    verbose=True,
    proxies={},
)


def strategy(data):
    signals = pd.Series(0, index=data.index)
    close = data["Close"]

    ema50 = ta.ema(close, timeperiod=50)
    ema200 = ta.ema(close, timeperiod=200)

    signals[ema50 > ema200] = 1
    signals[ema50 < ema200] = 0

    return signals


vb.run_strategy(strategy, verbose=True)
vb.plot_performance(mode="basic")
