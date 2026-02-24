from rich import print
import pandas as pd
import numpy as np

from trading.backtesting.backtesting import VectorizedBacktesting
import trading.technical_analysis as ta

from trading.brains.probability.models import EmpiricalDistribution

def strategy(data):
    upside_probabilities = pd.Series(0.0, index=data.index, dtype=float)

    for i in range(1, len(data)):
        available_data = data.iloc[:i]
        X = available_data['Close']
        if len(X) < 2:
            continue
        dist = EmpiricalDistribution(X)
        upside_probabilities.iloc[i] = 1 - dist.single_cdf(X.iloc[-1])
    signals = upside_probabilities
    return signals, (upside_probabilities, False)

vb = VectorizedBacktesting(
    instance_name="default",
    initial_capital=10000,
    slippage_pct=0.001,
    commission_fixed=0.00,
    leverage=1.0
)
vb.fetch_data(symbol="DOGE-USDT", days=1800, interval="1d", age_days=0, data_source="binance", cache_expiry_hours=9899)
vb.run_strategy(strategy, verbose=True)
vb.plot_performance(mode="standard")
