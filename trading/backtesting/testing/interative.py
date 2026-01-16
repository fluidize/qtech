from rich import print
import pandas as pd
import numpy as np

from trading.backtesting.backtesting import VectorizedBacktesting
from trading.backtesting.algorithm_optim import QuantitativeScreener
import trading.technical_analysis as ta

from trading.brains.probability.models import EmpiricalDistribution

def strategy(data):
    upside_probabilities = pd.Series(0.0, index=data.index, dtype=float)

    for i in range(1, len(data)):
        available_data = data.iloc[:i]
        X = available_data['Close']
        if len(X) < 2:
            continue
        counts, bin_edges = np.histogram(X['Close'], bins=100)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        current_X = X.iloc[i]
        total_counts = sum(counts)

        upside_probability = sum(counts[bin_centers > current_X]) / total_counts

        upside_probabilities.iloc[i] = upside_probability
    signals = upside_probabilities
    return signals, (upside_probabilities, False)

vb = VectorizedBacktesting(
    instance_name="default",
    initial_capital=10000,
    slippage_pct=0.001,
    commission_fixed=0.00,
    leverage=1.0
)
vb.fetch_data(symbol="DOGE-USDT", days=1800, interval="1d", age_days=0, data_source="binance", cache_expiry_hours=48)
vb.run_strategy(strategy, verbose=True)
vb.plot_performance(mode="standard")
