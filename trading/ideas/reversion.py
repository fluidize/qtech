import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("trading")
from backtesting.backtesting import VectorizedBacktesting



vb = VectorizedBacktesting(
    instance_name="default",
    initial_capital=10000,
    slippage_pct=0.00,
    commission_fixed=0.0,
    reinvest=False,
    leverage=1.0
)

def reversion_strategy(data):
    signals = pd.Series(2, index=data.index)
    body_max = np.maximum(data["Close"], data["Open"])
    short_signal = ((data["High"] - body_max) / body_max) > 0.25
    signals[short_signal] = 1
    return signals, ((data["High"] - body_max) / body_max)

equity_curves = {}
for symbol in ["CHEK", "ATCH"]:
    vb.fetch_data(
        symbol=symbol,
        days=365,
        interval="1d",
        age_days=0,
        data_source="yfinance",
        use_cache=True
    )
    equity_curve = vb.run_strategy(reversion_strategy, verbose=True)["Percentage_Return"]
    equity_curves[symbol] = equity_curve

sum_curve = sum(equity_curves.values()) / len(equity_curves) #equal weighted avg of all equity curves
plt.plot(sum_curve)
plt.show()
