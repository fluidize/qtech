import sys
sys.path.append("")

from trading.backtesting.backtesting import VectorizedBacktesting
import trading.technical_analysis as ta

import numpy as np
import pandas as pd

def indicator(data):
    signals = pd.Series(0, index=data.index)
    length = 100
    weight = ta.choppiness_index(data['High'], data['Low'], data['Close'])
    ma = (data['Close'] * (weight)).rolling(window=length).sum() / (1 / weight).rolling(window=length).sum()

    signals[data['Close'] > ma] = 3
    signals[data['Close'] < ma] = 2
    return signals, (ma, False)

vb = VectorizedBacktesting(
    instance_name="default",
    initial_capital=10000,
    slippage_pct=0.00,
    commission_fixed=0.0,
    leverage=1.0
)
vb.fetch_data(symbol="SOL-USDT", days=100, interval="1m", age_days=0, data_source="binance", cache_expiry_hours=0)
vb.run_strategy(indicator, verbose=True)
vb.plot_performance(mode="standard")