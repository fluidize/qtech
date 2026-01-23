from rich import print
import pandas as pd
import numpy as np

from trading.backtesting.backtesting import VectorizedBacktesting
from trading.backtesting.algorithm_optim import QuantitativeScreener
import trading.backtesting.testing.cstrats as cs
import trading.technical_analysis as ta

def strategy(data, ema27760_timeperiod, vwma27761_timeperiod, fisher_transform27762_timeperiod, LOGIC_vwma27761_Lt_const_27764_constant):
    signals = pd.Series(0, index=data.index)
    ema27760 = ta.ema(series=data['Close'], timeperiod=ema27760_timeperiod)
    vwma27761 = ta.vwma(series=data['Close'], volume=data['Volume'], timeperiod=vwma27761_timeperiod)
    fisher_transform27762 = ta.fisher_transform(series=data['Close'], timeperiod=fisher_transform27762_timeperiod)
    LOGIC_fisher_transform27762_Gt_Close = fisher_transform27762 > data['Close']
    LOGIC_vwma27761_Lt_const_27764 = vwma27761 < LOGIC_vwma27761_Lt_const_27764_constant
    LOGIC_ema27760_Gt_vwma27761 = ema27760 > vwma27761
    LOGIC_COMPOSITE_27765 = LOGIC_vwma27761_Lt_const_27764 & LOGIC_fisher_transform27762_Gt_Close
    LOGIC_COMPOSITE_27766 = LOGIC_ema27760_Gt_vwma27761 | LOGIC_COMPOSITE_27765
    LOGIC_COMPOSITE_27768 = LOGIC_ema27760_Gt_vwma27761 & LOGIC_vwma27761_Lt_const_27764
    signals[LOGIC_COMPOSITE_27766] = 1
    signals[LOGIC_COMPOSITE_27768] = 0
    return signals
params={'ema27760_timeperiod': 144, 'vwma27761_timeperiod': 150, 'fisher_transform27762_timeperiod': 153, 'LOGIC_vwma27761_Lt_const_27764_constant': 56}
search_space={'ema27760_timeperiod': (2, 200), 'vwma27761_timeperiod': (2, 200), 'fisher_transform27762_timeperiod': (2, 200), 'LOGIC_vwma27761_Lt_const_27764_constant': (-100, 100)}

#training
qs = QuantitativeScreener(
    symbols=["SOL-USDT"],
    days=1000,
    intervals=["30m"],
    age_days=100,
    data_source="binance",
    initial_capital=10000,
    slippage_pct=0.05,
    commission_fixed=0.00,
    cache_expiry_hours=48
)

qs.optimize(
    strategy_func=strategy,
    param_space=search_space,
    metric="Sharpe_Ratio",
    n_trials=100,
    direction="maximize",
    save_params=False
)
# qs.plot_best_performance(mode="standard")
best_set = qs.get_best()

vb = VectorizedBacktesting(
    instance_name="default",
    initial_capital=10000,
    slippage_pct=0.05,
    commission_fixed=0.00
)
vb.fetch_data(symbol="SOL-USDT", days=100, interval="30m", age_days=0, data_source="binance", cache_expiry_hours=0)

vb.run_strategy(strategy, verbose=True, **best_set["params"])
vb.plot_performance(mode="standard")