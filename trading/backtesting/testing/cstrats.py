import numpy as np
import pandas as pd

import sys
sys.path.append("")

import trading.backtesting.basic_strategies as bs
import trading.technical_analysis as ta
import trading.smc_analysis as smc

#cryptocurrency specific strategies

def param_space(**param_space):
    def decorator(func):
        func.param_space = param_space
        return func
    return decorator

@param_space(
    sd_window=(2, 100),
    sd_ma_window=(2, 100),
    fast_period=(2, 100),
    slow_period=(2, 100),
    signal_period=(2, 100)
)
def trend_reversal_strategy_v2(
        data: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        sd_window: int = 45,
        sd_ma_window: int = 3,
    ) -> pd.Series:
    signals = pd.Series(0, index=data.index)
    
    macd, signal, histogram = ta.macd(data['Close'], fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period)
    sd = ta.stddev(data['Close'], timeperiod=sd_window)
    sd_ma = ta.sma(sd, timeperiod=sd_ma_window)

    volatility_contraction = (sd > sd_ma) & (sd < sd.shift(1))

    buy_conditions = (macd < 0) & (volatility_contraction) #buy into bearish trend
    sell_conditions = (macd > 0) & (volatility_contraction) #sell into bullish trend

    signals[buy_conditions] = 3
    signals[sell_conditions] = 2
    
    return signals

@param_space(
    adx_sma_period=(2, 100)
)
def trend_strength_strategy(data: pd.DataFrame, adx_period: int = 14, adx_sma_period: int = 3, supertrend_period: int = 5, supertrend_multiplier: float = 2) -> pd.Series:
    signals = pd.Series(2, index=data.index)
    signals = pd.Series(0, index=data.index)
    adx, plus_di, minus_di = ta.adx(data['High'], data['Low'], data['Close'], timeperiod=adx_period)
    supertrend, supertrend_line = ta.supertrend(data['High'], data['Low'], data['Close'], period=supertrend_period, multiplier=supertrend_multiplier)
    buy_conditions =(adx > ta.sma(adx, timeperiod=adx_sma_period)) & (supertrend == 1)
    sell_conditions =(adx < ta.sma(adx, timeperiod=adx_sma_period)) & (supertrend == -1) 
    signals[buy_conditions] = 2
    signals[sell_conditions] = 3
    return signals

@param_space(
    vwma_period=(2, 200),
)
def vwma_trend_strategy(data: pd.DataFrame, vwma_period: int = 10) -> pd.Series:
    signals = pd.Series(0, index=data.index)

    vwma = (
        (data['Close'] * data['Volume']).rolling(vwma_period).sum() /
        data['Volume'].rolling(vwma_period).sum()
    )

    dist = (data['Close'] - vwma).abs() / data['Close']
    dist_ok = dist > dist.rolling(25).median()

    buy = (
        (data['Close'] > vwma) &
        dist_ok
    )

    sell = (
        (data['Close'] < vwma) &
        dist_ok
    )

    signals[buy] = 3
    signals[sell] = 2
    return signals