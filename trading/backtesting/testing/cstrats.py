import numpy as np
import pandas as pd

import sys
sys.path.append("")

import trading.backtesting.basic_strategies as bs
import trading.technical_analysis as ta
import trading.smc_analysis as smc

#cryptocurrency specific strategies

def trend_reversal_strategy_v1(
        data: pd.DataFrame,
        supertrend_window: int = 37,
        supertrend_multiplier: float = 2,
        ma_window: int = 35,
    ) -> pd.Series:
    signals = pd.Series(0, index=data.index)
    
    supertrend, supertrend_line = ta.supertrend(
        data['High'], 
        data['Low'], 
        data['Close'], 
        period=supertrend_window, 
        multiplier=supertrend_multiplier
    )

    hlc3 = (data["High"] + data["Low"] + data["Close"]) / 3

    price_ma_gap = ((hlc3 - ta.sma(hlc3, timeperiod=ma_window)) / hlc3)

    bullish_trend_reversal = price_ma_gap > 0 #short term trend reversal up
    bearish_trend_reversal = price_ma_gap < 0 #short term trend reversal down
    buy_conditions = (supertrend == -1) & (bullish_trend_reversal) #buy into bearish trend
    sell_conditions = (supertrend == 1) & (bearish_trend_reversal) #sell into bullish trend

    signals[buy_conditions] = 3
    signals[sell_conditions] = 2
    
    return signals

def trend_reversal_strategy_v2(
        data: pd.DataFrame,
        supertrend_window: int = 6,
        supertrend_multiplier: float = 1,
        bbdev: int = 7,
        bb_window: int = 45,
        bbw_ma_window: int = 38,
    ) -> pd.Series:
    #tuned for solusdt 15m

    signals = pd.Series(0, index=data.index)
    
    supertrend, supertrend_line = ta.supertrend(
        data['High'], 
        data['Low'], 
        data['Close'], 
        period=supertrend_window, 
        multiplier=supertrend_multiplier
    )

    upper, middle, lower = ta.bbands(data['Close'], timeperiod=bb_window, devup=bbdev, devdn=bbdev)
    bbw = (upper - lower) / middle #utilizing BBW as a volatility proxy
    bbw_ma = ta.sma(bbw, timeperiod=bbw_ma_window)

    volatility_contraction = (bbw > bbw_ma) & (bbw < bbw.shift(1))

    buy_conditions = (supertrend == -1) & (volatility_contraction) #buy into bearish trend
    sell_conditions = (supertrend == 1) & (volatility_contraction) #sell into bullish trend

    signals[buy_conditions] = 3
    signals[sell_conditions] = 2
    
    return signals

def directional_atr_strategy(data: pd.DataFrame,
        supertrend_window: int = 14,
        supertrend_multiplier: float = 1,
        threshold: float = 0.03,
    ) -> pd.Series:
    signals = pd.Series(0, index=data.index)
    supertrend, supertrend_line = ta.supertrend(
        data['High'], 
        data['Low'], 
        data['Close'], 
        period=supertrend_window, 
        multiplier=supertrend_multiplier
    )

    hlc3 = (data["High"] + data["Low"] + data["Close"]) / 3
    atr_gap = (hlc3 - supertrend_line) / hlc3

    overbought_conditions =  (atr_gap > threshold) #buy into bearish trend
    oversold_conditions = (atr_gap < -threshold) #sell into bullish trend

    signals[oversold_conditions] = 3
    signals[overbought_conditions] = 2

    return signals

def sott_strategy(data: pd.DataFrame, threshold: float = 0.0) -> None:
    signals = pd.Series(0, index=data.index)

    def efficient_work(close: pd.Series, length: int = 1) -> pd.Series:
        """Efficient Work: net move vs path length, range [-1, 1]."""
        change = close - close.shift(length)
        path = close.diff().abs().rolling(length).sum()
        ew = change / path
        return ew.fillna(0)

    def sott(df: pd.DataFrame) -> pd.Series:
        """
        Signs of the Times (SOTT) indicator.
        df must contain: ['open', 'high', 'low', 'close', 'volume'].
        Returns a Series of SOTT values in [-1, 1].
        """
        o, h, l, c, v = df['Open'], df['High'], df['Low'], df['Close'], df['Volume']

        bar_up = c > o
        bar_dn = c < o
        body   = (c - o).abs()
        wicks  = h - l - body
        rising_volume = v.diff(1) > 0

        # up weights
        up = (
            (bar_up.astype(int)) +
            (o > o.shift(1)).astype(int) +
            (h > h.shift(1)).astype(int) +
            (l > l.shift(1)).astype(int) +
            (c > c.shift(1)).astype(int) +
            ((bar_up & (body > body.shift(1))).astype(int)) +
            ((bar_up & (body > wicks)).astype(int)) +
            ((o > c.shift(1)) * 2).astype(int) +
            (np.maximum(0, efficient_work(c, 1)) * 2) +
            ((bar_up & rising_volume) * 2).astype(int)
        )

        # down weights
        dn = (
            (bar_dn.astype(int)) +
            (o < o.shift(1)).astype(int) +
            (h < h.shift(1)).astype(int) +
            (l < l.shift(1)).astype(int) +
            (c < c.shift(1)).astype(int) +
            ((bar_dn & (body > body.shift(1))).astype(int)) +
            ((bar_dn & (body > wicks)).astype(int)) +
            ((o < c.shift(1)) * 2).astype(int) +
            (np.abs(np.minimum(0, efficient_work(c, 1))) * 2) +
            ((bar_dn & rising_volume) * 2).astype(int)
        )

        # final sott line
        result = (up - dn) / 13.0

        return result

    sott = sott(data)
    sott = ta.sma(sott, timeperiod=20)
    signals[sott >= threshold] = 2
    signals[sott <= -threshold] = 3

    return signals

def wavetrend_strategy(data: pd.DataFrame, channel_length: int = 10, average_length: int = 21) -> pd.Series:
    signals = pd.Series(0, index=data.index)

    def wavetrend(data: pd.DataFrame, channel_length: int = 10, average_length: int = 21) -> pd.Series:
        n1 = channel_length
        n2 = average_length

        hlc3 = (data['High'] + data['Low'] + data['Close']) / 3

        ap = hlc3
        esa = ta.ema(ap, n1)
        d = ta.ema(abs(ap - esa), n1)
        ci = (ap - esa) / (0.015 * d)
        tci = ta.ema(ci, n2)

        wt1 = tci
        wt2 = ta.sma(wt1, 4)
        
        return wt1, wt2
    
    wt1, wt2 = wavetrend(data, channel_length, average_length) 

    buy_conditions = (wt1 > wt2)
    sell_conditions = (wt1 < wt2)
    signals[buy_conditions] = 3
    signals[sell_conditions] = 2

    return signals

def trend_oscillation_strategy(data: pd.DataFrame, fast_ma_period: int = 20, slow_ma_period: int = 75) -> pd.Series:
    signals = pd.Series(0, index=data.index)

    slow_ma = ta.sma(data['Close'], timeperiod=slow_ma_period)
    fast_ma = ta.ema(data['Close'], timeperiod=fast_ma_period)

    line = (fast_ma - slow_ma) / slow_ma
    smooth_line = ta.sma(line, timeperiod=4)
    
    long_condition = smooth_line > smooth_line.shift(1)
    short_condition = smooth_line < smooth_line.shift(1)

    signals[long_condition] = 3
    signals[short_condition] = 2

    return signals