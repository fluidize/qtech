import numpy as np
import pandas as pd

import sys
sys.path.append("")

import trading.backtesting.basic_strategies as bs
import trading.technical_analysis as ta
import trading.smc_analysis as smc

#cryptocurrency specific strategies

def trend_reversal_strategy(
        data: pd.DataFrame,
        supertrend_window: int = 10,
        supertrend_multiplier: float = 2,
        vol_ma_window: int = 13,
    ) -> pd.Series:
    signals = pd.Series(0, index=data.index)
    
    supertrend, supertrend_line = ta.supertrend(
        data['High'], 
        data['Low'], 
        data['Close'], 
        period=supertrend_window, 
        multiplier=supertrend_multiplier
    )

    # upper, middle, lower = ta.bbands(data['Close'], timeperiod=bb_window, devup=bb_dev, devdn=bb_dev)
    # bbw = (upper - lower) / middle #utilizing BBW as a volatility proxy
    # bbw_ma = ta.sma(bbw, timeperiod=bbw_ma_window)

    # volatility_contraction = (bbw < bbw_ma) & (bbw.shift(1) > bbw_ma.shift(1))
    hlc3 = (data["High"] + data["Low"] + data["Close"]) / 3

    volatility = ((hlc3 - ta.sma(hlc3, timeperiod=vol_ma_window)) / hlc3) * 667

    bullish_volatility_contraction = volatility > 0.1 #volatility spike upwards
    bearish_volatility_contraction = volatility < 0.1 #volatility spike downwards
    buy_conditions = (supertrend == -1) & (bullish_volatility_contraction) #buy into bearish trend
    sell_conditions = (supertrend == 1) & (bearish_volatility_contraction) #sell into bullish trend

    signals[buy_conditions] = 3
    signals[sell_conditions] = 2
    
    return signals, volatility

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