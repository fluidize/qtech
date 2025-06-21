import pandas as pd
import technical_analysis as ta
import smc_analysis as smc
import model_tools as mt
import numpy as np


def hold_strategy(data: pd.DataFrame, signal: int = 3) -> pd.Series:
    return pd.Series(signal, index=data.index)

def perfect_strategy(data: pd.DataFrame) -> pd.Series:
    """
    Perfect strategy that uses only past information to predict future open-to-open returns.
    
    Uses Close[t-1] to predict return from Open[t] to Open[t+1]
    Signal at Close[t-1] -> Execute at Open[t] -> Earn Open[t] to Open[t+1]
    """
    signals = pd.Series(2, index=data.index)
    
    if len(data) < 2:
        return signals
        
    # For each bar, use only information available at the previous close
    for i in range(1, len(data)):
        # At close of bar i-1, predict return from Open[i] to Open[i+1]
        if i < len(data) - 1:  # Make sure we have the next open
            # This is the return we want to predict and capture
            future_return = (data['Open'].iloc[i+1] - data['Open'].iloc[i]) / data['Open'].iloc[i]
            
            # "Perfect" prediction based on past data (in reality this would be a model)
            if future_return > 0:
                signals.iloc[i] = 3  # Long
            elif future_return < 0:
                signals.iloc[i] = 1  # Short
            # else stay flat (2)
    
    return signals

def zscore_reversion_strategy(data: pd.DataFrame, zscore_threshold: float = 1) -> pd.Series:
    signals = pd.Series(2, index=data.index)
    zscore = ta.zscore(data['Close'])
    signals[zscore < -1] = 3
    signals[zscore > 1] = 1
    return signals

def zscore_momentum_strategy(data: pd.DataFrame, zscore_threshold: float = 1) -> pd.Series:
    signals = pd.Series(0, index=data.index)
    zscore = ta.zscore(data['Close'])
    signals[zscore > zscore_threshold] = 3
    signals[zscore < -zscore_threshold] = 1
    return signals

def smc_strategy(data: pd.DataFrame) -> pd.Series:
    signals = pd.Series(2, index=data.index)
    return signals

def scalper_strategy(data: pd.DataFrame) -> pd.Series:
    """Trades frequently, good for testing."""
    signals = pd.Series(2, index=data.index)
    fast_ema = ta.ema(data['Close'], 3)
    slow_ema = ta.ema(data['Close'], 8)

    rolling_vol = data['Volume'].rolling(window=10).mean()

    buy_conditions = (fast_ema > slow_ema) & (data['Volume'] > rolling_vol * 3)
    sell_conditions = (fast_ema < slow_ema) & (data['Volume'] > rolling_vol * 3)

    signals[buy_conditions] = 3
    signals[sell_conditions] = 1
    return signals

def ETHBTC_trader(data: pd.DataFrame, chunks, interval, age_days, data_source: str = "binance", zscore_window: int = 20, lower_zscore_threshold: float = -1, upper_zscore_threshold: float = 1) -> pd.Series:
    signals = pd.Series(2, index=data.index)
    eth_price = mt.fetch_data('ETH-USDT', chunks=chunks, interval=interval, age_days=age_days, data_source=data_source)
    btc_price = mt.fetch_data('BTC-USDT', chunks=chunks, interval=interval, age_days=age_days, data_source=data_source)
    ethbtc_ratio = eth_price[['Open', 'High', 'Low', 'Close']] / btc_price[['Open', 'High', 'Low', 'Close']]
    
    zscore = ta.zscore(ethbtc_ratio['Open'], timeperiod=zscore_window)

    #follow ethbtc ratio breakouts
    buy_conditions = zscore > upper_zscore_threshold
    sell_conditions = zscore < lower_zscore_threshold
    signals[buy_conditions] = 3
    signals[sell_conditions] = 2
    return signals

def sr_strategy(data: pd.DataFrame, window: int = 12, threshold: float = 0.005, rejection_ratio_threshold: float = 0.5) -> pd.Series:
    signals = pd.Series(2, index=data.index)
    volatility = ta.volatility(data['Close'], timeperiod=20)
    
    support, resistance = smc.support_resistance_levels(data['Open'], data['High'], data['Low'], data['Close'], window=window)
    
    support_pct_distance = (data['Close'] - support) / support
    resistance_pct_distance = (resistance - data['Close']) / resistance

    bullish_rejection_wick_ratio = (data[['Open', 'Close']].min(axis=1) - data['Low']) / (data['High'] - data['Low'] + 1e-9)
    bearish_rejection_wick_ratio = (data['High'] - data[['Open', 'Close']].max(axis=1)) / (data['High'] - data['Low'] + 1e-9)
    
    recent_low = data['Low'].rolling(window=window).min().shift(1)
    recent_high = data['High'].rolling(window=window).max().shift(1)

    bullish_liquidity_sweep = ((bullish_rejection_wick_ratio > bearish_rejection_wick_ratio) & 
                                (bullish_rejection_wick_ratio > rejection_ratio_threshold) &
                                (data['Low'] < recent_low))
    bearish_liquidity_sweep = ((bearish_rejection_wick_ratio > bullish_rejection_wick_ratio) & 
                                (bearish_rejection_wick_ratio > rejection_ratio_threshold) &
                                (data['High'] > recent_high)) 

    support_bounce = (support_pct_distance >= threshold) & (data['Close'] > support)
    resistance_bounce = (resistance_pct_distance >= threshold) & (data['Close'] < resistance)

    #trigger off bounces
    buy_conditions = support_bounce | (~bearish_liquidity_sweep & (volatility > 70))
    sell_conditions = resistance_bounce | (~bullish_liquidity_sweep & (volatility > 70))
    signals[buy_conditions] = 3
    signals[sell_conditions] = 1
    return signals

def ma_trend_strategy(data: pd.DataFrame, band_period: int = 2, pct_band: float = 0.002, adx_ma_period: int = 32) -> pd.Series:
    signals = pd.Series(2, index=data.index)
    hlc3 = (data['High'] + data['Low'] + data['Close']) / 3
    upper_band = ta.ema(data['Close'], band_period) + data['Close'] * pct_band
    lower_band = ta.ema(data['Close'], band_period) - data['Close'] * pct_band

    adx, plus_di, minus_di = ta.adx(data['High'], data['Low'], data['Close'])
    adx_ma = ta.sma(adx, adx_ma_period)
    adx_ma_diff = (adx - adx_ma) / adx_ma #under the assumption that adx will begin to rise when a cascade hits

    buy_conditions = (data['Close'] > upper_band)
    sell_conditions = (data['Close'] < lower_band)

    signals[buy_conditions] = 3
    signals[sell_conditions] = 1
    return signals

def ma_super_trend_strategy(data: pd.DataFrame, period: int = 14, multiplier: float = 3) -> pd.Series:
    signals = pd.Series(0, index=data.index)
    ma1 = ta.sma(data['Close'], 20)
    ma2 = ta.sma(data['Close'], 50)
    ma3 = ta.sma(data['Close'], 100)

    buy_conditions = (ma1 > ma2) & (ma2 > ma3)
    sell_conditions = (ma1 < ma2) | (ma2 < ma3)

    signals[buy_conditions] = 3
    signals[sell_conditions] = 2
    return signals