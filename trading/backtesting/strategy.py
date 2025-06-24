import pandas as pd
import technical_analysis as ta
import smc_analysis as smc
import model_tools as mt
import numpy as np


def hold_strategy(data: pd.DataFrame, signal: int = 3) -> pd.Series:
    return pd.Series(signal, index=data.index)

def signal_spam(data: pd.DataFrame) -> pd.Series:
    signals = pd.Series(2, index=data.index)
    signals[data['Close'] > data['Open']] = 3
    signals[data['Close'] < data['Open']] = 2
    return signals

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

def zscore_momentum_strategy(data: pd.DataFrame, zscore_threshold: float = 0.1) -> pd.Series:
    signals = pd.Series(0, index=data.index)

    zscore = ta.zscore(data['Close'])
    signals[zscore > zscore_threshold] = 3
    signals[zscore < -zscore_threshold] = 2
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

def trend_strategy(data: pd.DataFrame,
                supertrend_window: int = 24,
                supertrend_multiplier: float = 1.2,
                ) -> pd.Series:
    signals = pd.Series(0, index=data.index)
    supertrend, supertrend_line = ta.supertrend(data['High'], data['Low'], data['Close'], period=supertrend_window, multiplier=supertrend_multiplier)

    supertrend_buy_conditions = (supertrend == 1)
    supertrend_sell_conditions = (supertrend == -1)

    signals[supertrend_buy_conditions] = 3
    signals[supertrend_sell_conditions] = 2
    return signals

def macd_dema_analysis(data: pd.DataFrame) -> pd.Series:

    signals = pd.Series(0, index=data.index)
    macd, macd_signal, macd_hist = ta.macd_dema(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    cci = ta.cci(data['High'], data['Low'], data['Close'], timeperiod=20)
    cci_ma = ta.sma(cci, timeperiod=14)

    macd_buy_conditions = (np.sign(macd_hist)-np.sign(macd_hist.shift(1)) >= 0) & (macd_hist > 0.01)
    macd_sell_conditions = (np.sign(macd_hist)-np.sign(macd_hist.shift(1)) <= 0) & (macd_hist < -0.01)
    cci_buy_conditions = (cci < -150) & (cci_ma < -150)
    cci_sell_conditions = (cci > 150) & (cci_ma > 150)

    signals[macd_buy_conditions & cci_buy_conditions] = 3
    signals[macd_sell_conditions & cci_sell_conditions] = 1

    return signals