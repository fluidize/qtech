import pandas as pd
import technical_analysis as ta
import smc_analysis as smc
import model_tools as mt
import numpy as np

def sl_tp(data: pd.DataFrame, signals: pd.Series, sl_pct: float, tp_pct: float) -> pd.Series:
    """
    Stop-loss and take-profit filter aligned with backtesting execution model.
    
    Backtesting execution model:
    - Signal at Close[t] → shifted to t+1 → execution position shifted to t+2 → executes at Open[t+2]
    - So signal at index i executes at Open[i+2]
    
    Args:
        data: DataFrame with OHLC data
        signals: Trading signals (0=hold, 1=short, 2=flat, 3=long)
        sl_pct: Stop-loss percentage (e.g., 0.01 for 1%)
        tp_pct: Take-profit percentage (e.g., 0.02 for 2%)
    
    Returns:
        pd.Series: Filtered signals
    """
        
    filtered_signals = signals.copy()
    entry_price = None
    current_position = 2  # Start flat
    
    for i in range(len(signals)):
        signal = signals.iloc[i]
        
        # New position signal
        if signal in [1, 3] and signal != current_position:
            current_position = signal
            # Signal at i executes at Open[i+2] due to double shift in backtesting
            if i + 2 < len(data):
                entry_price = data['Open'].iloc[i + 2]
        
        # Check SL/TP for active position
        elif current_position in [1, 3] and entry_price is not None:
            # Only check SL/TP after execution has occurred (i+2 or later)
            if i >= 2:  # Ensure we have executed
                current_price = data['Close'].iloc[i]
                price_change = (current_price - entry_price) / entry_price
                
                # Long position
                if current_position == 3:
                    if price_change >= tp_pct or price_change <= -sl_pct:
                        # Exit signal needs to account for the shifts
                        # Signal at i will execute at i+2, so we want to exit now
                        filtered_signals.iloc[i] = 2
                        current_position = 2
                        entry_price = None
                
                # Short position  
                elif current_position == 1:
                    if price_change <= -tp_pct or price_change >= sl_pct:
                        # Exit signal needs to account for the shifts
                        filtered_signals.iloc[i] = 2
                        current_position = 2
                        entry_price = None
    
    return filtered_signals


def hold_strategy(data: pd.DataFrame, signal: int = 3) -> pd.Series:
    signals = pd.Series(signal, index=data.index)
    return signals

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
        
    for i in range(2, len(data)-2):
        if i < len(data):
            future_return = (data['Open'].iloc[i+2] - data['Open'].iloc[i+1]) / data['Open'].iloc[i+1]
            
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
    
    supertrend, supertrend_line = ta.supertrend(
        data['High'], 
        data['Low'], 
        data['Close'], 
        period=supertrend_window, 
        multiplier=supertrend_multiplier
    )

    buy_conditions = (supertrend == 1)
    sell_conditions = (supertrend == -1)
    signals[buy_conditions] = 3
    signals[sell_conditions] = 2
    
    return signals

def ma_crossover_strategy(data: pd.DataFrame, ma_fast: int = 50, ma_slow: int = 200, slow_pct_shift: int = 0.01) -> pd.Series:
    signals = pd.Series(0, index=data.index)
    hlc3 = (data['High'] + data['Low'] + data['Close']) / 3

    ma_fast = ta.sma(data['Close'], timeperiod=ma_fast) 
    ma_slow = ta.sma(data['Close'], timeperiod=ma_slow) + hlc3 * slow_pct_shift
    signals[ma_fast < ma_slow] = 3
    signals[ma_fast > ma_slow] = 2

    return signals

def sr_strategy(data: pd.DataFrame, sr_window: int = 10) -> pd.Series:
    signals = pd.Series(0, index=data.index)

    support, resistance = smc.support_resistance_levels(data['Open'], data['High'], data['Low'], data['Close'], window=sr_window)

    buy_conditions = (data['Close'] < support.shift(1))
    sell_conditions = (data['Close'] > resistance.shift(1))
    signals[buy_conditions] = 3
    signals[sell_conditions] = 2

    return signals

def psar_strategy(data: pd.DataFrame, af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.2) -> pd.Series:
    signals = pd.Series(2, index=data.index)
    psar = ta.psar(data['High'], data['Low'], af_start, af_step, af_max)

    buy_conditions = (data['Close'] > psar)
    sell_conditions = (data['Close'] < psar)
    signals[buy_conditions] = 3
    signals[sell_conditions] = 2

    return signals

def ichimoku_strategy(data: pd.DataFrame, tenkan_period: int = 9, kijun_period: int = 26, senkou_period: int = 52, chikou_period: int = 26) -> pd.Series:
    signals = pd.Series(0, index=data.index)

    tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span = ta.ichimoku(
        data['High'], 
        data['Low'], 
        data['Close'], 
        tenkan_period, 
        kijun_period, 
        senkou_period, 
        chikou_period)
    
    buy_conditions = (data['Close'] > senkou_span_a) & (data['Close'] > senkou_span_b)
    sell_conditions = (data['Close'] < senkou_span_a) & (data['Close'] < senkou_span_b)

    signals[buy_conditions] = 3
    signals[sell_conditions] = 2

    return signals

def heikin_ashi_strategy(data: pd.DataFrame) -> pd.Series:
    signals = pd.Series(0, index=data.index)
    ha_data = ta.heikin_ashi_transform(data)

    slow_ma = ta.sma(ha_data['Close'], timeperiod=200)
    fast_ma = ta.sma(ha_data['Close'], timeperiod=50)

    buy_conditions = (fast_ma > slow_ma)
    sell_conditions = (fast_ma < slow_ma)
    signals[buy_conditions] = 3
    signals[sell_conditions] = 2

    return signals