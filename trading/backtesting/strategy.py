import pandas as pd
import technical_analysis as ta
import smc_analysis as smc
import model_tools as mt
import numpy as np

def stop_loss_filter(data: pd.DataFrame, signals: pd.Series, sl_pct: float) -> pd.Series:
    """
    Args:
        data: DataFrame with OHLC data
        signals: Trading signals (0=hold, 1=short, 2=flat, 3=long)
        sl_pct: Stop-loss percentage (e.g., 0.02 for 2%)
    
    Returns:
        pd.Series: Signals with stop-loss exits applied
    """
    filtered_signals = signals.copy()
    entry_price = None
    current_position = 2  # Start flat
    
    for i in range(len(signals)):
        signal = signals.iloc[i]
        
        # New position signal - track entry price
        if signal in [1, 3] and signal != current_position:
            current_position = signal
            # Signal at i executes at Open[i+1] due to shift in backtesting
            if i + 1 < len(data):
                entry_price = data['Open'].iloc[i + 1]
        
        # Check stop-loss for active position
        elif current_position in [1, 3] and entry_price is not None:
            current_price = data['Close'].iloc[i]
            price_change = (current_price - entry_price) / entry_price
            
            # Long position stop-loss
            if current_position == 3 and price_change <= -sl_pct:
                filtered_signals.iloc[i] = 2  # Exit to flat
                current_position = 2
                entry_price = None
            
            # Short position stop-loss
            elif current_position == 1 and price_change >= sl_pct:
                filtered_signals.iloc[i] = 2  # Exit to flat
                current_position = 2
                entry_price = None
        
        # Update position if signal changes to flat
        if signal == 2:
            current_position = 2
            entry_price = None
    
    return filtered_signals

def take_profit_filter(data: pd.DataFrame, signals: pd.Series, tp_pct: float) -> pd.Series:
    """
    Args:
        data: DataFrame with OHLC data
        signals: Trading signals (0=hold, 1=short, 2=flat, 3=long)
        tp_pct: Take-profit percentage (e.g., 0.03 for 3%)
    
    Returns:
        pd.Series: Signals with take-profit exits applied
    """
    filtered_signals = signals.copy()
    entry_price = None
    current_position = 2  # Start flat
    
    for i in range(len(signals)):
        signal = signals.iloc[i]
        
        # New position signal - track entry price
        if signal in [1, 3] and signal != current_position:
            current_position = signal
            # Signal at i executes at Open[i+1] due to shift in backtesting
            if i + 1 < len(data):
                entry_price = data['Open'].iloc[i + 1]
        
        # Check take-profit for active position
        elif current_position in [1, 3] and entry_price is not None:
            current_price = data['Close'].iloc[i]
            price_change = (current_price - entry_price) / entry_price
            
            # Long position take-profit
            if current_position == 3 and price_change >= tp_pct:
                filtered_signals.iloc[i] = 2  # Exit to flat
                current_position = 2
                entry_price = None
            
            # Short position take-profit
            elif current_position == 1 and price_change <= -tp_pct:
                filtered_signals.iloc[i] = 2  # Exit to flat
                current_position = 2
                entry_price = None
        
        # Update position if signal changes to flat
        if signal == 2:
            current_position = 2
            entry_price = None
    
    return filtered_signals

def hold_strategy(data: pd.DataFrame, signal: int = 3) -> pd.Series:
    signals = pd.Series(signal, index=data.index)
    signals = stop_loss_filter(data, signals, 0.01)
    return signals

def signal_spam(data: pd.DataFrame) -> pd.Series:
    signals = pd.Series(2, index=data.index)
    signals[data['Close'] > data['Open']] = 3
    signals[data['Close'] < data['Open']] = 1
    return signals

def perfect_strategy(data: pd.DataFrame) -> pd.Series:
    """
    Perfect strategy that uses all information.
    """
    signals = pd.Series(2, index=data.index)
    
    for i in range(2, len(data)-2):
        if data['Open'].iloc[i+2] > data['Open'].iloc[i+1]:
            signals.iloc[i] = 3
        elif data['Open'].iloc[i+2] < data['Open'].iloc[i+1]:
            signals.iloc[i] = 1
        else:
            signals.iloc[i] = 2
    
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

def scalper_strategy(
        data: pd.DataFrame,
        rsi_window: int = 2,
        psar_af_start: float = 0.02,
        psar_af_step: float = 0.02,
        psar_af_max: float = 0.2,
        rsi_threshold: float = 30,
        ma_window: int = 20,
    ) -> pd.Series:
    signals = pd.Series(0, index=data.index)

    rsi = ta.rsi(data['Close'], timeperiod=rsi_window)
    psar = ta.psar(data['High'], data['Low'], psar_af_start, psar_af_step, psar_af_max)

    long_rsi = rsi < rsi_threshold
    short_rsi = rsi > 100 - rsi_threshold

    long_psar = data['Close'] > psar
    short_psar = data['Close'] < psar

    long_ma = data['Close'] > ta.sma(data['Close'], timeperiod=ma_window)
    short_ma = data['Close'] < ta.sma(data['Close'], timeperiod=ma_window)

    long_score = long_rsi.astype(int) + long_psar.astype(int) + long_ma.astype(int)
    short_score = short_rsi.astype(int) + short_psar.astype(int) + short_ma.astype(int)

    signals[long_score == 3] = 3
    signals[short_score == 3] = 2

    return signals

def trend_reversal_strategy(
        data: pd.DataFrame,
        supertrend_window: int = 10,
        supertrend_multiplier: float = 2,
        bb_window: int = 20,
        bb_dev: float = 2,
        bbw_ma_window: int = 13,
    ) -> pd.Series:
    signals = pd.Series(0, index=data.index)
    
    supertrend, supertrend_line = ta.supertrend(
        data['High'], 
        data['Low'], 
        data['Close'], 
        period=supertrend_window, 
        multiplier=supertrend_multiplier
    )

    upper, middle, lower = ta.bbands(data['Close'], timeperiod=bb_window, devup=bb_dev, devdn=bb_dev)
    bbw = (upper - lower) / middle #utilizing BBW as a volatility proxy
    bbw_ma = ta.sma(bbw, timeperiod=bbw_ma_window)

    volatility_contraction = (bbw < bbw_ma) & (bbw.shift(1) > bbw_ma.shift(1))

    buy_conditions = (supertrend == -1) & (volatility_contraction)
    sell_conditions = (supertrend == 1) & (volatility_contraction)

    signals[buy_conditions] = 3
    signals[sell_conditions] = 2
    
    return signals

def ma_crossover_strategy(data: pd.DataFrame, fast_period: int = 50, slow_period: int = 200) -> pd.Series:
    signals = pd.Series(0, index=data.index)

    ma_fast = ta.sma(data['Close'], timeperiod=fast_period) 
    ma_slow = ta.sma(data['Close'], timeperiod=slow_period)
    signals[ma_fast > ma_slow] = 3
    signals[ma_fast < ma_slow] = 2

    return signals

def sr_strategy(data: pd.DataFrame, sr_window: int = 10) -> pd.Series:
    signals = pd.Series(0, index=data.index)

    support, resistance = smc.support_resistance_levels(data['Open'], data['High'], data['Low'], data['Close'], window=sr_window)

    buy_conditions = (data['Close'] < support.shift(1))
    sell_conditions = (data['Close'] > resistance.shift(1))
    signals[buy_conditions] = 3
    signals[sell_conditions] = 2

    return signals

def heikin_ashi_strategy(data: pd.DataFrame, fast_ma_window: int = 5, slow_ma_window: int = 20) -> pd.Series:
    """
    Use Heikin Ashi as a trend-following signal.
    """
    signals = pd.Series(0, index=data.index)
    ha_data = ta.heikin_ashi_transform(data)

    fast_ma = ta.sma(ha_data['Close'], timeperiod=fast_ma_window)
    slow_ma = ta.sma(ha_data['Close'], timeperiod=slow_ma_window)

    signals[fast_ma > slow_ma] = 3
    signals[fast_ma < slow_ma] = 2

    return signals

def supertrend_strategy(data: pd.DataFrame, supertrend_window: int = 10, supertrend_multiplier: float = 2) -> pd.Series:
    signals = pd.Series(0, index=data.index)

    supertrend, supertrend_line = ta.supertrend(
        data['High'], 
        data['Low'], 
        data['Close'], 
        period=supertrend_window, 
        multiplier=supertrend_multiplier
    )

    signals[supertrend == -1] = 3
    signals[supertrend == 1] = 2

    return signals