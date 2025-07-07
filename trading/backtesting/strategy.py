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
    signals[data['Close'] < data['Open']] = 2
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

def scalper_strategy(data: pd.DataFrame) -> pd.Series:
    """Trades frequently, good for testing."""
    signals = pd.Series(2, index=data.index)

    rsi = ta.rsi(data['Close'], timeperiod=2)
    adx, plus_di, minus_di = ta.adx(data['High'], data['Low'], data['Close'], timeperiod=2)

    buy_conditions = (rsi > adx) & (plus_di > minus_di)
    sell_conditions = (rsi < adx) & (minus_di > plus_di)

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

def rsi_divergence_strategy(data: pd.DataFrame, 
                           rsi_period: int = 14, 
                           lookback: int = 5,
                           min_strength: float = 0.1) -> pd.Series:
    """
    RSI Divergence Strategy
    
    Detects bullish and bearish divergences between price and RSI:
    - Bullish divergence: Price makes lower lows while RSI makes higher lows
    - Bearish divergence: Price makes higher highs while RSI makes lower highs
    
    Args:
        data: DataFrame with OHLC data
        rsi_period: Period for RSI calculation
        lookback: Number of bars to look back for divergence detection
        min_strength: Minimum strength threshold for divergence signals
    
    Returns:
        pd.Series: Trading signals (1=short, 2=flat, 3=long)
    """
    signals = pd.Series(2, index=data.index)  # Start with flat signals
    
    # Calculate RSI
    rsi = ta.rsi(data['Close'], timeperiod=rsi_period)
    
    # Handle NaN values
    rsi = rsi.fillna(50)
    
    # Vectorized divergence detection
    price_change = data['Close'] - data['Close'].shift(lookback)
    rsi_change = rsi - rsi.shift(lookback)
    
    # Bullish divergence: price down, RSI up, RSI oversold
    bullish_divergence = (price_change < 0) & (rsi_change > 0) & (rsi < 40)
    
    # Bearish divergence: price up, RSI down, RSI overbought
    bearish_divergence = (price_change > 0) & (rsi_change < 0) & (rsi > 60)
    
    # Generate signals - only set once per condition
    signals[bullish_divergence] = 3
    signals[bearish_divergence] = 1
    
    return signals

def comprehensive_trend_strategy(data: pd.DataFrame, 
                               rsi_period: int = 14,
                               rsi_range: float = 30,
                               supertrend_period: int = 10,
                               supertrend_multiplier: float = 3.0,
                               ma_fast: int = 21,
                               ma_slow: int = 50,
                               ma_trend: int = 200,
                               score_threshold: int = 3) -> pd.Series:
    """
    Comprehensive trend-following strategy combining multiple indicators.
    
    Entry Conditions (Long):
    1. RSI > oversold level (momentum confirmation)
    2. Supertrend is bullish (trend confirmation)
    3. Fast MA > Slow MA (short-term trend)
    4. Price > Trend MA (long-term trend filter)
    5. Heikin Ashi shows bullish trend
    
    Entry Conditions (Short):
    1. RSI < overbought level (momentum confirmation)
    2. Supertrend is bearish (trend confirmation)  
    3. Fast MA < Slow MA (short-term trend)
    4. Price < Trend MA (long-term trend filter)
    5. Heikin Ashi shows bearish trend
    """
    signals = pd.Series(2, index=data.index)  # Start with flat signals
    
    ha_data = ta.heikin_ashi_transform(data)
    
    rsi = ta.rsi(data['Close'], timeperiod=rsi_period)
    
    supertrend, supertrend_line = ta.supertrend(
        data['High'], 
        data['Low'], 
        data['Close'], 
        period=supertrend_period, 
        multiplier=supertrend_multiplier
    )

    hlc3 = (data['High'] + data['Low'] + data['Close']) / 3
    
    ma_fast_values = ta.ema(data['Close'], timeperiod=ma_fast)
    ma_slow_values = ta.ema(data['Close'], timeperiod=ma_slow)
    ma_trend_values = ta.sma(hlc3, timeperiod=ma_trend)
    
    rsi_bullish = rsi > 100 - rsi_range
    rsi_bearish = rsi < rsi_range
    
    supertrend_bullish = supertrend == 1
    supertrend_bearish = supertrend == -1
    
    ma_cross_bullish = ma_fast_values > ma_slow_values
    ma_cross_bearish = ma_fast_values < ma_slow_values
    
    trend_filter_bullish = hlc3 > ma_trend_values
    trend_filter_bearish = hlc3 < ma_trend_values
    
    ha_bullish = ha_data['Close'] > ha_data['Open']
    ha_bearish = ha_data['Close'] < ha_data['Open']
    
    
    long_score = (
        rsi_bullish.astype(int) + 
        supertrend_bullish.astype(int) + 
        ma_cross_bullish.astype(int) + 
        trend_filter_bullish.astype(int) + 
        ha_bullish.astype(int)
    )
    
    short_score = (
        rsi_bearish.astype(int) + 
        supertrend_bearish.astype(int) + 
        ma_cross_bearish.astype(int) + 
        trend_filter_bearish.astype(int) + 
        ha_bearish.astype(int)
    )
    
    long_conditions = long_score >= score_threshold
    short_conditions = short_score >= score_threshold
    
    signals[long_conditions] = 3
    signals[short_conditions] = 1
    
    # exit_long = supertrend_bearish | (ma_fast_values < ma_slow_values)
    # exit_short = supertrend_bullish | (ma_fast_values > ma_slow_values)
    
    # for i in range(len(signals)):
    #     if signals.iloc[i] == 3 and exit_long.iloc[i] and not long_conditions.iloc[i]:
    #         signals.iloc[i] = 2
    #     elif signals.iloc[i] == 1 and exit_short.iloc[i] and not short_conditions.iloc[i]:
    #         signals.iloc[i] = 2
    
    return signals