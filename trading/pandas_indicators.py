import pandas as pd
import numpy as np

def SMA(series, timeperiod=20):
    """Simple Moving Average"""
    return series.rolling(window=timeperiod).mean()

def EMA(series, timeperiod=20):
    """Exponential Moving Average"""
    return series.ewm(span=timeperiod, adjust=False).mean()

def RSI(series, timeperiod=14):
    """Relative Strength Index"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=timeperiod).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=timeperiod).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def MACD(series, fastperiod=12, slowperiod=26, signalperiod=9):
    """Moving Average Convergence Divergence"""
    exp1 = series.ewm(span=fastperiod, adjust=False).mean()
    exp2 = series.ewm(span=slowperiod, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signalperiod, adjust=False).mean()
    return macd, signal

def BBANDS(series, timeperiod=20, nbdevup=2, nbdevdn=2):
    """Bollinger Bands"""
    middle = SMA(series, timeperiod)
    std = series.rolling(window=timeperiod).std()
    upper = middle + (std * nbdevup)
    lower = middle - (std * nbdevdn)
    return upper, middle, lower

def STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3):
    """Stochastic Oscillator"""
    lowest_low = low.rolling(window=fastk_period).min()
    highest_high = high.rolling(window=fastk_period).max()
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=slowk_period).mean()
    return k, d

def ATR(high, low, close, timeperiod=14):
    """Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=timeperiod).mean()

def OBV(close, volume):
    """On Balance Volume"""
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv

def CCI(high, low, close, timeperiod=20):
    """Commodity Channel Index"""
    tp = (high + low + close) / 3
    sma = tp.rolling(window=timeperiod).mean()
    mad = tp.rolling(window=timeperiod).apply(lambda x: np.abs(x - x.mean()).mean())
    return (tp - sma) / (0.015 * mad)

def ADX(high, low, close, timeperiod=14):
    """Average Directional Index"""
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/timeperiod).mean() / tr.ewm(alpha=1/timeperiod).mean())
    minus_di = 100 * (minus_dm.ewm(alpha=1/timeperiod).mean() / tr.ewm(alpha=1/timeperiod).mean())
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/timeperiod).mean()
    
    return adx, plus_di, minus_di

def LOG(series):
    """Log Returns"""
    return np.log(series / series.shift(1))

def STDDEV(series, timeperiod=20):
    """Standard Deviation"""
    return series.rolling(window=timeperiod).std()

def ROC(series, timeperiod=10):
    """Rate of Change"""
    return series.pct_change(periods=timeperiod) * 100

def MOM(series, timeperiod=10):
    """Momentum"""
    return series.diff(timeperiod)

def WILLR(high, low, close, timeperiod=14):
    """Williams %R"""
    highest_high = high.rolling(window=timeperiod).max()
    lowest_low = low.rolling(window=timeperiod).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low)

def MFI(high, low, close, volume, timeperiod=14):
    """Money Flow Index"""
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    
    positive_flow = pd.Series(0, index=money_flow.index)
    negative_flow = pd.Series(0, index=money_flow.index)
    
    positive_flow[typical_price > typical_price.shift(1)] = money_flow
    negative_flow[typical_price < typical_price.shift(1)] = money_flow
    
    positive_mf = positive_flow.rolling(window=timeperiod).sum()
    negative_mf = negative_flow.rolling(window=timeperiod).sum()
    
    mfi = 100 - (100 / (1 + positive_mf / negative_mf))
    return mfi

def identify_candlestick_patterns(df, patterns=None):
    """
    Identify various candlestick patterns in the data.
    Args:
        df: DataFrame with OHLCV data
        patterns: List of patterns to identify. If None, identifies all patterns.
                 Available patterns: ['doji', 'hammer', 'shooting_star', 'engulfing', 
                                    'harami', 'morning_star', 'evening_star', 'three_white_soldiers',
                                    'three_black_crows', 'dark_cloud_cover', 'piercing_line']
    Returns:
        DataFrame with binary indicators (1 for pattern present, 0 for absent)
    """
    # Calculate basic candle properties
    df['body'] = df['Close'] - df['Open']
    df['upper_shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['lower_shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['total_range'] = df['High'] - df['Low']
    df['body_ratio'] = abs(df['body']) / df['total_range']
    
    # Available patterns
    available_patterns = ['doji', 'hammer', 'shooting_star', 'engulfing', 'harami', 
                         'morning_star', 'evening_star', 'three_white_soldiers',
                         'three_black_crows', 'dark_cloud_cover', 'piercing_line']
    
    # If no patterns specified, use all available patterns
    if patterns is None:
        patterns = available_patterns
    else:
        # Validate patterns
        invalid_patterns = set(patterns) - set(available_patterns)
        if invalid_patterns:
            raise ValueError(f"Invalid patterns: {invalid_patterns}. Available patterns: {available_patterns}")
    
    # Initialize pattern columns
    for pattern in patterns:
        df[pattern] = 0
    
    # Helper functions for pattern detection
    def is_doji(row):
        return abs(row['body']) < (row['total_range'] * 0.1)
    
    def is_hammer(row):
        return (row['lower_shadow'] > 2 * abs(row['body'])) and (row['upper_shadow'] < abs(row['body']))
    
    def is_shooting_star(row):
        return (row['upper_shadow'] > 2 * abs(row['body'])) and (row['lower_shadow'] < abs(row['body']))
    
    def is_engulfing(prev_row, curr_row):
        return (prev_row['body'] < 0 and curr_row['body'] > 0 and 
                curr_row['Open'] < prev_row['Close'] and curr_row['Close'] > prev_row['Open'])
    
    def is_harami(prev_row, curr_row):
        return (abs(prev_row['body']) > abs(curr_row['body']) and
                prev_row['Open'] > curr_row['Close'] and prev_row['Close'] < curr_row['Open'])
    
    def is_morning_star(prev2_row, prev_row, curr_row):
        return (prev2_row['body'] < 0 and 
                abs(prev_row['body']) < (prev_row['total_range'] * 0.1) and
                curr_row['body'] > 0 and
                curr_row['Close'] > prev2_row['Close'])
    
    def is_evening_star(prev2_row, prev_row, curr_row):
        return (prev2_row['body'] > 0 and 
                abs(prev_row['body']) < (prev_row['total_range'] * 0.1) and
                curr_row['body'] < 0 and
                curr_row['Close'] < prev2_row['Close'])
    
    def is_three_white_soldiers(prev2_row, prev_row, curr_row):
        return (prev2_row['body'] > 0 and prev_row['body'] > 0 and curr_row['body'] > 0 and
                prev_row['Close'] > prev2_row['Close'] and curr_row['Close'] > prev_row['Close'])
    
    def is_three_black_crows(prev2_row, prev_row, curr_row):
        return (prev2_row['body'] < 0 and prev_row['body'] < 0 and curr_row['body'] < 0 and
                prev_row['Close'] < prev2_row['Close'] and curr_row['Close'] < prev_row['Close'])
    
    def is_dark_cloud_cover(prev_row, curr_row):
        return (prev_row['body'] > 0 and curr_row['body'] < 0 and
                curr_row['Open'] > prev_row['High'] and
                curr_row['Close'] < (prev_row['Open'] + prev_row['Close']) / 2)
    
    def is_piercing_line(prev_row, curr_row):
        return (prev_row['body'] < 0 and curr_row['body'] > 0 and
                curr_row['Open'] < prev_row['Low'] and
                curr_row['Close'] > (prev_row['Open'] + prev_row['Close']) / 2)
    
    # Pattern detection
    for i in range(2, len(df)):
        prev2_row = df.iloc[i-2]
        prev_row = df.iloc[i-1]
        curr_row = df.iloc[i]
        
        # Single candle patterns
        if 'doji' in patterns and is_doji(curr_row):
            df.at[i, 'doji'] = 1
        
        if 'hammer' in patterns and is_hammer(curr_row):
            df.at[i, 'hammer'] = 1
        
        if 'shooting_star' in patterns and is_shooting_star(curr_row):
            df.at[i, 'shooting_star'] = 1
        
        # Two candle patterns
        if 'engulfing' in patterns and is_engulfing(prev_row, curr_row):
            df.at[i, 'engulfing'] = 1
        
        if 'harami' in patterns and is_harami(prev_row, curr_row):
            df.at[i, 'harami'] = 1
        
        if 'dark_cloud_cover' in patterns and is_dark_cloud_cover(prev_row, curr_row):
            df.at[i, 'dark_cloud_cover'] = 1
        
        if 'piercing_line' in patterns and is_piercing_line(prev_row, curr_row):
            df.at[i, 'piercing_line'] = 1
        
        # Three candle patterns
        if 'morning_star' in patterns and is_morning_star(prev2_row, prev_row, curr_row):
            df.at[i, 'morning_star'] = 1
        
        if 'evening_star' in patterns and is_evening_star(prev2_row, prev_row, curr_row):
            df.at[i, 'evening_star'] = 1
        
        if 'three_white_soldiers' in patterns and is_three_white_soldiers(prev2_row, prev_row, curr_row):
            df.at[i, 'three_white_soldiers'] = 1
        
        if 'three_black_crows' in patterns and is_three_black_crows(prev2_row, prev_row, curr_row):
            df.at[i, 'three_black_crows'] = 1
    
    return df[patterns]

def get_candlestick_patterns(df, patterns=None):
    """
    Get candlestick patterns for the given OHLCV data.
    Args:
        df: DataFrame with OHLCV data
        patterns: List of patterns to identify. If None, identifies all patterns.
    Returns:
        DataFrame with binary indicators for each pattern.
    """
    patterns_df = identify_candlestick_patterns(df, patterns)
    return patterns_df