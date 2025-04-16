import pandas as pd
import numpy as np

def sma(series, timeperiod=20):
    """Simple Moving Average"""
    return series.rolling(window=timeperiod).mean()

def ema(series, timeperiod=20):
    """Exponential Moving Average"""
    return series.ewm(span=timeperiod, adjust=False).mean()

def rsi(series, timeperiod=14):
    """Relative Strength Index"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=timeperiod).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=timeperiod).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(series, fastperiod=12, slowperiod=26, signalperiod=9):
    """Moving Average Convergence Divergence"""
    exp1 = series.ewm(span=fastperiod, adjust=False).mean()
    exp2 = series.ewm(span=slowperiod, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signalperiod, adjust=False).mean()
    return macd, signal

def bbands(series, timeperiod=20, nbdevup=2, nbdevdn=2):
    """Bollinger Bands"""
    middle = sma(series, timeperiod)
    std = series.rolling(window=timeperiod).std()
    upper = middle + (std * nbdevup)
    lower = middle - (std * nbdevdn)
    return upper, middle, lower

def stoch(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3):
    """Stochastic Oscillator"""
    lowest_low = low.rolling(window=fastk_period).min()
    highest_high = high.rolling(window=fastk_period).max()
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=slowk_period).mean()
    return k, d

def atr(high, low, close, timeperiod=14):
    """Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=timeperiod).mean()

def obv(close, volume):
    """On Balance Volume"""
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv

def cci(high, low, close, timeperiod=20):
    """Commodity Channel Index"""
    tp = (high + low + close) / 3
    sma = tp.rolling(window=timeperiod).mean()
    mad = tp.rolling(window=timeperiod).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma) / (0.015 * mad)

def adx(high, low, close, timeperiod=14):
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

def log_returns(series):
    """Log Returns"""
    return np.log(series / series.shift(1))

def stddev(series, timeperiod=20):
    """Standard Deviation"""
    return series.rolling(window=timeperiod).std()

def roc(series, timeperiod=10):
    """Rate of Change"""
    return series.pct_change(periods=timeperiod) * 100

def mom(series, timeperiod=10):
    """Momentum"""
    return series.diff(timeperiod)

def willr(high, low, close, timeperiod=14):
    """Williams %R"""
    highest_high = high.rolling(window=timeperiod).max()
    lowest_low = low.rolling(window=timeperiod).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low)

def mfi(high, low, close, volume, timeperiod=14):
    """Money Flow Index"""
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    
    positive_flow = pd.Series(0.0, index=money_flow.index, dtype='float64')
    negative_flow = pd.Series(0.0, index=money_flow.index, dtype='float64')
    
    positive_flow[typical_price > typical_price.shift(1)] = money_flow
    negative_flow[typical_price < typical_price.shift(1)] = money_flow
    
    positive_mf = positive_flow.rolling(window=timeperiod).sum()
    negative_mf = negative_flow.rolling(window=timeperiod).sum()
    
    mfi = 100 - (100 / (1 + positive_mf / negative_mf))
    return mfi

def kama(series, er_period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average"""
    change = abs(series - series.shift(er_period))
    volatility = series.diff().abs().rolling(window=er_period).sum()
    
    # Efficiency Ratio
    er = change / volatility
    
    # Smoothing Constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    # Scaling of Efficiency Ratio
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA series
    kama = pd.Series(index=series.index)
    
    # First KAMA value is the price
    first_idx = er.first_valid_index()
    if first_idx:
        kama.loc[first_idx] = series.loc[first_idx]
        
        # Calculate KAMA values
        for i in range(series.index.get_loc(first_idx) + 1, len(series)):
            idx = series.index[i]
            prev_idx = series.index[i-1]
            kama.loc[idx] = kama.loc[prev_idx] + sc.loc[idx] * (series.loc[idx] - kama.loc[prev_idx])
    
    return kama

def vwap(high, low, close, volume):
    """Volume Weighted Average Price"""
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    return vwap

def superTrend(high, low, close, period=14, multiplier=3):
    """SuperTrend indicator"""
    atr = ATR(high, low, close, period)
    
    upper_band = ((high + low) / 2) + (multiplier * atr)
    lower_band = ((high + low) / 2) - (multiplier * atr)
    
    supertrend = pd.Series(0, index=close.index)
    
    # Initialize with default direction (long)
    supertrend.iloc[0] = 1
    
    # Determine trend direction and supertrend values
    for i in range(1, len(close)):
        if close.iloc[i] > upper_band.iloc[i-1]:
            supertrend.iloc[i] = 1  # Uptrend
        elif close.iloc[i] < lower_band.iloc[i-1]:
            supertrend.iloc[i] = -1  # Downtrend
        else:
            supertrend.iloc[i] = supertrend.iloc[i-1]  # Continue previous trend
            
            # Update upper and lower bands
            if supertrend.iloc[i] == 1 and lower_band.iloc[i] < lower_band.iloc[i-1]:
                lower_band.iloc[i] = lower_band.iloc[i-1]
            if supertrend.iloc[i] == -1 and upper_band.iloc[i] > upper_band.iloc[i-1]:
                upper_band.iloc[i] = upper_band.iloc[i-1]
                
    return supertrend, upper_band, lower_band

def tsi(close, long_period=25, short_period=13, signal_period=13):
    """True Strength Index"""
    momentum = close.diff()  # 1-period price change
    
    # Double smoothed momentum
    smooth1 = momentum.ewm(span=long_period, adjust=False).mean()
    smooth2 = smooth1.ewm(span=short_period, adjust=False).mean()
    
    # Double smoothed absolute momentum
    abs_momentum = momentum.abs()
    abs_smooth1 = abs_momentum.ewm(span=long_period, adjust=False).mean()
    abs_smooth2 = abs_smooth1.ewm(span=short_period, adjust=False).mean()
    
    # TSI indicator and signal line
    tsi = 100 * (smooth2 / abs_smooth2)
    signal = tsi.ewm(span=signal_period, adjust=False).mean()
    
    return tsi, signal

def cmf(high, low, close, volume, period=20):
    """Chaikin Money Flow"""
    mfv = volume * ((close - low) - (high - close)) / (high - low)
    cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
    return cmf

def hma(series, period=16):
    """Hull Moving Average"""
    wma_half_period = WMA(series, period=period//2)
    wma_full_period = WMA(series, period=period)
    
    # HMA = WMA(2*WMA(period/2) - WMA(period), sqrt(period))
    sqrt_period = int(np.sqrt(period))
    return WMA(2 * wma_half_period - wma_full_period, period=sqrt_period)

def wma(series, period=20):
    """Weighted Moving Average"""
    weights = np.arange(1, period + 1)
    sum_weights = weights.sum()
    
    result = pd.Series(index=series.index)
    for i in range(period - 1, len(series)):
        result.iloc[i] = np.sum(series.iloc[i - period + 1:i + 1].values * weights) / sum_weights
    
    return result

def ichimoku(high, low, close, tenkan_period=9, kijun_period=26, senkou_period=52, chikou_period=26):
    """Ichimoku Cloud"""
    # Tenkan-sen (Conversion Line): (highest high + lowest low) / 2 for the past tenkan_period
    tenkan_sen = (high.rolling(window=tenkan_period).max() + low.rolling(window=tenkan_period).min()) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low) / 2 for the past kijun_period
    kijun_sen = (high.rolling(window=kijun_period).max() + low.rolling(window=kijun_period).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, shifted forward by kijun_period
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low) / 2 for the past senkou_period, shifted forward by kijun_period
    senkou_span_b = ((high.rolling(window=senkou_period).max() + low.rolling(window=senkou_period).min()) / 2).shift(kijun_period)
    
    # Chikou Span (Lagging Span): Current price, shifted backward by chikou_period
    chikou_span = close.shift(-chikou_period)
    
    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

def ppo(series, fast_period=12, slow_period=26, signal_period=9):
    """Percentage Price Oscillator"""
    fast_ema = EMA(series, timeperiod=fast_period)
    slow_ema = EMA(series, timeperiod=slow_period)
    
    ppo = ((fast_ema - slow_ema) / slow_ema) * 100
    signal = EMA(ppo, timeperiod=signal_period)
    histogram = ppo - signal
    
    return ppo, signal, histogram

def aobv(close, volume, fast_period=5, slow_period=10):
    """Adaptive On Balance Volume - OBV with smoothing and signal line"""
    # Calculate OBV
    obv = OBV(close, volume)
    
    # Generate fast and slow EMAs of OBV
    fast_obv = EMA(obv, timeperiod=fast_period)
    slow_obv = EMA(obv, timeperiod=slow_period)
    
    # Signal: fast OBV - slow OBV (like MACD with OBV)
    signal = fast_obv - slow_obv
    
    return obv, signal

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
    df = df.copy()
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
    pattern_df = pd.DataFrame(0, index=df.index, columns=patterns)
    
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
            pattern_df.at[df.index[i], 'doji'] = 1
        
        if 'hammer' in patterns and is_hammer(curr_row):
            pattern_df.at[df.index[i], 'hammer'] = 1
        
        if 'shooting_star' in patterns and is_shooting_star(curr_row):
            pattern_df.at[df.index[i], 'shooting_star'] = 1
        
        # Two candle patterns
        if 'engulfing' in patterns and is_engulfing(prev_row, curr_row):
            pattern_df.at[df.index[i], 'engulfing'] = 1
        
        if 'harami' in patterns and is_harami(prev_row, curr_row):
            pattern_df.at[df.index[i], 'harami'] = 1
        
        if 'dark_cloud_cover' in patterns and is_dark_cloud_cover(prev_row, curr_row):
            pattern_df.at[df.index[i], 'dark_cloud_cover'] = 1
        
        if 'piercing_line' in patterns and is_piercing_line(prev_row, curr_row):
            pattern_df.at[df.index[i], 'piercing_line'] = 1
        
        # Three candle patterns
        if 'morning_star' in patterns and is_morning_star(prev2_row, prev_row, curr_row):
            pattern_df.at[df.index[i], 'morning_star'] = 1
        
        if 'evening_star' in patterns and is_evening_star(prev2_row, prev_row, curr_row):
            pattern_df.at[df.index[i], 'evening_star'] = 1
        
        if 'three_white_soldiers' in patterns and is_three_white_soldiers(prev2_row, prev_row, curr_row):
            pattern_df.at[df.index[i], 'three_white_soldiers'] = 1
        
        if 'three_black_crows' in patterns and is_three_black_crows(prev2_row, prev_row, curr_row):
            pattern_df.at[df.index[i], 'three_black_crows'] = 1
    
    return pattern_df

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