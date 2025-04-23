import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

def sma(series, timeperiod=20) -> pd.Series:
    """Simple Moving Average"""
    return series.rolling(window=timeperiod).mean()

def ema(series, timeperiod=20) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=timeperiod, adjust=False).mean()

def rsi(series, timeperiod=14) -> pd.Series:
    """Relative Strength Index"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=timeperiod).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=timeperiod).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(series, fastperiod=12, slowperiod=26, signalperiod=9) -> tuple[pd.Series, pd.Series]:
    """Moving Average Convergence Divergence"""
    exp1 = series.ewm(span=fastperiod, adjust=False).mean()
    exp2 = series.ewm(span=slowperiod, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signalperiod, adjust=False).mean()
    return macd, signal

def bbands(series, timeperiod=20, nbdevup=2, nbdevdn=2) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands"""
    middle = sma(series, timeperiod)
    std = series.rolling(window=timeperiod).std()
    upper = middle + (std * nbdevup)
    lower = middle - (std * nbdevdn)
    return upper, middle, lower

def stoch(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3) -> tuple[pd.Series, pd.Series]:
    """Stochastic Oscillator"""
    lowest_low = low.rolling(window=fastk_period).min()
    highest_high = high.rolling(window=fastk_period).max()
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=slowk_period).mean()
    return k, d

def atr(high, low, close, timeperiod=14) -> pd.Series:
    """Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=timeperiod).mean()

def obv(close, volume) -> pd.Series:
    """On Balance Volume"""
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv

def cci(high, low, close, timeperiod=20) -> pd.Series:
    """Commodity Channel Index"""
    tp = (high + low + close) / 3
    sma = tp.rolling(window=timeperiod).mean()
    mad = tp.rolling(window=timeperiod).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma) / (0.015 * mad)

def adx(high, low, close, timeperiod=14) -> tuple[pd.Series, pd.Series, pd.Series]:
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

def log_returns(series) -> pd.Series:
    """Log Returns"""
    return np.log(series / series.shift(1))

def stddev(series, timeperiod=20) -> pd.Series:
    """Standard Deviation"""
    return series.rolling(window=timeperiod).std()

def roc(series, timeperiod=10) -> pd.Series:
    """Rate of Change"""
    return series.pct_change(periods=timeperiod) * 100

def mom(series, timeperiod=10) -> pd.Series:
    """Momentum"""
    return series.diff(timeperiod)

def willr(high, low, close, timeperiod=14) -> pd.Series:
    """Williams %R"""
    highest_high = high.rolling(window=timeperiod).max()
    lowest_low = low.rolling(window=timeperiod).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low)

def mfi(high, low, close, volume, timeperiod=14) -> pd.Series:
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

def kama(series, er_period=10, fast_period=2, slow_period=30) -> pd.Series:
    """Kaufman Adaptive Moving Average"""
    values = series.values
    change = abs(values - np.roll(values, er_period))
    volatility = np.abs(np.diff(values, prepend=values[0])).cumsum()
    
    # Efficiency Ratio
    er = change / volatility
    
    # Smoothing Constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    # Scaling of Efficiency Ratio
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA array
    kama = np.zeros_like(values)
    
    # First KAMA value is the price
    kama[0] = values[0]
    
    # Calculate KAMA values
    for i in range(1, len(values)):
        kama[i] = kama[i-1] + sc[i] * (values[i] - kama[i-1])
    
    return pd.Series(kama, index=series.index)

def vwap(high, low, close, volume) -> pd.Series:
    """Volume Weighted Average Price"""
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    return vwap

def supertrend(high, low, close, period=14, multiplier=3) -> tuple[pd.Series, pd.Series]:
    """SuperTrend indicator"""
    high_values = high.values
    low_values = low.values
    close_values = close.values
    
    # Calculate ATR
    tr1 = high_values - low_values
    tr2 = np.abs(high_values - np.roll(close_values, 1))
    tr3 = np.abs(low_values - np.roll(close_values, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).rolling(window=period).mean().values
    
    # Calculate basic upper and lower bands
    basic_upper = ((high_values + low_values) / 2) + (multiplier * atr)
    basic_lower = ((high_values + low_values) / 2) - (multiplier * atr)
    
    # Initialize arrays
    final_upper = np.zeros_like(close_values)
    final_lower = np.zeros_like(close_values)
    supertrend = np.zeros_like(close_values)
    
    # First value
    final_upper[0] = basic_upper[0]
    final_lower[0] = basic_lower[0]
    supertrend[0] = 1  # 1 for uptrend, -1 for downtrend
    
    # Calculate SuperTrend
    for i in range(1, len(close_values)):
        # Update upper band
        if close_values[i-1] > final_upper[i-1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = min(basic_upper[i], final_upper[i-1])
            
        # Update lower band
        if close_values[i-1] < final_lower[i-1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = max(basic_lower[i], final_lower[i-1])
            
        # Determine trend
        if close_values[i] > final_upper[i]:
            supertrend[i] = 1
        elif close_values[i] < final_lower[i]:
            supertrend[i] = -1
        else:
            supertrend[i] = supertrend[i-1]
    
    # Use upper band when trend is -1, lower band when trend is 1
    supertrend_line = np.where(supertrend == 1, final_lower, final_upper)
    
    return pd.Series(supertrend, index=close.index), pd.Series(supertrend_line, index=close.index)

def tsi(close, long_period=25, short_period=13, signal_period=13) -> tuple[pd.Series, pd.Series]:
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

def cmf(high, low, close, volume, period=20) -> pd.Series:
    """Chaikin Money Flow"""
    mfv = volume * ((close - low) - (high - close)) / (high - low)
    cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
    return cmf

def hma(series, period=16) -> pd.Series:
    """Hull Moving Average"""
    wma_half_period = wma(series, period=period//2)
    wma_full_period = wma(series, period=period)
    
    # HMA = WMA(2*WMA(period/2) - WMA(period), sqrt(period))
    sqrt_period = int(np.sqrt(period))
    return wma(2 * wma_half_period - wma_full_period, period=sqrt_period)

def wma(series, period=20) -> pd.Series:
    """Weighted Moving Average"""
    weights = np.arange(1, period + 1)
    sum_weights = weights.sum()
    
    result = pd.Series(index=series.index)
    for i in range(period - 1, len(series)):
        result.iloc[i] = np.sum(series.iloc[i - period + 1:i + 1].values * weights) / sum_weights
    
    return result

def ichimoku(high, low, close, tenkan_period=9, kijun_period=26, senkou_period=52, chikou_period=26) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
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

def ppo(series, fast_period=12, slow_period=26, signal_period=9) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Percentage Price Oscillator"""
    fast_ema = ema(series, timeperiod=fast_period)
    slow_ema = ema(series, timeperiod=slow_period)
    
    ppo = ((fast_ema - slow_ema) / slow_ema) * 100
    signal = ema(ppo, timeperiod=signal_period)
    histogram = ppo - signal
    
    return ppo, signal, histogram

def aobv(close, volume, fast_period=5, slow_period=10) -> tuple[pd.Series, pd.Series]:
    """Adaptive On Balance Volume - OBV with smoothing and signal line"""
    # Calculate OBV
    on_balance_volume = obv(close, volume)
    
    # Generate fast and slow EMAs of OBV
    fast_obv = ema(on_balance_volume, timeperiod=fast_period)
    slow_obv = ema(on_balance_volume, timeperiod=slow_period)
    
    # Signal: fast OBV - slow OBV (like MACD with OBV)
    signal = fast_obv - slow_obv
    
    return on_balance_volume, signal

def psar(high, low, acceleration_start=0.02, acceleration_step=0.02, max_acceleration=0.2) -> pd.Series:
    """Parabolic SAR"""
    n = len(high)
    psar = np.zeros(n, dtype=np.float64)  # Initialize PSAR as a NumPy array
    trend = np.empty(n, dtype=np.int32)  # Use np.empty instead of np.ones
    trend.fill(1)  # Fill with 1 for uptrend, -1 for downtrend
    ep = np.zeros(n, dtype=np.float64)  # Extreme point

    # Initialize first values
    psar[0] = low[0]
    ep[0] = high[0]
    af = acceleration_start

    for i in range(1, n):
        # Calculate SAR for current period
        if trend[i-1] == 1:
            diff = ep[i-1] - psar[i-1]
            accel_component = af * diff
            psar[i] = psar[i-1] + accel_component
        else:
            diff = psar[i-1] - ep[i-1]
            accel_component = af * diff
            psar[i] = psar[i-1] - accel_component

        trend_reversal = trend[i-1] == 1 and low[i] < psar[i]
        if trend[i-1] == 1:
            if low[i] < psar[i]:
                trend[i] = -1
                psar[i] = ep[i-1]
                ep[i] = low[i]
            else:
                trend[i] = 1
                if high[i] > ep[i-1]:
                    ep[i] = high[i]
                    af = min(af + acceleration_step, max_acceleration)
                else:
                    ep[i] = ep[i-1]
        else:
            if high[i] > psar[i]:
                trend[i] = 1
                psar[i] = ep[i-1]
                ep[i] = high[i]
            else:
                trend[i] = -1
                if low[i] < ep[i-1]:
                    ep[i] = low[i]
                    af = min(af + acceleration_step, max_acceleration)
                else:
                    ep[i] = ep[i-1]

    return psar

def identify_candlestick_patterns(open_prices, high_prices, low_prices, close_prices, patterns: list[str] = None) -> pd.DataFrame:
    """
    Identify various candlestick patterns in the data.
    Args:
        open_prices: NumPy array of open prices
        high_prices: NumPy array of high prices
        low_prices: NumPy array of low prices
        close_prices: NumPy array of close prices
        patterns: List of patterns to identify. If None, identifies all patterns.
    Returns:
        NumPy array of binary indicators for each pattern.
    """
    n = len(open_prices)
    body = close_prices - open_prices
    upper_shadow = high_prices - np.maximum(open_prices, close_prices)
    lower_shadow = np.minimum(open_prices, close_prices) - low_prices
    total_range = high_prices - low_prices
    body_ratio = np.abs(body) / total_range

    available_patterns = ['Doji', 'Hammer', 'Shooting_Star', 'Engulfing', 'Harami', 
                         'Morning_Star', 'Evening_Star', 'Three_White_Soldiers',
                         'Three_Black_Crows', 'Dark_Cloud_Cover', 'Piercing_Line']
    
    if patterns is None:
        patterns = available_patterns
    else:
        invalid_patterns = set(patterns) - set(available_patterns)
        if invalid_patterns:
            raise ValueError(f"Invalid patterns: {invalid_patterns}. Available patterns: {available_patterns}")

    pattern_results = np.zeros((n, len(patterns)), dtype=np.int32)

    for i in range(2, n):
        # Single candle patterns
        if 'Doji' in patterns and abs(body[i]) < (total_range[i] * 0.1):
            pattern_results[i, patterns.index('Doji')] = 1
        
        if 'Hammer' in patterns and (lower_shadow[i] > 2 * abs(body[i])) and (upper_shadow[i] < abs(body[i])):
            pattern_results[i, patterns.index('Hammer')] = 1
        
        if 'Shooting_Star' in patterns and (upper_shadow[i] > 2 * abs(body[i])) and (lower_shadow[i] < abs(body[i])):
            pattern_results[i, patterns.index('Shooting_Star')] = 1
        
        # Two candle patterns
        if 'Engulfing' in patterns and (body[i-1] < 0 and body[i] > 0 and 
                                      open_prices[i] < close_prices[i-1] and close_prices[i] > open_prices[i-1]):
            pattern_results[i, patterns.index('Engulfing')] = 1
        
        if 'Harami' in patterns and (abs(body[i-1]) > abs(body[i]) and
                                    open_prices[i-1] > close_prices[i] and close_prices[i-1] < open_prices[i]):
            pattern_results[i, patterns.index('Harami')] = 1
        
        if 'Dark_Cloud_Cover' in patterns and (body[i-1] > 0 and body[i] < 0 and
                                              open_prices[i] > high_prices[i-1] and
                                              close_prices[i] < (open_prices[i-1] + close_prices[i-1]) / 2):
            pattern_results[i, patterns.index('Dark_Cloud_Cover')] = 1
        
        if 'Piercing_Line' in patterns and (body[i-1] < 0 and body[i] > 0 and
                                           open_prices[i] < low_prices[i-1] and
                                           close_prices[i] > (open_prices[i-1] + close_prices[i-1]) / 2):
            pattern_results[i, patterns.index('Piercing_Line')] = 1
        
        # Three candle patterns
        if 'Morning_Star' in patterns and (body[i-2] < 0 and 
                                          abs(body[i-1]) < (total_range[i-1] * 0.1) and
                                          body[i] > 0 and
                                          close_prices[i] > close_prices[i-2]):
            pattern_results[i, patterns.index('Morning_Star')] = 1
        
        if 'Evening_Star' in patterns and (body[i-2] > 0 and 
                                          abs(body[i-1]) < (total_range[i-1] * 0.1) and
                                          body[i] < 0 and
                                          close_prices[i] < close_prices[i-2]):
            pattern_results[i, patterns.index('Evening_Star')] = 1
        
        if 'Three_White_Soldiers' in patterns and (body[i-2] > 0 and body[i-1] > 0 and body[i] > 0 and
                                                  close_prices[i-1] > close_prices[i-2] and close_prices[i] > close_prices[i-1]):
            pattern_results[i, patterns.index('Three_White_Soldiers')] = 1
        
        if 'Three_Black_Crows' in patterns and (body[i-2] < 0 and body[i-1] < 0 and body[i] < 0 and
                                               close_prices[i-1] < close_prices[i-2] and close_prices[i] < close_prices[i-1]):
            pattern_results[i, patterns.index('Three_Black_Crows')] = 1

    pattern_df = pd.DataFrame(pattern_results, index=range(len(open_prices)), columns=patterns)
    pattern_df.columns = [pattern for pattern in patterns]

    return pattern_df

def get_candlestick_patterns(df, patterns: list[str] = None) -> pd.DataFrame:
    """
    Get candlestick patterns for the given OHLCV data.
    Args:
        df: DataFrame with OHLCV data
        patterns: List of patterns to identify. If None, identifies all patterns.
    Returns:
        DataFrame with binary indicators for each pattern.
    """
    open_prices = df['Open'].values
    high_prices = df['High'].values
    low_prices = df['Low'].values
    close_prices = df['Close'].values

    pattern_df = identify_candlestick_patterns(open_prices, high_prices, low_prices, close_prices, patterns)
    
    return pattern_df

if __name__ == "__main__":
    # data = mt.fetch_data("BTC-USDT", 1, "1min", 0, kucoin=True)
    # data['SMA'] = sma(data['Close'])
    # data['EMA'] = ema(data['Close'])
    # data['RSI'] = rsi(data['Close'])
    # data['MACD'], data['MACD_Signal'] = macd(data['Close'])
    # data['BBands_Upper'], data['BBands_Middle'], data['BBands_Lower'] = bbands(data['Close'])
    # data['Stoch_K'], data['Stoch_D'] = stoch(data['High'], data['Low'], data['Close'])
    # data['ATR'] = atr(data['High'], data['Low'], data['Close'])
    # data['OBV'], data['OBV_Signal'] = aobv(data['Close'], data['Volume'])
    # data['CCI'] = cci(data['High'], data['Low'], data['Close'])
    # data['ADX'], data['ADX_Pos'], data['ADX_Neg'] = adx(data['High'], data['Low'], data['Close'])
    # data['Log_Returns'] = log_returns(data['Close'])
    # data['StdDev'] = stddev(data['Close'])
    # data['ROC'] = roc(data['Close'])
    # data['Momentum'] = mom(data['Close'])
    # data['WilliamsR'] = willr(data['High'], data['Low'], data['Close'])
    # data['MFI'] = mfi(data['High'], data['Low'], data['Close'], data['Volume'])
    # data['KAMA'] = kama(data['Close'])
    # data['VWAP'] = vwap(data['High'], data['Low'], data['Close'], data['Volume'])
    # data['SuperTrend'], data['SuperTrend_Upper'], data['SuperTrend_Lower'] = supertrend(data['High'], data['Low'], data['Close'])
    # data['TSI'], data['TSI_Signal'] = tsi(data['Close'])
    # data['CMF'] = cmf(data['High'], data['Low'], data['Close'], data['Volume'])
    # data['HMA'] = hma(data['Close'])
    # data['WMA'] = wma(data['Close'])
    # data['Ichimoku_Tenkan'], data['Ichimoku_Kijun'], data['Ichimoku_Senkou_A'], data['Ichimoku_Senkou_B'], data['Ichimoku_Chikou'] = ichimoku(data['High'], data['Low'], data['Close'])
    # data['PPO'], data['PPO_Signal'], data['PPO_Histogram'] = ppo(data['Close'])
    # data['AOBV'], data['AOBV_Signal'] = aobv(data['Close'], data['Volume'])
    
    # start_time = time.time()
    # data['Doji'], data['Hammer'], data['Shooting_Star'], data['Engulfing'], data['Harami'], data['Morning_Star'], data['Evening_Star'], data['Three_White_Soldiers'], data['Three_Black_Crows'], data['Dark_Cloud_Cover'], data['Piercing_Line'] = get_candlestick_patterns(df=data, patterns=['doji', 'hammer', 'shooting_star', 'engulfing', 'harami', 'morning_star', 'evening_star', 'three_white_soldiers', 'three_black_crows', 'dark_cloud_cover', 'piercing_line'])
    # end_time = time.time()
    # print(f"Time taken: {end_time - start_time} seconds")

    # start_time = time.time()
    # data['PSAR'] = psar(data['High'].values, data['Low'].values)
    # end_time = time.time()
    # print(f"Time taken: {end_time - start_time} seconds")
    # fig = go.Figure()
    # fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close']))
    # fig.add_trace(go.Scatter(x=data.index, y=data['PSAR'], mode='markers', name='PSAR'))
    # fig.show()

    start_time = time.time()
    data['SuperTrend'], data['SuperTrend_Line'] = supertrend(data['High'], data['Low'], data['Close'])
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close']))
    fig.add_trace(go.Scatter(x=data.index, y=data['SuperTrend_Line'], mode='lines', name='SuperTrend'))
    fig.show()
    
    
    
    
    
    
