import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from scipy.stats import percentileofscore
from scipy import signal

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

def log_return(series) -> pd.Series:
    """Log Returns"""
    return np.log(series / series.shift(1))

def dpo(series, timeperiod=20) -> pd.Series:
    """Detrended Price Oscillator
    A non-directional indicator that removes trend from price to identify cycles"""
    return series - series.shift(int(timeperiod/2) + 1).rolling(window=timeperiod).mean()

def dema(series, timeperiod=20) -> pd.Series:
    """Double Exponential Moving Average
    Reduces lag of traditional EMAs"""
    ema1 = ema(series, timeperiod)
    ema2 = ema(ema1, timeperiod)
    return 2 * ema1 - ema2

def tema(series, timeperiod=20) -> pd.Series:
    """Triple Exponential Moving Average
    Further reduces lag compared to DEMA"""
    ema1 = ema(series, timeperiod)
    ema2 = ema(ema1, timeperiod)
    ema3 = ema(ema2, timeperiod)
    return 3 * ema1 - 3 * ema2 + ema3

def fisher_transform(series, timeperiod=10) -> pd.Series:
    """Ehlers Fisher Transform
    Converts prices to a Gaussian normal distribution"""
    # Normalize price to range [-1, 1]
    max_high = series.rolling(window=timeperiod).max()
    min_low = series.rolling(window=timeperiod).min()
    
    # Avoid division by zero by adding small epsilon where high=low
    price_range = (max_high - min_low)
    # Replace zeros with a small number
    price_range = price_range.replace(0, 1e-10)
    
    normalized = 2 * ((series - min_low) / price_range - 0.5)
    
    # Limit input to tanh domain to avoid log(0) errors
    normalized = normalized.clip(-0.999, 0.999)
    
    # Apply Fisher Transform
    fisher = np.log((1 + normalized) / (1 - normalized))
    fisher.replace([np.inf, -np.inf], np.nan, inplace=True)
    fisher = fisher.ffill().bfill()
    
    return fisher

def aroon(high, low, timeperiod=14) -> tuple[pd.Series, pd.Series]:
    """Aroon Indicator
    Measures the strength of a trend by time from high/low"""
    periods = np.arange(timeperiod)
    
    # Calculate Aroon Up
    rolling_high = high.rolling(window=timeperiod)
    aroon_up = 100 * rolling_high.apply(lambda x: (timeperiod - 1 - np.argmax(x)) / (timeperiod - 1), raw=True)
    
    # Calculate Aroon Down
    rolling_low = low.rolling(window=timeperiod)
    aroon_down = 100 * rolling_low.apply(lambda x: (timeperiod - 1 - np.argmin(x)) / (timeperiod - 1), raw=True)
    
    return aroon_up, aroon_down

def awesome_oscillator(high, low, fast_period=5, slow_period=34) -> pd.Series:
    """Awesome Oscillator (AO)
    Measures market momentum"""
    median_price = (high + low) / 2
    ao = sma(median_price, fast_period) - sma(median_price, slow_period)
    return ao

def keltner_channels(high, low, close, timeperiod=20, atr_multiplier=2) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Keltner Channels
    Volatility-based bands with EMA and ATR"""
    middle = ema(close, timeperiod)
    atr_val = atr(high, low, close, timeperiod)
    
    upper = middle + (atr_multiplier * atr_val)
    lower = middle - (atr_multiplier * atr_val)
    
    return upper, middle, lower

def pvt(close, volume) -> pd.Series:
    """Price Volume Trend
    Cumulation of volume weighted by relative price change"""
    pct_change = close.pct_change()
    pvt = (pct_change * volume).cumsum()
    return pvt

def vwap_bands(high, low, close, volume, timeperiod=20, stdev_multiplier=2) -> tuple[pd.Series, pd.Series, pd.Series]:
    """VWAP with Standard Deviation Bands
    Adds upper and lower bands to VWAP based on standard deviation"""
    typical_price = (high + low + close) / 3
    vwap_data = vwap(high, low, close, volume)
    
    # Calculate standard deviation of price from VWAP
    deviation = np.sqrt(((typical_price - vwap_data) ** 2).rolling(window=timeperiod).mean())
    
    upper_band = vwap_data + (deviation * stdev_multiplier)
    lower_band = vwap_data - (deviation * stdev_multiplier)
    
    return upper_band, vwap_data, lower_band

def elder_ray(high, low, close, timeperiod=13) -> tuple[pd.Series, pd.Series]:
    """Elder Ray
    Shows buying and selling pressure"""
    ema_val = ema(close, timeperiod)
    
    bull_power = high - ema_val  # Buying pressure
    bear_power = low - ema_val   # Selling pressure
    
    return bull_power, bear_power

def rvi(open_price, high, low, close, timeperiod=10) -> pd.Series:
    """Relative Vigor Index
    Compares closing price to opening price to determine market vigor"""
    # Calculate numerator: (close-open)
    numerator = close - open_price
    # Calculate denominator: (high-low)
    denominator = high - low
    
    # Smooth using SMA
    num_sma = numerator.rolling(window=timeperiod).sum()
    den_sma = denominator.rolling(window=timeperiod).sum()
    
    # Avoid division by zero
    den_sma = den_sma.replace(0, np.nan)
    
    # Calculate RVI
    rvi_value = num_sma / den_sma
    rvi_value = rvi_value.ffill().bfill()
    
    return rvi_value

def choppiness_index(high, low, close, timeperiod=14) -> pd.Series:
    """Choppiness Index
    Determines if market is choppy (trading sideways) or trending"""
    atr_sum = atr(high, low, close, 1).rolling(window=timeperiod).sum()
    high_low_range = high.rolling(window=timeperiod).max() - low.rolling(window=timeperiod).min()
    
    ci = 100 * np.log10(atr_sum / high_low_range) / np.log10(timeperiod)
    return ci

def mass_index(high, low, timeperiod=25, ema_period=9) -> pd.Series:
    """Mass Index
    Identifies potential trend reversals by analyzing the narrowing and widening of trading ranges"""
    high_low_range = high - low
    
    # Calculate single and double EMAs of the range
    ema1 = ema(high_low_range, ema_period)
    ema2 = ema(ema1, ema_period)
    
    # Calculate ratio of EMAs
    ema_ratio = ema1 / ema2
    
    # Calculate Mass Index as sum of EMA ratios over period
    mass = ema_ratio.rolling(window=timeperiod).sum()
    
    return mass

def volume_zone_oscillator(close, volume, short_period=14, long_period=28) -> pd.Series:
    """Volume Zone Oscillator (VZO)
    Measures volume pressure with both price and volume data"""
    # Calculate price change direction
    price_change = close.diff()
    
    # Separate volume into positive and negative based on price change
    positive_volume = volume.copy()
    positive_volume[price_change <= 0] = 0
    
    negative_volume = volume.copy()
    negative_volume[price_change >= 0] = 0
    
    # Calculate EMAs of positive and negative volume
    pos_vol_ema_short = ema(positive_volume, short_period)
    neg_vol_ema_short = ema(negative_volume, short_period)
    
    pos_vol_ema_long = ema(positive_volume, long_period)
    neg_vol_ema_long = ema(negative_volume, long_period)
    
    # Calculate VZO
    short_ratio = pos_vol_ema_short / (pos_vol_ema_short + neg_vol_ema_short)
    long_ratio = pos_vol_ema_long / (pos_vol_ema_long + neg_vol_ema_long)
    
    # Replace NaN values with 0.5 (neutral)
    short_ratio = short_ratio.fillna(0.5)
    long_ratio = long_ratio.fillna(0.5)
    
    # Calculate VZO (normalized to -100 to +100 range)
    vzo = 100 * (short_ratio - long_ratio)
    
    return vzo

def volatility_ratio(high, low, close, roc_period=14, atr_period=14) -> pd.Series:
    """Volatility Ratio
    Compares price momentum to volatility to identify potential trend changes"""
    # Calculate ROC (momentum)
    roc_val = roc(close, roc_period)
    
    # Calculate ATR (volatility)
    atr_val = atr(high, low, close, atr_period)
    
    # Calculate Volatility Ratio
    vol_ratio = np.abs(roc_val) / (atr_val / close * 100)
    
    # Handle infinite or NaN values
    vol_ratio = vol_ratio.replace([np.inf, -np.inf], np.nan).fillna(1)
    
    return vol_ratio

def hurst_exponent(series, max_lag=20) -> pd.Series:
    """Hurst Exponent
    Measures the long-term memory of a time series and its tendency to mean revert or trend.
    - H < 0.5: Mean-reverting series
    - H = 0.5: Random walk
    - H > 0.5: Trending series
    """
    lags = range(2, max_lag)
    result = pd.Series(index=series.index)
    
    # Convert to numpy array for faster processing
    series_values = series.values
    
    for i in range(len(series_values) - max_lag):
        window = series_values[i:i+max_lag]
        tau = []; lag_var = []
        
        for lag in lags:
            # Calculate price difference with lag
            pp = np.subtract(window[lag:], window[:-lag])
            # Calculate variance - avoid zeros
            var = np.var(pp)
            if var > 0:  # Only include valid variances
                tau.append(lag)
                lag_var.append(var)
        
        # Linear fit on log-log plot - avoid empty arrays
        if len(tau) > 1 and len(lag_var) > 1:
            try:
                m, _ = np.polyfit(np.log(tau), np.log(lag_var), 1)
                # Convert to Hurst exponent
                h = m / 2.0
                
                if i + max_lag < len(series_values):
                    result.iloc[i + max_lag] = h
            except:
                # Skip in case of numerical issues
                pass
    
    # Forward-fill to handle initial NaN values
    return result.ffill()

def zscore(series, timeperiod=20) -> pd.Series:
    """Z-Score
    Measures how many standard deviations a value is from the mean"""
    mean = series.rolling(window=timeperiod).mean()
    std = series.rolling(window=timeperiod).std()
    z_score = (series - mean) / std
    return z_score

def percent_rank(series, timeperiod=14) -> pd.Series:
    """Percent Rank
    Ranks the current value within its recent history on a 0-100 scale"""
    def percentile_rank(window):
        if len(window) == 0:
            return np.nan
        return percentileofscore(window, window.iloc[-1]) 
    
    return series.rolling(window=timeperiod).apply(percentile_rank, raw=False)

def historical_volatility(close, timeperiod=20, annualization=252) -> pd.Series:
    """Historical Volatility
    Estimates future volatility based on past price movements (annualized)"""
    # Calculate daily log returns
    log_ret = np.log(close / close.shift(1))
    
    # Calculate standard deviation of log returns
    hist_vol = log_ret.rolling(window=timeperiod).std() * np.sqrt(annualization)
    
    return hist_vol

def fractal_indicator(high, low, n=2) -> tuple[pd.Series, pd.Series]:
    """Williams Fractal Indicator
    Identifies potential support and resistance points (local highs and lows)"""
    # Initialize empty series for up and down fractals
    up_fractals = pd.Series(index=high.index, data=False)
    down_fractals = pd.Series(index=low.index, data=False)
    
    # Convert to numpy arrays for faster processing in loops
    high_values = high.values
    low_values = low.values
    
    # Identify bearish (up) fractals: Center candle high is highest among 2n+1 candles
    for i in range(n, len(high_values)-n):
        if high_values[i] == np.max(high_values[i-n:i+n+1]):
            up_fractals.iloc[i] = True
    
    # Identify bullish (down) fractals: Center candle low is lowest among 2n+1 candles
    for i in range(n, len(low_values)-n):
        if low_values[i] == np.min(low_values[i-n:i+n+1]):
            down_fractals.iloc[i] = True
    
    return up_fractals, down_fractals

def donchian_channel(high, low, timeperiod=20) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Donchian Channel
    Shows the highest high and lowest low over a given period"""
    upper = high.rolling(window=timeperiod).max()
    lower = low.rolling(window=timeperiod).min()
    middle = (upper + lower) / 2
    
    return upper, middle, lower

def price_cycle(close, cycle_period=20) -> pd.Series:
    """Price Cycle Oscillator
    Attempts to isolate the cyclical component of price movements"""
    # Apply bandpass filter to isolate cyclical component
    # Parameters tuned to the given cycle period
    b, a = signal.butter(2, [0.5/cycle_period, 2.0/cycle_period], 'bandpass')
    
    # Apply filter to close prices - use ffill instead of deprecated method parameter
    close_filled = close.ffill().bfill().values
    cycle = signal.filtfilt(b, a, close_filled)
    
    return pd.Series(cycle, index=close.index)

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
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    
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

def cmf(high, low, close, volume, timeperiod=20) -> pd.Series:
    """Chaikin Money Flow"""
    mfv = volume * ((close - low) - (high - close)) / (high - low)
    cmf = mfv.rolling(window=timeperiod).sum() / volume.rolling(window=timeperiod).sum()
    return cmf

def hma(series, timeperiod=16) -> pd.Series:
    """Hull Moving Average"""
    wma_half_period = wma(series, timeperiod=timeperiod//2)
    wma_full_period = wma(series, timeperiod=timeperiod)
    
    # HMA = WMA(2*WMA(period/2) - WMA(period), sqrt(period))
    sqrt_period = int(np.sqrt(timeperiod))
    return wma(2 * wma_half_period - wma_full_period, timeperiod=sqrt_period)

def wma(series, timeperiod=20) -> pd.Series:
    """Weighted Moving Average"""
    weights = np.arange(1, timeperiod + 1)
    sum_weights = weights.sum()
    
    # Pre-allocate the result array with NaNs
    result = pd.Series(np.nan, index=series.index)
    
    # Convert to numpy array for faster processing
    series_values = series.values
    
    for i in range(timeperiod - 1, len(series_values)):
        # Directly use numpy array slicing instead of Series.iloc
        window_vals = series_values[i - timeperiod + 1:i + 1]
        result.iloc[i] = np.sum(window_vals * weights) / sum_weights
    
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
    import sys
    sys.path.append(r"trading")
    import model_tools as mt
    
    print("Benchmarking technical indicators...")
    data = mt.fetch_data("BTC-USDT", 365, "5min", 0, kucoin=True)
    
    # Dictionary to store execution times
    execution_times = {}
    
    # First test prepare_data_classifier to see which sections are slowest
    print("\nBenchmarking prepare_data_classifier sections...")
    start_time = time.time()
    X, y = mt.prepare_data_classifier(data, lagged_length=20)
    print(f"Total prepare_data_classifier time: {time.time() - start_time:.4f} seconds")
    
    print("\nBenchmarking individual indicator functions...")
    # Test individual complex indicators that might be slow
    functions_to_test = [
        ("hurst_exponent", lambda: hurst_exponent(data['Close'])),
        ("fractal_indicator", lambda: fractal_indicator(data['High'], data['Low'])),
        ("ichimoku", lambda: ichimoku(data['High'], data['Low'], data['Close'])),
        ("psar", lambda: psar(data['High'].values, data['Low'].values)),
        ("supertrend", lambda: supertrend(data['High'], data['Low'], data['Close'])),
        ("kama", lambda: kama(data['Close'])),
        ("wma", lambda: wma(data['Close'])),
        ("price_cycle", lambda: price_cycle(data['Close'])),
        ("percent_rank", lambda: percent_rank(data['Close'])),
        ("adx", lambda: adx(data['High'], data['Low'], data['Close'])),
        ("identify_candlestick_patterns", lambda: identify_candlestick_patterns(
            data['Open'].values, data['High'].values, data['Low'].values, data['Close'].values
        ))
    ]
    
    # Run benchmarks for individual functions
    for name, func in functions_to_test:
        start_time = time.time()
        result = func()
        end_time = time.time()
        execution_time = end_time - start_time
        execution_times[name] = execution_time
        print(f"{name}: {execution_time:.4f} seconds")
    
    # Find slowest function
    slowest_func = max(execution_times.items(), key=lambda x: x[1])
    print(f"\nSLOWEST FUNCTION: {slowest_func[0]} took {slowest_func[1]:.4f} seconds")
    
    # Sort functions by execution time
    print("\nAll functions sorted by execution time (slowest to fastest):")
    sorted_funcs = sorted(execution_times.items(), key=lambda x: x[1], reverse=True)
    for name, time_taken in sorted_funcs:
        print(f"{name}: {time_taken:.4f} seconds")
    
    
    
    
    
