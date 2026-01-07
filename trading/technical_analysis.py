import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
from typing import Optional
from numba import njit

# cores

@njit(cache=True)
def _sma_core(values: np.ndarray, timeperiod: int) -> np.ndarray:
    """Core SMA calculation using Numba"""
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(timeperiod - 1, n):
        result[i] = np.mean(values[i - timeperiod + 1:i + 1])
    return result

@njit(cache=True)
def _ema_core(values: np.ndarray, timeperiod: int) -> np.ndarray:
    """Core EMA calculation using Numba"""
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float64)
    alpha = 2.0 / (timeperiod + 1.0)
    result[0] = values[0]
    for i in range(1, n):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result

@njit(cache=True)
def _rsi_core(values: np.ndarray, timeperiod: int) -> np.ndarray:
    """Core RSI calculation using Numba"""
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < timeperiod + 1:
        return result
    
    deltas = np.diff(values)
    
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    avg_gain = np.mean(gains[:timeperiod])
    avg_loss = np.mean(losses[:timeperiod])
    
    if avg_loss == 0:
        result[timeperiod] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[timeperiod] = 100.0 - (100.0 / (1.0 + rs))
    
    alpha = 1.0 / timeperiod
    for i in range(timeperiod + 1, n):
        gain = gains[i - 1] if i - 1 < len(gains) else 0.0
        loss = losses[i - 1] if i - 1 < len(losses) else 0.0
        avg_gain = alpha * gain + (1 - alpha) * avg_gain
        avg_loss = alpha * loss + (1 - alpha) * avg_loss
        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return result

@njit(cache=True)
def _atr_core(high: np.ndarray, low: np.ndarray, close: np.ndarray, timeperiod: int) -> np.ndarray:
    """Core ATR calculation using Numba"""
    n = len(high)
    tr = np.full(n, np.nan, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i - 1])
        tr3 = abs(low[i] - close[i - 1])
        tr[i] = max(tr1, max(tr2, tr3))
    return _sma_core(tr, timeperiod)

@njit(cache=True)
def _wma_core(values: np.ndarray, timeperiod: int) -> np.ndarray:
    """Core WMA calculation using Numba"""
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float64)
    sum_weights = timeperiod * (timeperiod + 1) / 2.0
    for i in range(timeperiod - 1, n):
        weighted_sum = 0.0
        for j in range(timeperiod):
            weight = j + 1
            weighted_sum += values[i - j] * weight
        result[i] = weighted_sum / sum_weights
    return result

@njit(cache=True)
def _cci_core(high: np.ndarray, low: np.ndarray, close: np.ndarray, timeperiod: int) -> np.ndarray:
    """Core CCI calculation using Numba"""
    n = len(high)
    tp = (high + low + close) / 3.0
    result = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(timeperiod - 1, n):
        window = tp[i - timeperiod + 1:i + 1]
        sma_val = np.mean(window)
        mad = np.mean(np.abs(window - sma_val))
        if mad > 0:
            result[i] = (tp[i] - sma_val) / (0.015 * mad)
    
    return result

@njit(cache=True)
def _aroon_core(high: np.ndarray, low: np.ndarray, timeperiod: int) -> tuple:
    """Core Aroon calculation using Numba"""
    n = len(high)
    aroon_up = np.full(n, np.nan, dtype=np.float64)
    aroon_down = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(timeperiod - 1, n):
        high_window = high[i - timeperiod + 1:i + 1]
        low_window = low[i - timeperiod + 1:i + 1]
        
        periods_since_high = timeperiod - 1 - np.argmax(high_window)
        periods_since_low = timeperiod - 1 - np.argmin(low_window)
        
        aroon_up[i] = 100.0 * (timeperiod - periods_since_high) / timeperiod
        aroon_down[i] = 100.0 * (timeperiod - periods_since_low) / timeperiod
    
    return aroon_up, aroon_down

@njit(cache=True)
def _stoch_core(high: np.ndarray, low: np.ndarray, close: np.ndarray, fastk_period: int) -> np.ndarray:
    """Core Stochastic %K calculation using Numba"""
    n = len(high)
    k = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(fastk_period - 1, n):
        high_window = high[i - fastk_period + 1:i + 1]
        low_window = low[i - fastk_period + 1:i + 1]
        highest = np.max(high_window)
        lowest = np.min(low_window)
        if highest != lowest:
            k[i] = 100.0 * ((close[i] - lowest) / (highest - lowest))
    
    return k

@njit(cache=True)
def _bbands_core(values: np.ndarray, timeperiod: int, devup: float, devdn: float) -> tuple:
    """Core Bollinger Bands calculation using Numba"""
    n = len(values)
    upper = np.full(n, np.nan, dtype=np.float64)
    middle = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(timeperiod - 1, n):
        window = values[i - timeperiod + 1:i + 1]
        mean_val = np.mean(window)
        std_val = np.std(window)
        middle[i] = mean_val
        upper[i] = mean_val + (std_val * devup)
        lower[i] = mean_val - (std_val * devdn)
    
    return upper, middle, lower

@njit(cache=True)
def _adx_core(high: np.ndarray, low: np.ndarray, close: np.ndarray, timeperiod: int) -> tuple:
    """Core ADX calculation using Numba"""
    n = len(high)
    tr = np.full(n, np.nan, dtype=np.float64)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i - 1])
        tr3 = abs(low[i] - close[i - 1])
        tr[i] = max(tr1, max(tr2, tr3))
        
        high_diff = high[i] - high[i - 1]
        low_diff = low[i - 1] - low[i]
        
        if high_diff > 0 and high_diff > low_diff:
            plus_dm[i] = high_diff
        if low_diff > 0 and low_diff > high_diff:
            minus_dm[i] = low_diff
    
    tr_smooth = _ema_core(tr, timeperiod)
    plus_dm_smooth = _ema_core(plus_dm, timeperiod)
    minus_dm_smooth = _ema_core(minus_dm, timeperiod)
    
    plus_di = np.full(n, np.nan, dtype=np.float64)
    minus_di = np.full(n, np.nan, dtype=np.float64)
    dx = np.full(n, np.nan, dtype=np.float64)
    adx = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(timeperiod, n):
        if tr_smooth[i] > 0:
            plus_di[i] = 100.0 * (plus_dm_smooth[i] / tr_smooth[i])
            minus_di[i] = 100.0 * (minus_dm_smooth[i] / tr_smooth[i])
            
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 0:
                di_diff = abs(plus_di[i] - minus_di[i])
                dx[i] = 100.0 * (di_diff / di_sum)
    
    adx = _ema_core(dx, timeperiod)
    
    return adx, plus_di, minus_di

@njit(cache=True)
def _obv_core(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Core OBV calculation using Numba"""
    n = len(close)
    obv = np.zeros(n, dtype=np.float64)
    obv[0] = volume[0]
    for i in range(1, n):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]
    return obv

@njit(cache=True)
def _supertrend_core(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int, multiplier: float) -> tuple:
    """Core SuperTrend calculation using Numba"""
    n = len(close)
    atr_vals = _atr_core(high, low, close, period)
    
    basic_upper = np.zeros(n, dtype=np.float64)
    basic_lower = np.zeros(n, dtype=np.float64)
    final_upper = np.zeros(n, dtype=np.float64)
    final_lower = np.zeros(n, dtype=np.float64)
    supertrend = np.zeros(n, dtype=np.float64)
    
    hl_avg = (high + low) / 2.0
    for i in range(n):
        if not np.isnan(atr_vals[i]):
            basic_upper[i] = hl_avg[i] + (multiplier * atr_vals[i])
            basic_lower[i] = hl_avg[i] - (multiplier * atr_vals[i])
        else:
            basic_upper[i] = hl_avg[i]
            basic_lower[i] = hl_avg[i]
    
    final_upper[0] = basic_upper[0]
    final_lower[0] = basic_lower[0]
    supertrend[0] = 1.0
    
    for i in range(1, n):
        if close[i - 1] > final_upper[i - 1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = min(basic_upper[i], final_upper[i - 1])
        
        if close[i - 1] < final_lower[i - 1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = max(basic_lower[i], final_lower[i - 1])
        
        if close[i] > final_upper[i]:
            supertrend[i] = 1.0
        elif close[i] < final_lower[i]:
            supertrend[i] = -1.0
        else:
            supertrend[i] = supertrend[i - 1]
    
    supertrend_line = np.where(supertrend == 1.0, final_lower, final_upper)
    return supertrend, supertrend_line

@njit(cache=True)
def _kama_core(values: np.ndarray, er_period: int, fast_period: int, slow_period: int) -> np.ndarray:
    """Core KAMA calculation using Numba"""
    n = len(values)
    kama = np.zeros(n, dtype=np.float64)
    kama[0] = values[0]
    
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    for i in range(1, n):
        if i >= er_period:
            change = abs(values[i] - values[i - er_period])
            volatility = 0.0
            for j in range(i - er_period + 1, i + 1):
                volatility += abs(values[j] - values[j - 1])
            
            if volatility > 0:
                er = change / volatility
            else:
                er = 0.0
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        else:
            sc = slow_sc
        
        kama[i] = kama[i - 1] + sc * (values[i] - kama[i - 1])
    
    return kama

@njit(cache=True)
def _psar_core(high: np.ndarray, low: np.ndarray, acceleration_start: float, acceleration_step: float, max_acceleration: float) -> np.ndarray:
    """Core PSAR calculation using Numba"""
    n = len(high)
    psar = np.zeros(n, dtype=np.float64)
    trend = np.ones(n, dtype=np.int32)
    ep = np.zeros(n, dtype=np.float64)
    
    psar[0] = low[0]
    ep[0] = high[0]
    af = acceleration_start
    
    for i in range(1, n):
        if trend[i - 1] == 1:
            diff = ep[i - 1] - psar[i - 1]
            psar[i] = psar[i - 1] + af * diff
        else:
            diff = psar[i - 1] - ep[i - 1]
            psar[i] = psar[i - 1] - af * diff
        
        if trend[i - 1] == 1:
            if low[i] < psar[i]:
                trend[i] = -1
                psar[i] = ep[i - 1]
                ep[i] = low[i]
                af = acceleration_start
            else:
                trend[i] = 1
                if high[i] > ep[i - 1]:
                    ep[i] = high[i]
                    af = min(af + acceleration_step, max_acceleration)
                else:
                    ep[i] = ep[i - 1]
        else:
            if high[i] > psar[i]:
                trend[i] = 1
                psar[i] = ep[i - 1]
                ep[i] = high[i]
                af = acceleration_start
            else:
                trend[i] = -1
                if low[i] < ep[i - 1]:
                    ep[i] = low[i]
                    af = min(af + acceleration_step, max_acceleration)
                else:
                    ep[i] = ep[i - 1]
    
    return psar

@njit(cache=True)
def _zscore_core(values: np.ndarray, timeperiod: int) -> np.ndarray:
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(timeperiod - 1, n):
        window = values[i - timeperiod + 1:i + 1]
        mean_val = np.mean(window)
        std_val = np.std(window)
        if std_val > 0:
            result[i] = (values[i] - mean_val) / std_val
    return result

@njit(cache=True)
def _kalman_filter_core(values: np.ndarray, process_noise: float, measurement_noise: float) -> np.ndarray:
    n = len(values)
    x_estimate = np.zeros(n, dtype=np.float64)
    P = 1.0
    Q = process_noise
    R = measurement_noise
    
    x_estimate[0] = values[0]
    for i in range(1, n):
        x_prediction = x_estimate[i - 1]
        P_prediction = P + Q
        K = P_prediction / (P_prediction + R)
        x_estimate[i] = x_prediction + K * (values[i] - x_prediction)
        P = (1 - K) * P_prediction
    
    return x_estimate

@njit(cache=True)
def _mfi_core(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, timeperiod: int) -> np.ndarray:
    n = len(high)
    typical_price = (high + low + close) / 3.0
    money_flow = typical_price * volume
    
    positive_flow = np.zeros(n, dtype=np.float64)
    negative_flow = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        if typical_price[i] > typical_price[i - 1]:
            positive_flow[i] = money_flow[i]
        elif typical_price[i] < typical_price[i - 1]:
            negative_flow[i] = money_flow[i]
    
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(timeperiod - 1, n):
        pos_sum = np.sum(positive_flow[i - timeperiod + 1:i + 1])
        neg_sum = np.sum(negative_flow[i - timeperiod + 1:i + 1])
        if neg_sum > 0:
            result[i] = 100.0 - (100.0 / (1.0 + pos_sum / neg_sum))
        else:
            result[i] = 100.0
    
    return result

@njit(cache=True)
def _cmf_core(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, timeperiod: int) -> np.ndarray:
    n = len(high)
    mfv = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        hl_range = high[i] - low[i]
        if hl_range > 0:
            mfv[i] = volume[i] * ((close[i] - low[i]) - (high[i] - close[i])) / hl_range
    
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(timeperiod - 1, n):
        mfv_sum = np.sum(mfv[i - timeperiod + 1:i + 1])
        vol_sum = np.sum(volume[i - timeperiod + 1:i + 1])
        if vol_sum > 0:
            result[i] = mfv_sum / vol_sum
    
    return result

@njit(cache=True)
def _vwap_core(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    n = len(high)
    typical_price = (high + low + close) / 3.0
    vwap = np.zeros(n, dtype=np.float64)
    
    cum_price_volume = 0.0
    cum_volume = 0.0
    
    for i in range(n):
        cum_price_volume += typical_price[i] * volume[i]
        cum_volume += volume[i]
        if cum_volume > 0:
            vwap[i] = cum_price_volume / cum_volume
    
    return vwap

@njit(cache=True)
def _pvt_core(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    n = len(close)
    pvt = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        if close[i - 1] > 0:
            pct_change = (close[i] - close[i - 1]) / close[i - 1]
            pvt[i] = pvt[i - 1] + (pct_change * volume[i])
        else:
            pvt[i] = pvt[i - 1]
    
    return pvt

# wrappers

def heikin_ashi_transform(data: pd.DataFrame) -> pd.DataFrame:
    """Heikin Ashi Transform"""
    ha_data = data.copy()
    ha_data['Open'] = (ha_data['Open'] + ha_data['Close'].shift(1)) / 2
    ha_data['Close'] = (ha_data['Open'] + ha_data['High'] + ha_data['Low'] + ha_data['Close']) / 4
    ha_data['High'] = ha_data[['Open', 'Close', 'High']].max(axis=1)
    ha_data['Low'] = ha_data[['Open', 'Close', 'Low']].min(axis=1)
    return ha_data

def sma(series: pd.Series, timeperiod: int = 20) -> pd.Series:
    """Simple Moving Average"""
    values = np.asarray(series.values, dtype=np.float64)
    result = _sma_core(values, timeperiod)
    return pd.Series(result, index=series.index)

def ema(series: pd.Series, timeperiod: int = 20) -> pd.Series:
    """Exponential Moving Average"""
    values = np.asarray(series.values, dtype=np.float64)
    result = _ema_core(values, timeperiod)
    return pd.Series(result, index=series.index)

def vwma(series: pd.Series, volume: pd.Series, timeperiod: int = 20) -> pd.Series:
    """Volume Weighted Moving Average"""
    series_vals = np.asarray(series.values, dtype=np.float64)
    volume_vals = np.asarray(volume.values, dtype=np.float64)
    n = len(series_vals)
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(timeperiod - 1, n):
        window_series = series_vals[i - timeperiod + 1:i + 1]
        window_vol = volume_vals[i - timeperiod + 1:i + 1]
        result[i] = np.sum(window_series * window_vol) / np.sum(window_vol)
    return pd.Series(result, index=series.index)

def rsi(series: pd.Series, timeperiod: int = 14) -> pd.Series:
    """Relative Strength Index"""
    values = np.asarray(series.values, dtype=np.float64)
    result = _rsi_core(values, timeperiod)
    return pd.Series(result, index=series.index)

def macd(series: pd.Series, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9) -> tuple[pd.Series, pd.Series]:
    """Moving Average Convergence Divergence"""
    exp1 = ema(series, fastperiod)
    exp2 = ema(series, slowperiod)
    macd = exp1 - exp2
    signal = ema(macd, signalperiod)
    hist = macd - signal
    return macd, signal, hist

def macd_dema(series: pd.Series, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9) -> tuple[pd.Series, pd.Series]:
    """Moving Average Convergence Divergence with DEMA"""
    exp1 = dema(series, fastperiod)
    exp2 = dema(series, slowperiod)
    macd = exp1 - exp2
    signal = dema(macd, signalperiod)
    hist = macd - signal
    return macd, signal, hist

def bbands(series: pd.Series, timeperiod: int = 20, devup: int = 2, devdn: int = 2) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands"""
    values = np.asarray(series.values, dtype=np.float64)
    upper_vals, middle_vals, lower_vals = _bbands_core(values, timeperiod, float(devup), float(devdn))
    return pd.Series(upper_vals, index=series.index), pd.Series(middle_vals, index=series.index), pd.Series(lower_vals, index=series.index)

def stoch(high: pd.Series, low: pd.Series, close: pd.Series, fastk_period: int = 14, slowk_period: int = 3, slowd_period: int = 3) -> tuple[pd.Series, pd.Series]:
    """Stochastic Oscillator"""
    high_vals = np.asarray(high.values, dtype=np.float64)
    low_vals = np.asarray(low.values, dtype=np.float64)
    close_vals = np.asarray(close.values, dtype=np.float64)
    k_vals = _stoch_core(high_vals, low_vals, close_vals, fastk_period)
    k = pd.Series(k_vals, index=close.index)
    d = sma(k, slowk_period)
    return k, d

def atr(high: pd.Series, low: pd.Series, close: pd.Series, timeperiod: int = 14) -> pd.Series:
    """Average True Range"""
    high_vals = np.asarray(high.values, dtype=np.float64)
    low_vals = np.asarray(low.values, dtype=np.float64)
    close_vals = np.asarray(close.values, dtype=np.float64)
    result = _atr_core(high_vals, low_vals, close_vals, timeperiod)
    return pd.Series(result, index=high.index)

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On Balance Volume"""
    close_vals = np.asarray(close.values, dtype=np.float64)
    volume_vals = np.asarray(volume.values, dtype=np.float64)
    result = _obv_core(close_vals, volume_vals)
    return pd.Series(result, index=close.index)

def cci(high: pd.Series, low: pd.Series, close: pd.Series, timeperiod: int = 20) -> pd.Series:
    """Commodity Channel Index"""
    high_vals = np.asarray(high.values, dtype=np.float64)
    low_vals = np.asarray(low.values, dtype=np.float64)
    close_vals = np.asarray(close.values, dtype=np.float64)
    result = _cci_core(high_vals, low_vals, close_vals, timeperiod)
    return pd.Series(result, index=high.index)

def adx(high: pd.Series, low: pd.Series, close: pd.Series, timeperiod: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Average Directional Index"""
    high_vals = np.asarray(high.values, dtype=np.float64)
    low_vals = np.asarray(low.values, dtype=np.float64)
    close_vals = np.asarray(close.values, dtype=np.float64)
    adx_vals, plus_di_vals, minus_di_vals = _adx_core(high_vals, low_vals, close_vals, timeperiod)
    return pd.Series(adx_vals, index=high.index), pd.Series(plus_di_vals, index=high.index), pd.Series(minus_di_vals, index=high.index)

@njit(cache=True)
def _log_return_core(values: np.ndarray) -> np.ndarray:
    """Core log return calculation"""
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
        if values[i - 1] > 0:
            result[i] = np.log(values[i] / values[i - 1])
    return result

def log_return(series: pd.Series) -> pd.Series:
    """Log Returns"""
    values = np.asarray(series.values, dtype=np.float64)
    result = _log_return_core(values)
    return pd.Series(result, index=series.index)

def dpo(series: pd.Series, timeperiod: int = 20) -> pd.Series:
    """Detrended Price Oscillator
    A non-directional indicator that removes trend from price to identify cycles"""
    shift = int(timeperiod/2) + 1
    shifted = series.shift(shift)
    sma_shifted = sma(shifted, timeperiod)
    return series - sma_shifted

def dema(series: pd.Series, timeperiod: int = 20) -> pd.Series:
    """Double Exponential Moving Average"""
    ema1 = ema(series, timeperiod)
    ema2 = ema(ema1, timeperiod)
    return 2 * ema1 - ema2

def tema(series: pd.Series, timeperiod: int = 20) -> pd.Series:
    """Triple Exponential Moving Average"""
    ema1 = ema(series, timeperiod)
    ema2 = ema(ema1, timeperiod)
    ema3 = ema(ema2, timeperiod)
    return 3 * ema1 - 3 * ema2 + ema3

def fisher_transform(series: pd.Series, timeperiod: int = 10) -> pd.Series:
    """Ehlers Fisher Transform"""
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

def aroon(high: pd.Series, low: pd.Series, timeperiod: int = 14) -> tuple[pd.Series, pd.Series]:
    """Canonical Aroon (matches TA-Lib / TradingView)"""
    if timeperiod < 2:
        raise ValueError("timeperiod must be >= 2")
    n = len(high)
    if n < timeperiod:
        nan = pd.Series(np.nan, index=high.index)
        return nan, nan
    high_vals = np.asarray(high.values, dtype=np.float64)
    low_vals = np.asarray(low.values, dtype=np.float64)
    aroon_up_vals, aroon_down_vals = _aroon_core(high_vals, low_vals, timeperiod)
    return pd.Series(aroon_up_vals, index=high.index), pd.Series(aroon_down_vals, index=low.index)

def awesome_oscillator(high: pd.Series, low: pd.Series, fast_period: int = 5, slow_period: int = 34) -> pd.Series:
    """Awesome Oscillator (AO)"""
    median_price = (high + low) / 2
    ao = sma(median_price, fast_period) - sma(median_price, slow_period)
    return ao

def keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series, timeperiod: int = 20, atr_multiplier: int = 2) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Keltner Channels"""
    middle = ema(close, timeperiod)
    atr_val = atr(high, low, close, timeperiod)
    
    upper = middle + (atr_multiplier * atr_val)
    lower = middle - (atr_multiplier * atr_val)
    
    return upper, middle, lower

def pvt(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Price Volume Trend"""
    close_vals = np.asarray(close.values, dtype=np.float64)
    volume_vals = np.asarray(volume.values, dtype=np.float64)
    result = _pvt_core(close_vals, volume_vals)
    return pd.Series(result, index=close.index)

def vwap_bands(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, timeperiod: int = 20, stdev_multiplier: int = 2) -> tuple[pd.Series, pd.Series, pd.Series]:
    """VWAP with Standard Deviation Bands"""
    typical_price = (high + low + close) / 3
    vwap_data = vwap(high, low, close, volume)
    
    # Calculate standard deviation of price from VWAP
    deviation = np.sqrt(((typical_price - vwap_data) ** 2).rolling(window=timeperiod).mean())
    
    upper_band = vwap_data + (deviation * stdev_multiplier)
    lower_band = vwap_data - (deviation * stdev_multiplier)
    
    return upper_band, vwap_data, lower_band

def elder_ray(high: pd.Series, low: pd.Series, close: pd.Series, timeperiod: int = 13) -> tuple[pd.Series, pd.Series]:
    """Elder Ray"""
    ema_val = ema(close, timeperiod)
    
    bull_power = high - ema_val  # Buying pressure
    bear_power = low - ema_val   # Selling pressure
    
    return bull_power, bear_power

def rvi(open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, timeperiod: int = 10) -> pd.Series:
    """Relative Vigor Index"""
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

def choppiness_index(high: pd.Series, low: pd.Series, close: pd.Series, timeperiod: int = 14) -> pd.Series:
    """Choppiness Index"""
    atr_sum = atr(high, low, close, 1).rolling(window=timeperiod).sum()
    high_low_range = high.rolling(window=timeperiod).max() - low.rolling(window=timeperiod).min()
    
    ci = 100 * np.log10(atr_sum / high_low_range) / np.log10(timeperiod)
    return ci

def mass_index(high: pd.Series, low: pd.Series, timeperiod: int = 25, ema_period: int = 9) -> pd.Series:
    """Mass Index"""
    high_low_range = high - low
    
    # Calculate single and double EMAs of the range
    ema1 = ema(high_low_range, ema_period)
    ema2 = ema(ema1, ema_period)
    
    # Calculate ratio of EMAs
    ema_ratio = ema1 / ema2
    
    # Calculate Mass Index as sum of EMA ratios over period
    mass = ema_ratio.rolling(window=timeperiod).sum()
    
    return mass

def volume_zone_oscillator(close: pd.Series, volume: pd.Series, short_period: int = 14, long_period: int = 28) -> pd.Series:
    """Volume Zone Oscillator (VZO)"""
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

def volatility_ratio(high: pd.Series, low: pd.Series, close: pd.Series, roc_period: int = 14, atr_period: int = 14) -> pd.Series:
    """Volatility Ratio"""
    # Calculate ROC (momentum)
    roc_val = roc(close, roc_period)
    
    # Calculate ATR (volatility)
    atr_val = atr(high, low, close, atr_period)
    
    # Calculate Volatility Ratio
    vol_ratio = np.abs(roc_val) / (atr_val / close * 100)
    
    # Handle infinite or NaN values
    vol_ratio = vol_ratio.replace([np.inf, -np.inf], np.nan).fillna(1)
    
    return vol_ratio

def hurst_exponent(series: pd.Series, max_lag: int = 20) -> pd.Series:
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

def zscore(series: pd.Series, timeperiod: int = 20) -> pd.Series:
    """Rolling Z-Score"""
    values = np.asarray(series.values, dtype=np.float64)
    result = _zscore_core(values, timeperiod)
    return pd.Series(result, index=series.index)

def volatility(series: pd.Series, timeperiod: int = 20) -> pd.Series:
    """Volatility"""
    return series.rolling(window=timeperiod).std()

def get_date_range_periods(df: pd.DataFrame, unit: str = 'days') -> float:
    """
    Get the time span of a DataFrame in specified units.
    
    Args:
        df: DataFrame with datetime index or 'Datetime' column
        unit: 'days', 'hours', 'minutes', 'weeks'
    
    Returns:
        float: Time span in the specified unit
    """
    # Get datetime series
    if 'Datetime' in df.columns:
        datetime_series = pd.to_datetime(df['Datetime'])
    elif isinstance(df.index, pd.DatetimeIndex):
        datetime_series = df.index
    else:
        raise ValueError("DataFrame must have 'Datetime' column or DatetimeIndex")
    
    # Calculate total time span
    start_date = datetime_series.min()
    end_date = datetime_series.max()
    total_timedelta = end_date - start_date
    
    # Convert to requested unit
    total_seconds = total_timedelta.total_seconds()
    
    if unit == 'days':
        return total_seconds / (24 * 3600)
    elif unit == 'hours':
        return total_seconds / 3600
    elif unit == 'minutes':
        return total_seconds / 60
    elif unit == 'weeks':
        return total_seconds / (7 * 24 * 3600)
    else:
        raise ValueError("Unit must be 'days', 'hours', 'minutes', or 'weeks'")

def historical_volatility(close: pd.Series, output_period: float = -1, df: Optional[pd.DataFrame] = None) -> pd.Series:
    """
    Historical Volatility with optional automatic period detection
    Estimates future volatility based on past price movements
    
    Args:
        close: Close price series
        output_period: Output scaling period, -1 for no scaling (default 365 for annualization)
        df: DataFrame for automatic period detection (optional)
    """
    # Auto-detect input period if DataFrame provided
    if output_period == -1:
        input_period = output_period
    else:
        input_period = get_date_range_periods(df, unit='days')
    
    # Calculate daily log returns
    log_ret = np.log(close / close.shift(1))
    
    # Calculate standard deviation of log returns
    hist_vol = log_ret.std() * np.sqrt(output_period / input_period)
    
    return hist_vol

def fractal_indicator(high: pd.Series, low: pd.Series, n: int = 2) -> tuple[pd.Series, pd.Series]:
    """Williams Fractal Indicator"""
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

def donchian_channel(high: pd.Series, low: pd.Series, timeperiod: int = 20) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Donchian Channel"""
    upper = high.rolling(window=timeperiod).max()
    lower = low.rolling(window=timeperiod).min()
    middle = (upper + lower) / 2
    
    return upper, middle, lower

def price_cycle(close: pd.Series, cycle_period: int = 20) -> pd.Series:
    """Price Cycle Oscillator"""
    # Normalize frequencies to [0, 1] range (1 = Nyquist frequency)
    # Clamp frequencies to valid range
    low_freq = max(0.01, min(0.49, 0.5 / cycle_period))
    high_freq = max(low_freq + 0.01, min(0.99, 2.0 / cycle_period))
    
    b, a = signal.butter(2, [low_freq, high_freq], 'bandpass')

    close_filled = close.ffill().bfill().values
    cycle = signal.lfilter(b, a, close_filled)
    
    return pd.Series(cycle, index=close.index)

@njit(cache=True)
def _stddev_core(values: np.ndarray, timeperiod: int) -> np.ndarray:
    """Core standard deviation calculation"""
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(timeperiod - 1, n):
        window = values[i - timeperiod + 1:i + 1]
        result[i] = np.std(window)
    return result

def stddev(series: pd.Series, timeperiod: int = 20) -> pd.Series:
    """Standard Deviation"""
    values = np.asarray(series.values, dtype=np.float64)
    result = _stddev_core(values, timeperiod)
    return pd.Series(result, index=series.index)

@njit(cache=True)
def _roc_core(values: np.ndarray, timeperiod: int) -> np.ndarray:
    """Core rate of change calculation"""
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(timeperiod, n):
        if values[i - timeperiod] > 0:
            result[i] = ((values[i] - values[i - timeperiod]) / values[i - timeperiod]) * 100.0
    return result

def roc(series: pd.Series, timeperiod: int = 10) -> pd.Series:
    """Rate of Change"""
    values = np.asarray(series.values, dtype=np.float64)
    result = _roc_core(values, timeperiod)
    return pd.Series(result, index=series.index)

@njit(cache=True)
def _mom_core(values: np.ndarray, timeperiod: int) -> np.ndarray:
    """Core momentum calculation"""
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(timeperiod, n):
        result[i] = values[i] - values[i - timeperiod]
    return result

def mom(series: pd.Series, timeperiod: int = 10) -> pd.Series:
    """Momentum"""
    values = np.asarray(series.values, dtype=np.float64)
    result = _mom_core(values, timeperiod)
    return pd.Series(result, index=series.index)

@njit(cache=True)
def _willr_core(high: np.ndarray, low: np.ndarray, close: np.ndarray, timeperiod: int) -> np.ndarray:
    """Core Williams %R calculation"""
    n = len(high)
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(timeperiod - 1, n):
        high_window = high[i - timeperiod + 1:i + 1]
        low_window = low[i - timeperiod + 1:i + 1]
        highest = np.max(high_window)
        lowest = np.min(low_window)
        if highest != lowest:
            result[i] = -100.0 * (highest - close[i]) / (highest - lowest)
    return result

def willr(high: pd.Series, low: pd.Series, close: pd.Series, timeperiod: int = 14) -> pd.Series:
    """Williams %R"""
    high_vals = np.asarray(high.values, dtype=np.float64)
    low_vals = np.asarray(low.values, dtype=np.float64)
    close_vals = np.asarray(close.values, dtype=np.float64)
    result = _willr_core(high_vals, low_vals, close_vals, timeperiod)
    return pd.Series(result, index=close.index)

def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, timeperiod: int = 14) -> pd.Series:
    """Money Flow Index"""
    high_vals = np.asarray(high.values, dtype=np.float64)
    low_vals = np.asarray(low.values, dtype=np.float64)
    close_vals = np.asarray(close.values, dtype=np.float64)
    volume_vals = np.asarray(volume.values, dtype=np.float64)
    result = _mfi_core(high_vals, low_vals, close_vals, volume_vals, timeperiod)
    return pd.Series(result, index=close.index)

def kama(series: pd.Series, er_period: int = 10, fast_period: int = 2, slow_period: int = 30) -> pd.Series:
    """Kaufman Adaptive Moving Average"""
    values = np.asarray(series.values, dtype=np.float64)
    result = _kama_core(values, er_period, fast_period, slow_period)
    return pd.Series(result, index=series.index)

def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Volume Weighted Average Price"""
    high_vals = np.asarray(high.values, dtype=np.float64)
    low_vals = np.asarray(low.values, dtype=np.float64)
    close_vals = np.asarray(close.values, dtype=np.float64)
    volume_vals = np.asarray(volume.values, dtype=np.float64)
    result = _vwap_core(high_vals, low_vals, close_vals, volume_vals)
    return pd.Series(result, index=close.index)

def supertrend(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14, multiplier: int = 3) -> tuple[pd.Series, pd.Series]:
    """SuperTrend indicator"""
    high_vals = np.asarray(high.values, dtype=np.float64)
    low_vals = np.asarray(low.values, dtype=np.float64)
    close_vals = np.asarray(close.values, dtype=np.float64)
    supertrend_vals, supertrend_line_vals = _supertrend_core(high_vals, low_vals, close_vals, period, float(multiplier))
    return pd.Series(supertrend_vals, index=close.index), pd.Series(supertrend_line_vals, index=close.index)

def tsi(close: pd.Series, long_period: int = 25, short_period: int = 13, signal_period: int = 13) -> tuple[pd.Series, pd.Series]:
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

def cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, timeperiod: int = 20) -> pd.Series:
    """Chaikin Money Flow"""
    high_vals = np.asarray(high.values, dtype=np.float64)
    low_vals = np.asarray(low.values, dtype=np.float64)
    close_vals = np.asarray(close.values, dtype=np.float64)
    volume_vals = np.asarray(volume.values, dtype=np.float64)
    result = _cmf_core(high_vals, low_vals, close_vals, volume_vals, timeperiod)
    return pd.Series(result, index=close.index)

def hma(series: pd.Series, timeperiod: int = 16) -> pd.Series:
    """Hull Moving Average"""
    wma_half_period = wma(series, timeperiod=timeperiod//2)
    wma_full_period = wma(series, timeperiod=timeperiod)
    
    # HMA = WMA(2*WMA(period/2) - WMA(period), sqrt(period))
    sqrt_period = int(np.sqrt(timeperiod))
    return wma(2 * wma_half_period - wma_full_period, timeperiod=sqrt_period)

def wma(series: pd.Series, timeperiod: int = 20) -> pd.Series:
    """Weighted Moving Average"""
    values = np.asarray(series.values, dtype=np.float64)
    result = _wma_core(values, timeperiod)
    return pd.Series(result, index=series.index)

def ichimoku(high: pd.Series, low: pd.Series, close: pd.Series, tenkan_period: int = 9, kijun_period: int = 26, senkou_period: int = 52, chikou_period: int = 26) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Ichimoku Cloud"""
    tenkan_sen = (high.rolling(window=tenkan_period).max() + low.rolling(window=tenkan_period).min()) / 2
    
    kijun_sen = (high.rolling(window=kijun_period).max() + low.rolling(window=kijun_period).min()) / 2
    
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)
    
    senkou_span_b = ((high.rolling(window=senkou_period).max() + low.rolling(window=senkou_period).min()) / 2).shift(kijun_period)
    
    chikou_span = close.shift(-chikou_period)
    
    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

def ppo(series: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Percentage Price Oscillator"""
    fast_ema = ema(series, timeperiod=fast_period)
    slow_ema = ema(series, timeperiod=slow_period)
    
    ppo = ((fast_ema - slow_ema) / slow_ema) * 100
    signal = ema(ppo, timeperiod=signal_period)
    histogram = ppo - signal
    
    return ppo, signal, histogram

def aobv(close: pd.Series, volume: pd.Series, fast_period: int = 5, slow_period: int = 10) -> tuple[pd.Series, pd.Series]:
    """Adaptive On Balance Volume - OBV with smoothing and signal line"""
    # Calculate OBV
    on_balance_volume = obv(close, volume)
    
    # Generate fast and slow EMAs of OBV
    fast_obv = ema(on_balance_volume, timeperiod=fast_period)
    slow_obv = ema(on_balance_volume, timeperiod=slow_period)
    
    # Signal: fast OBV - slow OBV (like MACD with OBV)
    signal = fast_obv - slow_obv
    
    return on_balance_volume, signal

def psar(high: np.ndarray, low: np.ndarray, acceleration_start: float = 0.02, acceleration_step: float = 0.02, max_acceleration: float = 0.2) -> np.ndarray:
    """Parabolic SAR"""
    high_vals = np.asarray(high, dtype=np.float64)
    low_vals = np.asarray(low, dtype=np.float64)
    return _psar_core(high_vals, low_vals, acceleration_start, acceleration_step, max_acceleration)

def kalman_filter(series: pd.Series, process_noise: float = 0.01, measurement_noise: float = 0.1) -> pd.Series:
    """Kalman Filter"""
    values = np.asarray(series.values, dtype=np.float64)
    result = _kalman_filter_core(values, process_noise, measurement_noise)
    return pd.Series(result, index=series.index)

def identify_candlestick_patterns(open_prices: np.ndarray, high_prices: np.ndarray, low_prices: np.ndarray, close_prices: np.ndarray, patterns: Optional[list[str]] = None) -> pd.DataFrame:
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

def get_candlestick_patterns(df: pd.DataFrame, patterns: Optional[list[str]] = None) -> pd.DataFrame:
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