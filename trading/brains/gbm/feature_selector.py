import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import trading.technical_analysis as ta
import faulthandler
faulthandler.enable()

def generate_all_features(data: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=data.index)
    
    close = data['Close']
    high = data['High']
    low = data['Low']
    open_price = data['Open']
    volume = data['Volume']
    
    print("Generating features...")
    
    # Moving Averages
    for period in [5, 10, 20, 50, 100, 200]:
        features[f'sma_{period}'] = ta.sma(close, timeperiod=period)
        features[f'ema_{period}'] = ta.ema(close, timeperiod=period)
        features[f'wma_{period}'] = ta.wma(close, timeperiod=period)
        features[f'dema_{period}'] = ta.dema(close, timeperiod=period)
        features[f'tema_{period}'] = ta.tema(close, timeperiod=period)
        features[f'hma_{period}'] = ta.hma(close, timeperiod=period)
        features[f'kama_{period}'] = ta.kama(close, er_period=10, fast_period=2, slow_period=30)
    
    # Price-based indicators
    features['rsi_14'] = ta.rsi(close, timeperiod=14)
    features['rsi_21'] = ta.rsi(close, timeperiod=21)
    features['zscore_20'] = ta.zscore(close, timeperiod=20)
    features['zscore_50'] = ta.zscore(close, timeperiod=50)
    features['log_return'] = ta.log_return(close)
    features['roc_10'] = ta.roc(close, timeperiod=10)
    features['roc_20'] = ta.roc(close, timeperiod=20)
    features['mom_10'] = ta.mom(close, timeperiod=10)
    features['mom_20'] = ta.mom(close, timeperiod=20)
    features['volatility_20'] = ta.volatility(close, timeperiod=20)
    features['stddev_20'] = ta.stddev(close, timeperiod=20)
    features['stddev_50'] = ta.stddev(close, timeperiod=50)
    
    macd_line, macd_signal, macd_hist = ta.macd(close)
    features['macd_line'] = macd_line
    features['macd_signal'] = macd_signal
    features['macd_hist'] = macd_hist
    
    macd_dema_line, macd_dema_signal, macd_dema_hist = ta.macd_dema(close)
    features['macd_dema_line'] = macd_dema_line
    features['macd_dema_signal'] = macd_dema_signal
    features['macd_dema_hist'] = macd_dema_hist
    
    bb_upper, bb_middle, bb_lower = ta.bbands(close, timeperiod=20)
    features['bb_upper'] = bb_upper
    features['bb_middle'] = bb_middle
    features['bb_lower'] = bb_lower
    features['bb_width'] = (bb_upper - bb_lower) / bb_middle
    features['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
    
    stoch_k, stoch_d = ta.stoch(high, low, close)
    features['stoch_k'] = stoch_k
    features['stoch_d'] = stoch_d
    
    features['atr_14'] = ta.atr(high, low, close, timeperiod=14)
    features['atr_20'] = ta.atr(high, low, close, timeperiod=20)
    
    # ADX
    adx, plus_di, minus_di = ta.adx(high, low, close, timeperiod=14)
    features['adx'] = adx
    features['plus_di'] = plus_di
    features['minus_di'] = minus_di
    
    # Aroon
    aroon_up, aroon_down = ta.aroon(high, low, timeperiod=14)
    features['aroon_up'] = aroon_up
    features['aroon_down'] = aroon_down
    features['aroon_oscillator'] = aroon_up - aroon_down
    
    # CCI
    features['cci_20'] = ta.cci(high, low, close, timeperiod=20)
    
    # Williams %R
    features['willr_14'] = ta.willr(high, low, close, timeperiod=14)
    
    # Awesome Oscillator
    features['awesome_oscillator'] = ta.awesome_oscillator(high, low, fast_period=5, slow_period=34)
    
    # Keltner Channels
    kc_upper, kc_middle, kc_lower = ta.keltner_channels(high, low, close, timeperiod=20)
    features['kc_upper'] = kc_upper
    features['kc_middle'] = kc_middle
    features['kc_lower'] = kc_lower
    features['kc_width'] = (kc_upper - kc_lower) / kc_middle
    
    # Donchian Channel
    dc_upper, dc_middle, dc_lower = ta.donchian_channel(high, low, timeperiod=20)
    features['dc_upper'] = dc_upper
    features['dc_middle'] = dc_middle
    features['dc_lower'] = dc_lower
    
    # Supertrend
    supertrend, supertrend_direction = ta.supertrend(high, low, close, period=14, multiplier=3)
    features['supertrend'] = supertrend
    features['supertrend_direction'] = supertrend_direction
    
    # Ichimoku
    ichi_tenkan, ichi_kijun, ichi_senkou_a, ichi_senkou_b, ichi_chikou = ta.ichimoku(high, low, close)
    features['ichi_tenkan'] = ichi_tenkan
    features['ichi_kijun'] = ichi_kijun
    features['ichi_senkou_a'] = ichi_senkou_a
    features['ichi_senkou_b'] = ichi_senkou_b
    features['ichi_chikou'] = ichi_chikou
    
    ppo_line, ppo_signal, ppo_hist = ta.ppo(close)
    features['ppo_line'] = ppo_line
    features['ppo_signal'] = ppo_signal
    features['ppo_hist'] = ppo_hist
    
    tsi_line, tsi_signal = ta.tsi(close)
    features['tsi_line'] = tsi_line
    features['tsi_signal'] = tsi_signal
    
    features['obv'] = ta.obv(close, volume)
    features['mfi_14'] = ta.mfi(high, low, close, volume, timeperiod=14)
    features['cmf_20'] = ta.cmf(high, low, close, volume, timeperiod=20)
    features['vwap'] = ta.vwap(high, low, close, volume)
    features['pvt'] = ta.pvt(close, volume)
    
    vwma_20 = ta.vwma(close, volume, timeperiod=20)
    features['vwma_20'] = vwma_20
    
    vwap_upper, vwap_middle, vwap_lower = ta.vwap_bands(high, low, close, volume, timeperiod=20)
    features['vwap_upper'] = vwap_upper
    features['vwap_middle'] = vwap_middle
    features['vwap_lower'] = vwap_lower
    
    aobv_fast, aobv_slow = ta.aobv(close, volume)
    features['aobv_fast'] = aobv_fast
    features['aobv_slow'] = aobv_slow
    
    features['vzo'] = ta.volume_zone_oscillator(close, volume)
    
    features['dpo_20'] = ta.dpo(close, timeperiod=20)
    features['fisher_transform'] = ta.fisher_transform(close, timeperiod=10)
    features['elder_ray_bull'] = ta.elder_ray(high, low, close, timeperiod=13)[0]
    features['elder_ray_bear'] = ta.elder_ray(high, low, close, timeperiod=13)[1]
    features['rvi'] = ta.rvi(open_price, high, low, close, timeperiod=10)
    features['choppiness_index'] = ta.choppiness_index(high, low, close, timeperiod=14)
    features['mass_index'] = ta.mass_index(high, low, timeperiod=25)
    features['volatility_ratio'] = ta.volatility_ratio(high, low, close)
    features['hurst_exponent'] = ta.hurst_exponent(close, max_lag=20)
    features['price_cycle'] = ta.price_cycle(close, cycle_period=20)
    features['historical_volatility'] = ta.historical_volatility(close)
    
    fractal_up, fractal_down = ta.fractal_indicator(high, low, n=2)
    features['fractal_up'] = fractal_up
    features['fractal_down'] = fractal_down
    
    features['kalman_filter'] = ta.kalman_filter(close)
    
    features['high_low_ratio'] = high / low
    features['close_open_ratio'] = close / open_price
    features['body_size'] = abs(close - open_price) / close
    features['upper_shadow'] = (high - pd.concat([close, open_price], axis=1).max(axis=1)) / close
    features['lower_shadow'] = (pd.concat([close, open_price], axis=1).min(axis=1) - low) / close
    
    features['price_position'] = (close - low) / (high - low)
    
    features['returns_1'] = close.pct_change(1)
    features['returns_5'] = close.pct_change(5)
    features['returns_10'] = close.pct_change(10)
    
    print(f"Generated {len(features.columns)} features")
    return features


def select_features(
    data: pd.DataFrame,
    target: pd.Series,
    n_features: int = 50,
    task: str = 'classification',
    objective: Optional[str] = None,
    metric: Optional[str] = None,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    verbose: int = -1
) -> Tuple[List[str], pd.DataFrame]:
    print("Generating all features...")
    features = generate_all_features(data)
    
    target_aligned = target.reindex(features.index)
    
    aligned_data = pd.concat([features, target_aligned.to_frame('target')], axis=1)
    
    mask = ~aligned_data['target'].isna()
    
    if mask.sum() > 0:
        feature_nan_count = aligned_data[mask].drop(columns=['target']).isna().sum(axis=1)
        feature_mask = feature_nan_count < (len(features.columns) * 0.5)
        mask = mask & feature_mask
    
    X = aligned_data[mask].drop(columns=['target'])
    y = aligned_data[mask]['target']
    
    X = X.ffill().bfill()
    X = X.fillna(0)
    
    print(f"Training data: {len(X)} samples, {len(X.columns)} features")
    
    if objective is None:
        objective = 'binary' if task == 'classification' else 'regression'
    if metric is None:
        metric = 'binary_logloss' if task == 'classification' else 'rmse'
    
    print("Training LightGBM model for feature selection...")
    if task == 'classification':
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            objective=objective,
            metric=metric,
            verbose=verbose,
            random_state=42
        )
    else:
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            objective=objective,
            metric=metric,
            verbose=verbose,
            random_state=42
        )
    
    model.fit(X.values, y.values)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    

    selected_features = feature_importance.head(n_features)['feature'].tolist()
    
    print(f"\nSelected top {n_features} features:")
    print(feature_importance.head(n_features).to_string(index=False))
    
    return selected_features, feature_importance


def get_selected_features_data(
    data: pd.DataFrame,
    selected_features: List[str]
) -> pd.DataFrame:
    all_features = generate_all_features(data)
    return all_features[selected_features]

if __name__ == "__main__":
    import trading.model_tools as mt
    from trading.brains.gbm.feature_selector import select_features, get_selected_features_data
    from scipy.signal import savgol_filter
    
    data = mt.fetch_data(symbol="BTC-USDT", days=730, interval="1m", age_days=0, data_source="binance")
    
    target = pd.Series(savgol_filter(data['Close'].rolling(window=5).mean(), window_length=10, polyorder=4, deriv=1), index=data.index)
    target[target > 0] = 1
    target[target < 0] = 0
    
    selected_features, importance_df = select_features(
        data=data,
        target=target,
        n_features=50,
        task='classification'
    )
    
    feature_data = get_selected_features_data(data, selected_features)