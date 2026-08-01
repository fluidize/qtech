from trading.model_tools import ta_transform
import trading.model_tools as mt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

### metrics

def r2_between(x: pd.Series, y: pd.Series) -> float:
    return x.corr(y) ** 2

def shannon_entropy(series: pd.Series, bins: int = 50) -> float:
    values = series.dropna().to_numpy()
    counts, _ = np.histogram(values, bins=bins)
    probs = counts[counts > 0] / counts.sum()
    return -np.sum(probs * np.log2(probs))

### transforms

def difference_transform(x: pd.Series, y: pd.Series) -> np.ndarray:
    return x.sub(y).to_numpy()

def acceleration_transform(x: pd.Series) -> np.ndarray:
    return x.diff(1).to_numpy()

def log_transform(x: pd.Series) -> np.ndarray:
    return np.log(x.to_numpy())

def fft_transform(x: pd.Series) -> np.ndarray:
    return np.fft.fft(x.to_numpy())

transforms = [difference_transform, acceleration_transform, log_transform, fft_transform]

###

data = mt.fetch_data(
    symbols=["SOL-USDT", "BTC-USDT"],
    days=730,
    interval="1h",
    age_days=0,
    data_source="binance",
    cache_expiry_hours=-1,
    verbose=True,
)

X = ta_transform(data, add_ticker="BTC-USDT")

columns = X.columns

for col in columns:
    max_entropy = 0
    max_entropy_series = None
    for transform in transforms:
        X[col] = transform(X[col])
        if shannon_entropy(X[col]) > max_entropy:
            max_entropy = shannon_entropy(X[col])
            max_entropy_series = X[col]
    X[col] = max_entropy_series
