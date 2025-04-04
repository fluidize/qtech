import yfinance as yf
import pandas as pd
import numpy as np

def covariance(x: pd.DataFrame, y:pd.DataFrame):
    if len(x) != len(y):
        raise Exception(f"X and Y are not the same length. X:{len(x)} Y:{len(y)}")
    x_mean = x.mean()
    y_mean = y.mean()

    xy_dev_sum = 0
    for i in range(len(x)):
        xy_dev_sum += (x.iloc[i]-x_mean)*(y.iloc[i]-y_mean)
    
    cov = xy_dev_sum/(len(x)-1)

    return cov


m2_data = pd.read_csv(r"trading\ideas\m2\bitcoin-and-m2-growth-gl_short.csv")
m2_data["DateTime"] = pd.to_datetime(m2_data["DateTime"])
m2_data["DateTime"] = m2_data["DateTime"].dt.date

spx_data = pd.DataFrame(yf.download("VOO", start=m2_data["DateTime"].iloc[0], end=m2_data["DateTime"].iloc[-1], interval="1wk"))

shift = 108
x = m2_data["M2 Global Supply (USD)"].shift(shift).dropna()
y_1 = m2_data["BTC price"][shift:]
y_2 = spx_data["Close"][shift+2:]

m2_btc_cov = covariance(x, y_1)
m2_spx_cov = covariance(x, y_2)

normalized_btc_cov = m2_btc_cov/m2_btc_cov
normalized_spx_cov = m2_spx_cov.values/m2_btc_cov

print(normalized_btc_cov, normalized_spx_cov)