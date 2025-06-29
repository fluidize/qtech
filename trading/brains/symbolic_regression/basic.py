import pandas as pd
import numpy as np
from pysr import PySRRegressor

import sys
sys.path.append("trading")
import technical_analysis as ta
import model_tools as mt

df = mt.fetch_data(symbol="ETH-USDT", chunks=10, interval="15m", age_days=0, data_source="binance")

df['RSI'] = ta.rsi(df['Close'], timeperiod=14)
df['ZScore'] = ta.zscore(df['Close'], timeperiod=20)
df['Volatility'] = ta.volatility(df['Close'], timeperiod=10)

df['Target'] = ta.log_return(df['Close']).shift(-1)

df = df.dropna()

X = df[['RSI', 'ZScore', 'Volatility']].values
y = df['Target'].values

model = PySRRegressor(
    model_selection="best",
    niterations=1000,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["cos", "sin", "exp", "log", "abs"],
    maxsize=100,
    populations=100,
    progress=True,
    verbosity=0,
)

model.fit(X, y)

print(model.get_best())
