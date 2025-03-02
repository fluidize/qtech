import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Step 1: Fetch Bitcoin OHLCV Data
def fetch_btc_data():
    # Download Bitcoin OHLCV data (1 hour candles for 1 year)
    data = yf.download('BTC-USD', period='2y', interval='1h')  
    return data

# Step 2: Add Features (Moving Averages, RSI, Lagged Features)
def add_features(df):
    # 20-period and 50-period moving averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # RSI calculation (14-period)
    df['RSI'] = compute_rsi(df['Close'], 14)
    
    # Lag features (previous bars' OHLCV values)
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_High'] = df['High'].shift(1)
    df['Prev_Low'] = df['Low'].shift(1)
    df['Prev_Open'] = df['Open'].shift(1)
    df['Prev_Volume'] = df['Volume'].shift(1)
    
    # Remove rows with NaN values due to lagging
    df.dropna(inplace=True)
    
    return df

# Calculate 14-period RSI
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Step 3: Prepare Data (features and target)
def prepare_data(df):
    # Target is the next bar's close
    df['Target'] = df['Close'].shift(-1)
    
    # Features (OHLCV + technical indicators)
    X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'MA20', 'MA50', 'RSI', 'Prev_Close', 'Prev_High', 'Prev_Low', 'Prev_Open', 'Prev_Volume']]
    y = df['Target']
    
    # Remove the last row (no target)
    X = X[:-1]
    y = y[:-1]
    
    return X, y
