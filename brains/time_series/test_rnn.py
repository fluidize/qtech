import yfinance as yf
import pandas as pd
import numpy as np

import tensorflow as tf
from keras import layers, models, optimizers, callbacks, losses
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from datetime import datetime, timedelta
import os
import rich
import re

tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)

feature_scaler = MinMaxScaler()
price_scaler = MinMaxScaler()

os.chdir(os.path.dirname(os.path.abspath(__file__)))

########################################
## PRE-PROCESSING
########################################

def fetch_data(ticker, chunks, interval='1m', age_days=0):
    data = pd.DataFrame()
    temp_data = None
    for x in range(chunks):
        start_date = (datetime.now() - timedelta(days=8) - timedelta(days=8*x) - timedelta(days=age_days)).strftime('%Y-%m-%d')
        end_date = (datetime.now()- timedelta(days=8*x) - timedelta(days=age_days)).strftime('%Y-%m-%d')
        temp_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        data = pd.concat([data, temp_data])
    data.sort_index(inplace=True)
    return data

def add_features(df):
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    df['RSI'] = compute_rsi(df['Close'], 14)
    
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_High'] = df['High'].shift(1)
    df['Prev_Low'] = df['Low'].shift(1)
    df['Prev_Open'] = df['Open'].shift(1)
    df['Prev_Volume'] = df['Volume'].shift(1)
    
    df.dropna(inplace=True)

    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def prepare_data(df):
    ohlc_features = ['Open', 'High', 'Low', 'Close']
    other_features = ['Volume', 'MA20', 'MA50', 'RSI', 'Prev_Close', 'Prev_High', 'Prev_Low', 'Prev_Open', 'Prev_Volume']
    
    X_ohlc = df[ohlc_features]
    X_other = df[other_features]
    
    X_ohlc_scaled = pd.DataFrame(price_scaler.fit_transform(X_ohlc), columns=X_ohlc.columns)
    X_other_scaled = pd.DataFrame(feature_scaler.fit_transform(X_other), columns=X_other.columns)
    X = pd.concat([X_ohlc_scaled, X_other_scaled], axis=1)

    X_ohlc_scaled['Target'] = X_ohlc_scaled['Close'].shift(-1)
    y = X_ohlc_scaled[['Target']]
    
    X = X[:-1]
    y = y[:-1]
    
    return X, y

train_data = fetch_data('BTC-USD', 1, '5m', 0)
train_data = add_features(train_data)
X, y = prepare_data(train_data)

X = X.values.reshape((X.shape[0], 1, X.shape[1]))

########################################
## LOAD MODEL
########################################
print(" ".join(os.listdir(os.getcwd())))
result = re.search(r"BTC-USD_\dm_[0-9]+\.keras", " ".join(os.listdir(os.getcwd()))).group() #regex search for model in cwd

model = models.load_model(result)
rich.print(f"[bold purple]Using Model: {result}[/bold purple]")
print(model.summary())
rich.print(f"{model.loss}")

########################################
## PREDICT
########################################

yhat = model.predict(X)

yhat_expanded = np.zeros((yhat.shape[0], 4))
yhat_expanded[:,3] = yhat.squeeze()
yhat_inverse = price_scaler.inverse_transform(yhat_expanded)[:,3]

########################################
## ANALYZE
########################################

train_data.index = pd.to_datetime(train_data.index)
train_data = train_data.iloc[:-1]
train_data.reset_index(inplace=True)

fig = go.Figure()

fig.add_trace(go.Scatter(x=train_data['Datetime'], y=train_data["Close"].squeeze(), mode='lines', name='True', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=train_data['Datetime'], y=yhat_inverse, mode='lines', name='Prediction', line=dict(color='red')))

fig.update_layout(template='plotly_dark', title_text=f'Price Prediction | {model.loss.name}: {model.loss(y,yhat)}')

fig.show()