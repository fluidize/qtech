import yfinance as yf
import pandas as pd
import numpy as np

import tensorflow as tf
from keras import layers, models, optimizers, callbacks, losses
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from datetime import datetime, timedelta  # Ensure datetime is imported

tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)

feature_scaler = MinMaxScaler() #create scaler object in public to inverse in final output.
price_scaler = MinMaxScaler() #separate scaler object to store MinMax for OHLC features


########################################
## PRE-PROCESSING
########################################

def fetch_data(ticker, multiplier, interval='1m', age_days=0):
    data = pd.DataFrame()
    temp_data = None
    for x in range(multiplier):
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
    # Target is the next bar's close
    df['Target'] = df['Close'].shift(-1)

    # Features (OHLCV + technical indicators)
    X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'MA20', 'MA50', 'RSI', 'Prev_Close', 'Prev_High', 'Prev_Low', 'Prev_Open', 'Prev_Volume']]
    y = df[['Target']]
    
    # Remove the last row (no target)
    X = X[:-1]
    y = y[:-1]
    
    return X, y

data = fetch_data('BTC-USD', 5, '5m', 10)
scaled_data = add_features(data)
X, y = prepare_data(scaled_data)

X = X.values.reshape((X.shape[0], 1, X.shape[1]))  # Reshaping to (samples, time_steps, features)

########################################
## MODEL
########################################

model = models.load_model(input('Model Name: ') + '.h5')

########################################
## PREDICT
########################################

yhat = model.predict(X)# squeeze and reshape from ohlc scaler
print(yhat)
########################################
## ANALYZE
########################################

data.index = pd.to_datetime(data.index)
data = data.iloc[:-1]  # extras
data.reset_index(inplace=True)# Convert data.index to a column

fig = go.Figure()

fig.add_trace(go.Scatter(x=data['Datetime'], y=data["Close"].squeeze(), mode='lines', name='True', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=data['Datetime'], y=yhat.squeeze(), mode='lines', name='Prediction', line=dict(color='red')))
fig.update_layout(template='plotly_dark', title_text='Prediction vs True', xaxis_title='Datetime', yaxis_title='Price')

fig.show()