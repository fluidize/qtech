import yfinance as yf
import pandas as pd
import numpy as np

import tensorflow as tf
from keras import layers, models, optimizers, callbacks

import matplotlib.pyplot as plt
import seaborn as sns

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

def fetch_btc_data():
    # Download Bitcoin OHLCV data (1 minute candles for 8 days)
    data = yf.download('BTC-USD', period='8d', interval='1m')  
    return data

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

#preparation
data = fetch_btc_data()
data = add_features(data)
X, y = prepare_data(data)

# Reshape X into 3D for LSTM [samples, time steps, features]
# X.shape[0]: Number of rows (data points, samples).
# X.shape[1]: Number of columns (features, variables).
# X.shape[2] (only in 3D arrays): This could refer to additional dimensions, for example, in time-series data where the 3D shape might represent (samples, timesteps, features).
X = X.values.reshape((X.shape[0], 1, X.shape[1]))  # Reshaping to (samples, time_steps, features)

# LSTM MODEL
model = models.Sequential()
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss',  # Monitor validation loss
                              factor=0.2,  # Reduce by 80% (0.2)
                              patience=5,  # Wait for 5 epochs with no improvement
                              min_lr=1e-6)
lstm_width = 512

model.add(layers.Input(shape=(X.shape[1], X.shape[2]))) #LSTM INPUT SHAPE
model.add(layers.LSTM(units=lstm_width, return_sequences=True))  # First LSTM layer
model.add(layers.LSTM(units=lstm_width, return_sequences=False))

model.add(layers.Dense(256))
model.add(layers.Dense(256))
model.add(layers.Dense(256))
model.add(layers.Dense(1))  # Output layer

# compile and train
lossfn = tf.keras.losses.Huber()
model.compile(optimizer=optimizers.Adam(learning_rate=0.00001), loss=lossfn, metrics=['mean_squared_error'])
model.fit(X, y, epochs=100, batch_size=16)

#predict and plot
yhat = model.predict(X)
data.index = pd.to_datetime(data.index)
data = data.reset_index() # Alternatively, reset the index if the time is being used as a column
data = data.iloc[:-1] # extras


sns.set_style("darkgrid")

plt.figure(figsize=(10, 6))

sns.lineplot(x=data['Datetime'], y=y.to_numpy().ravel(), label="Actual Values (y)", color='blue', alpha=0.7)
sns.lineplot(x=data['Datetime'], y=yhat.flatten(), label="Predicted Values (y_hat)", color='red', alpha=0.7)

plt.title("Actual vs Predicted Values")
plt.xlabel("Time")
plt.ylabel("Price (USD)")

plt.legend()
plt.show()