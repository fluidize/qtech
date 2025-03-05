import yfinance as yf
import pandas as pd
import numpy as np

import tensorflow as tf
from keras import layers, models, optimizers, callbacks, losses
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)

feature_scaler = MinMaxScaler() #create scaler object in public to inverse in final output.
price_scaler = MinMaxScaler() #separate scaler object to store MinMax for OHLC features
close_scaler = MinMaxScaler() #separate scaler object to store MinMax for inversing output yhat


########################################
## PRE-PROCESSING
########################################

def fetch_btc_data():
    data = yf.download('BTC-USD', period='8d', interval='1m')  
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
    
    X_ohlc_scaled = pd.DataFrame(price_scaler.fit_transform(X_ohlc), columns=X_ohlc.columns) #SCALE OHLC SEPARATELY
    X_other_scaled = pd.DataFrame(feature_scaler.fit_transform(X_other), columns=X_other.columns) #SCALE OTHER FEATURES SEPARATELY
    X = pd.concat([X_ohlc_scaled, X_other_scaled], axis=1)

    X_ohlc_scaled['Target'] = X_ohlc_scaled['Close'].shift(-1) # Target is the next bar's close
    y = X_ohlc_scaled[['Target']]
    
    # Remove the last row (no target)
    X = X[:-1]
    y = y[:-1]
    
    return X, y

data = fetch_btc_data()
scaled_data = add_features(data)
X, y = prepare_data(scaled_data)

X = X.values.reshape((X.shape[0], 1, X.shape[1]))  # Reshaping to (samples, time_steps, features)

########################################
## MODEL
########################################

model = models.Sequential()
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss',  # Monitor validation loss
                              factor=0.2,
                              patience=5,
                              min_lr=1e-6)
early_stopping = callbacks.EarlyStopping(monitor='loss', mode='auto', patience=5, restore_best_weights=True)
rnn_width = 256
dense_width = 256

inputs = layers.Input(shape=(X.shape[1], X.shape[2])) # X.shape = (num_samples, num_time_steps, num_features)

rnn = layers.SimpleRNN(units=rnn_width, return_sequences=True)(inputs)
rnn = layers.SimpleRNN(units=rnn_width, return_sequences=True)(rnn)
rnn = layers.SimpleRNN(units=rnn_width, return_sequences=True)(rnn)

attention = layers.MultiHeadAttention(num_heads=13, key_dim=32)(rnn, rnn)

dense = layers.Dense(dense_width, activation='relu')(attention)
dense = layers.Dense(dense_width, activation='relu')(dense)

outputs = layers.Dense(1)(dense)

model = models.Model(inputs=inputs, outputs=outputs)
lossfn = losses.Huber(delta=5)
model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss=lossfn, metrics=['mean_squared_error'])
model_data = model.fit(X, y, epochs=3, batch_size=64, callbacks=[early_stopping, reduce_lr])

yhat = model.predict(X)

yhat_expanded = np.zeros((yhat.shape[0], 4))
yhat_expanded[:,3] = yhat.squeeze()
yhat_inverse = price_scaler.inverse_transform(yhat_expanded)[:,3]  # squeeze and reshape from ohlc scaler

########################################
## ANALYZE
########################################

data.index = pd.to_datetime(data.index)
data = data.iloc[:-1]  # extras

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
# Plot predictions
ax1.plot(data.index, data["Close"], label="True", color='blue')
ax1.plot(data.index, yhat_inverse, label="Prediction", color='red')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price')
ax1.set_title('Bitcoin Price Prediction')
ax1.legend()

# Plot loss
ax2.plot(model_data.history['loss'], label='Training Loss')
if 'val_loss' in model_data.history:
    ax2.plot(model_data.history['val_loss'], label='Validation Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title(f'Model Loss {lossfn.name} | MSE: {mean_squared_error(y.squeeze(), yhat.squeeze())}')
ax2.legend()
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.tight_layout()
plt.show()