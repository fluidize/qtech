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

tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

class TimeSeriesPredictor:
    def __init__(self, rnn_width, dense_width, ticker='BTC-USD', chunks=5, interval='5m', age_days=10):
        self.rnn_width = rnn_width
        self.dense_width = dense_width

        self.ticker = ticker
        self.chunks = chunks
        self.interval = interval
        self.age_days = age_days
        self.feature_scaler = MinMaxScaler()
        self.price_scaler = MinMaxScaler()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

    ########################################
    ## PRE-PROCESSING
    ########################################

    def _fetch_data(self):
        data = pd.DataFrame()
        temp_data = None
        for x in range(self.chunks):
            start_date = (datetime.now() - timedelta(days=8) - timedelta(days=8*x) - timedelta(days=self.age_days)).strftime('%Y-%m-%d')
            end_date = (datetime.now()- timedelta(days=8*x) - timedelta(days=self.age_days)).strftime('%Y-%m-%d')
            temp_data = yf.download(self.ticker, start=start_date, end=end_date, interval=self.interval)
            data = pd.concat([data, temp_data])
        data.sort_index(inplace=True)
        return data

    def _add_features(self, df):
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        df['RSI'] = self._compute_rsi(df['Close'], 14)
        
        df['Prev_Close'] = df['Close'].shift(1)
        df['Prev_High'] = df['High'].shift(1)
        df['Prev_Low'] = df['Low'].shift(1)
        df['Prev_Open'] = df['Open'].shift(1)
        df['Prev_Volume'] = df['Volume'].shift(1)
        
        df.dropna(inplace=True)

        return df

    def _compute_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _prepare_data(self, df):
        ohlc_features = ['Open', 'High', 'Low', 'Close']
        other_features = ['Volume', 'MA20', 'MA50', 'RSI', 'Prev_Close', 'Prev_High', 'Prev_Low', 'Prev_Open', 'Prev_Volume']
        
        X_ohlc = df[ohlc_features]
        X_other = df[other_features]
        
        X_ohlc_scaled = pd.DataFrame(self.price_scaler.fit_transform(X_ohlc), columns=X_ohlc.columns)
        X_other_scaled = pd.DataFrame(self.feature_scaler.fit_transform(X_other), columns=X_other.columns)
        X = pd.concat([X_ohlc_scaled, X_other_scaled], axis=1)

        X_ohlc_scaled['Target'] = X_ohlc_scaled['Close'].shift(-1)
        y = X_ohlc_scaled[['Target']]
        
        X = X[:-1]
        y = y[:-1]
        
        return X, y

    ########################################
    ## TRAIN MODEL
    ########################################

    def _train_model(self, X, y, rnn_width=512, dense_width=512):
        X = X.values.reshape((X.shape[0], 1, X.shape[1]))

        model = models.Sequential()
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-8)
        early_stopping = callbacks.EarlyStopping(monitor='loss', min_delta=1e-4, mode='auto', patience=3, restore_best_weights=True)
        inputs = layers.Input(shape=(X.shape[1], X.shape[2]))

        rnn = layers.LSTM(units=rnn_width, return_sequences=True)(inputs)
        rnn = layers.GRU(units=rnn_width, return_sequences=True)(rnn)
        rnn = layers.LSTM(units=rnn_width, return_sequences=True)(rnn)
        rnn = layers.GRU(units=rnn_width, return_sequences=True)(rnn)

        attention = layers.MultiHeadAttention(num_heads=13, key_dim=32)(rnn, rnn)

        dense = layers.Dense(dense_width, activation='relu')(attention)
        dense = layers.Dense(dense_width, activation='relu')(dense)

        outputs = layers.Dense(1)(dense)

        model = models.Model(inputs=inputs, outputs=outputs)
        lossfn = losses.Huber(delta=5.0)
        model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss=lossfn, metrics=['mean_squared_error'])
        model_data = model.fit(X, y, epochs=50, batch_size=64, callbacks=[early_stopping, reduce_lr])

        return model, model_data

    ########################################
    ## PREDICT
    ########################################

    def _predict(self, model, X):
        yhat = model.predict(X)

        yhat_expanded = np.zeros((yhat.shape[0], 4))
        yhat_expanded[:,3] = yhat.squeeze()
        yhat_inverse = self.price_scaler.inverse_transform(yhat_expanded)[:,3]

        return yhat_inverse

    ########################################
    ## ANALYZE
    ########################################

    def _analyze(self, train_data, yhat_inverse, model_data, y):
        train_data.index = pd.to_datetime(train_data.index)
        train_data = train_data.iloc[:-1]
        train_data.reset_index(inplace=True)

        loss_history = model_data.history['loss']

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=('Price Prediction', f'Loss: {loss_history[-1]} | MSE: {mean_squared_error(y, yhat_inverse)}'))

        fig.add_trace(go.Scatter(x=train_data['Datetime'], y=train_data["Close"].squeeze(), mode='lines', name='True', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=train_data['Datetime'], y=yhat_inverse, mode='lines', name='Prediction', line=dict(color='red')), row=1, col=1)

        fig.add_trace(go.Scatter(x=list(range(len(loss_history))), y=loss_history, mode='lines', name='Loss', line=dict(color='orange')), row=2, col=1)

        fig.update_layout(template='plotly_dark', title_text='Price Prediction and Model Loss')
        fig.update_xaxes(tickmode='linear', tick0=0, dtick=1, row=2, col=1)

        fig.show()

    def run(self):
        train_data = self._fetch_data()
        train_data = self._add_features(train_data)
        X, y = self._prepare_data(train_data)
        model, model_data = self._train_model(X, y)
        print("Training complete")
        yhat_inverse = self._predict(model, X)
        self._analyze(train_data, yhat_inverse, model_data, y)

        if input('Save Model? (Y/N): ').lower() == 'y':
            model.save(f'{self.ticker}_{self.interval}_{model.count_params()}.keras', overwrite=True)
            print(f'Model Saved to {os.getcwd()}')