import yfinance as yf
import pandas as pd
import numpy as np

import tensorflow as tf
from keras import layers, models, optimizers, callbacks, losses
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import mean_squared_error

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from datetime import datetime, timedelta
import os

tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

def check_for_nan(data, stage):
    nan_count = np.sum(np.isnan(data))
    print(f"NaNs found at {stage}: {nan_count}")
    return nan_count

class TimeSeriesPredictor:
    def __init__(self, epochs, rnn_width, dense_width, ticker='BTC-USD', chunks=5, interval='5m', age_days=10):
        self.epochs = epochs
        self.rnn_width = rnn_width
        self.dense_width = dense_width

        self.ticker = ticker
        self.chunks = chunks
        self.interval = interval
        self.age_days = age_days
        self.feature_scaler = QuantileTransformer()
        self.ohlcv_scaler = QuantileTransformer()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

    def _fetch_data(self, ticker, chunks, interval, age_days):
        data = pd.DataFrame()
        for x in range(chunks):
            start_date = (datetime.now() - timedelta(days=8) - timedelta(days=8*x) - timedelta(days=age_days)).strftime('%Y-%m-%d')
            end_date = (datetime.now() - timedelta(days=8*x) - timedelta(days=age_days)).strftime('%Y-%m-%d')
            temp_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
            data = pd.concat([data, temp_data])
        data.sort_index(inplace=True)
        data.columns = data.columns.droplevel(1)
        data.reset_index(inplace=True)
        data.rename(columns={'index': 'Datetime'}, inplace=True)
        return data

    def _compute_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _prepare_data(self, data, train_split=True):
        df = data.copy()

        # Remove indicator columns if they exist
        df = df.drop(columns=['MA50', 'MA20', 'MA10', 'RSI'], errors='ignore')

        df['Prev_Close'] = df['Close'].shift(1)
        df['Prev_High'] = df['High'].shift(1)
        df['Prev_Low'] = df['Low'].shift(1)
        df['Prev_Open'] = df['Open'].shift(1)
        df['Prev_Volume'] = df['Volume'].shift(1)

        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = self._compute_rsi(df['Close'], 14)

        df.dropna(inplace=True)

        ohlcv_features = ['Open', 'High', 'Low', 'Close', 'Volume']
        other_features = ['MA50', 'MA20', 'MA10', 'RSI', 'Prev_Close', 'Prev_High', 'Prev_Low', 'Prev_Open', 'Prev_Volume']
        
        X_ohlcv = df[ohlcv_features]
        X_other = df[other_features]
        
        X_ohlcv_scaled = pd.DataFrame(self.ohlcv_scaler.fit_transform(X_ohlcv), columns=X_ohlcv.columns)
        X_other_scaled = pd.DataFrame(self.feature_scaler.fit_transform(X_other), columns=X_other.columns)
        
        if train_split:
            X = pd.concat([X_ohlcv_scaled, X_other_scaled], axis=1)
            y = X_ohlcv_scaled.shift(-1)  # Shift the target variable to predict the next time step
            
            X = X[:-1]
            y = y[:-1]
            return X, y
        
        return df

    def _train_model(self, X, y):
        X = X.values.reshape((X.shape[0], 1, X.shape[1]))

        model = models.Sequential()
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-8)
        early_stopping = callbacks.EarlyStopping(monitor='loss', min_delta=1e-4, mode='auto', patience=3, restore_best_weights=True)
        inputs = layers.Input(shape=(X.shape[1], X.shape[2]))

        rnn = layers.Bidirectional(layers.LSTM(units=self.rnn_width, return_sequences=True))(inputs)
        rnn = layers.GRU(units=self.rnn_width, return_sequences=True)(rnn)
        rnn = layers.GRU(units=self.rnn_width, return_sequences=True)(rnn)

        attention = layers.MultiHeadAttention(num_heads=14, key_dim=32)(rnn, rnn)

        dense = layers.Dense(self.dense_width)(attention)
        dense = layers.LeakyReLU(alpha=0.3)(dense)
        dense = layers.Dense(self.dense_width)(dense)
        dense = layers.LeakyReLU(alpha=0.3)(dense)
        dense = layers.Dense(self.dense_width)(dense)
        dense = layers.LeakyReLU(alpha=0.3)(dense)
        
        outputs = layers.Dense(5)(dense)  # Adjusted to output 5 features (OHLCV)

        model = models.Model(inputs=inputs, outputs=outputs)
        lossfn = losses.Huber(delta=5.0)
        model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss=lossfn, metrics=['mean_squared_error'])
        model_data = model.fit(X, y, epochs=self.epochs, batch_size=64, callbacks=[early_stopping, reduce_lr])

        return model, model_data

    def _predict(self, model, X):
        X = X.values.reshape((X.shape[0], 1, X.shape[1]))
        yhat = model.predict(X)
        
        # Inverse transform predictions
        yhat = self.ohlcv_scaler.inverse_transform(yhat.squeeze())
        return yhat

    def create_plot(self, data, yhat, model_data, show_graph=False, save_image=False):
        data = self._prepare_data(data.copy(), train_split=False) #NA columns can get removed
        data.reset_index(inplace=True)

        loss_history = model_data.history['loss']

        X_plot = data["Close"][:-1]
        y_plot = yhat[:, 3]

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=('Price Prediction', f'Loss: {loss_history[-1]} | MSE: {mean_squared_error(X_plot, y_plot)}'))

        fig.add_trace(go.Scatter(x=data['Datetime'], y=X_plot.squeeze(), mode='lines', name='True Close', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data['Datetime'], y=y_plot, mode='lines', name='Predicted Close', line=dict(color='red')), row=1, col=1)

        fig.add_trace(go.Scatter(x=list(range(len(loss_history))), y=loss_history, mode='lines', name='Loss', line=dict(color='orange')), row=2, col=1)

        fig.update_layout(template='plotly_dark', title_text='Price Prediction and Model Loss')
        fig.update_xaxes(tickmode='linear', tick0=0, dtick=1, row=2, col=1)
        
        if show_graph:
            fig.show()
        if save_image:
            fig.write_image(f"images/{self.ohlcv_scaler}")
        return fig

    def run(self, save=False):
        data = self._fetch_data(self.ticker, self.chunks, self.interval, self.age_days)
        X, y = self._prepare_data(data)
        print(X)
        model, model_data = self._train_model(X, y)
        yhat = self._predict(model, X)
        if save: 
            model.save(f'{self.ticker}_{self.interval}_{model.count_params()}.keras', overwrite=True)
            print(f'Model Saved to {os.getcwd()}')
        return data, yhat, model_data

class ModelTesting(TimeSeriesPredictor):
    def __init__(self, ticker, chunks, interval, age_days):
        super().__init__(epochs=0, rnn_width=0, dense_width=0, ticker=ticker, chunks=chunks, interval=interval, age_days=age_days)
        self.model = None #doesn't exist in TimeSeriesPredictor

    def load_model(self, model_name):
        self.model = models.load_model(model_name) #models is a keras function
        self.model.summary()
        self.model_filename = model_name #store it to get interval

    def create_test_plot(self, data, yhat, show_graph=False):
        fig = go.Figure() 

        fig.add_trace(go.Scatter(x=yhat['Datetime'], y=data['Close'].squeeze(), mode='lines', name='True Close', line=dict(color='blue'), connectgaps=True))
        fig.add_trace(go.Scatter(x=yhat['Datetime'], y=yhat['Close'], mode='lines', name='Predicted Close', line=dict(color='red'), connectgaps=True))

        fig.update_layout(template='plotly_dark', title_text='Price Prediction')
        
        if show_graph:
            fig.show()

        return fig

    def _extended_predict(self, model, data, interval, extension=10):
        # PARSE INTO A TIMEDELTA
        number = int(interval[:-1])  # Take all characters except the last one as the number
        unit = interval[-1]  # Last character will be the unit (e.g., 'm', 'h')

        if unit == 'm':
            extended_time = timedelta(minutes=number)
        elif unit == 'h':
            extended_time = timedelta(hours=number)
        elif unit == 'd':
            extended_time = timedelta(days=number)
        else:
            raise ValueError("Unsupported unit, please use 'm' for minutes or 'h' for hours.")
          
        original_data = data.copy()
        predicted_data = data

        for i in range(extension):
            # Use only the last row for prediction
            X, _ = self._prepare_data(predicted_data)
            yhat = self._predict(model, X)
            
            new_data = pd.DataFrame(yhat[-1].reshape(1, -1), columns=original_data.columns[1:6]) #only ohlcv
            new_data['Datetime'] = predicted_data['Datetime'].iloc[-1] + extended_time
            predicted_data = pd.concat([predicted_data, new_data], axis=0)
            print(predicted_data)

        return original_data, predicted_data

    def run(self, extension=10):
        model_interval = self.model_filename.split("_")[1]
        data = self._fetch_data(self.ticker, self.chunks, self.interval, self.age_days)
        data = self._extended_predict(self.model, data, model_interval, extension)
        return data

test_client = TimeSeriesPredictor(epochs=3, rnn_width=1, dense_width=1, ticker='BTC-USD', chunks=5, interval='5m', age_days=1)
data, yhat, model_data = test_client.run(save=False)
test_client.create_plot(data, yhat, model_data, show_graph=True)

# test_client = ModelTesting(ticker='BTC-USD', chunks=1, interval='5m', age_days=0)
# test_client.load_model(model_name="BTC-USD_5m_2212325.keras")
# original_data, predicted_data = test_client.run(extension=100)
# test_client.create_test_plot(original_data, predicted_data, show_graph=True)