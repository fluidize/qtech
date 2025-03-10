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
    def __init__(self, epochs, rnn_width, dense_width, ticker='BTC-USD', chunks=5, interval='5m', age_days=10):
        self.epochs = epochs
        self.rnn_width = rnn_width
        self.dense_width = dense_width

        self.ticker = ticker
        self.chunks = chunks
        self.interval = interval
        self.age_days = age_days
        self.ohlcv_scaler = MinMaxScaler()
        self.data = None
        self.X = None
        self.y = None
        self.model_data = None
        self.yhat = None

        self.model_filename = None
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

    def _fetch_data(self):
        data = pd.DataFrame()
        temp_data = None
        for x in range(self.chunks):
            start_date = (datetime.now() - timedelta(days=8) - timedelta(days=8*x) - timedelta(days=self.age_days)).strftime('%Y-%m-%d')
            end_date = (datetime.now()- timedelta(days=8*x) - timedelta(days=self.age_days)).strftime('%Y-%m-%d')
            temp_data = yf.download(self.ticker, start=start_date, end=end_date, interval=self.interval)
            data = pd.concat([data, temp_data])
        data.sort_index(inplace=True)
        data.columns = data.columns.droplevel(1)
        data.reset_index(inplace=True)
        data.rename(columns={'index': 'Datetime'}, inplace=True)
        self.data = data

    def _prepare_data(self):
        df = self.data
        ohlcv_features = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        X_ohlcv = df[ohlcv_features]
        
        X_ohlcv_scaled = pd.DataFrame(self.ohlcv_scaler.fit_transform(X_ohlcv), columns=X_ohlcv.columns)
        self.X = X_ohlcv_scaled

        self.y = X_ohlcv_scaled.shift(-1)  # Shift the target variable to predict the next time step
        
        self.X = self.X[:-1]
        self.y = self.y[:-1]

    def _train_model(self):
        X = self.X.values.reshape((self.X.shape[0], 1, self.X.shape[1]))
        y = self.y.values.reshape((self.y.shape[0], 1, self.y.shape[1]))

        model = models.Sequential()
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-8)
        early_stopping = callbacks.EarlyStopping(monitor='loss', min_delta=1e-4, mode='auto', patience=3, restore_best_weights=True)
        inputs = layers.Input(shape=(X.shape[1], X.shape[2]))

        rnn = layers.LSTM(units=self.rnn_width, return_sequences=True)(inputs)
        rnn = layers.GRU(units=self.rnn_width, return_sequences=True)(rnn)
        rnn = layers.LSTM(units=self.rnn_width, return_sequences=True)(rnn)
        rnn = layers.GRU(units=self.rnn_width, return_sequences=True)(rnn)

        attention = layers.MultiHeadAttention(num_heads=13, key_dim=32)(rnn, rnn)

        dense = layers.Dense(self.dense_width, activation='relu')(attention)
        dense = layers.Dense(self.dense_width, activation='relu')(dense)
        dense = layers.Dense(self.dense_width, activation='relu')(dense) 

        outputs = layers.Dense(5, activation='relu')(dense)  # Adjusted to output 5 features (OHLCV)

        model = models.Model(inputs=inputs, outputs=outputs)
        lossfn = losses.Huber(delta=5.0)
        model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss=lossfn, metrics=['mean_squared_error'])
        self.model_data = model.fit(X, y, epochs=self.epochs, batch_size=64, callbacks=[early_stopping, reduce_lr])

        return model

    def _predict(self, model):
        X = self.X.values.reshape((self.X.shape[0], 1, self.X.shape[1]))
        yhat = model.predict(X)
        
        # Inverse transform predictions
        self.yhat = self.ohlcv_scaler.inverse_transform(yhat.squeeze())

    def create_plot(self, show_graph=False):
        data = self.data
        yhat = self.yhat[:, 3]  # Extract the predicted Close values
        model_data = self.model_data

        data.index = pd.to_datetime(data.index)
        data = data.iloc[:-1]
        data.reset_index(inplace=True)

        loss_history = model_data.history['loss']

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=('Price Prediction', f'Loss: {loss_history[-1]} | MSE: {mean_squared_error(data["Close"], yhat)}'))

        fig.add_trace(go.Scatter(x=data['Datetime'], y=data['Close'].squeeze(), mode='lines', name='True Close', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data['Datetime'], y=yhat, mode='lines', name='Predicted Close', line=dict(color='red')), row=1, col=1)

        fig.add_trace(go.Scatter(x=list(range(len(loss_history))), y=loss_history, mode='lines', name='Loss', line=dict(color='orange')), row=2, col=1)

        fig.update_layout(template='plotly_dark', title_text='Price Prediction and Model Loss')
        fig.update_xaxes(tickmode='linear', tick0=0, dtick=1, row=2, col=1)
        
        if show_graph:
            fig.show()

        return fig

    def run(self, save=False):
        self._fetch_data()
        self._prepare_data()
        model = self._train_model()
        self._predict(model)  # Output is already inverse transformed
        if save: 
            model.save(f'{self.ticker}_{self.interval}_{model.count_params()}.keras', overwrite=True)
            print(f'Model Saved to {os.getcwd()}')

class ModelTesting(TimeSeriesPredictor):
    def __init__(self, ticker, chunks, interval, age_days):
        super().__init__(epochs=0, rnn_width=0, dense_width=0, ticker=ticker, chunks=chunks, interval=interval, age_days=age_days)
        self.model = None #doesn't exist in TimeSeriesPredictor

    def load_model(self, model_name):
        self.model = models.load_model(model_name) #models is a keras function
        self.model.summary()
        self.model_filename = model_name #store it to get interval

    def create_test_plot(self, show_graph=False):
        data = self.data
        yhat = self.yhat[:, 3]  # Extract the predicted Close values

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=data['Datetime'], y=data['Close'].squeeze(), mode='lines', name='True Close', line=dict(color='blue'), connectgaps=True))
        fig.add_trace(go.Scatter(x=data['Datetime'], y=yhat, mode='lines', name='Predicted Close', line=dict(color='red'), connectgaps=True))

        fig.update_layout(template='plotly_dark', title_text='Price Prediction')
        
        if show_graph:
            fig.show()

        return fig

    def _extended_predict(self, model, interval, extension=10):
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

        for i in range(extension):
            self._predict(model)  #set inverse and set self.yhat
            # Append the new predictions to the data
            new_data = pd.DataFrame(self.yhat, columns=self.data.columns[1:6]) #only ohlcv
            new_data['Datetime'] = self.data['Datetime'].iloc[-1] + extended_time
            self.data = pd.concat([self.data, new_data], axis=0)

    def run(self, extension=10):
        model_interval = self.model_filename.split("_")[1]
        self._fetch_data()
        self._prepare_data()
        self._extended_predict(model=self.model, interval=model_interval, extension=extension)

model = TimeSeriesPredictor(epochs=5, rnn_width=128, dense_width=128, ticker='BTC-USD', chunks=7, interval='5m', age_days=3)
model.run(save=True)
model.create_plot(show_graph=True)

# Example usage
test_client = ModelTesting(ticker='BTC-USD', chunks=1, interval='5m', age_days=0)
test_client.load_model(model_name=input("Enter the model name: "))
test_client.run(extension=10)
test_client.create_test_plot(show_graph=True)

