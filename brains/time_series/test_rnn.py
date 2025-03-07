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

from train_rnn import TimeSeriesPredictor

tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)

class ModelTesting:
    def __init__(self, ticker, chunks, interval, age_days):
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
        return TimeSeriesPredictor.add_features(self, df)

    def _compute_rsi(self, series, period=14):
        return TimeSeriesPredictor.compute_rsi(self, series, period)

    def _prepare_data(self, df):
        X, y = TimeSeriesPredictor.prepare_data(self, df)
        return X, y

    ########################################
    ## LOAD MODEL
    ########################################

    def _load_model(self):
        print(" ".join(os.listdir(os.getcwd())))
        result = re.search(r"BTC-USD_\dm_[0-9]+\.keras", " ".join(os.listdir(os.getcwd()))).group() #regex search for model in cwd
        model = models.load_model(result)
        rich.print(f"[bold purple]Using Model: {result}[/bold purple]")
        print(model.summary())
        rich.print(f"{model.loss}")
        return model

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

    def _analyze(self, train_data, yhat_inverse, model, y):
        train_data.index = pd.to_datetime(train_data.index)
        train_data = train_data.iloc[:-1]
        train_data.reset_index(inplace=True)

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=train_data['Datetime'], y=train_data["Close"].squeeze(), mode='lines', name='True', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=train_data['Datetime'], y=yhat_inverse, mode='lines', name='Prediction', line=dict(color='red')))

        fig.update_layout(template='plotly_dark', title_text=f'Price Prediction | {model.loss.name}: {model.loss(y,yhat_inverse)}')

        fig.show()

    def run(self):
        train_data = self._fetch_data()
        train_data = self._add_features(train_data)
        X, y = self._prepare_data(train_data)
        X = X.values.reshape((X.shape[0], 1, X.shape[1]))
        model = self._load_model()
        yhat_inverse = self._predict(model, X)
        self._analyze(train_data, yhat_inverse, model, y)

# Example usage
tester = ModelTesting(ticker='BTC-USD', chunks=1, interval='5m', age_days=0)
tester.run()