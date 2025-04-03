import pandas as pd
import numpy as np
import os

import tensorflow as tf
from tensorflow import keras
import torch

import plotly.graph_objects as go

from single_predictors.price_data import fetch_data, prepare_data
from single_predictors.single_pytorch_model import SingleModel

os.chdir(os.path.dirname(os.path.abspath(__file__)))

class TimeSeriesPredictor:
    def __init__(self, ticker='SOL-USDT', chunks=5, interval='1min', age_days=0):
        self.model_type = None
        self.ticker = ticker
        self.chunks = chunks
        self.interval = interval
        self.age_days = age_days
        self.model = None
        self.data = None

        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_keras_model(self, model_path):
        self.model = keras.load_model(model_path)
        self.model_type = "keras"
    def load_pytorch_model(self, model_path):
        self.model = torch.load(model_path, weights_only=False)
        self.model.eval()
        self.model.to(self.DEVICE)
        self.model_type = "pytorch"

    def fetch_data(self):
        self.data = prepare_data(fetch_data(self.ticker, self.chunks, self.interval, self.age_days, kucoin=True), train_split=True)

    def predict(self):
        if (self.model is None) or (self.model_type is None) or (self.data is None):
            raise ValueError("Model not loaded.")
        
        print(f"TRAINING MODEL {self.model_type}")

        if self.model_type == "pytorch":
            X, y = self.data
            X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.DEVICE)
            with torch.no_grad():
                predictions = self.model(X_tensor)
            return predictions

    def create_plot(self, actual, predicted):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=actual['Datetime'], y=actual['Close'], mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=actual['Datetime'], y=predicted, mode='lines', name='Predicted'))
        fig.update_layout(title='Price Prediction', xaxis_title='Date', yaxis_title='Price')
        fig.show()

    def run(self):
        self.fetch_data()
        self.load_pytorch_model(r"single_predictors\trained_model.pth")
        predictions = self.predict()
        print(predictions)

if __name__ == "__main__":
    # Example usage
    predictor = TimeSeriesPredictor()
    predictor.run()