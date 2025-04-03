import pandas as pd
import numpy as np
import os

import tensorflow as tf
from tensorflow import keras

import plotly.graph_objects as go
from datetime import datetime, timedelta

class TimeSeriesPredictor:
    def __init__(self, model_type, ticker='BTC-USD', chunks=5, interval='5m', age_days=10):
        self.model_type = model_type
        self.ticker = ticker
        self.chunks = chunks
        self.interval = interval
        self.age_days = age_days
        self.model = None
        self.data = None

    def load_model(self, model_path):
        self.model = keras.load_model(model_path)

    def fetch_data(self):
        # Implement data fetching logic based on model type
        if self.model_type == '1only':
            # Fetch data for 1only model
            pass  # Replace with actual data fetching logic
        elif self.model_type == 'ohlcv':
            # Fetch data for ohlcv model
            pass  # Replace with actual data fetching logic

    def predict(self):
        if self.model is None:
            raise ValueError("Model not loaded.")
        predictions = self.model.predict(self.data)
        return predictions

    def create_plot(self, actual, predicted):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=actual['Datetime'], y=actual['Close'], mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=actual['Datetime'], y=predicted, mode='lines', name='Predicted'))
        fig.update_layout(title='Price Prediction', xaxis_title='Date', yaxis_title='Price')
        fig.show()

if __name__ == "__main__":
    # Example usage
    predictor = TimeSeriesPredictor(model_type='1only')
    predictor.load_model('path_to_1only_model.keras')
    predictor.fetch_data()
    predictions = predictor.predict()
    predictor.create_plot(predictor.data, predictions) 