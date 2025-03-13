import yfinance as yf
import pandas as pd
import numpy as np

import tensorflow as tf
from keras import layers, models, optimizers, callbacks, losses
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

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
        self.scalers = {
            'price': MinMaxScaler(feature_range=(0, 1)),
            'volume': QuantileTransformer(output_distribution='normal'),
            'technical': StandardScaler()
        }
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
        gain = (delta.where(delta > 0, 0)).ewm(span=period, min_periods=period).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=period, min_periods=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _compute_bollinger_bands(self, series, period=20, std=2):
        middle_band = series.rolling(window=period).mean()
        std_dev = series.rolling(window=period).std()
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)
        return upper_band, middle_band, lower_band

    def _compute_macd(self, series, fast=12, slow=26, signal=9):
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    def _compute_atr(self, high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def _add_time_features(self, df):
        df['Hour'] = pd.to_datetime(df['Datetime']).dt.hour
        df['DayOfWeek'] = pd.to_datetime(df['Datetime']).dt.dayofweek
        df['Hour_sin'] = np.sin(2 * np.pi * df['Hour']/24)
        df['Hour_cos'] = np.cos(2 * np.pi * df['Hour']/24)
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek']/7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek']/7)
        return df.drop(['Hour', 'DayOfWeek'], axis=1)

    def _prepare_data(self, data, train_split=True):
        df = data.copy()
        indicator_columns = ['MA50', 'MA20', 'MA10', 'RSI']
        df = df.drop(columns=indicator_columns, errors='ignore')
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
        df['Volume_Change'] = df['Volume'].pct_change()
        df['RSI'] = self._compute_rsi(df['Close'], 14)
        upper_bb, middle_bb, lower_bb = self._compute_bollinger_bands(df['Close'], 20, 2)
        df['BB_Upper'] = upper_bb
        df['BB_Middle'] = middle_bb
        df['BB_Lower'] = lower_bb
        df['BB_Width'] = (upper_bb - lower_bb) / middle_bb
        macd, signal_line = self._compute_macd(df['Close'])
        df['MACD'] = macd
        df['MACD_Signal'] = signal_line
        df['MACD_Hist'] = macd - signal_line
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA10_MA20_Cross'] = df['MA10'] - df['MA20']
        df['ATR'] = self._compute_atr(df['High'], df['Low'], df['Close'], 14)
        df['Volatility'] = df['Close'].rolling(window=20).std()
        df['ROC'] = df['Close'].pct_change(periods=10) * 100
        df['Price_Momentum'] = df['Close'] - df['Close'].rolling(window=10).mean()
        df = self._add_time_features(df)
        for period in [1, 2, 3]:
            df[f'Prev{period}_Close'] = df['Close'].shift(period)
            df[f'Prev{period}_Volume'] = df['Volume'].shift(period)
        df.dropna(inplace=True)
        
        if train_split:
            price_features = ['Open', 'High', 'Low', 'Close']
            volume_features = ['Volume', 'Volume_Change'] + [f'Prev{i}_Volume' for i in [1, 2, 3]]
            technical_features = [col for col in df.columns if col not in price_features + volume_features + ['Datetime']]
            df[price_features] = self.scalers['price'].fit_transform(df[price_features])
            df[volume_features] = df[volume_features].replace([np.inf, -np.inf], np.nan)
            df[volume_features] = df[volume_features].fillna(df[volume_features].mean())
            df[volume_features] = self.scalers['volume'].fit_transform(df[volume_features])
            df[technical_features] = self.scalers['technical'].fit_transform(df[technical_features])
            X = df.drop(['Datetime'], axis=1)
            y = df[['Close']].shift(-1)
            self.output_dimensions = len(y.columns)
            X = X[:-1]
            y = y[:-1]
            return X, y
        return df

    def _train_model(self, X, y):
        X = X.values.reshape((X.shape[0], 1, X.shape[1]))
        print(f"Training shapes - X: {X.shape}, y: {y.shape}")
        inputs = layers.Input(shape=(X.shape[1], X.shape[2]))
        x = layers.LSTM(units=self.rnn_width, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(units=self.rnn_width//2, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(self.dense_width, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(self.dense_width//2, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        outputs = layers.Dense(self.output_dimensions)(x)
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])
        callbacks_list = [
            callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
            callbacks.ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
        ]
        model_data = model.fit(X, y, epochs=self.epochs, batch_size=32, validation_split=0.2, callbacks=callbacks_list, verbose=1)
        print("\nTraining Results:")
        print(f"Final loss: {model_data.history['loss'][-1]:.6f}")
        print(f"Final validation loss: {model_data.history['val_loss'][-1]:.6f}")
        return model, model_data

    def _predict(self, model, X):
        X = X.values.reshape((X.shape[0], 1, X.shape[1]))
        yhat = model.predict(X)
        yhat_expanded = np.zeros((yhat.shape[0], 4))
        yhat_expanded[:, 3] = yhat.squeeze()
        yhat = self.scalers['price'].inverse_transform(yhat_expanded)[:, 3]
        print(f"Prediction shape: {yhat.shape}")
        print(f"Prediction range: min={np.min(yhat):.2f}, max={np.max(yhat):.2f}")
        print(f"Prediction std: {np.std(yhat):.2f}")
        return yhat

    def create_plot(self, data, yhat, model_data, show_graph=False, save_image=False):
        df = self._prepare_data(data.copy(), train_split=False)
        actual_prices = df['Close'].values[:-1]
        dates = df['Datetime'].values[:-1]
        mse = mean_squared_error(actual_prices, yhat)
        mape = np.mean(np.abs((actual_prices - yhat) / actual_prices)) * 100
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
            subplot_titles=('Price Prediction', f'Prediction Error (MSE: {mse:.2f}, MAPE: {mape:.2f}%)',
                          f'Training Loss History (Final Loss: {model_data.history["loss"][-1]:.6f})'))
        fig.add_trace(go.Scatter(x=dates, y=actual_prices, mode='lines', name='Actual Close', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=yhat, mode='lines', name='Predicted Close', line=dict(color='red')), row=1, col=1)
        error = actual_prices - yhat
        fig.add_trace(go.Scatter(x=dates, y=error, mode='lines', name='Prediction Error', line=dict(color='orange')), row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        fig.add_trace(go.Scatter(x=list(range(len(model_data.history['loss']))), y=model_data.history['loss'],
                               mode='lines', name='Training Loss', line=dict(color='green')), row=3, col=1)
        fig.update_layout(height=900, template='plotly_dark', showlegend=True, title_text='Cryptocurrency Price Prediction Analysis')
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Error", row=2, col=1)
        fig.update_yaxes(title_text="Loss", row=3, col=1)
        if show_graph:
            fig.show()
        if save_image:
            fig.write_image(f"images/prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        return fig

    def run(self, save=False):
        data = self._fetch_data(self.ticker, self.chunks, self.interval, self.age_days)
        X, y = self._prepare_data(data)
        print(f"Training data shape: X={X.shape}, y={y.shape}")
        print(f"X range: min={np.min(X.values):.2f}, max={np.max(X.values):.2f}")
        print(f"y range: min={np.min(y.values):.2f}, max={np.max(y.values):.2f}")
        model, model_data = self._train_model(X, y)
        yhat = self._predict(model, X)
        if save: 
            model.save(f'{self.ticker}_{self.interval}_{model.count_params()}.keras', overwrite=True)
            print(f'Model Saved to {os.getcwd()}')
        return data, yhat, model_data

class ModelTesting(TimeSeriesPredictor):
    def __init__(self, ticker, chunks, interval, age_days):
        super().__init__(epochs=0, rnn_width=0, dense_width=0, ticker=ticker, chunks=chunks, interval=interval, age_days=age_days)
        self.model = None

    def load_model(self, model_name):
        self.model = models.load_model(model_name)
        self.model.summary()
        self.model_filename = model_name

    def create_test_plot(self, data, yhat, show_graph=False):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=yhat['Datetime'], y=data['Close'].squeeze(), mode='lines', name='True Close', line=dict(color='blue'), connectgaps=True))
        fig.add_trace(go.Scatter(x=yhat['Datetime'], y=yhat['Close'], mode='lines', name='Predicted Close', line=dict(color='red'), connectgaps=True))
        fig.update_layout(template='plotly_dark', title_text='Price Prediction')
        if show_graph:
            fig.show()
        return fig

    def _extended_predict(self, model, data, interval, extension=10):
        number = int(interval[:-1])
        unit = interval[-1]
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
            X, _ = self._prepare_data(predicted_data)
            yhat = self._predict(model, X)
            new_row = pd.DataFrame(index=[0])
            new_row['Datetime'] = predicted_data['Datetime'].iloc[-1] + extended_time
            new_row['Close'] = yhat[-1]
            new_row['Open'] = predicted_data['Open'].iloc[-1]
            new_row['High'] = predicted_data['High'].iloc[-1]
            new_row['Low'] = predicted_data['Low'].iloc[-1]
            new_row['Volume'] = predicted_data['Volume'].iloc[-1]
            predicted_data = pd.concat([predicted_data, new_row], axis=0)
        return original_data, predicted_data

    def run(self, extension=10):
        model_interval = self.model_filename.split("_")[1]
        data = self._fetch_data(self.ticker, self.chunks, self.interval, self.age_days)
        data = self._extended_predict(self.model, data, model_interval, extension)
        return data

test_client = TimeSeriesPredictor(epochs=50, rnn_width=128, dense_width=64, ticker='BTC-USD', chunks=10, interval='5m', age_days=2)
data, yhat, model_data = test_client.run(save=True)
test_client.create_plot(data, yhat, model_data, show_graph=True)