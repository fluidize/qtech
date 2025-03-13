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
        self.feature_scaler = QuantileTransformer()
        self.ohlcv_scaler = QuantileTransformer()
        # Initialize scalers dictionary
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
        # Convert to sine and cosine for cyclical features
        df['Hour_sin'] = np.sin(2 * np.pi * df['Hour']/24)
        df['Hour_cos'] = np.cos(2 * np.pi * df['Hour']/24)
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek']/7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek']/7)
        return df.drop(['Hour', 'DayOfWeek'], axis=1)

    def _prepare_data(self, data, train_split=True):
        df = data.copy()
        
        # Remove existing indicator columns if they exist
        indicator_columns = ['MA50', 'MA20', 'MA10', 'RSI']
        df = df.drop(columns=indicator_columns, errors='ignore')

        # Price and volume features
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Technical indicators
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
        
        # Moving averages and crosses
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA10_MA20_Cross'] = df['MA10'] - df['MA20']
        
        # Volatility indicators
        df['ATR'] = self._compute_atr(df['High'], df['Low'], df['Close'], 14)
        df['Volatility'] = df['Close'].rolling(window=20).std()
        
        # Price momentum and trend features
        df['ROC'] = df['Close'].pct_change(periods=10) * 100
        df['Price_Momentum'] = df['Close'] - df['Close'].rolling(window=10).mean()
        
        # Add time-based features
        df = self._add_time_features(df)
        
        # Previous period features
        for period in [1, 2, 3]:
            df[f'Prev{period}_Close'] = df['Close'].shift(period)
            df[f'Prev{period}_Volume'] = df['Volume'].shift(period)
        
        df.dropna(inplace=True)
        
        if train_split:
            # Separate features for different scaling approaches
            price_features = ['Open', 'High', 'Low', 'Close']
            volume_features = ['Volume', 'Volume_Change'] + [f'Prev{i}_Volume' for i in [1, 2, 3]]
            technical_features = [col for col in df.columns if col not in price_features + volume_features + ['Datetime']]
            
            # Scale price features
            price_scaler = MinMaxScaler(feature_range=(0, 1))
            df[price_features] = price_scaler.fit_transform(df[price_features])
            
            # Scale volume features using robust scaler due to outliers
            volume_scaler = QuantileTransformer(output_distribution='normal')
            # Handle infinite values before scaling
            df[volume_features] = df[volume_features].replace([np.inf, -np.inf], np.nan)
            df[volume_features] = df[volume_features].fillna(df[volume_features].mean())
            df[volume_features] = volume_scaler.fit_transform(df[volume_features])
            
            # Scale technical features
            tech_scaler = StandardScaler()
            df[technical_features] = tech_scaler.fit_transform(df[technical_features])
            
            # Store scalers for later use
            self.scalers = {
                'price': price_scaler,
                'volume': volume_scaler,
                'technical': tech_scaler
            }
            
            X = df.drop(['Datetime'], axis=1)
            y = df[['Close']].shift(-1)  # Predict next period's close price
            self.output_dimensions = len(y.columns)
            
            X = X[:-1]  # Remove last row since we won't have target for it
            y = y[:-1]  # Remove last row to match X
            
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
        
        outputs = layers.Dense(self.output_dimensions)(dense)

        model = models.Model(inputs=inputs, outputs=outputs)
        lossfn = losses.Huber(delta=5.0)
        model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss=lossfn, metrics=['mean_squared_error'])
        model_data = model.fit(X, y, epochs=self.epochs, batch_size=64, callbacks=[early_stopping, reduce_lr])

        return model, model_data

    def _predict(self, model, X):
        X = X.values.reshape((X.shape[0], 1, X.shape[1]))
        yhat = model.predict(X)
        
        # Create a dummy array with the same number of columns as price features
        yhat_expanded = np.zeros((yhat.shape[0], 4))  # 4 columns for OHLC
        yhat_expanded[:, 3] = yhat.squeeze()  # Put predictions in Close position
        
        # Use the price scaler for inverse transform
        yhat = self.scalers['price'].inverse_transform(yhat_expanded)[:, 3]  # Get Close price back
        return yhat

    def create_plot(self, data, yhat, model_data, show_graph=False, save_image=False):
        data = self._prepare_data(data.copy(), train_split=False) #NA columns can get removed
        data.reset_index(inplace=True)

        loss_history = model_data.history['loss']

        X_plot = data["Close"][:-1]
        y_plot = yhat  # Only plot the Close values

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

test_client = TimeSeriesPredictor(epochs=10, rnn_width=256, dense_width=256, ticker='BTC-USD', chunks=5, interval='5m', age_days=1)
data, yhat, model_data = test_client.run(save=False)
test_client.create_plot(data, yhat, model_data, show_graph=True)

# test_client = ModelTesting(ticker='BTC-USD', chunks=1, interval='5m', age_days=0)
# test_client.load_model(model_name="BTC-USD_5m_2212325.keras")
# original_data, predicted_data = test_client.run(extension=100)
# test_client.create_test_plot(original_data, predicted_data, show_graph=True)