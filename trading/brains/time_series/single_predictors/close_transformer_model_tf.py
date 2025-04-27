import yfinance as yf
import pandas as pd
import numpy as np
import os

import tensorflow as tf
from keras import layers, models, optimizers, losses, callbacks, regularizers

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append("trading")
from model_tools import fetch_data

tf.config.threading.set_intra_op_parallelism_threads(8)

class TransformerModel:
    def __init__(self, epochs, ticker='BTC-USDT', chunks=5, interval='5min', age_days=10):
        self.epochs = epochs
        self.sequence_length = 25 
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
        self.data = fetch_data(self.ticker, self.chunks, self.interval, self.age_days)

    def _compute_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _prepare_data(self, data, train_split=True):
        df = data.copy()
        df = df.drop(columns=['MA50', 'MA20', 'MA10', 'RSI'], errors='ignore')

        # Price-based features
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
        
        # Create sequences of previous values
        self.lagged_length = 5
        
        # Create all lagged features at once using pd.concat
        lagged_features = []
        for col in ['Close', 'Volume', 'High', 'Low', 'Open']:
            for i in range(1, self.lagged_length):
                lagged_features.append(pd.DataFrame({
                    f'Prev{i}_{col}': df[col].shift(i)
                }))
        
        # Concatenate all lagged features at once
        if lagged_features:
            df = pd.concat([df] + lagged_features, axis=1)
        
        std = df['Close'].std()
        df['Close_ZScore'] = (df['Close'] - df['Close'].mean()) / std  # Z-score
        
        # Moving averages (normalized by current price)
        df['MA10'] = df['Close'].rolling(window=10).mean() / df['Close']
        df['MA20'] = df['Close'].rolling(window=20).mean() / df['Close']
        df['MA50'] = df['Close'].rolling(window=50).mean() / df['Close']
        df['MA10_MA20_Cross'] = df['MA10'] - df['MA20']
        
        # RSI and other technical indicators
        df['RSI'] = self._compute_rsi(df['Close'], 14)

        df.dropna(inplace=True)
        
        if train_split:
            # Group features by their characteristics
            price_features = ['Open', 'High', 'Low', 'Close']
            volume_features = ['Volume'] + [f'Prev{i}_Volume' for i in range(1, self.lagged_length)]
            bounded_features = ['RSI']  # Features that are already bounded (e.g., 0-100)
            normalized_features = ['MA10', 'MA20', 'MA50', 'Price_Range', 'MA10_MA20_Cross', 'Close_ZScore']
            
            # Remaining features need standardization
            technical_features = [col for col in df.columns 
                               if col not in (price_features + volume_features + bounded_features + 
                                            normalized_features + ['Datetime'])]
            
            # Scale absolute price values with MinMaxScaler
            df[price_features] = self.scalers['price'].fit_transform(df[price_features])
            
            # Transform volume features to normal distribution
            df[volume_features] = df[volume_features].replace([np.inf, -np.inf], np.nan)
            df[volume_features] = df[volume_features].fillna(df[volume_features].mean())
            df[volume_features] = self.scalers['volume'].fit_transform(df[volume_features])
            
            # Standardize technical features that aren't already normalized
            if technical_features:
                df[technical_features] = self.scalers['technical'].fit_transform(df[technical_features])
            
            X = df.drop(['Datetime'], axis=1)
            y = df['Close'].shift(-1)
            
            X = X[:-1]  # Remove last row since we don't have target for it
            y = y[:-1]  # Remove last row since we don't have target for it
            return X, y
        
        return df

    def _train_model(self, X, y):
        n_samples = X.shape[0] - self.sequence_length + 1
        n_features = X.shape[1]
        
        X_sequences = np.zeros((n_samples, self.sequence_length, n_features), dtype=np.float32)
        for i in range(n_samples):
            X_sequences[i] = X[i:i + self.sequence_length]
        
        y = y[self.sequence_length - 1:]
        
        l1_reg = 1e-4
        l2_reg = 1e-4
        self.rnn_count = 3
        self.dense_count = 5
        embedding_dim = 32
        num_heads = X_sequences.shape[1]
        ff_dim = 64
        growth_rate = 32

        inputs = layers.Input(shape=(X_sequences.shape[1], X_sequences.shape[2]), name='input_layer')
    
        x = layers.Dense(embedding_dim, name='embedding_projection')(inputs)
        
        transformer_outputs = [x]
        for i in range(2):
            concat_inputs = layers.Concatenate(name=f'transformer_{i}_concat')(transformer_outputs)
            self_attention = layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=embedding_dim // num_heads,
                name=f'transformer_{i}_self_attention'
            )(
                query=concat_inputs,
                key=concat_inputs,
                value=concat_inputs
            )
            
            if i > 0:
                cross_attention = layers.MultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=embedding_dim // num_heads,
                    name=f'transformer_{i}_cross_attention'
                )(
                    query=self_attention,
                    key=transformer_outputs[0],
                    value=transformer_outputs[0]
                )
                self_attention = layers.Add(name=f'transformer_{i}_cross_merge')([self_attention, cross_attention])
            
            # Position-wise feed-forward network
            ffn = layers.Dense(ff_dim, activation="relu", name=f'transformer_{i}_ffn_1')(self_attention)  # expand to higher dimension and compress back
            ffn = layers.Dense(embedding_dim, name=f'transformer_{i}_ffn_2')(ffn)
            
            transformer_outputs.append(ffn)
        x = layers.Concatenate(name='transformer_final_concat')(transformer_outputs)
        
        gru_outputs = [x]
        for i in range(self.rnn_count):
            concat_inputs = layers.Concatenate(axis=-1, name=f'gru_{i}_concat')(gru_outputs)
            
            gru_out = layers.Bidirectional(layers.GRU(
                units=64,
                return_sequences=True,
                kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                name=f'gru_layer_{i}'
            ), name=f'bidirectional_gru_{i}')(concat_inputs)
            
            avg_pool = layers.GlobalAveragePooling1D(name=f'avg_pool_{i}')(gru_out)
            max_pool = layers.GlobalMaxPooling1D(name=f'max_pool_{i}')(gru_out)
            
            gru_out = layers.Concatenate(axis=-1, name=f'gru_features_{i}')([
                gru_out,
                layers.RepeatVector(self.sequence_length)(avg_pool),
                layers.RepeatVector(self.sequence_length)(max_pool)
            ])
            
            gru_outputs.append(gru_out)
        final_outputs = []
        for output in gru_outputs:
            avg_features = layers.GlobalAveragePooling1D(name=f'final_avg_pool_{len(final_outputs)}')(output)
            max_features = layers.GlobalMaxPooling1D(name=f'final_max_pool_{len(final_outputs)}')(output)
            final_outputs.extend([avg_features, max_features])
        x = layers.Concatenate(axis=-1, name='gru_final_concat')(final_outputs)
        
        dense_outputs = [x]
        for i in range(self.dense_count): 
            concat_inputs = layers.Concatenate(name=f'dense_{i}_concat')(dense_outputs)
            x = layers.Dense(growth_rate, name=f'dense_layer_{i}')(concat_inputs)
            x = layers.LeakyReLU(name=f'leaky_relu_layer_{i}')(x)
            
            dense_outputs.append(x)
        x = layers.Concatenate(name='final_concat')(dense_outputs)
        
        x = layers.Dense(32, activation='relu', name='transition_layer')(x)
        
        outputs = layers.Dense(1, name='output_layer')(x)
        
        lossfn = losses.Huber(delta=5.0)
        optimizer = optimizers.Adam(learning_rate=1e-3)
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=optimizer,
                      loss=lossfn,
                      metrics=['mae'])
        
        model_data = model.fit(X_sequences, y, 
                                epochs=self.epochs,
                                batch_size=32,
                                validation_split=0.1,
                                callbacks=[
                                    callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                                    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)
                                ],
                                verbose=1)
        
        return model, model_data

    def _predict(self, new_data):
        prepared_data = self._prepare_data(new_data, train_split=False)
        n_samples = prepared_data.shape[0] - self.sequence_length + 1
        n_features = prepared_data.shape[1]
        
        X_sequences = np.zeros((n_samples, self.sequence_length, n_features), dtype=np.float32)
        for i in range(n_samples):
            X_sequences[i] = prepared_data[i:i + self.sequence_length]
        
        predictions = self.model.predict(X_sequences)
        return predictions
    
    def prompt_save(self, model):
        if input("Save? y/n ").lower() == 'y':
            model.save("1only.keras")

    def run(self):
        X, y = self._prepare_data(self.data)
        model, model_data = self._train_model(X, y)
        self.prompt_save(model)
        return model, model_data

if __name__ == "__main__":
    train = TransformerModel(epochs=10, ticker='SOL-USDT', chunks=10, interval='1min', age_days=0)
    model, model_data = train.run()