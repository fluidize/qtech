import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast
import pandas as pd
import numpy as np

from rich import print
from tqdm import tqdm
import os
# from model_tools import fetch_data, prepare_data, create_plot

from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler
from datetime import datetime, timedelta
import requests
import yfinance as yf
from tqdm import tqdm
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from rich import print

def fetch_data(ticker, chunks, interval, age_days, kucoin: bool = True):
    print("[green]DOWNLOADING DATA[/green]")
    if not kucoin:
        data = pd.DataFrame()
        times = []
        for x in range(chunks):
            chunksize = 1
            start_date = datetime.now() - timedelta(days=chunksize) - timedelta(days=chunksize*x) - timedelta(days=age_days)
            end_date = datetime.now() - timedelta(days=chunksize*x) - timedelta(days=age_days)
            temp_data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval=interval, progress=False)
            if data.empty:
                data = temp_data
            else:
                data = pd.concat([data, temp_data])
            times.append(start_date)
            times.append(end_date)
        
        earliest = min(times)
        latest = max(times)
        difference = latest - earliest
        print(f"\n{ticker} | {difference.days} days {difference.seconds//3600} hours {difference.seconds//60%60} minutes {difference.seconds%60} seconds")

        data.sort_index(inplace=True)
        data.columns = data.columns.droplevel(1)
        data.reset_index(inplace=True)
        data.rename(columns={'index': 'Datetime'}, inplace=True)
        data.rename(columns={'Date': 'Datetime'}, inplace=True)
        data = pd.DataFrame(data)  
    elif kucoin:
        data = pd.DataFrame(columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])
        times = []
        
        progress_bar = tqdm(total=chunks, desc="KUCOIN PROGRESS")
        for x in range(chunks):
            chunksize = 1440  # 1d of 1m data
            end_time = datetime.now() - timedelta(minutes=chunksize*x)
            start_time = end_time - timedelta(minutes=chunksize)
            
            params = {
                "symbol": ticker,
                "type": interval,
                "startAt": str(int(start_time.timestamp())),
                "endAt": str(int(end_time.timestamp()))
            }
            
            request = requests.get("https://api.kucoin.com/api/v1/market/candles", params=params).json()
            request_data = request["data"]  # list of lists
            
            records = []
            for dochltv in request_data:
                records.append({
                    "Datetime": dochltv[0],
                    "Open": float(dochltv[1]),
                    "Close": float(dochltv[2]),
                    "High": float(dochltv[3]),
                    "Low": float(dochltv[4]),
                    "Volume": float(dochltv[6])
                })
            
            temp_data = pd.DataFrame(records)
            if data.empty:
                data = temp_data
            else:
                data = pd.concat([data, temp_data])
            times.append(start_time)
            times.append(end_time)

            progress_bar.update(1)
        progress_bar.close()
        
        earliest = min(times)
        latest = max(times)
        difference = latest - earliest
        print(f"\n{ticker} | {difference.days} days {difference.seconds//3600} hours {difference.seconds//60%60} minutes {difference.seconds%60} seconds | {data.shape[0]} bars")
        
        data["Datetime"] = pd.to_datetime(pd.to_numeric(data['Datetime']), unit='s')
        data.sort_values('Datetime', inplace=True)
        data.reset_index(drop=True, inplace=True)
        
    return data

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def prepare_data(data, train_split=True):
    scalers = {
        'price': MinMaxScaler(feature_range=(0, 1)),
        'volume': QuantileTransformer(output_distribution='normal'),
        'technical': StandardScaler()
    }

    df = data.copy()
    df = df.drop(columns=['MA50', 'MA20', 'MA10', 'RSI'], errors='ignore')

    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
    
    lagged_length = 5
    
    lagged_features = []
    for col in ['Close', 'Volume', 'High', 'Low', 'Open']:
        for i in range(1, lagged_length):
            lagged_features.append(pd.DataFrame({
                f'Prev{i}_{col}': df[col].shift(i)
            }))
    
    if lagged_features:
        df = pd.concat([df] + lagged_features, axis=1)
    
    std = df['Close'].std()
    df['Close_ZScore'] = (df['Close'] - df['Close'].mean()) / std 
    
    df['MA10'] = df['Close'].rolling(window=10).mean() / df['Close']
    df['MA20'] = df['Close'].rolling(window=20).mean() / df['Close']
    df['MA50'] = df['Close'].rolling(window=50).mean() / df['Close']
    df['MA10_MA20_Cross'] = df['MA10'] - df['MA20']
    
    df['RSI'] = compute_rsi(df['Close'], 14)

    # Check for NaN values
    if df.isnull().any().any():
        # Fill NaN values with the mean of the column
        df = df.fillna(df.mean())
        # Alternatively, drop rows with NaN values
        # df = df.dropna()

    df.dropna(inplace=True)
    
    if train_split:
        price_features = ['Open', 'High', 'Low', 'Close']
        volume_features = ['Volume'] + [f'Prev{i}_Volume' for i in range(1, lagged_length)]
        bounded_features = ['RSI']  # Features that are already bounded (e.g., 0-100)
        normalized_features = ['MA10', 'MA20', 'MA50', 'Price_Range', 'MA10_MA20_Cross', 'Close_ZScore']
        
        technical_features = [col for col in df.columns 
                            if col not in (price_features + volume_features + bounded_features + 
                                        normalized_features + ['Datetime'])]
        
        df[price_features] = scalers['price'].fit_transform(df[price_features])
        
        df[volume_features] = df[volume_features].replace([np.inf, -np.inf], np.nan)
        df[volume_features] = df[volume_features].fillna(df[volume_features].mean())
        df[volume_features] = scalers['volume'].fit_transform(df[volume_features])
        
        if technical_features:
            df[technical_features] = scalers['technical'].fit_transform(df[technical_features])
        
        X = df.drop(['Datetime'], axis=1)
        y = df['Close'].shift(-1)  # Target is the next day's close price
        
        X = X[:-1]  # Remove last row since we don't have target for it
        y = y[:-1]  # Remove last row since we don't have target for it
        return X, y, scalers
    
    return df, scalers

def create_plot(actual, predicted):
    difference = len(actual)-len(predicted) #trimmer
    actual = actual[difference:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual['Datetime'], y=actual['Close'], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=actual['Datetime'], y=predicted, mode='lines', name='Predicted'))
    fig.update_layout(title='Price Prediction', xaxis_title='Date', yaxis_title='Price')
    fig.show()

class SingleModel(nn.Module):
    def __init__(self, epochs=10, train: bool = True):
        super().__init__()

        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if train:
            self.epochs = epochs
            self.data = fetch_data("BTC-USDT", 29, "1min", 0, kucoin=True) #set train data here
            X, y, scalers = prepare_data(self.data, train_split=True) #train split to calculate dims

            input_dim = X.shape[1]
        else:
            input_dim = 33

        # MODEL ARCHITECTURE
        rnn_width = 256
        dense_width = 256
        self.em1 = nn.Embedding(num_embeddings=33, embedding_dim=32)
        self.rnn1 = nn.LSTM(input_dim, rnn_width, num_layers=3, bidirectional=True, dropout=0.2)
        self.mha = nn.MultiheadAttention(embed_dim=rnn_width*2, num_heads=8, batch_first=True)
        self.fc1 = nn.Linear(rnn_width*2, dense_width)
        self.fc2 = nn.Linear(dense_width, dense_width)
        self.output = nn.Linear(dense_width, 1)

    def forward(self, x):
        x, _ = self.rnn1(x)  # Unpack the output and hidden state
        # x, _ = self.mha(x, x, x)
        x = torch.flatten(x, 1) #go thru dense layers
        x = self.fc1(x)
        x = nn.LeakyReLU(negative_slope=0.3)(x)
        x = self.fc2(x)
        x = nn.LeakyReLU(negative_slope=0.3)(x)
        x = self.output(x)
        return x

    def train_model(self, model, prompt_save=False):
        X, y, scalers = prepare_data(self.data, train_split=True)

        model.to(self.DEVICE)

        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)  # Reshape y to (batch_size, 1) to remove reduction warning

        train_dataset = TensorDataset(X_tensor, y_tensor)
        
        batch_size = 64
        criterion = nn.MSELoss()
        lr = 1e-5
        l2_reg = 1e-5
        optimizer = optim.Adam(model.parameters(),
                               lr=lr,
                               weight_decay=l2_reg,
                               amsgrad=True,
                               )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        best_loss = float('inf')
        for epoch in range(model.epochs):
            model.train()  # set model to training mode
            total_loss = 0

            print()
            progress_bar = tqdm(total=len(train_loader), desc="EPOCH PROGRESS")
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.DEVICE), y_batch.to(self.DEVICE)

                optimizer.zero_grad()  # reset the gradients of all parameters before backpropagation
                outputs = model(X_batch)  # forward pass
                loss = criterion(outputs, y_batch)  # Calculate loss
                loss.backward()  # back pass
                optimizer.step()  # update weights

                total_loss += loss.item() 
                progress_bar.update(1)
            progress_bar.close()
            if total_loss < best_loss:
                best_loss = total_loss
                # Save model checkpoint here if needed

            print(f"\nEpoch {epoch + 1}/{model.epochs}, Train Loss: {total_loss:.4f}")
            scheduler.step(total_loss)

        # Prompt to save the model
        if ((input("Save? y/n ").lower() == 'y') if prompt_save else False):
            torch.save(model, input("Enter save path: "))
    
    def predict(self, model , data):
        #input unprepared data
        X, y, scalers = prepare_data(data)
        X = torch.tensor(X.values, dtype=torch.float32).to(self.DEVICE)

        model.eval()
        model.to(self.DEVICE)
        with torch.no_grad(), autocast('cuda'): #automatically set precision since gpu is giving up
            yhat = model(X)
            yhat = yhat.cpu().numpy()
        
        temp_arr = np.zeros((len(yhat),4))
        temp_arr[:, 3] = yhat.squeeze()
        
        predictions = scalers['price'].inverse_transform(temp_arr)[:, 3]

        return predictions

def load_model(model_path: str) -> 'SingleModel':
    """
    Load a trained SingleModel from a saved checkpoint.
    
    Args:
        model_path (str): Path to the saved model file
        
    Returns:
        SingleModel: Loaded model instance
    """
    import torch
    import os
    
    # Create a new model instance
    model = SingleModel(train=False)
    
    # Load the model with weights_only=False since we trust our own model file
    # This is safe because we're loading our own trained model
    state_dict = torch.load(model_path, 
                          map_location=torch.device('cpu'),
                          weights_only=False)  # Explicitly set to False
    
    # Load the state dict into the model
    
    # Set to evaluation mode
    model.eval()
    
    return model

if __name__ == "__main__":
    model = load_model(r"trading\brains\time_series\single_predictors\model.pth")
    predictions = model.predict(model, model.data)
    create_plot(model.data, predictions)
