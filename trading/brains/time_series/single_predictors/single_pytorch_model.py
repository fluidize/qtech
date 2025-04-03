import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
# from price_data import fetch_data, prepare_data

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
            data = pd.concat([data, temp_data])
            times.append(start_date)
            times.append(end_date)
        
        earliest = min(times)
        latest = max(times)
        difference = latest - earliest
        print(f"{ticker} | {difference.days} days {difference.seconds//3600} hours {difference.seconds//60%60} minutes {difference.seconds%60} seconds")

        data.sort_index(inplace=True)
        data.columns = data.columns.droplevel(1)
        data.reset_index(inplace=True)
        data.rename(columns={'index': 'Datetime'}, inplace=True)
        data.rename(columns={'Date': 'Datetime'}, inplace=True)
        data = pd.DataFrame(data)  
    elif kucoin:
        data = pd.DataFrame(columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])
        times = []
        
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
            data = pd.concat([data, temp_data])
            times.append(start_time)
            times.append(end_time)
        
        earliest = min(times)
        latest = max(times)
        difference = latest - earliest
        print(f"{ticker} | {difference.days} days {difference.seconds//3600} hours {difference.seconds//60%60} minutes {difference.seconds%60} seconds")
        
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
        y = df['Close'].shift(-1)
        
        X = X[:-1]  # Remove last row since we don't have target for it
        y = y[:-1]  # Remove last row since we don't have target for it
        return X, y
    
    return df

class SimpleNN(nn.Module):
    def __init__(self, batch_size=64, learning_rate=0.001, epochs=10):
        super().__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.X, self.y = prepare_data(fetch_data("BTC-USDT", 10, "1min", 0, kucoin=True), train_split=True)
        
        print(self.X.head())
        print(self.X.shape)

        input_dim = self.X.shape[1]

        # MODEL ARCHITECTURE
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def train_model(self, model, save_path):
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[green]USING DEVICE: {DEVICE}[/green]")
        model.to(DEVICE)

        # Convert the data to tensors
        X_tensor = torch.tensor(self.X.values, dtype=torch.float32)
        y_tensor = torch.tensor(self.y.values, dtype=torch.float32)

        # Create a DataLoader for the training set
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=model.batch_size, shuffle=True)

        # Define the loss function and optimizer
        criterion = nn.MSELoss()  # Using MSELoss for regression
        optimizer = optim.Adam(model.parameters(), lr=model.learning_rate)

        for epoch in range(model.epochs):
            model.train()  # Set model to training mode
            total_loss = 0

            # Training loop
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

                optimizer.zero_grad()  # Zero the gradients
                outputs = model(X_batch)  # Forward pass
                loss = criterion(outputs, y_batch)  # Calculate loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights

                total_loss += loss.item()  # Accumulate total loss for this epoch

            # Print training loss for each epoch
            print(f"Epoch {epoch + 1}/{model.epochs}, Train Loss: {total_loss:.4f}")

        # Prompt to save the model
        if input("Save? y/n ").lower() == 'y':
            torch.save(model, save_path)
            print(f"Model saved to {save_path}")


if __name__ == "__main__":
    model = SimpleNN(batch_size=128, learning_rate=0.0005, epochs=100)
    model.train_model(model, "trained_model.pth")
