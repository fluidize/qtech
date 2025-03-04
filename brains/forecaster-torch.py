import yfinance as yf
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fetch_btc_data():
    data = yf.download('BTC-USD', period='1d', interval='1m')  
    return data

def add_features(df):
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = compute_rsi(df['Close'], 14)
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_High'] = df['High'].shift(1)
    df['Prev_Low'] = df['Low'].shift(1)
    df['Prev_Open'] = df['Open'].shift(1)
    df['Prev_Volume'] = df['Volume'].shift(1)
    df.dropna(inplace=True)
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def prepare_data(df):
    df['Target'] = df['Close'].shift(-1)
    X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'MA20', 'MA50', 'RSI', 'Prev_Close', 'Prev_High', 'Prev_Low', 'Prev_Open', 'Prev_Volume']]
    y = df['Target']
    X = X[:-1]
    y = y[:-1]
    return X, y

data = fetch_btc_data()
data = add_features(data)
X, y = prepare_data(data)

X = X.values.reshape((X.shape[0], 1, X.shape[1]))  

X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1).to(device) #Neural networks in PyTorch often expect the target labels (y) to be in a specific shape (e.g., (batch_size, 1)).

dataset = TensorDataset(X_tensor, y_tensor) #CONV TO TENSOR
train_loader = DataLoader(dataset, batch_size=64, shuffle=True) #LOAD TO MEM

class Model(nn.Module):
    def __init__(self, input_size, rnn_width=2048, dense_width=4096, num_heads=13, key_dim=128):
        super(Model, self).__init__()
        self.prefc1 = nn.Linear(input_size, dense_width)
        self.prefc2 = nn.Linear(dense_width, dense_width)
        self.prefc3 = nn.Linear(dense_width, rnn_width)

        self.lstm1 = nn.LSTM(dense_width, rnn_width, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(rnn_width * 2, rnn_width, batch_first=True, bidirectional=True)
        
        self.gru1 = nn.GRU(rnn_width, rnn_width * 2, batch_first=True)
        self.gru2 = nn.GRU(rnn_width * 2, rnn_width * 2, batch_first=True)
        self.gru3 = nn.GRU(rnn_width * 2, rnn_width, batch_first=True)
        
        self.fc1 = nn.Linear(rnn_width, dense_width)
        self.fc2 = nn.Linear(dense_width, dense_width)
        self.fc3 = nn.Linear(dense_width, 1)

    def forward(self, x):
        x = torch.relu(self.prefc1(x))
        x = torch.relu(self.prefc2(x))
        x = torch.relu(self.prefc2(x))
        x = torch.relu(self.prefc2(x))
        x = torch.relu(self.prefc2(x))
        x = torch.relu(self.prefc2(x))
        x = torch.relu(self.prefc2(x))
        x = torch.relu(self.prefc3(x))
        
        # # LSTM layers (Bidirectional)
        # x, _ = self.lstm1(x)
        # x, _ = self.lstm2(x)

        # GRU layers
        x, _ = self.gru1(x)  # Revert back to (batch_size, seq_len, embed_dim)
        x, _ = self.gru2(x)
        x, _ = self.gru3(x)
        #: This means "select all samples in the batch."
        #-1 This means "select the last time step in the sequence."
        #: This means "select all the features (hidden size) for the last time step."

        x = torch.relu(self.fc1(x[:, -1, :])) #x[:, -1, :] = last time step in the sequence for each sample in the batch
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc2(x))

        x = self.fc3(x)
        return x

input_size = X.shape[2]
model = Model(input_size)
model.to(device) #USE GPU

loss_fn = nn.HuberLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
from tqdm import tqdm

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device) #USE GPU
        optimizer.zero_grad() #clear previous gradients
        outputs = model(X_batch) #fwd pass to get predictions
        loss = loss_fn(outputs, y_batch) #calculate loss
        loss.backward() #backpropogate loss to calculate gradients
        optimizer.step() #update model parameters
        torch.cuda.empty_cache()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

model.eval()
with torch.no_grad():
    yhat = model(X_tensor).cpu().numpy() #MOVE FROM GPU TO CPU

data.index = pd.to_datetime(data.index)
data = data.reset_index()  
data = data.iloc[:-1]  

sns.set_style("darkgrid")
plt.figure(figsize=(10, 6))

sns.lineplot(x=data['Datetime'], y=y.to_numpy().ravel(), label="Actual Values (y)", color='blue', alpha=0.7)
sns.lineplot(x=data['Datetime'], y=yhat.flatten(), label="Predicted Values (y_hat)", color='red', alpha=0.7)

plt.title("Actual vs Predicted Values")
plt.xlabel("Time")
plt.ylabel("Price (USD)")

plt.legend()
plt.show()
