import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

import scipy

import matplotlib.pyplot as plt

import sys
sys.path.append("")
import trading.model_tools as mt
import trading.technical_analysis as ta

class ExtremumDetector(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features=features)
        self.dense_1 = nn.Linear(features, 64)
        self.dense_2 = nn.Linear(64, 64)
        self.dense_3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.bn(x)
        x = self.dense_1(x)
        x = F.relu(x)
        x = self.dense_2(x)
        x = F.relu(x)
        x = self.dense_3(x)
        return self.sigmoid(x).squeeze()

data = mt.fetch_data(symbol="BTC-USDT", days=100, interval="15m", age_days=0, data_source="binance").drop(columns=["Datetime"], axis=1)
data["Kalman"] = ta.kalman_filter(data["Close"], process_noise=0.01, measurement_noise=0.01)

for i in range(1, 10):
    for column in data.columns:
        data[f"{column}_shift_{i}"] = data[column].shift(i)
data = data.dropna()

X = torch.tensor(data.values, dtype=torch.float32).to("cuda")

peaks = scipy.signal.argrelextrema(data["Close"].values, np.greater)
troughs = scipy.signal.argrelextrema(data["Close"].values, np.less)
y = np.zeros_like(data["Close"].values)
y[peaks] = 1
y[troughs] = 1
y = torch.tensor(y, dtype=torch.float32).to("cuda")

model = ExtremumDetector(features=X.shape[1]).to("cuda")

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

losses = []
accuracies = []

for epoch in tqdm(range(10000)):
    optimizer.zero_grad()
    predictions = model(X)
    loss = criterion(predictions, y)
    loss.backward()
    optimizer.step()

    correct = predictions.round() == y
    accuracy = correct.sum() / len(correct)
    losses.append(loss.item())
    accuracies.append(accuracy.cpu())

with torch.no_grad():
    predictions = model(X)
    print(predictions)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(losses)
    ax2.plot(accuracies)
    plt.show()