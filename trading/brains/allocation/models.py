import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from scipy.signal import savgol_filter

import trading.technical_analysis as ta

class PriceDataset(Dataset):
    def __init__(self, data: pd.DataFrame, seq_len: int = 10):
        self.seq_len = seq_len
        self.data = data

        close = pd.Series(ta.tema(self.data['Close'], timeperiod=2), index=self.data.index)
        high = pd.Series(ta.tema(self.data['High'], timeperiod=2), index=self.data.index)
        low = pd.Series(ta.tema(self.data['Low'], timeperiod=2), index=self.data.index)

        macd_hist = ta.macd(close)[2]
        macd_dema_hist = ta.macd_dema(close)[2]
        aroon_up, aroon_down = ta.aroon(high, low, timeperiod=14)
        stoch_k, stoch_d = ta.stoch(high, low, close)

        ret_1 = close.pct_change(1)
        ret_5 = close.pct_change(5)
        ret_20 = close.pct_change(20)
        log_ret = ta.log_return(close)

        self.X = pd.DataFrame({
            'ret_1': ret_1,
            'ret_5': ret_5,
            'ret_20': ret_20,
            'log_ret': log_ret,
            'roc_10': ta.roc(close, timeperiod=10),
            'mom_10': ta.mom(close, timeperiod=10),

            'stoch_k': stoch_k,
            'stoch_d': stoch_d,
            'macd_hist': macd_hist,
            'macd_dema_hist': macd_dema_hist,
            'aroon_osc': aroon_up - aroon_down,
        })
        self.X = self.X.dropna(axis=0)
        self.valid_indices = self.X.index[self.seq_len - 1:]

        feat_arr = self.X.values.astype(np.float32)

        sequences = []
        for i in range(len(feat_arr) - self.seq_len + 1):
            sequences.append(feat_arr[i:i + self.seq_len])

        self.X = torch.from_numpy(np.array(sequences))
        # Transpose to (num_sequences, num_features, seq_len)
        self.X = self.X.transpose(1, 2)

        # (num_sequences, num_features, seq_len)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Return both the data and the corresponding dataframe index
        return self.X[idx], idx

class StochasticLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if self.training:
            x += torch.randn_like(x)*0.4
        return x

class ConvolutionHead(nn.Module):
    def __init__(self, channels, width, hidden_channel_size=64, hidden_linear_size=256, out_size=256):
        super().__init__()

        self.channels = channels
        self.width = width
        self.hidden_channel_size = hidden_channel_size
        self.hidden_linear_size = hidden_linear_size
        self.out_size = out_size

        self.convolver = nn.Sequential(
            nn.Conv1d(channels, hidden_channel_size, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channel_size),
            nn.ReLU(),
            nn.Conv1d(hidden_channel_size, hidden_channel_size, kernel_size=3, padding=1, groups=hidden_channel_size),
            nn.GroupNorm(8, hidden_channel_size),
            nn.ReLU(),
            nn.Conv1d(hidden_channel_size, hidden_channel_size * 2, kernel_size=1),
            nn.GroupNorm(8, hidden_channel_size * 2),
            nn.ReLU(),
            nn.Conv1d(hidden_channel_size * 2, hidden_channel_size * 2, kernel_size=3, padding=1, groups=hidden_channel_size * 2),
            nn.GroupNorm(8, hidden_channel_size * 2),
            nn.ReLU(),
            nn.Conv1d(hidden_channel_size * 2, hidden_channel_size, kernel_size=1),
            nn.GroupNorm(8, hidden_channel_size),
            nn.ReLU(),
            nn.Conv1d(hidden_channel_size, hidden_channel_size, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channel_size),
            nn.ReLU(),
        )

        self.embedding = nn.Sequential(
            nn.Linear(hidden_channel_size * self.width, hidden_linear_size),
            nn.GroupNorm(8, hidden_linear_size),
            nn.ReLU(),
            nn.Linear(hidden_linear_size, hidden_linear_size),
            nn.GroupNorm(8, hidden_linear_size),
            nn.ReLU(),
            nn.Linear(hidden_linear_size, hidden_linear_size),
            nn.GroupNorm(8, hidden_linear_size),
            nn.ReLU(),
            nn.Linear(hidden_linear_size, self.out_size),
        )

    def forward(self, x):
        x = self.convolver(x)
        x = x.flatten(1, 2)
        embedding = self.embedding(x)
        return embedding

class LSTMHead(nn.Module):
    def __init__(self, channels, width, hidden_size=64, out_size=256):
        super().__init__()

        self.channels = channels
        self.width = width
        self.hidden_size = hidden_size
        self.out_size = out_size

        self.lstm = nn.LSTM(channels, hidden_size, batch_first=True)
        self.embedding = nn.Sequential(
            nn.Linear(hidden_size * width, 128),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Linear(128, out_size),
        )

    def forward(self, x):
        x = x.transpose(1, 2)  #(B, W, C)
        x, _ = self.lstm(x)
        x = x.flatten(1, 2)  #(B, hidden_size * W)
        embedding = self.embedding(x)
        return embedding

class Allocator(nn.Module):
    def __init__(
            self,
            channels,
            width,
            conv_hidden_channel_size=64,
            conv_hidden_linear_size=64,
            conv_out_size=64,
            lstm_hidden_size=64,
            lstm_out_size=64
        ):

        super().__init__()

        self.conv = ConvolutionHead(channels, width, hidden_channel_size=conv_hidden_channel_size, hidden_linear_size=conv_hidden_linear_size, out_size=conv_out_size)
        self.lstm = LSTMHead(channels, width, hidden_size=lstm_hidden_size, out_size=lstm_out_size)

        self.out_size = self.conv.out_size + self.lstm.out_size

        self.fc = nn.Sequential(
            nn.Linear(self.out_size, 128),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )
    
    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.lstm(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.fc(x)
        return x.squeeze(-1)
