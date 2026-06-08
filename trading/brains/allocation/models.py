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
        self.X = self.X.dropna(axis=1)

        feat_arr = self.X.values.astype(np.float32)

        sequences = []
        for i in range(len(feat_arr) - self.seq_len + 1):
            sequences.append(feat_arr[i:i + self.seq_len])

        self.X = torch.from_numpy(np.array(sequences))
        # Transpose to (num_sequences, num_features, seq_len)
        self.X = self.X.transpose(1, 2)
        self.valid_indices = self.data.index[self.seq_len - 1:]

        # (num_sequences, num_features, seq_len)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

class StochasticLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if self.training:
            x += torch.randn_like(x)*0.4
        return x

class BasicModel(nn.Module):
    def __init__(self, channels, width):
        super().__init__()
        self.seq_len = width
        self.convolver = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.embedding = nn.Sequential(
            nn.Linear(32 * self.seq_len, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.convolver(x)
        x = x.flatten(1, 2)  #(B, C * W)
        embedding = self.embedding(x)
        return F.tanh(embedding).squeeze(-1)