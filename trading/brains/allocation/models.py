import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

from scipy.signal import savgol_filter

import trading.technical_analysis as ta

DROPOUT = 0.15


def ta_transform(data: pd.DataFrame, add_ticker: str):
    close = data["Close"]
    high = data["High"]
    low = data["Low"]

    macd_hist = ta.macd(close)[2]
    aroon_up, aroon_down = ta.aroon(high, low, timeperiod=14)
    stoch_k, stoch_d = ta.stoch(high, low, close)

    ret_1 = close.pct_change(1)
    ret_5 = close.pct_change(5)
    ret_20 = close.pct_change(20)
    log_ret = ta.log_return(close)

    add_close = data[f"add_{add_ticker}_Close"]
    add_high = data[f"add_{add_ticker}_High"]
    add_low = data[f"add_{add_ticker}_Low"]

    add_macd_hist = ta.macd(add_close)[2]
    add_aroon_up, add_aroon_down = ta.aroon(add_high, add_low, timeperiod=14)
    add_stoch_k, add_stoch_d = ta.stoch(add_high, add_low, add_close)

    add_ret_1 = add_close.pct_change(1)
    add_ret_5 = add_close.pct_change(5)
    add_ret_20 = add_close.pct_change(20)
    add_log_ret = ta.log_return(add_close)


    return pd.DataFrame(
        {
            "ret_1": ret_1,
            "ret_5": ret_5,
            "ret_20": ret_20,
            "log_ret": log_ret,
            "roc_10": ta.roc(close, timeperiod=10),
            "mom_10": ta.mom(close, timeperiod=10),
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
            "macd_hist": macd_hist,
            "aroon_osc": aroon_up - aroon_down,

            "add_ret_1": add_ret_1,
            "add_ret_5": add_ret_5,
            "add_ret_20": add_ret_20,
            "add_log_ret": add_log_ret,
            "add_roc_10": ta.roc(add_close, timeperiod=10),
            "add_mom_10": ta.mom(add_close, timeperiod=10),
            "add_stoch_k": add_stoch_k,
            "add_stoch_d": add_stoch_d,
            "add_macd_hist": add_macd_hist,
            "add_aroon_osc": add_aroon_up - add_aroon_down,
        }
    ).dropna(axis=0)


class PriceDataset(Dataset):
    def __init__(self, data: pd.DataFrame, seq_len: int = 10):
        self.seq_len = seq_len
        self.data = data

        self.X = ta_transform(self.data, add_ticker="BTC-USDT")
        self.valid_indices = self.X.index[self.seq_len - 1 :]

        feat_arr = self.X.values.astype(np.float32)

        sequences = []
        for i in range(len(feat_arr) - self.seq_len + 1):
            sequences.append(feat_arr[i : i + self.seq_len])

        self.X = torch.from_numpy(np.array(sequences))
        # Transpose to (num_sequences, num_features, seq_len)
        self.X = self.X.transpose(1, 2)

        # (num_sequences, num_features, seq_len)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Return both the data and the corresponding dataframe index
        return self.X[idx], idx


class ConvolutionEncoder(nn.Module):
    def __init__(
        self,
        channels,
        width,
        hidden_channel_size=64,
        hidden_linear_size=256,
        out_size=256,
    ):
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
            nn.Conv1d(
                hidden_channel_size,
                hidden_channel_size,
                kernel_size=3,
                padding=1,
                groups=hidden_channel_size,
            ),
            nn.GroupNorm(8, hidden_channel_size),
            nn.ReLU(),
            nn.Conv1d(hidden_channel_size, hidden_channel_size * 2, kernel_size=1),
            nn.GroupNorm(8, hidden_channel_size * 2),
            nn.ReLU(),
            nn.Conv1d(
                hidden_channel_size * 2,
                hidden_channel_size * 2,
                kernel_size=3,
                padding=1,
                groups=hidden_channel_size * 2,
            ),
            nn.GroupNorm(8, hidden_channel_size * 2),
            nn.ReLU(),
            nn.Conv1d(hidden_channel_size * 2, hidden_channel_size, kernel_size=1),
            nn.GroupNorm(8, hidden_channel_size),
            nn.ReLU(),
            nn.Conv1d(
                hidden_channel_size, hidden_channel_size, kernel_size=3, padding=1
            ),
            nn.GroupNorm(8, hidden_channel_size),
            nn.ReLU(),
        )

        self.encoder = nn.Sequential(
            nn.Linear(hidden_channel_size * self.width, hidden_linear_size),
            nn.GroupNorm(8, hidden_linear_size),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_linear_size, hidden_linear_size),
            nn.GroupNorm(8, hidden_linear_size),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_linear_size, hidden_linear_size),
            nn.GroupNorm(8, hidden_linear_size),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_linear_size, self.out_size),
        )

    def forward(self, x):
        x = self.convolver(x)
        x = x.flatten(1, 2)
        embedding = self.encoder(x)
        return embedding


class LSTMEncoder(nn.Module):
    def __init__(self, channels, width, hidden_size=64, out_size=256):
        super().__init__()

        self.channels = channels
        self.width = width
        self.hidden_size = hidden_size
        self.out_size = out_size

        self.lstm = nn.LSTM(channels, hidden_size, batch_first=True)
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size * width, 128),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 128),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 128),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, out_size),
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, W, C)
        x, _ = self.lstm(x)
        x = x.flatten(1, 2)  # (B, hidden_size * W)
        embedding = self.encoder(x)
        return embedding


class AllocatorPolicy(nn.Module):
    def __init__(
        self,
        channels,
        width,
        conv_hidden_channel_size=64,
        conv_hidden_linear_size=64,
        conv_out_size=64,
        lstm_hidden_size=64,
        lstm_out_size=64,
    ):

        super().__init__()

        self.conv = ConvolutionEncoder(
            channels,
            width,
            hidden_channel_size=conv_hidden_channel_size,
            hidden_linear_size=conv_hidden_linear_size,
            out_size=conv_out_size,
        )
        self.lstm = LSTMEncoder(
            channels, width, hidden_size=lstm_hidden_size, out_size=lstm_out_size
        )

        self.out_size = self.conv.out_size + self.lstm.out_size

        self.dist_encoder = nn.Sequential(
            nn.Linear(self.out_size, 128),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 128),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 128),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )

        self.mean_head = nn.Linear(128, 1)
        self.log_std_head = nn.Linear(128, 1)

    def get_action(self, obs):
        mean, log_std = self.forward(obs)
        if self.training:
            log_std = torch.clamp(log_std, -40, 20)
            std = torch.exp(log_std)

            epsilon = torch.randn_like(mean)
            action = mean + std * epsilon
        else:
            action = mean

        ### tanh squish
        return F.tanh(action).squeeze(-1)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.lstm(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.dist_encoder(x)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)

        return mean, log_std
