import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter

import trading.technical_analysis as ta

DROPOUT = 1/8


def ta_transform(data: pd.DataFrame, add_ticker: str):
    scaler = MinMaxScaler()
    data = ta.heikin_ashi_transform(data)
    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    aroon_up, aroon_down = ta.aroon(high, low, timeperiod=14)
    stoch_k, stoch_d = ta.stoch(high, low, close)

    ### add data
    add_data = data[[f"add_{add_ticker}_Open", f"add_{add_ticker}_High", f"add_{add_ticker}_Low", f"add_{add_ticker}_Close", f"add_{add_ticker}_Volume"]].copy()
    add_data.columns = ["Open", "High", "Low", "Close", "Volume"]
    add_data = ta.heikin_ashi_transform(add_data)
    
    add_close = add_data["Close"]
    add_high = add_data["High"]
    add_low = add_data["Low"]
    add_volume = add_data["Volume"]

    add_aroon_up, add_aroon_down = ta.aroon(add_high, add_low, timeperiod=14)
    add_stoch_k, add_stoch_d = ta.stoch(add_high, add_low, add_close)

    df = pd.DataFrame(
        {
            "ret_5": close.pct_change(5),
            "ret_20": close.pct_change(20),
            "log_ret": ta.log_return(close),
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
            "macd_hist": ta.macd_hist(close),
            "aroon_osc": aroon_up - aroon_down,
            "rsi": ta.rsi(close, timeperiod=14),
            "vol_ratio": ta.vol_ratio(volume),
            "atr": ta.atr(high, low, close, timeperiod=14),

            "add_ret_5": add_close.pct_change(5),
            "add_ret_20": add_close.pct_change(20),
            "add_log_ret": ta.log_return(add_close),
            "add_stoch_k": add_stoch_k,
            "add_stoch_d": add_stoch_d,
            "add_macd_hist": ta.macd_hist(add_close),
            "add_aroon_osc": add_aroon_up - add_aroon_down,
            "add_rsi": ta.rsi(add_close, timeperiod=14),
            "add_vol_ratio": ta.vol_ratio(add_volume),
            "add_atr": ta.atr(add_high, add_low, add_close, timeperiod=14),
        }
    ).dropna(axis=0)
    df[df.columns] = scaler.fit_transform(df[df.columns])
    return df


class PriceDataset(Dataset):
    def __init__(self, data: pd.DataFrame, add_ticker: str, seq_len: int = 10):
        self.seq_len = seq_len
        self.data = data

        self.X = ta_transform(self.data, add_ticker=add_ticker)
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

class MultiScalePooling(nn.Module):
    def __init__(self, reduction_dim=128):
        super().__init__()

        self.reduction_dim = reduction_dim
        
        self.aap_1 = nn.AdaptiveAvgPool1d(1)
        self.aap_2 = nn.AdaptiveAvgPool1d(4)
        self.aap_3 = nn.AdaptiveAvgPool1d(8)

        self.channel_reducer = nn.Conv1d(64, self.reduction_dim, 1)

        self.output_dim = self.reduction_dim*(1+4+8)

    def forward(self, x):
        x1 = self.aap_1(x)
        x1 = self.channel_reducer(x1).flatten(1) #(B,64)

        x2 = self.aap_2(x)
        x2 = self.channel_reducer(x2).flatten(1) #(B,256)

        x3 = self.aap_3(x)
        x3 = self.channel_reducer(x3).flatten(1) #(B,512)

        x = torch.cat([x1,x2,x3], dim=1)

        return x

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

        self.msp = MultiScalePooling(reduction_dim=hidden_channel_size)

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
            # nn.Linear(hidden_channel_size * self.width, hidden_linear_size),
            nn.Linear(self.msp.output_dim, hidden_linear_size),
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
        x = self.msp(x)
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


class Booster(nn.Module):
    def __init__(self, lstm_embedding_size, conv_embedding_size, out_size=1):
        super().__init__()

        self.lstm_embedding_size = lstm_embedding_size
        self.conv_embedding_size = conv_embedding_size
        self.out_size = out_size

        self.encoder = nn.Sequential(
            nn.Linear(lstm_embedding_size + conv_embedding_size, 128),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 128),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, out_size),
        )
    
    def forward(self, conv_embedding, lstm_embedding):
        x = torch.cat([conv_embedding, lstm_embedding], dim=1)
        output = self.encoder(x)
        return output
        


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

        self.mean_booster = Booster(lstm_out_size, conv_out_size, out_size=1)
        self.log_std_booster = Booster(lstm_out_size, conv_out_size, out_size=1)

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

        mean = self.mean_head(x) + self.mean_booster(x1, x2)
        log_std = self.log_std_head(x) + self.log_std_booster(x1, x2)

        return mean, log_std
