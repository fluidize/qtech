import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter

import trading.technical_analysis as ta


class PriceDataset(Dataset):
    def __init__(self, data: pd.DataFrame, seq_len: int = 32):
        self.data = data
        self.seq_len = seq_len

        close = pd.Series(ta.tema(self.data['Close'], timeperiod=2), index=self.data.index)
        high = pd.Series(ta.tema(self.data['High'], timeperiod=2), index=self.data.index)
        low = pd.Series(ta.tema(self.data['Low'], timeperiod=2), index=self.data.index)
        open_price = pd.Series(ta.tema(self.data['Open'], timeperiod=2), index=self.data.index)
        volume = pd.Series(ta.tema(self.data['Volume'], timeperiod=2), index=self.data.index)

        macd_line, macd_signal, macd_hist = ta.macd(close)
        macd_dema_line, macd_dema_signal, macd_dema_hist = ta.macd_dema(close)
        aroon_up, aroon_down = ta.aroon(high, low, timeperiod=14)
        stoch_k, stoch_d = ta.stoch(high, low, close)
        tsi_line, tsi_signal = ta.tsi(close)
        ppo_line, ppo_signal, ppo_hist = ta.ppo(close)
        bb_upper, bb_middle, bb_lower = ta.bbands(close, timeperiod=20)
        kc_upper, kc_middle, kc_lower = ta.keltner_channels(high, low, close, timeperiod=20)

        ret_1 = close.pct_change(1)
        ret_5 = close.pct_change(5)
        ret_20 = close.pct_change(20)
        log_ret = ta.log_return(close)
        vol_10 = ta.volatility(close, timeperiod=10)
        vol_20 = ta.volatility(close, timeperiod=20)
        vol_ratio = vol_10 / vol_20.replace(0, np.nan).clip(lower=1e-12)
        vol_ma = volume.rolling(20, min_periods=1).mean()
        volume_ratio = volume / vol_ma.replace(0, np.nan)
        roll_20_high = high.rolling(20, min_periods=1).max()
        roll_20_low = low.rolling(20, min_periods=1).min()
        range_20 = (roll_20_high - roll_20_low).replace(0, np.nan)
        price_in_range = (close - roll_20_low) / range_20

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

        velocity = savgol_filter(self.data['Open'], window_length=25, polyorder=3, deriv=1)
        y_velocity = pd.Series(velocity, index=self.data.index)

        regime_observations = savgol_filter(self.data['Open'], window_length=75, polyorder=3, deriv=1)
        y_regime = pd.Series(0, index=self.data.index)
        y_regime[regime_observations > 0] = 1

        self.X = self.X.dropna(axis=1)
        combined = pd.concat([self.X, y_velocity.rename('y_velocity'), y_regime.rename('y_regime')], axis=1).dropna()
        aligned_data = self.data.loc[combined.index]

        feature_cols = list(self.X.columns)
        feat_arr = combined[feature_cols].values.astype(np.float32)
        yv = combined['y_velocity'].values.astype(np.float32)
        yr = combined['y_regime'].values.astype(np.float32)

        n = len(combined)
        n_windows = n - seq_len + 1
        X_windows = np.zeros((n_windows, seq_len, feat_arr.shape[1]), dtype=np.float32)
        for k in range(n_windows):
            X_windows[k] = feat_arr[k : k + seq_len]

        self.X = torch.from_numpy(X_windows)
        self.y_velocity = torch.tensor(yv[seq_len - 1 :], dtype=torch.float32)
        self.y_regime = torch.tensor(yr[seq_len - 1 :], dtype=torch.float32)

        self.data = aligned_data.iloc[seq_len - 1 :].copy()
        self.valid_indices = self.data.index

    def split(self, test_size=0.25):
        train_idx, val_idx = train_test_split(range(len(self.X)), test_size=test_size, shuffle=False)
        return (
            TensorSubset(
                self.data.iloc[train_idx].reset_index(drop=True),
                self.X[train_idx],
                self.y_velocity[train_idx],
                self.y_regime[train_idx],
            ),
            TensorSubset(
                self.data.iloc[val_idx].reset_index(drop=True),
                self.X[val_idx],
                self.y_velocity[val_idx],
                self.y_regime[val_idx],
            ),
        )


class TensorSubset(Dataset):
    def __init__(self, data: pd.DataFrame, X: torch.Tensor, y_velocity: torch.Tensor, y_regime: torch.Tensor):
        self.data = data
        self.X = X
        self.y_velocity = y_velocity
        self.y_regime = y_regime
        self.valid_indices = self.data.index

    def all_to_device(self, device='cuda'):
        self.X = self.X.to(device=device)
        self.y_velocity = self.y_velocity.to(device=device)
        self.y_regime = self.y_regime.to(device=device)

    def __len__(self):
        return len(self.X)


def _gru_last_hidden(gru: nn.GRU, x: torch.Tensor, dropout: nn.Dropout) -> torch.Tensor:
    _, h_n = gru(x)
    return dropout(h_n[-1])


class VelocityDistributionPredictor(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, hidden_dim: int = 256, dropout: float = 0.05):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        flat_dim = seq_len * input_dim
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.mean_skip = nn.Linear(flat_dim, 1)
        self.std_head = nn.Linear(hidden_dim, 1)
        self.std_skip = nn.Linear(flat_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = _gru_last_hidden(self.gru, x, self.dropout)
        flat = x.flatten(1)
        mean = self.mean_head(h).squeeze(-1) + self.mean_skip(flat).squeeze(-1)
        std = F.softplus(self.std_head(h).squeeze(-1) + self.std_skip(flat).squeeze(-1))
        return torch.stack([mean, std], dim=1)


class AllocationModel(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, hidden_dim: int = 256, dropout: float = 0.05):
        super().__init__()
        self.input_dim = input_dim
        flat_dim = seq_len * input_dim
        self.net = nn.Sequential(
            nn.Linear(flat_dim + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, directional_estimate: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(1)
        h = torch.cat([flat, directional_estimate], dim=1)
        return torch.sigmoid(self.net(h)).squeeze(-1)


class CombinedModelWrapper(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        alloc_dropout: float = 0.05,
        dist_dropout: float = 0.05,
    ):
        super().__init__()
        self.distribution_model = VelocityDistributionPredictor(input_dim, seq_len, dropout=dist_dropout)
        self.alloc = AllocationModel(input_dim, seq_len, dropout=alloc_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dist_estimate = self.distribution_model(x)
        return self.alloc(x, dist_estimate)
