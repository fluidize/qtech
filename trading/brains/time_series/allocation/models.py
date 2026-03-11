import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter
import torch.nn.functional as F

import trading.technical_analysis as ta


class PriceDataset(Dataset):
    def __init__(self, data: pd.DataFrame, shift: int = 0):
        self.data = data

        close = pd.Series(ta.tema(self.data['Close'], timeperiod=2), index=self.data.index)
        high = pd.Series(ta.tema(self.data['High'], timeperiod=2), index=self.data.index)
        low = pd.Series(ta.tema(self.data['Low'], timeperiod=2), index=self.data.index)
        open_price = pd.Series(ta.tema(self.data['Open'], timeperiod=2), index=self.data.index)
        volume = pd.Series(ta.tema(self.data['Volume'], timeperiod=2), index=self.data.index)
        
        # Oscillators
        macd_line, macd_signal, macd_hist = ta.macd(close)
        macd_dema_line, macd_dema_signal, macd_dema_hist = ta.macd_dema(close)
        aroon_up, aroon_down = ta.aroon(high, low, timeperiod=14)
        stoch_k, stoch_d = ta.stoch(high, low, close)
        tsi_line, tsi_signal = ta.tsi(close)
        ppo_line, ppo_signal, ppo_hist = ta.ppo(close)
        bb_upper, bb_middle, bb_lower = ta.bbands(close, timeperiod=20)
        kc_upper, kc_middle, kc_lower = ta.keltner_channels(high, low, close, timeperiod=20)
        
        # Multi-horizon returns (different time scales)
        ret_1 = close.pct_change(1)
        ret_5 = close.pct_change(5)
        ret_20 = close.pct_change(20)
        log_ret = ta.log_return(close)
        # Volatility regime: short vs long vol
        vol_10 = ta.volatility(close, timeperiod=10)
        vol_20 = ta.volatility(close, timeperiod=20)
        vol_ratio = vol_10 / vol_20.replace(0, np.nan).clip(lower=1e-12)
        # Volume context (unusual volume)
        vol_ma = volume.rolling(20, min_periods=1).mean()
        volume_ratio = volume / vol_ma.replace(0, np.nan)
        # Price position in recent range (0–1)
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

        shifted_cols = {}
        for i in range(shift):
            for col in self.X.columns:
                shifted_cols[col + f'_{i}'] = self.X[col].shift(i)

        if shifted_cols:
            self.X = pd.concat([self.X, pd.DataFrame(shifted_cols)], axis=1)
        
        self.y = pd.Series(savgol_filter(self.data['Open'], window_length=20, polyorder=3, deriv=1))

        self.X = self.X.dropna(axis=1)
        combined = self.X.join(self.y.rename('y')).dropna()
        self.valid_indices = combined.index

        self.X = torch.tensor(combined[self.X.columns].values.astype(np.float32))
        self.y = torch.tensor(combined['y'].values.astype(np.float32))

    def split(self, test_size=0.2):
        train_idx, val_idx = train_test_split(range(len(self.X)), test_size=test_size, shuffle=False)
        return TensorSubset(self.data.iloc[train_idx], self.X[train_idx], self.y[train_idx]), TensorSubset(self.data.iloc[val_idx], self.X[val_idx], self.y[val_idx])

class TensorSubset(Dataset):
    def __init__(self, data: pd.DataFrame, X: torch.Tensor, y: torch.Tensor):
        self.data = data.reset_index(drop=True)
        self.X = X
        self.y = y
        self.valid_indices = self.data.index

    def __len__(self):
        return len(self.X)

class DistributionPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.mean_skip = nn.Linear(input_dim, 1)
        self.std_head = nn.Linear(hidden_dim, 1)
        self.std_skip = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        mean = self.mean_head(h).squeeze(-1) + self.mean_skip(x).squeeze(-1)
        std = F.softplus(self.std_head(h).squeeze(-1) + self.std_skip(x).squeeze(-1))
        return torch.stack([mean, std], dim=1)

class AllocationModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.05):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim + 2, hidden_dim),
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
        h = torch.cat([x, directional_estimate], dim=1)
        return torch.sigmoid(self.net(h)).squeeze(-1)


class CombinedModelWrapper(nn.Module):
    def __init__(self, input_dim: int, alloc_dropout: float = 0.1, dist_dropout: float = 0.3):
        super().__init__()
        self.distribution_model = DistributionPredictor(input_dim, dropout=dist_dropout)
        self.alloc = AllocationModel(input_dim, dropout=alloc_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dist_estimate = self.distribution_model(x)
        return self.alloc(x, dist_estimate)
