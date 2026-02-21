import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter

import matplotlib.pyplot as plt

import trading.model_tools as mt
import trading.technical_analysis as ta

from trading.backtesting.backtesting import VectorizedBacktesting
import loss_functions as lf

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
        return TensorSubset(self.data, self.X[train_idx], self.y[train_idx]), TensorSubset(self.data, self.X[val_idx], self.y[val_idx])

class TensorSubset(Dataset):
    def __init__(self, data: pd.DataFrame, X: torch.Tensor, y: torch.Tensor):
        self.data = data
        self.X = X
        self.y = y
        self.valid_indices = range(len(self.X))

    def __len__(self):
        return len(self.X)

class DirectionalConfidencePredictor(nn.Module):
    def __init__(self, input_dim, dropout=0.03):
        super().__init__()
        self.input_dim = input_dim
        
        self.main_network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.main_network(x) #y2 = upper bound, y1 = lower bound
        return x.squeeze()

class AllocationModel(nn.Module):
    def __init__(self, input_dim, dropout=0.03):
        super().__init__()
        self.input_dim = input_dim
        
        self.main_network = nn.Sequential(
            nn.Linear(input_dim + 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor, directional_estimate: torch.Tensor) -> torch.Tensor:
        x = self.main_network(torch.cat([x, directional_estimate], dim=1))
        x = nn.Sigmoid()(x).squeeze()
        return x

def model_wrapper(data, alloc_model, directional_model, device):
    dataset = PriceDataset(data, shift=SHIFTS)
    alloc_model.eval()
    directional_model.eval()
    with torch.no_grad():
        X_tensor = dataset.X.to(device)
        directional_estimate = directional_model(X_tensor)
        predictions = alloc_model(X_tensor, directional_estimate).cpu().numpy()

    signals = pd.Series(0.0, index=data.index)
    signals.loc[dataset.valid_indices] = predictions
    return signals
### Training ###
if __name__ == "__main__":
    EPOCHS = 10000
    SHIFTS = 100
    DATA = {
        "symbol": "SOL-USDT",
        "days":365,
        "interval": "1h",
        "age_days": 0,
        "data_source": "binance",
        "cache_expiry_hours": 999,
        "verbose": True
    }
    DEVICE = 'cuda'

    data = mt.fetch_data(**DATA)
    full_dataset = PriceDataset(data, shift=SHIFTS)
    train_dataset, val_dataset = full_dataset.split(test_size=0.5)

    alloc_model = AllocationModel(input_dim=train_dataset.X.shape[1]).to(DEVICE)
    directional_confidence_model = DirectionalConfidencePredictor(input_dim=train_dataset.X.shape[1]).to(DEVICE)

    alloc_optimizer = optim.Adam(alloc_model.parameters(), weight_decay=1e-5)  
    directional_confidence_optimizer = optim.Adam(directional_confidence_model.parameters(), weight_decay=1e-5)

    alloc_scheduler = optim.lr_scheduler.CosineAnnealingLR(alloc_optimizer, T_max=EPOCHS, eta_min=1e-8)
    directional_confidence_scheduler = optim.lr_scheduler.CosineAnnealingLR(directional_confidence_optimizer, T_max=EPOCHS, eta_min=1e-8)

    best_val_alloc_loss = float('inf')
    best_alloc_model_state = None

    alloc_loss_fn = lf.SharpeLoss(device=DEVICE)
    directional_confidence_loss_fn = lf.IntervalLoss(device=DEVICE)

    progress_bar = tqdm(total=EPOCHS, desc="Training")
    alloc_train_losses = []
    alloc_val_losses = []
    directional_train_losses = []
    directional_val_losses = []
    for epoch in range(EPOCHS):
        directional_confidence_model.train()
        alloc_model.train()

        directional_confidence_optimizer.zero_grad()
        alloc_optimizer.zero_grad()

        directional_confidence_train_loss = directional_confidence_loss_fn(
            directional_confidence_model(train_dataset.X.to(DEVICE)), train_dataset.y.to(DEVICE)
        )
        directional_confidence_train_loss.backward()

        alloc_train_loss = alloc_loss_fn(alloc_model, directional_confidence_model, train_dataset)
        alloc_train_loss.backward()

        directional_confidence_optimizer.step()
        alloc_optimizer.step()

        directional_confidence_scheduler.step()
        alloc_scheduler.step()

        # evaluation
        alloc_model.eval()
        directional_confidence_model.eval()
        with torch.no_grad():
            val_alloc_loss = alloc_loss_fn(alloc_model, directional_confidence_model, val_dataset)
            val_directional_loss = directional_confidence_loss_fn(
                directional_confidence_model(val_dataset.X.to(DEVICE)), val_dataset.y.to(DEVICE)
            )

        alloc_train_losses.append(alloc_train_loss.item())
        alloc_val_losses.append(val_alloc_loss.item())
        directional_train_losses.append(directional_confidence_train_loss.item())
        directional_val_losses.append(val_directional_loss.item())

        if val_alloc_loss < best_val_alloc_loss:
            best_val_alloc_loss = val_alloc_loss
            best_alloc_model_state = alloc_model.state_dict().copy()

        progress_bar.set_description(
            f"Epoch {epoch+1} | alloc T: {alloc_train_losses[-1]:.4f} V: {alloc_val_losses[-1]:.4f} | "
            f"dir T: {directional_train_losses[-1]:.4f} V: {directional_val_losses[-1]:.4f}"
        )
        progress_bar.update(1)
    progress_bar.close()

    # if best_alloc_model_state is not None:
        # alloc_model.load_state_dict(best_alloc_model_state)
        # print(f"Loaded best model with validation loss: {best_val_alloc_loss:.6f}")

    fig, (ax_alloc, ax_dir) = plt.subplots(1, 2, figsize=(12, 4))
    ax_alloc.plot(alloc_train_losses, label='Train')
    ax_alloc.plot(alloc_val_losses, label='Val')
    ax_alloc.set_xlabel('Epoch')
    ax_alloc.set_ylabel('Loss')
    ax_alloc.set_title('Alloc')
    ax_alloc.legend()
    ax_alloc.grid(True)
    ax_dir.plot(directional_train_losses, label='Train')
    ax_dir.plot(directional_val_losses, label='Val')
    ax_dir.set_xlabel('Epoch')
    ax_dir.set_ylabel('Loss')
    ax_dir.set_title('Directional')
    ax_dir.legend()
    ax_dir.grid(True)
    plt.tight_layout()
    plt.show(block=False)

    vb = VectorizedBacktesting(
        instance_name="AllocationModel",
        initial_capital=10000,
        slippage_pct=0.00,
        commission_fixed=0.0,
        leverage=1.0
    )
    vb.load_data(train_dataset.data, symbol=DATA["symbol"], interval=DATA["interval"], age_days=DATA["age_days"])
    vb.run_strategy(model_wrapper, verbose=True, alloc_model=alloc_model, directional_model=directional_confidence_model, device=DEVICE)
    vb.plot_performance(mode="basic")
