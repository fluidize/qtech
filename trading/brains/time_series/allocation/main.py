import numpy as np
import pandas as pd
from tqdm import tqdm
from pandas._libs.tslibs.ccalendar import DAYS
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from scipy.signal import savgol_filter

import trading.model_tools as mt
import trading.technical_analysis as ta

class PriceDataset(Dataset):
    def __init__(self, data: pd.DataFrame, shift: int = 0):
        self.data = data

        close = self.data['Close']
        high = self.data['High']
        low = self.data['Low']
        open_price = self.data['Open']
        volume = self.data['Volume']
        
        adx, plus_di, minus_di = ta.adx(high, low, close, timeperiod=14)
        macd_line, macd_signal, macd_hist = ta.macd(close)
        macd_dema_line, macd_dema_signal, macd_dema_hist = ta.macd_dema(close)
        aroon_up, aroon_down = ta.aroon(high, low, timeperiod=14)
        stoch_k, stoch_d = ta.stoch(high, low, close)
        tsi_line, tsi_signal = ta.tsi(close)
        ppo_line, ppo_signal, ppo_hist = ta.ppo(close)
        bb_upper, bb_middle, bb_lower = ta.bbands(close, timeperiod=20)
        kc_upper, kc_middle, kc_lower = ta.keltner_channels(high, low, close, timeperiod=20)
        vwap_upper, vwap_middle, vwap_lower = ta.vwap_bands(high, low, close, volume, timeperiod=20)
        aobv_fast, aobv_slow = ta.aobv(close, volume)
        elder_ray_bull, elder_ray_bear = ta.elder_ray(high, low, close, timeperiod=13)
        
        self.X = pd.DataFrame({
            'returns_5': close.pct_change(5),
            'macd_dema_hist': macd_dema_hist,
            'awesome_oscillator': ta.awesome_oscillator(high, low, fast_period=5, slow_period=34),
            'elder_ray_bear': elder_ray_bear,
            'elder_ray_bear': elder_ray_bear,
            'elder_ray_bear': elder_ray_bear,
            'elder_ray_bear': elder_ray_bear,
            'hurst_exponent': ta.hurst_exponent(close, max_lag=20),
            'hurst_exponent': ta.hurst_exponent(close, max_lag=20),
            'elder_ray_bear': elder_ray_bear,
            'vzo': ta.volume_zone_oscillator(close, volume),
            'vzo': ta.volume_zone_oscillator(close, volume),
            'hurst_exponent': ta.hurst_exponent(close, max_lag=20),
            'hurst_exponent': ta.hurst_exponent(close, max_lag=20),
            'choppiness_index': ta.choppiness_index(high, low, close, timeperiod=14),
            'mass_index': ta.mass_index(high, low, timeperiod=25),
            'vzo': ta.volume_zone_oscillator(close, volume),
            'mass_index': ta.mass_index(high, low, timeperiod=25),
            'vzo': ta.volume_zone_oscillator(close, volume),
            'hurst_exponent': ta.hurst_exponent(close, max_lag=20),
            'minus_di': minus_di,
            'aobv_slow': aobv_slow,
            'mass_index': ta.mass_index(high, low, timeperiod=25),
            'aobv_slow': aobv_slow,
            'mass_index': ta.mass_index(high, low, timeperiod=25),
            'stoch_d': stoch_d,
            'vzo': ta.volume_zone_oscillator(close, volume),
            'stoch_d': stoch_d,
            'dpo_20': ta.dpo(close, timeperiod=20),
            'elder_ray_bull': elder_ray_bull,
            'aobv_slow': aobv_slow,
            'dpo_20': ta.dpo(close, timeperiod=20),
            'fisher_transform': ta.fisher_transform(close, timeperiod=10),
            'obv': ta.obv(close, volume),
            'aobv_slow': aobv_slow,
            'stoch_d': stoch_d,
            'mass_index': ta.mass_index(high, low, timeperiod=25),
            'pvt': ta.pvt(close, volume),
            'obv': ta.obv(close, volume),
            'dpo_20': ta.dpo(close, timeperiod=20),
            'cmf_20': ta.cmf(high, low, close, volume, timeperiod=20),
            'stoch_d': stoch_d,
            'plus_di': plus_di,
            'price_cycle': ta.price_cycle(close, cycle_period=20),
            'pvt': ta.pvt(close, volume),
            'obv': ta.obv(close, volume),
            'dpo_20': ta.dpo(close, timeperiod=20),
            'mfi_14': ta.mfi(high, low, close, volume, timeperiod=14),
            'aobv_slow': aobv_slow,
            'kc_width': (kc_upper - kc_lower) / kc_middle,
            'pvt': ta.pvt(close, volume),
            'roc_10': ta.roc(close, timeperiod=10),
            'bb_width': (bb_upper - bb_lower) / bb_middle,
            'mom_10': ta.mom(close, timeperiod=10),
            'aroon_down': aroon_down,
            'ppo_hist': ppo_hist,
            'vwap_lower': vwap_lower,
            'ppo_signal': ppo_signal,
            'macd_line': macd_line,
            'macd_signal': macd_signal,
            'tsi_line': tsi_line,
            'log_return': ta.log_return(close),
            'mom_20': ta.mom(close, timeperiod=20),
            'macd_dema_signal': macd_dema_signal,
            'stoch_k': stoch_k,
            'zscore_20': ta.zscore(close, timeperiod=20),
            'aroon_oscillator': aroon_up - aroon_down,
            'rsi_14': ta.rsi(close, timeperiod=14),
            'macd_dema_line': macd_dema_line,
            'rsi_21': ta.rsi(close, timeperiod=21),
            'vwap': ta.vwap(high, low, close, volume),
            'willr_14': ta.willr(high, low, close, timeperiod=14),
            'ppo_line': ppo_line,
            'vwap_lower': vwap_lower,
            'ppo_signal': ppo_signal,
            'macd_line': macd_line,
            'macd_signal': macd_signal,
            'price_cycle': ta.price_cycle(close, cycle_period=20),
            'zscore_50': ta.zscore(close, timeperiod=50),
            'rvi': ta.rvi(open_price, high, low, close, timeperiod=10),
            'macd_hist': macd_hist,
            'aroon_up': aroon_up,
            'tsi_signal': tsi_signal,
            'volatility_20': ta.volatility(close, timeperiod=20),
            'roc_10': ta.roc(close, timeperiod=10),
            'atr_14': ta.atr(high, low, close, timeperiod=14),
            'mom_10': ta.mom(close, timeperiod=10),
            'roc_20': ta.roc(close, timeperiod=20),
            'atr_20': ta.atr(high, low, close, timeperiod=20),
            'tsi_line': tsi_line,
            'log_return': ta.log_return(close),
        })

        shifted_cols = {}
        for i in range(shift):
            for col in self.X.columns:
                shifted_cols[col + f'_{i}'] = self.X[col].shift(i)

        if shifted_cols:
            self.X = pd.concat([self.X, pd.DataFrame(shifted_cols)], axis=1)

        y = pd.Series(savgol_filter(self.data['Close'].rolling(window=5).mean(), window_length=20, polyorder=4, deriv=1), index=self.data.index)
        y[y > 0] = 1
        y[y < 0] = 0

        mask = ~(self.X.isna().any(axis=1) | y.isna())
        self.valid_indices = self.data.index[mask]
        self.X = self.X[mask].values.astype(np.float32)
        self.y = y[mask].values.astype(np.float32) 

        
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class AllocationModel(nn.Module):
    def __init__(self, input_dim: int, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        
        self.feature_block = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.main_network = nn.Sequential(
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_block(x)
        x = self.main_network(x)
        x = nn.Sigmoid()(x)
        return x.squeeze()

def train_model(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = loss_fn(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(dataloader.dataset)

def evaluate_model(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            loss = loss_fn(output, y_batch)
            total_loss += loss.item() * len(X_batch)
    return total_loss / len(dataloader.dataset)

def evaluate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            predictions = (output > 0.5).float()
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)
    return correct / total

if __name__ == "__main__":
    from trading.backtesting.backtesting import VectorizedBacktesting
    import matplotlib.pyplot as plt
    import faulthandler
    faulthandler.enable()

    epochs = 10000
    shifts = 10
    
    dataset = PriceDataset(mt.fetch_data(symbol="BTC-USDT", days=10, interval="30m", age_days=0, data_source="binance"), shift=shifts)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    model = AllocationModel(input_dim=dataset.X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = model.to(device)

    progress_bar = tqdm(total=epochs, desc="Training")
    losses = []
    for epoch in range(epochs):
        loss = train_model(model, dataloader, loss_fn, optimizer, device)
        val_loss = evaluate_model(model, dataloader, loss_fn, device)
        accuracy = evaluate_accuracy(model, dataloader, device)

        progress_bar.set_description(f"Epoch {epoch+1}, Loss: {loss}, Accuracy: {accuracy}")
        losses.append(loss)
        progress_bar.update(1)
    progress_bar.close()

    plt.plot(losses)
    plt.show()

    def model_wrapper(data):
        dataset = PriceDataset(data, shift=shifts)
        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(dataset.X, dtype=torch.float32).to(device)
            predictions = model(X_tensor).cpu().numpy()
        
        signals = pd.Series(0.0, index=data.index)
        signals.loc[dataset.valid_indices] = predictions
        return signals

    vb = VectorizedBacktesting(
        instance_name="AllocationModel",
        initial_capital=10000,
        slippage_pct=0.00,
        commission_fixed=0.0,
        leverage=1.0
    )
    vb.fetch_data(symbol="BTC-USDT", days=10, interval="30m", age_days=0, data_source="binance")
    vb.run_strategy(model_wrapper, verbose=True)
    vb.plot_performance(mode="basic")
