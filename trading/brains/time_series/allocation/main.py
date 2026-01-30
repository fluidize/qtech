import numpy as np
import pandas as pd
from sklearn.utils.validation import validate_data
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

import faulthandler
faulthandler.enable()

import trading.model_tools as mt
import trading.technical_analysis as ta
from trading.backtesting.backtesting import VectorizedBacktesting

class PriceDataset(Dataset):
    def __init__(self, data: pd.DataFrame, shift: int = 0):
        self.data = data

        close = self.data['Close']
        high = self.data['High']
        low = self.data['Low']
        open_price = self.data['Open']
        volume = self.data['Volume']
        
        # Oscillators
        macd_line, macd_signal, macd_hist = ta.macd(close)
        macd_dema_line, macd_dema_signal, macd_dema_hist = ta.macd_dema(close)
        aroon_up, aroon_down = ta.aroon(high, low, timeperiod=14)
        stoch_k, stoch_d = ta.stoch(high, low, close)
        tsi_line, tsi_signal = ta.tsi(close)
        ppo_line, ppo_signal, ppo_hist = ta.ppo(close)
        bb_upper, bb_middle, bb_lower = ta.bbands(close, timeperiod=20)
        kc_upper, kc_middle, kc_lower = ta.keltner_channels(high, low, close, timeperiod=20)
        
        # Momentum
        # (computed inline)
        
        # Volatility
        # (computed inline)
        
        self.X = pd.DataFrame({
            # Oscillators
            'stoch_k': stoch_k,
            'stoch_d': stoch_d,
            'rsi_14': ta.rsi(close, timeperiod=14),
            'rsi_21': ta.rsi(close, timeperiod=21),
            'willr_14': ta.willr(high, low, close, timeperiod=14),
            'awesome_oscillator': ta.awesome_oscillator(high, low, fast_period=5, slow_period=34),
            'aroon_oscillator': aroon_up - aroon_down,
            'fisher_transform': ta.fisher_transform(close, timeperiod=10),
            'vzo': ta.volume_zone_oscillator(close, volume),
            'mfi_14': ta.mfi(high, low, close, volume, timeperiod=14),
            'cmf_20': ta.cmf(high, low, close, volume, timeperiod=20),
            'rvi': ta.rvi(open_price, high, low, close, timeperiod=10),
            'tsi_line': tsi_line,
            'tsi_signal': tsi_signal,
            'ppo_line': ppo_line,
            'ppo_signal': ppo_signal,
            'ppo_hist': ppo_hist,
            'macd_line': macd_line,
            'macd_signal': macd_signal,
            'macd_hist': macd_hist,
            'macd_dema_line': macd_dema_line,
            'macd_dema_signal': macd_dema_signal,
            'macd_dema_hist': macd_dema_hist,
            
            # Momentum
            'returns_5': close.pct_change(5),
            'log_return': ta.log_return(close),
            'roc_10': ta.roc(close, timeperiod=10),
            'roc_20': ta.roc(close, timeperiod=20),
            'mom_10': ta.mom(close, timeperiod=10),
            'mom_20': ta.mom(close, timeperiod=20),
            'dpo_20': ta.dpo(close, timeperiod=20),
            
            # Volatility
            'volatility_20': ta.volatility(close, timeperiod=20),
            'atr_14': ta.atr(high, low, close, timeperiod=14),
            'atr_20': ta.atr(high, low, close, timeperiod=20),
            'bb_width': (bb_upper - bb_lower) / bb_middle,
            'kc_width': (kc_upper - kc_lower) / kc_middle,
            'choppiness_index': ta.choppiness_index(high, low, close, timeperiod=14),
        })

        shifted_cols = {}
        for i in range(shift):
            for col in self.X.columns:
                shifted_cols[col + f'_{i}'] = self.X[col].shift(i)

        if shifted_cols:
            self.X = pd.concat([self.X, pd.DataFrame(shifted_cols)], axis=1)

        y = pd.Series(savgol_filter(self.data['Close'], window_length=100, polyorder=3, deriv=1), index=self.data.index)

        #y = pd.Series(self.data['Close'].pct_change())
        # y[y > 0] = 1
        # y[y < 0] = 0

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
        
        self.main_network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
#            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
#            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
#            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.main_network(x)
        # x = nn.Sigmoid()(x)
        return x.squeeze()

class Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        # loss = -(target * torch.log(output) + (1 - target) * torch.log(1 - output)).mean()
        loss = torch.abs(output - target).mean()
        return loss

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

def evaluate_loss(model, dataloader, loss_fn, device):
    """Evaluate model on validation/test set and return loss."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            loss = loss_fn(output, y_batch)
            total_loss += loss.item() * len(X_batch)
            correct += ((output > 0.5) == y_batch).sum().item()
            total += y_batch.size(0)
    return total_loss / len(dataloader.dataset), correct / total

def model_wrapper(data):
    dataset = PriceDataset(data, shift=SHIFTS)
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(dataset.X, dtype=torch.float32).to(device)
        predictions = model(X_tensor).cpu().numpy()
    
    # predictions[predictions > 0] = 1
    # predictions[predictions < 0] = -1

    signals = pd.Series(0.0, index=data.index)
    signals.loc[dataset.valid_indices] = predictions
    return signals

### Training ###

EPOCHS = 500
SHIFTS = 100
DATA = {
    "symbol": "SOL-USDT",
    "days": 180,
    "interval": "30m",
    "age_days": 0,
    "data_source": "binance",
    "cache_expiry_hours": 999,
    "verbose": True
}

data = mt.fetch_data(**DATA)
train_data, val_data = train_test_split(data, test_size=0.2, shuffle=False) #do not shuffle

train_dataset = PriceDataset(train_data, shift=SHIFTS)
val_dataset = PriceDataset(val_data, shift=SHIFTS)
train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)



model = AllocationModel(input_dim=train_dataset.X.shape[1])
optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)  
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-8)

loss_fn = nn.L1Loss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
model = model.to(device)

# Early stopping parameters
patience = 50  # Number of epochs to wait before stopping
best_val_loss = float('inf')
best_val_accuracy = 0
best_model_state = None

progress_bar = tqdm(total=EPOCHS, desc="Training")
train_losses = []
val_losses = []
val_accuracies = []
for epoch in range(EPOCHS):
    train_loss = train_model(model, train_dataloader, loss_fn, optimizer, device)
    val_loss, val_accuracy = evaluate_loss(model, val_dataloader, loss_fn, device)
    scheduler.step()
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_accuracy = val_accuracy
        best_model_state = model.state_dict().copy()
    
    progress_bar.set_description(f"Epoch {epoch+1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val Accuracy: {val_accuracy:.6f}")
    progress_bar.update(1)

progress_bar.close()

# Load best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"Loaded best model with validation loss: {best_val_loss:.6f} and validation accuracy: {best_val_accuracy:.6f}")

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training, Validation Loss, and Accuracy')
plt.legend()
plt.grid(True)
plt.show(block=False)

vb = VectorizedBacktesting(
    instance_name="AllocationModel",
    initial_capital=10000,
    slippage_pct=0.00,
    commission_fixed=0.0,
    leverage=1.0
)
vb.load_data(val_data, symbol=DATA["symbol"], interval=DATA["interval"], age_days=DATA["age_days"])
vb.run_strategy(model_wrapper, verbose=True)
vb.plot_performance(mode="basic")
