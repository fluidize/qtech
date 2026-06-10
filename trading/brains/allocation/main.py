import numpy as np
import pandas as pd
from tqdm import tqdm
from rich import print

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import trading.model_tools as mt
import loss_functions as lf
from models import Allocator, PriceDataset
from trading.backtesting.backtesting import VectorizedBacktest

def model_to_signals(model, dataset, device: str = "cuda", batch_size: int = 32, eval_mode: bool = True):

    dataloader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False,
    )
    
    num_sequences = len(dataset)
    predictions = torch.zeros(num_sequences, dtype=torch.float32, device=device)

    with (torch.enable_grad() if not eval_mode else torch.no_grad()):
        for batch_X, batch_indices in dataloader:
            batch_X = batch_X.to(device)
            batch_predictions = model(batch_X)
            predictions[batch_indices] = batch_predictions.float()

    raw_signals_t = torch.zeros(len(dataset.data), dtype=torch.float32, device=device)
    valid_positions = torch.tensor(
        dataset.data.index.get_indexer(dataset.valid_indices),
        dtype=torch.long, device=device
    )
    raw_signals_t[valid_positions] = predictions

    return raw_signals_t

if __name__ == "__main__":
    EPOCHS = 1000
    SEQ_LEN = 16
    BATCH_SIZE = 8192
    DATA = {
        "symbol": "BTC-USDT",
        "days": 1095,
        "interval": "30m",
        "age_days": 0,
        "data_source": "binance",
        "cache_expiry_hours": -1,
        "verbose": True
    }
    LEARNING_RATE = 5e-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    data = mt.fetch_data(**DATA)
    train_dataset_raw, val_dataset_raw = train_test_split(
        data,
        test_size=0.25,
        shuffle=False,
    )

    train_dataset = PriceDataset(train_dataset_raw, seq_len=SEQ_LEN)
    val_dataset = PriceDataset(val_dataset_raw, seq_len=SEQ_LEN)

    num_features = train_dataset.X.shape[1]
    sequence_length = train_dataset.X.shape[2]

    model = Allocator(
        channels=num_features,
        width=sequence_length,
        conv_hidden_channel_size=64,
        conv_hidden_linear_size=64,
        conv_out_size=64,
        lstm_hidden_size=64,
        lstm_out_size=64,
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = lf.AllocationLoss(device=DEVICE)

    summary(model)

    train_losses = []
    val_losses = []

    progress_bar = tqdm(total=EPOCHS, desc="Training")
    for epoch in range(EPOCHS):
        model.train()
        train_signals = model_to_signals(model, train_dataset, device=DEVICE, batch_size=BATCH_SIZE, eval_mode=False)
        train_loss = loss_fn(train_signals, train_dataset)
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        model.eval()
        with torch.no_grad():
            val_signals = model_to_signals(model, val_dataset, device=DEVICE, batch_size=BATCH_SIZE, eval_mode=True)
            val_loss = loss_fn(val_signals, val_dataset)
        
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        progress_bar.set_description(
            f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss.item():.4f} - Val Loss: {val_loss.item():.4f}"
        )
        progress_bar.update(1)
    progress_bar.close()

    def model_wrapper(data, model, device, seq_len=10, batch_size=32):
        dataset = PriceDataset(data, seq_len=seq_len)
        raw_signals = model_to_signals(model, dataset, device=device, batch_size=batch_size, eval_mode=True)
        signals = pd.Series(raw_signals.cpu().numpy(), index=data.index)
        return signals

    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    vb = VectorizedBacktest(
        instance_name="AllocationModel",
        initial_capital=10000,
        slippage_pct=0.0,
        commission_fixed=0.0,
        leverage=1.0,
    )
    vb.load_data(val_dataset_raw, symbol=DATA["symbol"], interval=DATA["interval"], age_days=DATA["age_days"])
    vb.run_strategy(model_wrapper, verbose=True, model=model, device=DEVICE, seq_len=SEQ_LEN, batch_size=BATCH_SIZE)

    print("Backtest metrics:", vb.get_performance_metrics())
    vb.plot_performance(mode="basic")
