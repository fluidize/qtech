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
from models import AllocatorPolicy, PriceDataset
from trading.backtesting.backtesting import VectorizedBacktest

if __name__ == "__main__":
    EPOCHS = 1024
    SEQ_LEN = 16
    BATCH_SIZE = 2 ** 10
    
    DATA = {
        "symbols": ["SOL-USDT", "BTC-USDT"],
        "days": 730,
        "interval": "1h",
        "age_days": 0,
        "data_source": "binance",
        "cache_expiry_hours": -1,
        "verbose": True
    }
    LEARNING_RATE = 5e-6
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    data = mt.fetch_data(**DATA)
    train_dataset_raw, val_dataset_raw = train_test_split(
        data,
        test_size=0.25,
        shuffle=False,
    )

    train_dataset = PriceDataset(train_dataset_raw, add_ticker=DATA["symbols"][1], seq_len=SEQ_LEN)
    val_dataset = PriceDataset(val_dataset_raw, add_ticker=DATA["symbols"][1], seq_len=SEQ_LEN)

    num_features = train_dataset.X.shape[1]
    sequence_length = train_dataset.X.shape[2]

    model = AllocatorPolicy(
        channels=num_features,
        width=sequence_length,
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = lf.AllocationLoss(device=DEVICE)

    summary(model)

    train_losses = []
    val_losses = []

    progress_bar = tqdm(total=EPOCHS, desc="Training")
    for epoch in range(EPOCHS):
        ### train
        model.train()

        train_signals = lf.model_to_signals(model, train_dataset, device=DEVICE, batch_size=BATCH_SIZE, eval_mode=False)
        train_loss = loss_fn(train_signals, train_dataset)
        train_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        ### val
        model.eval()
        with torch.no_grad():
            val_signals = lf.model_to_signals(model, val_dataset, device=DEVICE, batch_size=BATCH_SIZE, eval_mode=True)
            val_loss = loss_fn(val_signals, val_dataset)
        
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        progress_bar.set_description(
            f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss.item():.4f} - Val Loss: {val_loss.item():.4f}"
        )
        progress_bar.update(1)
    progress_bar.close()

    def model_wrapper(data, model, device, seq_len=10, batch_size=32):
        dataset = PriceDataset(data, add_ticker=DATA["symbols"][1], seq_len=seq_len)
        raw_signals = lf.model_to_signals(model, dataset, device=device, batch_size=batch_size, eval_mode=True)
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
    vb.load_data(val_dataset_raw, symbols=DATA["symbols"], interval=DATA["interval"], age_days=DATA["age_days"])
    vb.run_strategy(model_wrapper, verbose=True, model=model, device=DEVICE, seq_len=SEQ_LEN, batch_size=BATCH_SIZE)

    print("Backtest metrics:", vb.get_performance_metrics())
    vb.plot_performance(mode="basic")
