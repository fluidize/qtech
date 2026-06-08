import numpy as np
import pandas as pd
from tqdm import tqdm
from rich import print

import torch
import torch.optim as optim
from torchinfo import summary

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import trading.model_tools as mt
import loss_functions as lf
from models import PriceDataset, BasicModel
from trading.backtesting.backtesting import VectorizedBacktest

plt.ioff()

def train_loop(epochs, data_dict, device, split_size=0.25, seq_len=10):
    data = mt.fetch_data(**data_dict)
    train_dataset_raw, val_dataset_raw = train_test_split(
        data,
        test_size=split_size,
        shuffle=False,
    )

    train_dataset = PriceDataset(train_dataset_raw, seq_len=seq_len)
    val_dataset = PriceDataset(val_dataset_raw, seq_len=seq_len)

    num_features = train_dataset.X.shape[1]
    sequence_length = train_dataset.X.shape[2]
    model = BasicModel(
        channels=num_features,
        width=sequence_length,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-6)
    loss_fn = lf.SharpeLoss(device=device)

    print(summary(model))

    train_losses = []
    val_losses = []

    progress_bar = tqdm(total=epochs, desc="Training")
    for epoch in range(epochs):
        ### training
        model.train()

        train_loss = loss_fn(model, train_dataset)
        train_loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        ###

        ### validation
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model, val_dataset)
        
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        ###

        progress_bar.set_description(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss.item():.4f} - Val Loss: {val_loss.item():.4f}"
        )
        progress_bar.update(1)
    progress_bar.close()

    return (
        model, train_dataset_raw, val_dataset_raw, train_losses, val_losses,
    )


### Training ###
if __name__ == "__main__":
    EPOCHS = 1000
    SEQ_LEN = 16
    DATA = {
        "symbol": "SOL-USDT",
        "days": 1095,
        "interval": "30m",
        "age_days": 0,
        "data_source": "binance",
        "cache_expiry_hours": -1,
        "verbose": True
    }
    DEVICE = 'cuda'

    def model_wrapper(data, model, device, seq_len=10):
        dataset = PriceDataset(data, seq_len=seq_len)
        model.eval()
        with torch.no_grad():
            X_tensor = dataset.X.to(device)
            predictions = model(X_tensor).cpu().numpy()

        signals = pd.Series(0.0, index=data.index)
        signals.loc[dataset.valid_indices] = predictions
        return signals

    model, train_dataset_raw, val_dataset_raw, train_losses, val_losses = train_loop(
        EPOCHS, DATA, DEVICE, split_size=0.25, seq_len=SEQ_LEN,
    )

    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()

    vb = VectorizedBacktest(
        instance_name="AllocationModel",
        initial_capital=10000,
        slippage_pct=0.0,
        commission_fixed=0.0,
        leverage=1.0,
    )
    vb.load_data(val_dataset_raw, symbol=DATA["symbol"], interval=DATA["interval"], age_days=DATA["age_days"])
    vb.run_strategy(model_wrapper, verbose=True, model=model, device=DEVICE, seq_len=SEQ_LEN)

    print("Backtest metrics:", vb.get_performance_metrics())
    vb.plot_performance(mode="basic")

