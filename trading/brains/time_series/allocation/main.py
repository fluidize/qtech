import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.optim as optim

import matplotlib.pyplot as plt

import trading.model_tools as mt
import loss_functions as lf
from models import PriceDataset, AllocationWrapper, DistributionPredictor
from trading.backtesting.backtesting import VectorizedBacktest

### Training ###
if __name__ == "__main__":
    EPOCHS = 1000
    SHIFTS = 100
    DATA = {
        "symbol": "BTC-USDT",
        "days":365,
        "interval": "1h",
        "age_days": 0,
        "data_source": "binance",
        "cache_expiry_hours": 999,
        "verbose": True
    }
    DEVICE = 'cuda'

    def model_wrapper(data, model, device):
        dataset = PriceDataset(data, shift=SHIFTS)
        model.eval()
        with torch.no_grad():
            X_tensor = dataset.X.to(device)
            predictions = model(X_tensor).cpu().numpy()

        signals = pd.Series(0.0, index=data.index)
        signals.loc[dataset.valid_indices] = predictions
        return signals

    data = mt.fetch_data(**DATA)
    full_dataset = PriceDataset(data, shift=SHIFTS)
    train_dataset, val_dataset = full_dataset.split(test_size=0.5)

    model = AllocationWrapper(input_dim=train_dataset.X.shape[1]).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-8)

    alloc_loss_fn = lf.SharpeLoss(device=DEVICE)
    distribution_loss_fn = lf.NegativeLogLikelihoodLoss(device=DEVICE)

    best_val_loss = float('inf')
    best_model_state = None
    alloc_train_losses = []
    alloc_val_losses = []
    distribution_train_losses = []
    distribution_val_losses = []

    progress_bar = tqdm(total=EPOCHS, desc="Training")
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        distribution_train_loss = distribution_loss_fn(
            model.distribution_model(train_dataset.X.to(DEVICE)), train_dataset.y.to(DEVICE)
        )
        distribution_train_loss.backward()

        alloc_train_loss = alloc_loss_fn(model, train_dataset)
        alloc_train_loss.backward()

        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_alloc_loss = alloc_loss_fn(model, val_dataset)
            val_distribution_loss = distribution_loss_fn(
                model.distribution_model(val_dataset.X.to(DEVICE)), val_dataset.y.to(DEVICE)
            )

        alloc_train_losses.append(alloc_train_loss.item())
        alloc_val_losses.append(val_alloc_loss.item())
        distribution_train_losses.append(distribution_train_loss.item())
        distribution_val_losses.append(val_distribution_loss.item())

        val_loss = val_alloc_loss.item() + val_distribution_loss.item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        progress_bar.set_description(
            f"Epoch {epoch+1} | alloc T: {alloc_train_losses[-1]:.4f} V: {alloc_val_losses[-1]:.4f} | "
            f"dist T: {distribution_train_losses[-1]:.4f} V: {distribution_val_losses[-1]:.4f}"
        )
        progress_bar.update(1)
    progress_bar.close()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    fig, (ax_alloc, ax_distribution) = plt.subplots(1, 2, figsize=(12, 4))
    ax_alloc.plot(alloc_train_losses, label='Train')
    ax_alloc.plot(alloc_val_losses, label='Val')
    ax_alloc.set_xlabel('Epoch')
    ax_alloc.set_ylabel('Loss')
    ax_alloc.set_title('Alloc')
    ax_alloc.legend()
    ax_alloc.grid(True)
    ax_distribution.plot(distribution_train_losses, label='Train')
    ax_distribution.plot(distribution_val_losses, label='Val')
    ax_distribution.set_xlabel('Epoch')
    ax_distribution.set_ylabel('Loss')
    ax_distribution.set_title('Distribution')
    ax_distribution.legend()
    ax_distribution.grid(True)
    plt.tight_layout()
    plt.show(block=False)

    vb = VectorizedBacktest(
        instance_name="AllocationModel",
        initial_capital=10000,
        slippage_pct=0.00,
        commission_fixed=0.0,
        leverage=1.0
    )
    vb.load_data(val_dataset.data, symbol=DATA["symbol"], interval=DATA["interval"], age_days=DATA["age_days"])
    vb.run_strategy(model_wrapper, verbose=True, model=model, device=DEVICE)
    vb.plot_performance(mode="basic")
