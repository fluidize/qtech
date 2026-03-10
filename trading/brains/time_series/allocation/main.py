import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.optim as optim

import matplotlib.pyplot as plt

import trading.model_tools as mt
import loss_functions as lf
from models import PriceDataset, AllocationModel, DirectionalConfidencePredictor

from trading.backtesting.backtesting import VectorizedBacktest
import loss_functions as lf
from models import PriceDataset, TensorSubset, DirectionalConfidencePredictor, AllocationModel

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

    vb = VectorizedBacktest(
        instance_name="AllocationModel",
        initial_capital=10000,
        slippage_pct=0.00,
        commission_fixed=0.0,
        leverage=1.0
    )
    vb.load_data(val_dataset.data, symbol=DATA["symbol"], interval=DATA["interval"], age_days=DATA["age_days"])
    vb.run_strategy(model_wrapper, verbose=True, alloc_model=alloc_model, directional_model=directional_confidence_model, device=DEVICE)
    vb.plot_performance(mode="basic")
