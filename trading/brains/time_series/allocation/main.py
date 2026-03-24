import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.optim as optim

import matplotlib.pyplot as plt

import trading.model_tools as mt
import loss_functions as lf
from models import PriceDataset, CombinedModelWrapper
from trading.backtesting.backtesting import VectorizedBacktest


def train_loop(epochs, shifts, data_dict, device, split_size=0.25):
    data = mt.fetch_data(**data_dict)
    full_dataset = PriceDataset(data, shift=shifts)
    train_dataset, val_dataset = full_dataset.split(test_size=split_size)

    model = CombinedModelWrapper(input_dim=train_dataset.X.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)

    alloc_loss_fn = lf.SharpeLoss(device=device)
    distribution_loss_fn = lf.StudentTLoss(dof=5.0)
    regime_loss_fn = lf.WeightedCrossEntropyLoss(device=device, num_classes=2, target=train_dataset.y_regime)

    alloc_train_losses = []
    alloc_val_losses = []
    distribution_train_losses = []
    distribution_val_losses = []
    regime_train_losses = []
    regime_val_losses = []

    progress_bar = tqdm(total=epochs, desc="Training")
    for epoch in range(epochs):
        train_dataset.all_to_device(device=device)
        val_dataset.all_to_device(device=device)

        model.train()
        optimizer.zero_grad()

        distribution_train_loss = distribution_loss_fn(model.distribution_model(train_dataset.X), train_dataset.y_velocity)
        distribution_train_loss.backward()

        regime_train_loss = regime_loss_fn(model.regime_model(train_dataset.X), train_dataset.y_regime)
        regime_train_loss.backward()

        alloc_train_loss = alloc_loss_fn(model, train_dataset)
        alloc_train_loss.backward()

        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_alloc_loss = alloc_loss_fn(model, val_dataset)
            val_distribution_loss = distribution_loss_fn(model.distribution_model(val_dataset.X), val_dataset.y_velocity)
            val_regime_loss = regime_loss_fn(model.regime_model(val_dataset.X), val_dataset.y_regime)

        alloc_train_losses.append(alloc_train_loss.item())
        alloc_val_losses.append(val_alloc_loss.item())
        distribution_train_losses.append(distribution_train_loss.item())
        distribution_val_losses.append(val_distribution_loss.item())
        regime_train_losses.append(regime_train_loss.item())
        regime_val_losses.append(val_regime_loss.item())

        progress_bar.set_description(
            f"Epoch {epoch+1} | alloc T: {alloc_train_losses[-1]:.4f} V: {alloc_val_losses[-1]:.4f}"
        )
        progress_bar.update(1)
    progress_bar.close()

    return (
        alloc_train_losses, alloc_val_losses,
        distribution_train_losses, distribution_val_losses,
        regime_train_losses, regime_val_losses,
        model, train_dataset, val_dataset,
    )


### Training ###
if __name__ == "__main__":
    EPOCHS = 1000
    SHIFTS = 10
    DATA = {
        "symbol": "SOL-USDT",
        "days": 1095,
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

    (
        alloc_train_losses, alloc_val_losses,
        distribution_train_losses, distribution_val_losses,
        regime_train_losses, regime_val_losses,
        model, train_dataset, val_dataset,
    ) = train_loop(
        EPOCHS, SHIFTS, DATA, DEVICE, split_size=0.25
    )

    fig, (ax_alloc, ax_distribution, ax_regime) = plt.subplots(1, 3, figsize=(14, 4))
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
    ax_regime.plot(regime_train_losses, label='Train')
    ax_regime.plot(regime_val_losses, label='Val')
    ax_regime.set_xlabel('Epoch')
    ax_regime.set_ylabel('Loss')
    ax_regime.set_title('Regime')
    ax_regime.legend()
    ax_regime.grid(True)
    plt.tight_layout()
    plt.show(block=False)

    def run_vb(dataset):
        vb = VectorizedBacktest(instance_name="AllocationModel", initial_capital=10000, slippage_pct=0.0, commission_fixed=0.0, leverage=1.0)
        vb.load_data(dataset.data, symbol=DATA["symbol"], interval=DATA["interval"], age_days=DATA["age_days"])
        vb.run_strategy(model_wrapper, verbose=True, model=model, device=DEVICE)
        return vb

    vb_val = run_vb(val_dataset)
    vb_train = run_vb(train_dataset)

    fig, ((ax_val0, ax_train0), (ax_val1, ax_train1)) = plt.subplots(2, 2, figsize=(14, 8))
    for vb, ax0, ax1, label in [(vb_val, ax_val0, ax_val1, "Val"), (vb_train, ax_train0, ax_train1, "Train")]:
        s = vb.get_performance_metrics()
        ax0.plot(vb.data['Datetime'], vb.data['Portfolio_Value'], color='orange')
        ax0.plot(vb.data['Datetime'], vb.initial_capital * (1 + vb.data['Open_Return']).cumprod(), color='blue')
        ax0.set_title(f"{label} | TR: {s['Total_Return']*100:.3f}% | Sharpe: {s['Sharpe_Ratio']:.3f} | Max DD: {s['Max_Drawdown']*100:.3f}%")
        ax1.plot(vb.data['Datetime'], vb.data['Position'], color='green')
        ax1.set_title("Position")
    plt.tight_layout()
    plt.show()
