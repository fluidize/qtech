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
from models.basic_models import LogisticUnit
from datasets import PriceDataset
from trading.backtesting.backtesting import VectorizedBacktest

if __name__ == "__main__":
    EPOCHS = 128
    SEQ_LEN = 1
    BATCH_SIZE = 2 ** 16

    DATA = {
        "symbols": ["SOL-USDT", "BTC-USDT"],
        "days": 90,
        "interval": "1m",
        "age_days": 0,
        "data_source": "binance",
        "cache_expiry_hours": -1,
        "verbose": True
    }
    LEARNING_RATE = 5e-3
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

    model_types = ['linear', 'log', 'exp', 'poly', 'sqrt']
    models = {}
    optimizers = {}
    train_losses_dict = {}
    val_losses_dict = {}

    for model_type in model_types:
        models[model_type] = LogisticUnit(
            num_features=num_features,
            model_type=model_type
        ).to(DEVICE)
        optimizers[model_type] = optim.RMSprop(models[model_type].parameters(), lr=LEARNING_RATE)
        train_losses_dict[model_type] = []
        val_losses_dict[model_type] = []

    loss_fn = lf.HullTacLoss(device=DEVICE)

    progress_bar = tqdm(total=EPOCHS, desc="Training")
    for epoch in range(EPOCHS):
        for model_type in model_types:
            model = models[model_type]
            optimizer = optimizers[model_type]

            ### train
            model.train()
            train_signals = lf.model_to_signals(model, train_dataset, device=DEVICE, batch_size=BATCH_SIZE, eval_mode=False, epoch=epoch, total_epochs=EPOCHS)
            train_loss = loss_fn(train_signals, train_dataset)
            train_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            ### val
            model.eval()
            with torch.no_grad():
                val_signals = lf.model_to_signals(model, val_dataset, device=DEVICE, batch_size=BATCH_SIZE, eval_mode=True, epoch=epoch, total_epochs=EPOCHS)
                val_loss = loss_fn(val_signals, val_dataset)

            train_losses_dict[model_type].append(train_loss.item())
            val_losses_dict[model_type].append(val_loss.item())

        progress_bar.set_description(
            f"Epoch {epoch+1}/{EPOCHS}"
        )
        progress_bar.update(1)
    progress_bar.close()

    # Plot train and val losses in 2-column subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    for model_type in model_types:
        ax1.plot(train_losses_dict[model_type], label=f'{model_type}')
        ax2.plot(val_losses_dict[model_type], label=f'{model_type}')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Train Losses')
    ax1.legend()
    ax1.grid(True)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Validation Losses')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('model_losses.png')
    plt.show()

    # Run backtests for all models
    def model_wrapper(data, model, device, seq_len=10, batch_size=32):
        dataset = PriceDataset(data, add_ticker=DATA["symbols"][1], seq_len=seq_len)
        raw_signals = lf.model_to_signals(model, dataset, device=device, batch_size=batch_size, eval_mode=True, epoch=EPOCHS, total_epochs=EPOCHS)
        signals = pd.Series(raw_signals.cpu().numpy(), index=data.index)
        return signals

    backtest_results = {}
    for model_type in model_types:
        vb = VectorizedBacktest(
            instance_name=f"AllocationModel_{model_type}",
            initial_capital=10000,
            slippage_pct=0.0,
            commission_fixed=0.0,
            leverage=1.0,
        )
        vb.load_data(val_dataset_raw, symbols=DATA["symbols"], interval=DATA["interval"], age_days=DATA["age_days"])
        vb.run_strategy(model_wrapper, verbose=False, model=models[model_type], device=DEVICE, seq_len=SEQ_LEN, batch_size=BATCH_SIZE)

        backtest_metrics = vb.get_performance_metrics()
        backtest_results[model_type] = {
            'metrics': backtest_metrics,
            'equity_curve': vb.get_equity_curve()
        }
        print(f"{model_type} - Sharpe: {backtest_metrics['Sharpe_Ratio']:.4f}")

    # Plot all strategy curves together
    plt.figure(figsize=(12, 6))
    for model_type in model_types:
        equity_curve = backtest_results[model_type]['equity_curve']
        plt.plot(equity_curve.index, equity_curve.values, label=f'{model_type} (Sharpe: {backtest_results[model_type]["metrics"]["Sharpe_Ratio"]:.2f})')

    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.title('Strategy Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('strategy_comparison.png')
    plt.show()