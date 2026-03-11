import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from trading.model_tools import fetch_data
from loss_functions import NegativeLogLikelihoodLoss
from models import PriceDataset, TensorSubset, DistributionPredictor
import matplotlib.pyplot as plt
from tqdm import tqdm

EPOCHS = 5000
DEVICE = 'cuda'
DATA = {
    "symbol": "BTC-USDT",
    "days":365,
    "interval": "1h",
    "age_days": 0,
    "data_source": "binance",
    "cache_expiry_hours": 999,
    "verbose": True
}

data = fetch_data(**DATA)
full_dataset = PriceDataset(data, shift=10)
train_dataset, val_dataset = full_dataset.split(test_size=0.2)

model = DistributionPredictor(input_dim=train_dataset.X.shape[1]).to(DEVICE)
optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-8)
loss_fn = NegativeLogLikelihoodLoss()

train_losses = []
val_losses = []
best_val_loss = float('inf')
best_model_state = None

for i in tqdm(range(EPOCHS)):
    model.train()
    optimizer.zero_grad()
    yhat = model(train_dataset.X.to(DEVICE))
    loss = loss_fn(yhat, train_dataset.y.to(DEVICE))
    loss.backward()
    optimizer.step()
    scheduler.step()
    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_yhat = model(val_dataset.X.to(DEVICE))
        val_loss = loss_fn(val_yhat, val_dataset.y.to(DEVICE)).item()
    val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict().copy()

if best_model_state is not None:
    model.load_state_dict(best_model_state)

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training, Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

model.eval()
with torch.no_grad():
    test_y = model(val_dataset.X.to(DEVICE)).detach().cpu().numpy()

mean = test_y[:, 0]
std = test_y[:, 1]
upper = mean + std
lower = mean - std

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(val_dataset.y.numpy(), label='True')
ax1.plot(upper, label='Upper band', color='blue')
ax1.plot(lower, label='Lower band', color='blue')
ax1.plot(mean, label='Predicted mean', color='orange')
ax1.fill_between(range(len(mean)), lower, upper, alpha=0.3)
ax1.set_xlabel('Time')
ax1.set_ylabel('Price')
ax1.set_title('True vs Predicted (1 std range)')
ax1.legend()
ax1.grid(True)

ax2.plot(std, label='Predicted std', color='blue')
ax2.set_xlabel('Time')
ax2.set_ylabel('Std')
ax2.set_title('Model Std')
ax2.legend()
ax2.grid(True)
plt.tight_layout()
plt.show()