import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from trading.model_tools import fetch_data

from models import PriceDataset, RegimeClassifier
import loss_functions as lf

import matplotlib.pyplot as plt
from tqdm import tqdm

EPOCHS = 1000
SHIFTS = 10
DATA = {
    "symbol": "SOL-USDT",
    "days": 365,
    "interval": "4h",
    "age_days": 0,
    "data_source": "binance",
    "cache_expiry_hours": 999,
    "verbose": True
}
DEVICE = 'cuda'

data = fetch_data(**DATA)
full_dataset = PriceDataset(data, shift=10)
train_dataset, val_dataset = full_dataset.split(test_size=0.75)

model = RegimeClassifier(input_dim=train_dataset.X.shape[1]).to(DEVICE)
optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-8)
loss_fn = lf.WeightedCrossEntropyLoss(device=DEVICE, num_classes=2, target=val_dataset.y_regime)

train_losses = []
val_losses = []
best_val_loss = float('inf')
best_model_state = None

for i in tqdm(range(EPOCHS)):
    train_dataset.all_to_device(device=DEVICE)
    val_dataset.all_to_device(device=DEVICE)

    model.train()
    optimizer.zero_grad()
    yhat = model(train_dataset.X)
    loss = loss_fn(yhat, train_dataset.y_regime.long())
    loss.backward()
    optimizer.step()
    scheduler.step()
    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_yhat = model(val_dataset.X)
        val_loss = loss_fn(val_yhat, val_dataset.y_regime.long()).item()
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
    logits_val = model(val_dataset.X.to(DEVICE)).detach().cpu()
    logits_train = model(train_dataset.X.to(DEVICE)).detach().cpu()
pred_state_val = logits_val.argmax(dim=1).numpy()
pred_state_train = logits_train.argmax(dim=1).numpy()
n_states = max(pred_state_val.max(), pred_state_train.max()) + 1
cmap = plt.cm.viridis

fig, (ax_val, ax_train) = plt.subplots(1, 2, figsize=(14, 5))

for ax, pred_state, dataset, title in (
    (ax_val, pred_state_val, val_dataset, 'Validation'),
    (ax_train, pred_state_train, train_dataset, 'Train'),
):
    price = dataset.data['Close'].values
    time_index = np.arange(len(pred_state))
    baseline = np.full_like(price, price.min())
    ax.plot(time_index, price, color='black', linewidth=1)
    i = 0
    while i < len(pred_state):
        j = i
        while j < len(pred_state) and pred_state[j] == pred_state[i]:
            j += 1
        color = cmap(pred_state[i] / n_states)
        ax.fill_between(time_index[i:j], price[i:j], baseline[i:j], color=color, alpha=0.3)
        i = j
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.set_title(f'{title}: price by predicted regime')

fig.tight_layout()
plt.show()
