import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from trading.model_tools import fetch_data
from models import PriceDataset, RegimeClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm

EPOCHS = 1000
DEVICE = 'cuda'
DATA = {
    "symbol": "SOL-USDT",
    "days": 1095,
    "interval": "30m",
    "age_days": 0,
    "data_source": "binance",
    "cache_expiry_hours": 999,
    "verbose": True
}

data = fetch_data(**DATA)
full_dataset = PriceDataset(data, shift=10)
train_dataset, val_dataset = full_dataset.split(test_size=0.2)

model = RegimeClassifier(input_dim=train_dataset.X.shape[1]).to(DEVICE)
optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-8)
loss_fn = nn.CrossEntropyLoss()

train_losses = []
val_losses = []
best_val_loss = float('inf')
best_model_state = None

for i in tqdm(range(EPOCHS)):
    model.train()
    optimizer.zero_grad()
    yhat = model(train_dataset.X.to(DEVICE))
    loss = loss_fn(yhat, train_dataset.y_regime.to(DEVICE).long())
    loss.backward()
    optimizer.step()
    scheduler.step()
    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_yhat = model(val_dataset.X.to(DEVICE))
        val_loss = loss_fn(val_yhat, val_dataset.y_regime.to(DEVICE).long()).item()
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
    logits = model(val_dataset.X.to(DEVICE)).detach().cpu()
pred_state = logits.argmax(dim=1).numpy()
price = val_dataset.data['Close'].values
time_index = np.arange(len(pred_state))
baseline = np.full_like(price, price.min())
cmap = plt.cm.viridis
n_states = pred_state.max() + 1

fig, ax = plt.subplots(figsize=(12, 5))
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
ax.set_title('Price by Predicted Regime')
fig.tight_layout()
plt.show()
