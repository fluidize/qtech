import numpy as np
import matplotlib.pyplot as plt

from trading.model_tools import fetch_data
from models import PriceDataset

DATA = {
    "symbol": "SOL-USDT",
    "days": 365,
    "interval": "4h",
    "age_days": 0,
    "data_source": "binance",
    "cache_expiry_hours": 999,
    "verbose": True
}

data = fetch_data(**DATA)
dataset = PriceDataset(data, seq_len=10)

true_state = dataset.y_regime.numpy()
price = dataset.data.loc[dataset.valid_indices]['Close'].values
time_index = np.arange(len(true_state))
baseline = np.full_like(price, price.min())
cmap = plt.cm.viridis
n_states = 2

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(time_index, price, color='black', linewidth=1)
i = 0
while i < len(true_state):
    j = i
    while j < len(true_state) and true_state[j] == true_state[i]:
        j += 1
    color = cmap(true_state[i] / n_states)
    ax.fill_between(time_index[i:j], price[i:j], baseline[i:j], color=color, alpha=0.3)
    i = j
ax.set_xlabel('Time')
ax.set_ylabel('Price')
ax.set_title('Price by True Regime (0=downtrend, 1=uptrend)')
fig.tight_layout()
plt.show()
