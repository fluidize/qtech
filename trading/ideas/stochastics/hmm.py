import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
from scipy.signal import savgol_filter

import sys
sys.path.append("trading")
import model_tools as mt

data = mt.fetch_data("BTC-USDT", days=100, interval="1h", age_days=0,
                    data_source="binance").drop("Datetime", axis=1)

X = savgol_filter(data['Close'], window_length=40, polyorder=3, deriv=1)
X = np.expand_dims(np.asarray(X), axis=1)
price = data['Close'].values

model = hmm.GaussianHMM(
    n_components=2,
    covariance_type="diag",
    n_iter=10000,
    init_params="kmeans",
    random_state=42
)
model.fit(X)

states = model.predict(X)
print(states)
time_index = np.arange(len(states))
baseline = np.full_like(price, price.min())
cmap = plt.cm.viridis
n_states = states.max() + 1

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(time_index, price, color='black', linewidth=1)

i = 0
while i < len(states):
    j = i
    while j < len(states) and states[j] == states[i]:
        j += 1
    color = cmap(states[i] / n_states)
    ax.fill_between(time_index[i:j], price[i:j], baseline[i:j], color=color, alpha=0.3)
    i = j

ax.set_xlabel('Time')
ax.set_ylabel('Price')
ax.set_title('Price by HMM State')
fig.tight_layout()
plt.show()