import scipy
import matplotlib.pyplot as plt

import sys
sys.path.append("")

import trading.model_tools as mt

#use the savitzky golay filter to estimate price acceleration

data = mt.fetch_data("BTC-USDT", days=1, interval="5m", age_days=0, data_source="binance")

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), sharex=True)

window = 7

ax1.plot(data["Close"])
ax1.plot(scipy.signal.savgol_filter(data["Close"], window_length=25, polyorder=2, deriv=0), color="red")
ax2.plot(scipy.signal.savgol_filter(data["Close"], window_length=10, polyorder=2, deriv=1) / (data["Close"])) 

plt.show()
