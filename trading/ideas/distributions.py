import matplotlib.pyplot as plt
import numpy as np

import trading.technical_analysis as ta
import trading.model_tools as mt

data = mt.fetch_data(symbol="DOGE-USDT", days=1800, interval="1d", age_days=0, data_source="binance")

X, _, _ = ta.adx(data['High'], data['Low'], data['Close'], timeperiod=14)
X.dropna(inplace=True)

counts, bin_edges = np.histogram(X, bins=200)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
current_price = X.iloc[-1]

print(f"Historical Upside Probability: {sum(counts[bin_centers>current_price]) / sum(counts)}")
print(f"Historical Downside Probability: {sum(counts[bin_centers<current_price]) / sum(counts)}")

plt.bar(bin_centers, counts, width=np.diff(bin_edges))
plt.axvline(x=current_price, color='red', linestyle='--')
plt.show()

