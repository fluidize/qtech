import matplotlib.pyplot as plt
import trading.model_tools as mt

data = mt.fetch_data(symbol="DOGE-USDT", days=1800, interval="1d", age_days=0, data_source="binance")

X = data['Close']
data = data.dropna()

plt.hist(X, bins=100)
plt.axvline(x=X.iloc[-1], color='red', linestyle='--')
plt.show()