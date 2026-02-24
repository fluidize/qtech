import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import print

import sys
sys.path.append("")

import trading.model_tools as mt

data = mt.fetch_data(symbol="SHIB-USDT", interval="1d", days=1825, age_days=0, data_source="binance")
X = data['Close'].pct_change()

print(f"Mean: {X.mean():.2f}, Std: {X.std():.2f}")
plt.hist(X, bins=100)
plt.show()