import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.append("")

import trading.model_tools as mt

data = mt.fetch_data(symbol="SHIB-USDT", interval="1d", days=1000, age_days=0)

plt.hist(data['Close'].pct_change(), bins=100)
plt.show()