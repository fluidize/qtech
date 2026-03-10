import trading.model_tools as mt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = mt.fetch_data("DOGE-USDT", days=1800, interval="1d", age_days=0, data_source="binance")
X = data['Close'].pct_change()

pct = 0.10
pct_prob = sum(X > pct) / len(X)
expected_days = 1 / pct_prob

print(f"Chance of getting a {pct*100}% gain in 1 day: {pct_prob}")
print(f"Expected days to get a {pct*100}% gain: {expected_days}")

kelly_criterion = (pct_prob * pct - (1 - pct_prob)) / pct
print(f"Kelly criterion: {kelly_criterion}")