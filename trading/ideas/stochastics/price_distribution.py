import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np
import sys
sys.path.append("trading")
import model_tools as mt
import pandas_indicators as ta
import monte_carlo as mc
from scipy.stats import ks_2samp

data = mt.fetch_data("BTC-USDT", 1, "1min", 0, kucoin=True)

actual_log_returns = ta.log_returns(data['Close'])
simulated_returns = mc.monte_carlo_simulation(n_steps=len(actual_log_returns), n_paths=1000, time_unit=1)
simulated_log_returns = ta.log_returns(simulated_returns)

fig = sp.make_subplots(rows=2, cols=1)
fig.add_trace(go.Histogram(x=actual_log_returns, nbinsx=1000), row=1, col=1)
fig.add_trace(go.Histogram(x=simulated_log_returns.values.flatten(), nbinsx=1000), row=2, col=1)

fig.show()