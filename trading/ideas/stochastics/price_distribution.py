import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np
import sys
sys.path.append("trading")
import model_tools as mt
import pandas_indicators as ta
import wiener_process as wiener

data = mt.fetch_data("BTC-USDT", 29, "1min", 0, kucoin=True)

actual_log_returns = ta.log_returns(data['Close'])
simulated_log_returns = np.exp(wiener.wiener_process(n_steps=len(actual_log_returns), time_unit=1, gbm=True))

fig = sp.make_subplots(rows=2, cols=1)
fig.add_trace(go.Histogram(x=actual_log_returns, nbinsx=1000), row=1, col=1)
fig.add_trace(go.Histogram(x=simulated_log_returns, nbinsx=1000), row=2, col=1)
fig.show()