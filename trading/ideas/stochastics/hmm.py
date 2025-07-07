import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from hmmlearn import hmm

import sys
sys.path.append("trading")
import model_tools as mt

pio.renderers.default = "browser"

data = mt.fetch_data("BTC-USDT", chunks=730, interval="1d", age_days=0, 
                    data_source="binance").drop("Datetime", axis=1)

X = data[['Close']].pct_change().dropna()

model = hmm.GaussianHMM(
    n_components=2, 
    covariance_type="diag", 
    n_iter=10000,
    init_params="kmeans",
    random_state=42
)
model.fit(X)

samples, states = model.sample(1000)
time_index = np.arange(len(samples))

fig = go.Figure()
print(samples)
fig.add_trace(go.Scatter(
    x=time_index,
    y=samples[:,0],
    mode="lines",
    name="Generated Prices",
    line=dict(color='blue')
))

fig.add_trace(go.Scatter(
    x=time_index,
    y=states,
    mode="markers",
    name="Hidden States",
    yaxis="y2",
    marker=dict(color=states, colorscale="Viridis", size=5)
))

fig.update_layout(
    title="HMM Simulation",
    yaxis=dict(title="Price"),
    yaxis2=dict(
        title="State",
        overlaying="y",
        side="right",
    )
)

fig.show()