import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv("solana\jupiter\log.csv")

fig = go.Figure()

fig.add_trace(go.Scatter(x=df["Datetime"], y=df["Portfolio"], mode="lines"))
fig.show()