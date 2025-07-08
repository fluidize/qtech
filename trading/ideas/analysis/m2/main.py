import pandas as pd
import numpy as np
import plotly.express as px

data = pd.read_csv(r"C:\Users\MBXia\OneDrive\Documents\GitHub\qtech\trading\ideas\analysis\m2\bitcoin-and-m2-growth-gl.csv").drop(columns=["DateTime", "M2 Growth YoY (%)"])
data = data.rename(columns={"BTC price": "BTCUSD", "M2 Global Supply (USD)": "M2"})

data["BTCUSD"] = data["BTCUSD"]

btc_columns = ["BTCUSD"]
for shift in range(72, 108, 2):
    data[f"BTCUSD_lag_{shift}"] = data["BTCUSD"].shift(shift)
    btc_columns.append(f"BTCUSD_lag_{shift}")

# Create lagged variables for M2
m2_columns = ["M2"]
for shift in range(72, 110, 2):
    data[f"M2_lag_{shift}"] = data["M2"].shift(shift)
    m2_columns.append(f"M2_lag_{shift}")

# Create correlation matrix between BTC and M2 variables only
correlation_matrix = data[btc_columns + m2_columns].corr()

# Extract only the cross-correlations (BTC vs M2)
btc_m2_corr = correlation_matrix.loc[btc_columns, m2_columns]

# Create the heatmap
fig = px.imshow(btc_m2_corr, 
                color_continuous_scale="RdBu",
                aspect="auto",
                title="BTC vs M2 Correlation Matrix")

fig.update_layout(
    xaxis_title="M2 Variables",
    yaxis_title="BTC Variables"
)

fig.show(renderer="browser")
