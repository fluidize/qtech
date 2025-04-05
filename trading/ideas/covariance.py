import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def covariance(x: pd.DataFrame, y: pd.DataFrame):
    if len(x) != len(y):
        raise Exception(f"X and Y are not the same length. X:{len(x)} Y:{len(y)}")
    x_mean = x.mean()
    y_mean = y.mean()

    xy_dev_sum = 0
    for i in range(len(x)):
        xy_dev_sum += (x.iloc[i] - x_mean) * (y.iloc[i] - y_mean)
    
    cov = xy_dev_sum / (len(x) - 1)

    return cov

def covariance_to_correlation(cov, x_std, y_std):
    return cov / (x_std * y_std)

def rolling_covariance(x, y, window_size=30):
    # Initialize an empty list to store the rolling covariance values
    rolling_cov = []
    
    # Iterate over the data in windows
    for i in range(len(x) - window_size + 1):
        # Extract the current window
        x_window = x.iloc[i:i + window_size]
        y_window = y.iloc[i:i + window_size]
        
        # Calculate covariance for the current window
        cov = covariance(x_window, y_window)
        rolling_cov.append(cov)
    
    # Convert the list to a pandas Series
    return pd.Series(rolling_cov, index=x.index[window_size - 1:])

def rolling_correlation(x, y, window_size=30):
    # Initialize an empty list to store the rolling correlation values
    rolling_corr = []
    
    # Iterate over the data in windows
    for i in range(len(x) - window_size + 1):
        # Extract the current window
        x_window = x.iloc[i:i + window_size]
        y_window = y.iloc[i:i + window_size]
        
        # Calculate covariance for the current window
        cov = covariance(x_window, y_window)
        
        # Calculate standard deviations
        x_std = x_window.std()
        y_std = y_window.std()
        
        # Calculate correlation
        if x_std > 0 and y_std > 0:  # Avoid division by zero
            corr = covariance_to_correlation(cov, x_std, y_std)
        else:
            corr = 0  # If std is zero, correlation is undefined
        
        rolling_corr.append(corr)
    
    # Convert the list to a pandas Series
    return pd.Series(rolling_corr, index=x.index[window_size - 1:])

m2_data = pd.read_csv(r"trading\ideas\m2\bitcoin-and-m2-growth-gl.csv")
m2_data["DateTime"] = pd.to_datetime(m2_data["DateTime"])
m2_data["DateTime"] = m2_data["DateTime"].dt.date

shift = 108
x = m2_data["M2 Global Supply (USD)"].shift(shift).dropna()
y_1 = m2_data["BTC price"][shift:]

x = x.squeeze()
y_1 = y_1.squeeze()

# Calculate rolling covariance
rolling_cov_btc = rolling_covariance(x, y_1, window_size=10)

# Calculate rolling correlation
rolling_corr_btc = rolling_correlation(x, y_1, window_size=10)

normalized_m2 = (x - x.min()) / (x.max() - x.min())
normalized_rolling_cov = (rolling_cov_btc - rolling_cov_btc.min()) / (rolling_cov_btc.max() - rolling_cov_btc.min())
normalized_rolling_corr = (rolling_corr_btc - rolling_corr_btc.min()) / (rolling_corr_btc.max() - rolling_corr_btc.min())
normalized_btc_price = (y_1 - y_1.min()) / (y_1.max() - y_1.min())

# Plot the rolling covariance and normalized rolling covariance
fig = go.Figure()
fig.add_trace(go.Scatter(x=m2_data["DateTime"][shift:], y=normalized_m2, mode='lines', name='Normalized M2 Global Supply'))
fig.add_trace(go.Scatter(x=m2_data["DateTime"][shift:], y=normalized_rolling_cov, mode='lines', name='Normalized Rolling Covariance'))
fig.add_trace(go.Scatter(x=m2_data["DateTime"][shift:], y=normalized_btc_price, mode='lines', name='Normalized BTC Price'))
fig.update_layout(title='Normalized Rolling Covariance of M2 Global Supply', xaxis_title='Date', yaxis_title='Normalized Covariance')
fig.show()

# Plot the rolling correlation and normalized rolling correlation
fig = go.Figure()
fig.add_trace(go.Scatter(x=m2_data["DateTime"][shift:], y=normalized_m2, mode='lines', name='Normalized M2 Global Supply'))
fig.add_trace(go.Scatter(x=m2_data["DateTime"][shift:], y=normalized_rolling_corr, mode='lines', name='Normalized Rolling Correlation'))
fig.add_trace(go.Scatter(x=m2_data["DateTime"][shift:], y=normalized_btc_price, mode='lines', name='Normalized BTC Price'))
fig.update_layout(title='Normalized Rolling Correlation of M2 Global Supply', xaxis_title='Date', yaxis_title='Normalized Correlation')
fig.show()