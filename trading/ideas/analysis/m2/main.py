import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import datetime as dt
import seaborn as sns

def get_data(tickers: List[str], start_date: dt.date, end_date: dt.date):
    data = yf.download(tickers, start=start_date, end=end_date, interval="1d", progress=False)
    data.drop(columns=["Open", "High", "Low", "Volume"], inplace=True)
    
    # Remove MultiIndex by flattening column names
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [f"{col[1]}_{col[0]}" for col in data.columns]
    
    return data

def plot_correlation_matrix(data):
    # No need to select 'Close' columns anymore as we've flattened the structure
    # Just use all columns directly
    returns = data.pct_change().dropna()
    corr_matrix = returns.corr()
    
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(corr_matrix, annot=True, cmap='viridis', 
                linewidths=0.5, fmt='.4f', 
                xticklabels=corr_matrix.columns, 
                yticklabels=corr_matrix.columns)
    
    plt.title('Correlation Matrix of Daily Returns')
    plt.tight_layout()
    plt.show()
    
    return corr_matrix

if __name__ == "__main__":
    index_tickers = [
        "^GSPC",  # S&P 500
        "^DJI",   # Dow Jones Industrial Average
        "^IXIC",  # NASDAQ Composite
        "^RUT",   # Russell 2000
        "^OEX",   # S&P 100
        "^MID",   # S&P MidCap 400
        "^FTSE",  # FTSE 100 (UK)
        "^GDAXI", # DAX 40 (Germany)
        "^FCHI",  # CAC 40 (France)
        "^IBEX",  # IBEX 35 (Spain)
        "^SSMI",  # SMI (Switzerland)
        "^N225",  # Nikkei 225 (Japan)
        "^HSI",   # Hang Seng Index (Hong Kong)
        "^AXJO",  # ASX 200 (Australia)
        "^NSEI",  # Nifty 50 (India)
        "^KS11",  # KOSPI (South Korea)
    ]
    forex_tickers = [
        "BTC-USD",
        "ETH-USD",
        "EURUSD=X",
        "GBPUSD=X",
        "USDJPY=X",
        "USDCAD=X",
        "AUDUSD=X",
    ]
    # data = get_data(["BTC-USD", "ETH-USD", "SOL-USD"], dt.datetime.now() - dt.timedelta(days=365), dt.datetime.now())
    
    data = pd.read_csv(r"trading\ideas\analysis\bitcoin-and-m2-growth-gl.csv")
    print(data.head())
    data = data[["BTC price", "M2 Global Supply (USD)"]]
    for column in data.columns:
        for i in range(72, 108):
            if column == "M2 Global Supply (USD)":
                data[f"{column}_shift_{i}"] = data[column].shift(-i)
        
    print("Data columns (flattened structure):")
    print(data.columns.tolist())
    print("\nSample data:")
    print(data.head())

    corr_matrix = plot_correlation_matrix(data)
    print("\nCorrelation Matrix:")
    print(corr_matrix)