import yfinance as yf
import pandas as pd
import numpy as np
import requests
import random

from rich import print
from tqdm import tqdm
import matplotlib.pyplot as plt

import trading.model_tools as mt

stocks = requests.get("https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt").text.strip().split("\n")
DAYS = 1095

def get_financials(ticker: str):
    ticker = yf.Ticker(ticker)
    financials = ticker.quarterly_financials
    sorted_financials = financials[financials.columns[::-1]] #sort date
    return sorted_financials

def get_metric(financials: pd.DataFrame, metric: str):
    try:
        return financials.loc[metric].pct_change()
    except:
        return None

def backtest(data: pd.DataFrame):
    return data['Close'] / data['Close'].iloc[0]

sample = random.sample(stocks, k=100)

growth_candidate_equity_curves = pd.DataFrame()

for stock in tqdm(sample):
    financials = get_financials(stock)
    metric = get_metric(financials, "Diluted EPS")
    if metric is not None:
        if metric.mean() > 0:
            try:
                growth_candidate_equity_curves[stock] = backtest(mt.fetch_data(symbol=stock, days=DAYS, interval="1d", age_days=0, data_source="yfinance"))
            except:
                pass

plt.plot(growth_candidate_equity_curves.mean(axis=1), label="Growth Candidates")
plt.plot(backtest(mt.fetch_data(symbol="SPY", days=DAYS, interval="1d", age_days=0, data_source="yfinance")), label="Benchmark")
plt.legend()
plt.show()