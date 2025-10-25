import sys

sys.path.append("")
import trading.model_tools as mt
import trading.backtesting.vb_metrics as vb_metrics

stock1 = input("Enter the first stock symbol: ")
stock2 = input("Enter the second stock symbol: ")

stock1_data = mt.fetch_data(stock1, days=365, interval="1d", age_days=0, data_source="yfinance")
stock2_data = mt.fetch_data(stock2, days=365, interval="1d", age_days=0, data_source="yfinance")


alpha, beta = vb_metrics.get_alpha_beta(stock1_data['Close'].pct_change(), stock2_data['Close'].pct_change(), n_days=365, return_interval="1d")

print(f"{stock1} vs {stock2} Beta: {beta:.2f}")