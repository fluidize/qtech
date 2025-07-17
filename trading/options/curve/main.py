import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

def get_option_chain(ticker, expiration=None):
    ticker_obj = yf.Ticker(ticker)
    if expiration is None:
        expirations = ticker_obj.options
        if not expirations:
            raise ValueError(f"No options available for {ticker}")
        expiration = expirations[0]
    chain = ticker_obj.option_chain(expiration)
    spot = ticker_obj.history(period="1d")['Close'].iloc[-1]
    return chain, spot, expiration

def filter_otm(options, spot):
    calls = options.calls.copy()
    puts = options.puts.copy()
    otm_calls = calls[calls['strike'] > spot]
    otm_puts = puts[puts['strike'] < spot]
    return otm_calls, otm_puts

def plot_vol_curve(otm_calls, otm_puts, spot, ticker, expiration):
    plt.figure(figsize=(10,6))
    if 'impliedVolatility' in otm_calls.columns:
        plt.scatter(otm_calls['strike'], otm_calls['impliedVolatility'], label='OTM Calls', color='green', alpha=0.7)
    if 'impliedVolatility' in otm_puts.columns:
        plt.scatter(otm_puts['strike'], otm_puts['impliedVolatility'], label='OTM Puts', color='red', alpha=0.7)
    plt.axvline(spot, color='black', linestyle='--', label='Spot Price')
    plt.xlabel('Strike Price')
    plt.ylabel('Implied Volatility')
    plt.title(f'Volatility Curve for {ticker} (Exp: {expiration})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    ticker = input("Enter ticker:")
    chain, spot, expiration = get_option_chain(ticker)
    print(f"Spot price: {spot:.2f}, Expiration: {expiration}")
    otm_calls, otm_puts = filter_otm(chain, spot)
    print(f"OTM Calls: {len(otm_calls)}, OTM Puts: {len(otm_puts)}")
    plot_vol_curve(otm_calls, otm_puts, spot, ticker, expiration)

if __name__ == "__main__":
    main() 