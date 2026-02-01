import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib import cbook, cm
from matplotlib.colors import LightSource
import numpy as np
from datetime import datetime
from scipy.interpolate import griddata
from rich import print

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

def get_option_iv(option):
    return option.impliedVolatility

def filter_otm(options, spot):
    calls = options.calls.copy()
    puts = options.puts.copy()
    otm_calls = calls[calls['strike'] > spot]
    otm_puts = puts[puts['strike'] < spot]
    return otm_calls, otm_puts

def days_to_expiry(expiration_str):
    """Convert expiration date string to days from today"""
    expiry_date = datetime.strptime(expiration_str, '%Y-%m-%d')
    today = datetime.now()
    return (expiry_date - today).days

def volatility_curve():
    ticker = input("Enter ticker: ")
    chain, spot, expiration = get_option_chain(ticker)
    otm_calls, otm_puts = filter_otm(chain, spot)

    otm_calls_iv = get_option_iv(otm_calls)
    otm_puts_iv = get_option_iv(otm_puts)

    otm_calls_strike = otm_calls['strike']
    otm_puts_strike = otm_puts['strike']

    plt.figure(figsize=(10, 6))

    plt.plot(otm_calls_strike, otm_calls_iv, label='OTM Calls', color='green', marker='o')
    plt.plot(otm_puts_strike, otm_puts_iv, label='OTM Puts', color='red', marker='o')
    plt.axvline(spot, color='black', linestyle='--', label='Spot Price')

    plt.xlabel('Strike Price')
    plt.ylabel('Implied Volatility')
    plt.title(f'Volatility Curve for {ticker} (Exp: {expiration})')
    plt.legend()
    plt.grid(True)
    plt.show()

def volatility_surface():
    ticker = input("Enter ticker: ")
    ticker_obj = yf.Ticker(ticker)

    x_calls = np.array([])
    y_calls = np.array([])
    z_calls = np.array([])
    x_puts = np.array([])
    y_puts = np.array([])
    z_puts = np.array([])

    for expiration in ticker_obj.options:
        chain, spot, expiration = get_option_chain(ticker, expiration)
        print(f"Expiration: {expiration}")
        otm_calls, otm_puts = filter_otm(chain, spot)

        otm_calls_iv = get_option_iv(otm_calls)
        otm_puts_iv = get_option_iv(otm_puts)

        otm_calls_strike = otm_calls['strike']
        otm_puts_strike = otm_puts['strike']

        days_exp = days_to_expiry(expiration)

        x_calls = np.append(x_calls, otm_calls_strike)
        y_calls = np.append(y_calls, np.full(len(otm_calls_strike), days_exp))
        z_calls = np.append(z_calls, otm_calls_iv)
        
        x_puts = np.append(x_puts, otm_puts_strike)
        y_puts = np.append(y_puts, np.full(len(otm_puts_strike), days_exp))
        z_puts = np.append(z_puts, otm_puts_iv)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x_calls, y_calls, z_calls, c='green', marker='o', label='OTM Calls', s=10, alpha=0.5)
    ax.scatter(x_puts, y_puts, z_puts, c='red', marker='o', label='OTM Puts', s=10, alpha=0.5)

    x_all = np.concatenate([x_calls, x_puts])
    y_all = np.concatenate([y_calls, y_puts])
    z_all = np.concatenate([z_calls, z_puts])
    
    xi = np.linspace(x_all.min(), x_all.max(), 50)
    yi = np.linspace(y_all.min(), y_all.max(), 50)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    zi_grid = griddata((x_all, y_all), z_all, (xi_grid, yi_grid), method='linear')
    
    surf = ax.plot_surface(xi_grid, yi_grid, zi_grid, cmap='viridis', alpha=0.5)

    x_plane = np.array([spot, spot])
    y_plane = np.array([0, y_all.max()])
    
    ax.plot(x_plane, y_plane, np.zeros_like(x_plane), color='black', label='Spot Price')

    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Days to Expiry')
    ax.set_zlabel('Implied Volatility')
    ax.set_title(f'Volatility Surface for {ticker}')
    ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    volatility_surface()