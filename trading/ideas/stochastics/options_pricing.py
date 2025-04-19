import math
from scipy.stats import norm
from datetime import datetime

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """
    Black-Scholes Option Pricing Model

    Parameters:
    - S: Spot price of the underlying asset
    - K: Strike price
    - T: Time to expiry in years
    - r: Risk-free interest rate (annualized)
    - sigma: Volatility of the underlying asset (annualized)
    - option_type: 'call' or 'put'

    Returns:
    - Option price
    """

    if T <= 0:
        return max(0.0, (S - K) if option_type == 'call' else (K - S))

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price

# Inputs
S = 84000         # Current stock price
K = 175000         # Strike price
expiry_date = datetime(2027, 12, 27)
today = datetime(2025, 4, 19)
T = (expiry_date - today).days / 365  # Time to expiry in years
r = 0.6      # Risk-free interest rate (3%)
sigma = 0.53    # Volatility (25%)

# Get price
call_price = black_scholes_price(S, K, T, r, sigma, option_type='call')
put_price = black_scholes_price(S, K, T, r, sigma, option_type='put')
coin_per_contract = 0.05714

call_price *= coin_per_contract
put_price *= coin_per_contract

print(f"Call Price: {call_price:.2f}")
print(f"Put Price: {put_price:.2f}")
