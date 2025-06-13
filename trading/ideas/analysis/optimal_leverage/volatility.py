import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import mplfinance as mpf

import sys
sys.path.append("trading")
import model_tools as mt
import technical_analysis as ta

data = mt.fetch_data(ticker="BTC-USDT", interval="1day", chunks=100, age_days=0)

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def bs_price(S, K, T, r, sigma, option_type='c'):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'c':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def fetch_deribit_data(ticker: str, days_to_contract_expiry: int) -> pd.DataFrame:
    """Fetch Deribit data for a given ticker and days to contract expiry."""
    # TODO: Implement Deribit data fetching
    return pd.DataFrame()

class VolatilityAnalysis:
    """Using various sources of volatility, estimate the optimal leverage to hold a perpetual contract at."""
    def __init__(self, data: pd.DataFrame, days_to_contract_expiry: int):
        self.data = data
        self.days_to_contract_expiry = days_to_contract_expiry
        self.T = days_to_contract_expiry / 365  # Time to expiry in years
        self.r = 0.0  # Risk-free rate

    def get_historical_volatility(self) -> float:
        """Unscaled historical volatility (returns float or last value of Series)"""
        hv = ta.historical_volatility(ta.log_return(self.data['Close']), df=self.data, output_period=365)
        if isinstance(hv, pd.Series):
            return hv.iloc[-1]
        return float(hv)
    
    def get_implied_vol(self, option_type: str = 'c') -> float:
        """
        Calculate Black-Scholes implied volatility for an ATM option (strike = current price).
        If market_price is None, estimate it using historical volatility.
        """
        S = self.data['Close'].iloc[-1]
        K = S  # ATM
        T = self.T
        r = self.r
        
        hist_vol = self.get_historical_volatility()
        market_price = bs_price(S, K, T, r, hist_vol, option_type)
        
        def objective(sigma):
            return bs_price(S, K, T, r, sigma, option_type) - market_price
        try:
            iv = brentq(objective, 1e-6, 5.0)
            return iv
        except Exception as e:
            print(f"Error calculating Black-Scholes implied vol: {e}")
            print(f"Parameters: S={S}, K={K}, T={T}, market_price={market_price}")
            return None

    def plot_volatility_bands(self):
        """
        Plotly: Plot candlesticks for the full data in log space and overlay 1σ, 2σ, 3σ volatility bands for the next days_to_contract_expiry.
        Bands are calculated using historical volatility (log returns, annualized).
        """
        import plotly.graph_objects as go
        df = self.data.copy()
        # Compute log prices
        df['log_Open'] = np.log(df['Open'])
        df['log_High'] = np.log(df['High'])
        df['log_Low'] = np.log(df['Low'])
        df['log_Close'] = np.log(df['Close'])
        last_log_close = df['log_Close'].iloc[-1]
        hist_vol = self.get_historical_volatility()
        daily_vol = hist_vol / np.sqrt(365)
        days = np.arange(1, self.days_to_contract_expiry + 1)
        band_colors = ['#66c2a5', '#fc8d62', '#8da0cb']
        future_dates = pd.date_range(df['Datetime'].iloc[-1], periods=self.days_to_contract_expiry+1, freq='D')[1:]
        # Candlestick trace in log space
        fig = go.Figure(data=[
            go.Candlestick(
                x=df['Datetime'],
                open=df['log_Open'],
                high=df['log_High'],
                low=df['log_Low'],
                close=df['log_Close'],
                name='Log Candles',
                increasing_line_color='green',
                decreasing_line_color='red',
                showlegend=False
            )
        ])
        # Overlay bands for the next N days (already in log space)
        for k, color in zip([1,2,3], band_colors):
            upper = last_log_close + k * daily_vol * np.sqrt(days)
            lower = last_log_close - k * daily_vol * np.sqrt(days)
            fig.add_trace(go.Scatter(
                x=future_dates, y=upper, mode='lines',
                line=dict(color=color, dash='dash'),
                name=f'+{k}σ',
            ))
            fig.add_trace(go.Scatter(
                x=future_dates, y=lower, mode='lines',
                line=dict(color=color, dash='dash'),
                name=f'-{k}σ',
            ))
        fig.update_layout(
            title='Volatility Bands (Log Price, Plotly)',
            yaxis_title='log(Price)',
            xaxis_title='Date',
            xaxis_rangeslider_visible=False,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        fig.show()

VA = VolatilityAnalysis(data, 15)
VA.plot_volatility_bands()
