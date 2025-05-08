import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize

def get_data(tickers: list, start_date: str, end_date: str):
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    return data

class PortfolioOptimizer:
    def __init__(self, price_df: pd.DataFrame):
        self.price_df = price_df
        self.returns = price_df.pct_change().dropna()
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()

    def portfolio_performance(self, weights, risk_free_rate=0.0):
        weights = np.array(weights)
        port_return = np.sum(self.mean_returns * weights) * 252  # annualized
        port_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
        sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0
        return port_return, port_vol, sharpe

    def optimize(self, objective='sharpe', risk_free_rate=0.0):
        num_assets = len(self.mean_returns)
        args = (risk_free_rate,)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        x0 = np.array(num_assets * [1. / num_assets])

        if objective == 'sharpe':
            def neg_sharpe(weights, risk_free_rate):
                return -self.portfolio_performance(weights, risk_free_rate)[2]
            result = minimize(neg_sharpe, x0, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
        elif objective == 'volatility':
            def port_vol(weights, _):
                return self.portfolio_performance(weights)[1]
            result = minimize(port_vol, x0, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
        else:
            raise ValueError(f"Unknown objective: {objective}")

        if not result.success:
            raise RuntimeError("Optimization failed: " + result.message)
        return result.x

    def summary(self, weights, risk_free_rate=0.0):
        port_return, port_vol, sharpe = self.portfolio_performance(weights, risk_free_rate)
        return {
            'weights': weights,
            'annual_return': port_return,
            'annual_volatility': port_vol,
            'sharpe_ratio': sharpe
        }

if __name__ == "__main__":
    # Define assets and date range
    assets = ['AAPL']
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    
    # Fetch historical data using get_data function
    print(f"Fetching historical data for {len(assets)} assets...")
    data = get_data(assets, start_date, end_date)
    
    # Extract adjusted close prices
    prices = data['Adj Close']
    
    # Check for missing data
    missing = prices.isna().sum()
    if missing.sum() > 0:
        print(f"Warning: Missing data detected:\n{missing[missing > 0]}")
        prices = prices.ffill().bfill()  # Forward and backward fill missing values
    
    print(f"Data loaded: {len(prices)} rows from {prices.index[0]} to {prices.index[-1]}")
    
    # Optimize portfolio
    optimizer = PortfolioOptimizer(prices)
    
    # Optimize for maximum Sharpe ratio
    weights_sharpe = optimizer.optimize(objective='sharpe', risk_free_rate=0.02)
    sharpe_summary = optimizer.summary(weights_sharpe, risk_free_rate=0.02)
    
    print("\nOptimal Portfolio (Max Sharpe Ratio):")
    for asset, weight in zip(assets, weights_sharpe):
        print(f"  {asset}: {weight:.2%}")
    print(f"Expected Annual Return: {sharpe_summary['annual_return']:.2%}")
    print(f"Expected Annual Volatility: {sharpe_summary['annual_volatility']:.2%}")
    print(f"Sharpe Ratio: {sharpe_summary['sharpe_ratio']:.2f}")
    
    # Optimize for minimum volatility
    weights_vol = optimizer.optimize(objective='volatility')
    vol_summary = optimizer.summary(weights_vol)
    
    print("\nOptimal Portfolio (Min Volatility):")
    for asset, weight in zip(assets, weights_vol):
        print(f"  {asset}: {weight:.2%}")
    print(f"Expected Annual Return: {vol_summary['annual_return']:.2%}")
    print(f"Expected Annual Volatility: {vol_summary['annual_volatility']:.2%}")
    print(f"Sharpe Ratio: {vol_summary['sharpe_ratio']:.2f}")
    
    # Visualize the portfolio weights
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.pie(weights_sharpe, labels=assets, autopct='%1.1f%%', startangle=90)
    plt.title('Max Sharpe Ratio Portfolio')
    
    plt.subplot(1, 2, 2)
    plt.pie(weights_vol, labels=assets, autopct='%1.1f%%', startangle=90)
    plt.title('Min Volatility Portfolio')
    
    plt.tight_layout()
    plt.savefig('optimal_portfolio_allocation.png')
    plt.show()
    
    # Plot efficient frontier
    print("\nGenerating efficient frontier...")
    returns = []
    volatilities = []
    sharpe_ratios = []
    
    # Generate random portfolios
    num_portfolios = 5000
    for i in range(num_portfolios):
        weights = np.random.random(len(assets))
        weights = weights / np.sum(weights)
        ret, vol, sr = optimizer.portfolio_performance(weights, risk_free_rate=0.02)
        returns.append(ret)
        volatilities.append(vol)
        sharpe_ratios.append(sr)
        
        # Show progress occasionally
        if (i+1) % 1000 == 0:
            print(f"  Generated {i+1}/{num_portfolios} random portfolios")
    
    # Plot efficient frontier
    plt.figure(figsize=(10, 6))
    plt.scatter(volatilities, returns, c=sharpe_ratios, cmap='viridis', alpha=0.5)
    plt.colorbar(label='Sharpe Ratio')
    
    # Plot max Sharpe and min vol portfolios
    max_sharpe_ret, max_sharpe_vol, _ = optimizer.portfolio_performance(weights_sharpe, risk_free_rate=0.02)
    min_vol_ret, min_vol_vol, _ = optimizer.portfolio_performance(weights_vol)
    
    plt.scatter(max_sharpe_vol, max_sharpe_ret, c='red', marker='*', s=300, label='Max Sharpe')
    plt.scatter(min_vol_vol, min_vol_ret, c='green', marker='*', s=300, label='Min Volatility')
    
    plt.title('Portfolio Optimization - Efficient Frontier')
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('efficient_frontier.png')
    plt.show()
    