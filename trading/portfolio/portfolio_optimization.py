import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
from typing import List, Dict, Tuple
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich import print as rprint

class PortfolioOptimizer:
    """
    Optimizes a stock-based portfolio based on a specified objective.
    """
    @staticmethod
    def fetch_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        console = Console()
        with console.status(f"[bold blue]Fetching data for {len(tickers)} assets...") as status:
            data = yf.download(tickers, start=start_date, end=end_date, progress=False)
            prices = data['Close']
            prices = prices.dropna(axis=1)
            
            console.print(Panel(f"""
[green]Data Loading Complete[/green]
• Using [bold]{len(prices.columns)}[/bold] assets after filtering
• Loaded [bold]{len(prices)}[/bold] rows
• Date Range: [bold]{prices.index[0].strftime('%Y-%m-%d')}[/bold] to [bold]{prices.index[-1].strftime('%Y-%m-%d')}[/bold]
"""))
            return prices

    def __init__(self, price_df: pd.DataFrame = None, tickers: List[str] = None, 
                 start_date: str = None, end_date: str = None):
        """Initialize optimizer with either price data or ticker information."""
        self.console = Console()
        
        if price_df is not None:
            self.price_df = price_df
        elif all(x is not None for x in [tickers, start_date, end_date]):
            self.price_df = self.fetch_data(tickers, start_date, end_date)
        else:
            raise ValueError("Must provide either price_df or (tickers, start_date, end_date)")

        self.returns = self.price_df.pct_change().dropna()
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()
        self.assets = list(self.mean_returns.index)

    def portfolio_performance(self, weights, risk_free_rate=0.0) -> Tuple[float, float, float]:
        weights = np.array(weights)
        portfolio_return = np.sum(self.mean_returns * weights) * 252  # annualized
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
        sharpe = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
        return portfolio_return, portfolio_vol, sharpe

    def optimize(self, objective='sharpe', risk_free_rate=0.0) -> np.ndarray:
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
                return self.portfolio_performance(weights, risk_free_rate)[1]
            result = minimize(port_vol, x0, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
        else:
            raise ValueError(f"Unknown objective: {objective}")

        if not result.success:
            raise RuntimeError("Optimization failed: " + result.message)
        return result.x

    def summary(self, weights, risk_free_rate=0.0) -> Dict:
        portfolio_return, portfolio_vol, sharpe = self.portfolio_performance(weights, risk_free_rate)
        return {
            'weights': weights,
            'annual_return': portfolio_return,
            'annual_volatility': portfolio_vol,
            'sharpe_ratio': sharpe
        }

    def print_portfolio_summary(self, weights: np.ndarray, portfolio_type: str, risk_free_rate=0.0):
        summary = self.summary(weights, risk_free_rate)
        
        # Create weights table
        weights_table = Table(title=f"\n{portfolio_type} Portfolio Weights", show_header=True, header_style="bold magenta")
        weights_table.add_column("Asset", style="cyan")
        weights_table.add_column("Weight", justify="right", style="green")
        
        # Sort weights by value for better visualization
        sorted_weights = sorted(zip(self.assets, weights), key=lambda x: x[1], reverse=True)
        for asset, weight in sorted_weights:
            if weight >= 0.01:  # Only show weights >= 1%
                weights_table.add_row(asset, f"{weight:.2%}")
        
        # Create metrics table
        metrics_table = Table(show_header=True, header_style="bold magenta")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", justify="right", style="green")
        
        metrics_table.add_row("Expected Annual Return", f"{summary['annual_return']:.2%}")
        metrics_table.add_row("Expected Annual Volatility", f"{summary['annual_volatility']:.2%}")
        metrics_table.add_row("Sharpe Ratio", f"{summary['sharpe_ratio']:.2f}")
        
        # Print tables
        self.console.print(weights_table)
        self.console.print(metrics_table)

    def plot_portfolio_weights(self, weights_sharpe, weights_vol, save_path=None):
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.pie(weights_sharpe, labels=self.assets, autopct='%1.1f%%', startangle=90)
        plt.title('Max Sharpe Ratio Portfolio')
        
        plt.subplot(1, 2, 2)
        plt.pie(weights_vol, labels=self.assets, autopct='%1.1f%%', startangle=90)
        plt.title('Min Volatility Portfolio')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def generate_random_portfolios(self, num_portfolios=5000, risk_free_rate=0.0):
        returns = []
        volatilities = []
        sharpe_ratios = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("Generating random portfolios...", total=num_portfolios)
            
            for i in range(num_portfolios):
                weights = np.random.random(len(self.assets))
                weights = weights / np.sum(weights)
                ret, vol, sr = self.portfolio_performance(weights, risk_free_rate)
                returns.append(ret)
                volatilities.append(vol)
                sharpe_ratios.append(sr)
                
                progress.update(task, advance=1)
                
        return returns, volatilities, sharpe_ratios

    def plot_efficient_frontier(self, weights_sharpe, weights_vol, risk_free_rate=0.0, num_portfolios=5000, save_path=None):
        print("\nGenerating efficient frontier...")
        returns, volatilities, sharpe_ratios = self.generate_random_portfolios(num_portfolios, risk_free_rate)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(volatilities, returns, c=sharpe_ratios, cmap='viridis', alpha=0.5)
        plt.colorbar(label='Sharpe Ratio')
        
        max_sharpe_ret, max_sharpe_vol, _ = self.portfolio_performance(weights_sharpe, risk_free_rate)
        min_vol_ret, min_vol_vol, _ = self.portfolio_performance(weights_vol)
        
        plt.scatter(max_sharpe_vol, max_sharpe_ret, c='red', marker='*', s=300, label='Max Sharpe')
        plt.scatter(min_vol_vol, min_vol_ret, c='green', marker='*', s=300, label='Min Volatility')
        
        plt.title('Portfolio Optimization - Efficient Frontier')
        plt.xlabel('Expected Volatility')
        plt.ylabel('Expected Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def optimize_and_plot(self, risk_free_rate=0.0, num_portfolios=5000):
        """Convenience method to run the full optimization and visualization pipeline."""
        with self.console.status("[bold blue]Optimizing portfolios...") as status:
            weights_sharpe = self.optimize(objective='sharpe', risk_free_rate=risk_free_rate)
            weights_vol = self.optimize(objective='volatility', risk_free_rate=risk_free_rate)

        self.console.rule("[bold blue]Portfolio Optimization Results")
        
        self.print_portfolio_summary(weights_sharpe, "Maximum Sharpe Ratio", risk_free_rate)
        self.console.print()  # Add spacing
        self.print_portfolio_summary(weights_vol, "Minimum Volatility", risk_free_rate)

        self.console.print("\n[bold blue]Generating visualizations...[/bold blue]")
        self.plot_portfolio_weights(weights_sharpe, weights_vol, save_path='optimal_portfolio_allocation.png')
        self.plot_efficient_frontier(weights_sharpe, weights_vol, risk_free_rate=risk_free_rate, 
                                   num_portfolios=num_portfolios, save_path='efficient_frontier.png')

        return weights_sharpe, weights_vol

if __name__ == "__main__":
    assets = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'NVDA', 'NFLX', 'WMT', 'JNJ', 'VZ', 'IBM', 'MMM', 'PFE', 'BA', 'CAT', 'CSCO', 'TM', 'V', 'WBA', 'DIS', 'GE', 'GS', 'JPM', 'MCD', 'MRK', 'MS', 'NKE', 'ORCL', 'QCOM', 'RTX', 'TXN', 'UNH', 'VZ', 'WMT', 'XOM']
    
    # Initialize optimizer with tickers and date range
    optimizer = PortfolioOptimizer(
        tickers=assets,
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    
    # Run the full optimization and visualization pipeline
    optimizer.optimize_and_plot(risk_free_rate=0.00)
    