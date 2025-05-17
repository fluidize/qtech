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
from matplotlib.colors import LightSource


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

        self.day_returns = self.price_df.pct_change().dropna()
        self.total_returns = (1 + self.day_returns).cumprod() - 1
        self.mean_returns = self.day_returns.mean()
        self.cov_matrix = self.day_returns.cov()
        self.assets = list(self.mean_returns.index)

    def portfolio_performance(self, weights, risk_free_rate=0.0) -> Dict[str, float]:
        weights = np.array(weights)
        
        # Calculate portfolio returns and standard deviation
        portfolio_returns = np.dot(self.day_returns, weights)  # daily returns
        total_return = (1 + portfolio_returns).prod() - 1  # total return over period
        annual_return = (1 + total_return) ** (252/len(self.day_returns)) - 1  # annualized return
        annual_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
        
        # Calculate Sharpe ratio
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        
        # Calculate downside volatility and Sortino ratio
        # Use daily risk-free rate for threshold
        daily_rf = risk_free_rate / 252
        downside_returns = portfolio_returns[portfolio_returns < daily_rf]
        
        if len(downside_returns) > 0:
            # Calculate semi-deviation focusing only on returns below target (risk-free rate)
            downside_volatility = np.sqrt(np.mean((downside_returns - daily_rf) ** 2) * 252)
            sortino_ratio = (annual_return - risk_free_rate) / downside_volatility
        else:
            # If no downside returns, set high Sortino ratio
            downside_volatility = 0.0
            sortino_ratio = np.inf if annual_return > risk_free_rate else 0
        
        return {
            'weights': weights,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'downside_volatility': downside_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio
        }

    def optimize(self, objective='sharpe', risk_free_rate=0.0) -> np.ndarray:
        num_assets = len(self.mean_returns)
        args = (risk_free_rate,)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        x0 = np.array(num_assets * [1. / num_assets])

        if objective == 'sharpe':
            def neg_sharpe(weights, risk_free_rate):
                return -self.portfolio_performance(weights, risk_free_rate)["sharpe_ratio"]
            result = minimize(neg_sharpe, x0, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
        elif objective == 'sortino':
            def neg_sortino(weights, risk_free_rate):
                return -self.portfolio_performance(weights, risk_free_rate)["sortino_ratio"]
            result = minimize(neg_sortino, x0, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
        else:
            raise ValueError(f"Unknown objective: {objective}")

        if not result.success:
            raise RuntimeError("Optimization failed: " + result.message)
        return result.x

    def print_portfolio_summary(self, weights: np.ndarray, portfolio_type: str, risk_free_rate=0.0):
        perf = self.portfolio_performance(weights, risk_free_rate)
        
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
        
        metrics_table.add_row("Expected Annual Return", f"{perf['annual_return']:.2%}")
        metrics_table.add_row("Expected Annual Volatility", f"{perf['annual_volatility']:.2%}")
        metrics_table.add_row("Downside Volatility", f"{perf['downside_volatility']:.2%}")
        metrics_table.add_row("Sharpe Ratio", f"{perf['sharpe_ratio']:.2f}")
        metrics_table.add_row("Sortino Ratio", f"{perf['sortino_ratio']:.2f}")
        
        # Print tables
        self.console.print(weights_table)
        self.console.print(metrics_table)

    def plot_portfolio_weights(self, weights_sharpe, weights_sortino, save_path=None):
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.pie(weights_sharpe, labels=self.assets, autopct='%1.1f%%', startangle=90)
        plt.title('Max Sharpe Ratio Portfolio')
        
        plt.subplot(1, 2, 2)
        plt.pie(weights_sortino, labels=self.assets, autopct='%1.1f%%', startangle=90)
        plt.title('Max Sortino Ratio Portfolio')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def generate_random_portfolios(self, num_portfolios=5000, risk_free_rate=0.0) -> Tuple[List[float], List[float], List[float], List[float]]:
        returns = []
        volatilities = []
        sharpe_ratios = []
        sortino_ratios = []
        
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
                outputs = self.portfolio_performance(weights, risk_free_rate)
                returns.append(outputs["annual_return"])
                volatilities.append(outputs["annual_volatility"])
                sharpe_ratios.append(outputs["sharpe_ratio"])
                sortino_ratios.append(outputs["sortino_ratio"])
                
                progress.update(task, advance=1)
                
        return returns, volatilities, sharpe_ratios, sortino_ratios

    def plot_efficient_frontier(self, weights_sharpe, weights_sortino, risk_free_rate=0.0, num_portfolios=50000, save_path=None):
        print("\nGenerating efficient frontier...")
        returns, volatilities, sharpe_ratios, sortino_ratios = self.generate_random_portfolios(num_portfolios, risk_free_rate)
        sharpe_perf = self.portfolio_performance(weights_sharpe, risk_free_rate)
        sortino_perf = self.portfolio_performance(weights_sortino, risk_free_rate)
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.scatter(x=volatilities, y=returns, c=sharpe_ratios, cmap='viridis', alpha=0.5)
        max_graph_volatility = max(np.concatenate([volatilities, np.array([sharpe_perf["annual_volatility"]]), np.array([sortino_perf["annual_volatility"]])]))
        linereg_model = np.polyfit(volatilities, returns, deg=2)
        linereg_model_y = np.polyval(linereg_model, np.linspace(min(volatilities), max_graph_volatility, 100))
        plt.plot(np.linspace(min(volatilities), max_graph_volatility, 100), linereg_model_y, linestyle="--", alpha=0.5, label=f'Curve(deg={len(linereg_model)}): {linereg_model}')
        plt.colorbar(label='Sharpe Ratio')
        
        plt.scatter(sharpe_perf["annual_volatility"], sharpe_perf["annual_return"], 
                   c='red', marker='*', s=300, label='Max Sharpe')
        plt.scatter(sortino_perf["annual_volatility"], sortino_perf["annual_return"], 
                   c='green', marker='*', s=300, label='Max Sortino')
        
        plt.title(f'Efficient Frontier (Sharpe) | {num_portfolios} samples')
        plt.xlabel('Expected Volatility')
        plt.ylabel('Expected Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def optimize_and_plot(self, risk_free_rate=0.0, num_portfolios=5000):
        """Convenience method to run the full optimization and visualization pipeline."""
        with self.console.status("[bold blue]Optimizing portfolios...") as status:
            weights_sharpe = self.optimize(objective='sharpe', risk_free_rate=risk_free_rate)
            weights_sortino = self.optimize(objective='sortino', risk_free_rate=risk_free_rate)

        self.console.rule("[bold blue]Portfolio Optimization Results")
        
        self.print_portfolio_summary(weights_sharpe, "Maximum Sharpe Ratio", risk_free_rate)
        self.console.print()  # Add spacing
        self.print_portfolio_summary(weights_sortino, "Maximum Sortino Ratio", risk_free_rate)

        self.console.print("\n[bold blue]Generating visualizations...[/bold blue]")
        self.plot_portfolio_weights(weights_sharpe, weights_sortino, save_path='optimal_portfolio_allocation.png')
        self.plot_efficient_frontier(weights_sharpe, weights_sortino, risk_free_rate=risk_free_rate, 
                                   num_portfolios=num_portfolios, save_path='efficient_frontier.png')

        return weights_sharpe, weights_sortino

if __name__ == "__main__":
    assets = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'NVDA', 'NFLX', 'WMT', 'JNJ', 'VZ', 'IBM', 'MMM', 'PFE', 'BA', 'CAT', 'CSCO', 'TM', 'V', 'WBA', 'DIS', 'GE', 'GS', 'JPM', 'MCD', 'MRK', 'MS', 'NKE', 'ORCL', 'QCOM', 'RTX', 'TXN', 'UNH', 'VZ', 'WMT', 'XOM']
    
    optimizer = PortfolioOptimizer(
        tickers=assets,
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    
    optimizer.optimize_and_plot(risk_free_rate=0.00)
    