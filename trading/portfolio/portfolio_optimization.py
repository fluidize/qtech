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
from datetime import datetime, timedelta


class PortfolioOptimizer:
    """
    Optimizes a stock-based portfolio based on a specified objective.
    """
    @staticmethod
    def fetch_data(tickers: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        console = Console()
        tickers = [ticker.replace(".", "-") for ticker in tickers]
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        with console.status(f"[bold blue]Fetching data for {len(tickers)} assets...") as status:
            data = yf.download(tickers, start=start_str, end=end_str, progress=False)
            prices = data['Close']
            prices = prices.dropna(axis=1)
            
            console.print(Panel(f"""
[green]Data Loading Complete[/green]
• Using [bold]{len(prices.columns)}[/bold] assets after filtering
• Loaded [bold]{len(prices)}[/bold] rows
• Date Range: [bold]{prices.index[0].strftime('%Y-%m-%d')}[/bold] to [bold]{prices.index[-1].strftime('%Y-%m-%d')}[/bold]
"""))
            return prices

    def __init__(self, tickers: List[str] = None, 
                 start_date: datetime = None, end_date: datetime = None, backtest_start_date: datetime = None, backtest_end_date: datetime = None, rfr: float = 0.0):
        """Initialize optimizer with either price data or ticker information."""
        self.console = Console()
        self.price_df = self.fetch_data(tickers, start_date, end_date)
        self.backtest_price_df = self.fetch_data(tickers, backtest_start_date, backtest_end_date)

        # Keep only shared columns (tickers)
        shared_cols = list(set(self.price_df.columns) & set(self.backtest_price_df.columns))
        self.price_df = self.price_df[shared_cols]
        self.backtest_price_df = self.backtest_price_df[shared_cols]
        self.assets = shared_cols
        self.rfr = rfr

    def portfolio_performance(self, weights, backtest=False) -> Dict[str, float]:
        weights = np.array(weights)
        day_returns = self.price_df.pct_change().dropna() if not backtest else self.backtest_price_df.pct_change().dropna()
        cov_matrix = day_returns.cov()
        portfolio_returns = np.dot(day_returns, weights)  # daily returns
        total_return = (1 + portfolio_returns).prod() - 1  # total return over period
        annual_return = (1 + total_return) ** (252/len(day_returns)) - 1  # annualized return
        annual_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        sharpe_ratio = (annual_return - self.rfr) / annual_volatility if annual_volatility > 0 else 0
        daily_rf = self.rfr / 252
        downside_returns = portfolio_returns[portfolio_returns < daily_rf]
        if len(downside_returns) > 0:
            downside_volatility = np.sqrt(np.mean((downside_returns - daily_rf) ** 2) * 252)
            sortino_ratio = (annual_return - self.rfr) / downside_volatility
        else:
            downside_volatility = 0.0
            sortino_ratio = np.inf if annual_return > self.rfr else 0
        return {
            'weights': weights,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'downside_volatility': downside_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio
        }

    def optimize(self, objective='sharpe', weights_threshold=0.01) -> np.ndarray:
        day_returns = self.price_df.pct_change().dropna()
        mean_returns = day_returns.mean()
        num_assets = len(mean_returns)
        args = ()
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        x0 = np.array(num_assets * [1. / num_assets])

        if objective == 'sharpe':
            def neg_sharpe(weights):
                return -self.portfolio_performance(weights, backtest=False)["sharpe_ratio"]
            result = minimize(neg_sharpe, x0, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
        elif objective == 'sortino':
            def neg_sortino(weights):
                return -self.portfolio_performance(weights, backtest=False)["sortino_ratio"]
            result = minimize(neg_sortino, x0, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
        else:
            raise ValueError(f"Unknown objective: {objective}")

        if not result.success:
            raise RuntimeError("Optimization failed: " + result.message)
        for i, weight in enumerate(result.x):
            if round(weight, 2) < weights_threshold:
                result.x[i] = 0.00
        return result.x

    def print_expected_portfolio_summary(self, weights: np.ndarray, portfolio_type: str):
        perf = self.portfolio_performance(weights, backtest=False)
        weights_table = Table(title=f"\n{portfolio_type} Portfolio Weights", show_header=True, header_style="bold magenta")
        weights_table.add_column("Asset", style="cyan")
        weights_table.add_column("Weight", justify="right", style="green")
        sorted_weights = sorted(zip(self.assets, weights), key=lambda x: x[1], reverse=True)
        for asset, weight in sorted_weights:
            weights_table.add_row(asset, f"{weight:.2%}")
        metrics_table = Table(show_header=True, header_style="bold magenta")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", justify="right", style="green")
        metrics_table.add_row("Expected Annual Return", f"{perf['annual_return']:.2%}")
        metrics_table.add_row("Expected Annual Volatility", f"{perf['annual_volatility']:.2%}")
        metrics_table.add_row("Downside Volatility", f"{perf['downside_volatility']:.2%}")
        metrics_table.add_row("Sharpe Ratio", f"{perf['sharpe_ratio']:.2f}")
        metrics_table.add_row("Sortino Ratio", f"{perf['sortino_ratio']:.2f}")
        self.console.print(weights_table)
        self.console.print(metrics_table)

    def plot_portfolio_weights(self, weights_sharpe: np.ndarray, weights_sortino: np.ndarray, save_path=None):
        """Plot the portfolio weights for the maximum Sharpe and Sortino portfolios with a pie chart."""
        plt.figure(figsize=(12, 6))

        sharpe_labels = np.array(self.assets)
        sortino_labels = np.array(self.assets)

        trimmed_weights_sharpe = weights_sharpe[weights_sharpe != 0.00]
        trimmed_weights_sortino = weights_sortino[weights_sortino != 0.00]

        sharpe_labels = sharpe_labels[weights_sharpe != 0.00]
        sortino_labels = sortino_labels[weights_sortino != 0.00]

        plt.subplot(1, 2, 1)
        plt.pie(trimmed_weights_sharpe, labels=sharpe_labels, autopct='%1.1f%%', startangle=90)
        plt.title('Max Sharpe Ratio Portfolio')
        
        plt.subplot(1, 2, 2)
        plt.pie(trimmed_weights_sortino, labels=sortino_labels, autopct='%1.1f%%', startangle=90)
        plt.title('Max Sortino Ratio Portfolio')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def generate_random_portfolios(self, num_portfolios=5000) -> np.ndarray:
        """Generate random portfolio weights only."""
        weights_list = []
        for _ in range(num_portfolios):
            weights = np.random.random(len(self.assets))
            weights = weights / np.sum(weights)
            weights_list.append(weights)
        return np.array(weights_list) # 2D

    def evaluate_random_portfolios(self, weights_list: np.ndarray, backtest=False) -> Tuple[list, list, list, list]:
        """Evaluate performance metrics for a list of portfolio weights."""
        returns = []
        volatilities = []
        sharpe_ratios = []
        sortino_ratios = []
        with Progress(TextColumn("[bold blue]Evaluating random portfolios..."), BarColumn(), TaskProgressColumn()) as progress:
            task = progress.add_task(description="[bold blue]Evaluating random portfolios...", total=len(weights_list))
            for i, weights in enumerate(weights_list):
                outputs = self.portfolio_performance(weights, backtest=backtest)
                returns.append(outputs["annual_return"])
                volatilities.append(outputs["annual_volatility"])
                sharpe_ratios.append(outputs["sharpe_ratio"])
                sortino_ratios.append(outputs["sortino_ratio"])
                progress.update(task, completed=i+1)
        return returns, volatilities, sharpe_ratios, sortino_ratios

    def plot_efficient_frontier(self, weights_sharpe, weights_sortino, num_portfolios=10000, save_path=None):
        random_weights = self.generate_random_portfolios(num_portfolios)
        plt.figure(figsize=(12, 6))

        # expected efficient frontier
        expected_returns, expected_volatilities, expected_sharpe_ratios, expected_sortino_ratios = self.evaluate_random_portfolios(random_weights, backtest=False)
        expected_sharpe_perf = self.portfolio_performance(weights_sharpe, backtest=False)
        expected_sortino_perf = self.portfolio_performance(weights_sortino, backtest=False)
        
        plt.subplot(1, 2, 1)
        plt.scatter(x=expected_volatilities, y=expected_returns, c=expected_sharpe_ratios, cmap='viridis', alpha=0.5)
        expected_max_volatility = max(np.concatenate([expected_volatilities, np.array([expected_sharpe_perf["annual_volatility"]]), np.array([expected_sortino_perf["annual_volatility"]])]))
        expected_linereg_model = np.polyfit(expected_volatilities, expected_returns, deg=2)
        expected_linereg_model_y = np.polyval(expected_linereg_model, np.linspace(min(expected_volatilities), expected_max_volatility, 100))
        plt.plot(np.linspace(min(expected_volatilities), expected_max_volatility, 100), expected_linereg_model_y, linestyle="--", alpha=0.5, label=f'Curve(deg={len(expected_linereg_model)}): {expected_linereg_model}')

        plt.colorbar(label='Sharpe Ratio')
        plt.scatter(expected_sharpe_perf["annual_volatility"], expected_sharpe_perf["annual_return"], 
                   c='red', marker='*', s=300, label='Max Sharpe')
        plt.scatter(expected_sortino_perf["annual_volatility"], expected_sortino_perf["annual_return"], 
                   c='green', marker='*', s=300, label='Max Sortino')
        plt.title(f'Expected Efficient Frontier (Sharpe) | {num_portfolios} samples')
        plt.xlabel('Expected Volatility')
        plt.ylabel('Expected Return')
        plt.legend()

        # backtested efficient frontier
        backtest_returns, backtest_volatilities, backtest_sharpe_ratios, backtest_sortino_ratios = self.evaluate_random_portfolios(random_weights, backtest=True)
        backtest_sharpe_perf = self.portfolio_performance(weights_sharpe, backtest=True)
        backtest_sortino_perf = self.portfolio_performance(weights_sortino, backtest=True)

        plt.subplot(1, 2, 2)
        plt.scatter(x=backtest_volatilities, y=backtest_returns, c=backtest_sharpe_ratios, cmap='viridis', alpha=0.5)
        backtest_max_volatility = max(np.concatenate([backtest_volatilities, np.array([backtest_sharpe_perf["annual_volatility"]]), np.array([backtest_sortino_perf["annual_volatility"]])]))
        backtest_linereg_model = np.polyfit(backtest_volatilities, backtest_returns, deg=2)
        backtest_linereg_model_y = np.polyval(backtest_linereg_model, np.linspace(min(backtest_volatilities), backtest_max_volatility, 100))
        plt.plot(np.linspace(min(backtest_volatilities), backtest_max_volatility, 100), backtest_linereg_model_y, linestyle="--", alpha=0.5, label=f'Curve(deg={len(backtest_linereg_model)}): {backtest_linereg_model}')
        plt.colorbar(label='Sharpe Ratio')
        plt.scatter(backtest_sharpe_perf["annual_volatility"], backtest_sharpe_perf["annual_return"], 
                   c='red', marker='*', s=300, label='Max Sharpe')
        plt.scatter(backtest_sortino_perf["annual_volatility"], backtest_sortino_perf["annual_return"], 
                   c='green', marker='*', s=300, label='Max Sortino')
        plt.title(f'Backtested Efficient Frontier (Sharpe) | {num_portfolios} samples')
        plt.xlabel('Backtested Volatility')
        plt.ylabel('Backtested Return')
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def optimize_and_plot(self, num_portfolios=10000):
        """Convenience method to run the full optimization and visualization pipeline."""
        with self.console.status("[bold blue]Optimizing portfolio...") as status:
            weights_sharpe = self.optimize(objective='sharpe')
            weights_sortino = self.optimize(objective='sortino')

        self.console.rule("[bold blue]Portfolio Optimization Results")
        
        self.print_expected_portfolio_summary(weights_sharpe, "Maximum Sharpe Ratio")
        self.console.print()
        self.print_expected_portfolio_summary(weights_sortino, "Maximum Sortino Ratio")

        self.console.print("\n[bold blue]Generating visualizations...[/bold blue]")
        self.plot_portfolio_weights(weights_sharpe, weights_sortino, save_path='optimal_portfolio_allocation.png')
        self.plot_efficient_frontier(weights_sharpe, weights_sortino, num_portfolios=num_portfolios, save_path='efficient_frontier.png')

        return weights_sharpe, weights_sortino

if __name__ == "__main__":
    assets = pd.read_csv(r'trading\portfolio\assets.csv')['Ticker'].values.tolist()
    # assets = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'NVDA', 'NFLX', 'WMT', 'JNJ', 'VZ', 'IBM', 'MMM', 'PFE', 'BA', 'CAT', 'CSCO', 'TM', 'V', 'WBA', 'DIS', 'GE', 'GS', 'JPM', 'MCD', 'MRK', 'MS', 'NKE', 'ORCL', 'QCOM', 'RTX', 'TXN', 'UNH', 'VZ', 'WMT', 'XOM']

    optimizer = PortfolioOptimizer(
        tickers=assets,
        start_date=datetime.now() - timedelta(days=365*2),
        end_date=datetime.now() - timedelta(days=365),
        backtest_start_date=datetime.now() - timedelta(days=366),
        backtest_end_date=datetime.now()
    )
    
    optimizer.optimize_and_plot(num_portfolios=10000)
    