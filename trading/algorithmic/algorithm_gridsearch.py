import itertools
import pandas as pd
from typing import Dict, List, Any, Callable
from rich import print
from rich.table import Table
from rich.console import Console

import sys, os
sys.path.append("trading")
from vectorized_backtesting import VectorizedBacktesting

class AlgorithmGridSearch:
    def __init__(
        self,
        backtest: VectorizedBacktesting,
        strategy_func: Callable,
        param_grid: Dict[str, List[Any]],
        metric: str = "Total Return"
    ):
        """
        Initialize the grid search framework.
        
        Args:
            backtest: VectorizedBacktesting instance
            strategy_func: The strategy function to optimize
            param_grid: Dictionary of parameters to search
            metric: Performance metric to optimize (default: Total Return)
        """
        self.backtest = backtest
        self.strategy_func = strategy_func
        self.param_grid = param_grid
        self.metric = metric
        self.results = []
        self.console = Console()

    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        """Generate all possible combinations of parameters."""
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
        return combinations

    def run(self) -> pd.DataFrame:
        """Run the grid search and return results sorted by the specified metric."""
        param_combinations = self._generate_param_combinations()
        total_combinations = len(param_combinations)
        
        table = Table(title="Grid Search Progress")
        table.add_column("Combination", justify="right")
        table.add_column("Parameters", style="cyan")
        table.add_column("Performance", style="green")
        
        for i, params in enumerate(param_combinations, 1):
            self.backtest.run_strategy(self.strategy_func, **params)
            metrics = self.backtest.get_performance_metrics()
            
            result = {
                "parameters": params,
                "metrics": metrics
            }
            self.results.append(result)
            
            table.add_row(
                f"{i}/{total_combinations}",
                str(params),
                f"{metrics[self.metric]:.2%}"
            )
            self.console.clear()
            self.console.print(table)
        
        results_df = pd.DataFrame([
            {
                **result["parameters"],
                **result["metrics"]
            }
            for result in self.results
        ])
        
        return results_df.sort_values(by=self.metric, ascending=False)

    def get_best_params(self) -> Dict[str, Any]:
        """Get the best parameters based on the specified metric."""
        if not self.results:
            raise ValueError("No results available. Run the grid search first.")
        
        best_result = max(self.results, key=lambda x: x["metrics"][self.metric])
        return best_result["parameters"]

    def get_best_metrics(self) -> Dict[str, float]:
        """Get the metrics for the best parameter combination."""
        if not self.results:
            raise ValueError("No results available. Run the grid search first.")
        
        best_result = max(self.results, key=lambda x: x["metrics"][self.metric])
        return best_result["metrics"]

    def plot_best_performance(self, show_graph: bool = True, advanced: bool = False):
        """Plot the performance of the best parameter combination."""
        best_params = self.get_best_params()
        self.backtest.run_strategy(self.strategy_func, **best_params)
        return self.backtest.plot_performance(show_graph=show_graph, advanced=advanced)

if __name__ == "__main__":
    backtest = VectorizedBacktesting(
        symbol="BTC-USDT",
        initial_capital=10000.0,
        chunks=5,
        interval="5min",
        age_days=10
    )
    
    backtest.fetch_data(kucoin=True)
    
    param_grid = {
        "oversold": [20, 25, 30, 35],
        "overbought": [65, 70, 75, 80]
    }
    
    grid_search = AlgorithmGridSearch(
        backtest=backtest,
        strategy_func=backtest.rsi_strategy,
        param_grid=param_grid,
        metric="Total Return"
    )
    
    results = grid_search.run()
    
    print("\nTop 5 Parameter Combinations:")
    print(results.head())
    
    grid_search.plot_best_performance(advanced=True)
