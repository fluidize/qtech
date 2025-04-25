import itertools
import pandas as pd
from typing import Dict, List, Any, Callable
from rich import print
from rich.table import Table
from rich.console import Console
from tqdm import tqdm
import concurrent.futures
from functools import partial

import sys, os
sys.path.append("trading")
from vectorized_backtesting import VectorizedBacktesting

class AlgorithmGridSearch:
    def __init__(
        self,
        backtest: VectorizedBacktesting,
        strategy_func: Callable,
        param_grid: Dict[str, List[Any]],
        metric: str = "Total Return",
        max_workers: int = None
    ):
        """
        Initialize the grid search framework.
        
        Args:
            backtest: VectorizedBacktesting instance
            strategy_func: The strategy function to optimize
            param_grid: Dictionary of parameters to search
            metric: Performance metric to optimize (default: Total Return)
            max_workers: Maximum number of parallel workers (default: None, uses system's default)
        """
        self.backtest = backtest
        self.strategy_func = strategy_func
        self.param_grid = param_grid
        self.metric = metric
        self.results = []
        self.console = Console()
        self.max_workers = max_workers

    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        """Generate all possible combinations of parameters."""
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
        
        if self.strategy_func.__name__ == "rsi_strategy":
            valid_combinations = [
                param for param in combinations 
                if param["oversold"] < param["overbought"]
            ]
        if self.strategy_func.__name__ == "macd_strategy":
            valid_combinations = [
                param for param in combinations 
                if param["fastperiod"] < param["slowperiod"]
            ]
        
        return valid_combinations

    def _run_single_combination(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single parameter combination and return the results."""
        self.backtest.run_strategy(self.strategy_func, **params)
        metrics = self.backtest.get_performance_metrics()
        return {
            "parameters": params,
            "metrics": metrics
        }

    def run(self, multiprocessing: bool = False) -> pd.DataFrame:
        """Run the grid search in parallel and return results sorted by the specified metric."""
        param_combinations = self._generate_param_combinations()
        total_combinations = len(param_combinations)
        
        table = Table(title="Grid Search Results")
        table.add_column("Parameters", style="cyan")
        table.add_column("Performance", style="green")
        
        progress_bar = tqdm(total=total_combinations, desc="Grid Search Progress")
        
        def update_progress(future):
            result = future.result()
            self.results.append(result)
            progress_bar.update(1)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._run_single_combination, params) for params in param_combinations]
            
            for future in futures:
                future.add_done_callback(update_progress)
            
            concurrent.futures.wait(futures)
        
        progress_bar.close()
        
        # Sort results by metric in descending order
        sorted_results = sorted(self.results, key=lambda x: x["metrics"][self.metric], reverse=True)
        
        # Print the results table
        for i, result in enumerate(sorted_results, 1):
            table.add_row(
                str(result["parameters"]),
                f"{result['metrics'][self.metric]:.2%}"
            )
        self.console.print(table)
        
        # Create and return the results DataFrame
        results_df = pd.DataFrame([
            {
                **result["parameters"],
                **result["metrics"]
            }
            for result in sorted_results
        ])
        
        return results_df

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
        chunks=15,
        interval="1min",
        age_days=0
    )
    
    backtest.fetch_data(kucoin=True)
    
    param_grid = {
        "fastperiod": list(range(1, 100, 10)),
        "slowperiod": list(range(1, 100, 10)),
        "signalperiod": list(range(1, 100, 10))
    }
    
    grid_search = AlgorithmGridSearch(
        backtest=backtest,
        strategy_func=backtest.macd_strategy,
        param_grid=param_grid,
        metric="Alpha",
        max_workers=24  # Adjust based on your system's capabilities
    )
    
    results = grid_search.run()
    
    grid_search.plot_best_performance(advanced=True)
    print(grid_search.get_best_params())
