import itertools
import pandas as pd
from typing import Dict, List, Any, Callable
from rich import print
from rich.table import Table
from rich.console import Console
from tqdm import tqdm
import numpy as np

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
        else:
            valid_combinations = combinations
        
        return valid_combinations

    def _run_single_combination(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single parameter combination and return the results."""
        # Reset the backtest to ensure a clean environment
        self.backtest.reset_backtest() if hasattr(self.backtest, 'reset_backtest') else None
        
        # Alternatively, recreate the entire backtest with the same settings
        if not hasattr(self.backtest, 'reset_backtest'):
            # Store original settings
            symbol = self.backtest.symbol
            initial_capital = self.backtest.initial_capital
            data = self.backtest.data  # Assuming data is stored in the backtest
            
            # Create a fresh backtest instance with the same data
            temp_backtest = VectorizedBacktesting(
                symbol=symbol, 
                initial_capital=initial_capital
            )
            # If data is already fetched, reuse it instead of fetching again
            if hasattr(self.backtest, 'data') and self.backtest.data is not None:
                temp_backtest.data = self.backtest.data.copy()
                self.backtest = temp_backtest
        
        # Run the strategy with the current parameters
        self.backtest.run_strategy(self.strategy_func, **params)
        metrics = self.backtest.get_performance_metrics()
        return {
            "parameters": params,
            "metrics": metrics
        }

    def run(self) -> pd.DataFrame:
        """Run the grid search sequentially and return results sorted by the specified metric."""
        param_combinations = self._generate_param_combinations()
        total_combinations = len(param_combinations)
        
        table = Table(title="Grid Search Results")
        table.add_column("Parameters", style="cyan")
        table.add_column("Metric", style="green")
        table.add_column("Alpha", style="blue")
        
        self.results = []
        
        with tqdm(total=total_combinations, desc="Grid Search Progress") as progress_bar:
            for params in param_combinations:
                result = self._run_single_combination(params)
                self.results.append(result)
                progress_bar.update(1)
        
        sorted_results = sorted(self.results, key=lambda x: x["metrics"][self.metric], reverse=True)
        
        for i, result in enumerate(sorted_results, 1):
            table.add_row(
                str(result["parameters"]),
                f"{result['metrics'][self.metric]:.2%}",
                f"{result['metrics']['Alpha']:.2%}"
            )
        self.console.print(table)
        
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
        chunks=365,
        interval="5min",
        age_days=0
    )
    
    backtest.fetch_data(kucoin=True)
    
    param_grid = {
        "confidence_threshold": np.arange(0, 100, 1)/100
    }
    
    grid_search = AlgorithmGridSearch(
        backtest=backtest,
        strategy_func=backtest.nn_strategy_batch,
        param_grid=param_grid,
        metric="PT_Ratio",
    )
    
    results = grid_search.run()
    
    grid_search.plot_best_performance(advanced=True)
    print(grid_search.get_best_params())
