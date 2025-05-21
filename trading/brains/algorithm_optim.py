import itertools
import pandas as pd
from typing import Dict, List, Any, Callable
from rich import print
from rich.table import Table
from rich.console import Console
from tqdm import tqdm
import numpy as np
from scipy.optimize import minimize
import sys, os
sys.path.append("trading")
from vectorized_backtesting import VectorizedBacktesting
from skopt import gp_minimize
from skopt.space import Integer

class AlgorithmGridSearch:
    def __init__(
        self,
        engine: VectorizedBacktesting,
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
        self.engine = engine
        self.strategy_func = strategy_func
        self.param_grid = param_grid
        self.metric = metric
        self.results = []
        self.console = Console()

    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        """Generate all possible combinations of parameters. Function constraints are applied here."""
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
        elif self.strategy_func.__name__ == "ema_cross_strategy":
            valid_combinations = [
                param for param in combinations 
                if param["fast_period"] < param["slow_period"]
            ]
        else:
            valid_combinations = combinations
        
        return valid_combinations

    def _run_single_combination(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single parameter combination and return the results."""
        self.engine.run_strategy(self.strategy_func, **params) #previous data gets overwritten
        metrics = self.engine.get_performance_metrics()
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
        table.add_column("Active Returns", style="magenta")

        self.results = []
        
        table_limit = 50
        with tqdm(total=total_combinations, desc="Grid Search Progress") as progress_bar:
            for count, params in enumerate(param_combinations):
                result = self._run_single_combination(params)
                self.results.append(result)
                progress_bar.update(1)
                if count >= table_limit-1:
                    break

        sorted_results = sorted(self.results, key=lambda x: x["metrics"][self.metric], reverse=True)
        
        for i, result in enumerate(sorted_results, 1):
            table.add_row(
                str(result["parameters"]),
                f"{result['metrics'][self.metric]:.2%}",
                f"{result['metrics']['Alpha']:.2%}",
                f"{result['metrics']['Active_Returns']:.2%}"
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
        self.engine.run_strategy(self.strategy_func, **best_params)
        return self.engine.plot_performance(show_graph=show_graph, advanced=advanced)

class AlgorithmGradientDescent:
    def __init__(self, engine: VectorizedBacktesting, strategy_func: Callable, start_params: Dict[str, Any], metric: str = "Total Return", n_calls: int = 30):
        self.engine = engine
        self.strategy_func = strategy_func
        self.start_params = start_params
        self.metric = metric
        self.param_names = list(start_params.keys())
        self.best_params = None
        self.best_metrics = None
        self.n_calls = n_calls
        # Use Integer space for all params, using the min/max from start_params or a default range
        self.bounds = [(1, 50) for _ in self.param_names]  # You can customize this per param if needed
        self.dimensions = [Integer(b[0], b[1], name=k) for k, b in zip(self.param_names, self.bounds)]

    def _array_to_param_dict(self, arr):
        return {k: int(round(v)) for k, v in zip(self.param_names, arr)}

    def run(self) -> pd.DataFrame:
        def objective(params):
            param_dict = self._array_to_param_dict(params)
            self.engine.run_strategy(self.strategy_func, **param_dict)
            metrics = self.engine.get_performance_metrics()
            return -metrics[self.metric]  # Negate if maximizing

        result = gp_minimize(
            objective,
            self.dimensions,
            n_calls=self.n_calls,
            random_state=42,
        )
        self.best_params = self._array_to_param_dict(result.x)
        self.best_metrics = -result.fun
        return self.best_metrics

    def plot_best_performance(self, show_graph: bool = True, advanced: bool = False):
        self.engine.run_strategy(self.strategy_func, **self.best_params)
        return self.engine.plot_performance(show_graph=show_graph, advanced=advanced)

    def get_best_params(self) -> Dict[str, Any]:
        return self.best_params

    def get_best_metrics(self) -> Dict[str, float]:
        return self.best_metrics

class AlgorithmAssignmentOptimizer:
    """ Instead of optimizing the algorithm itself, we find the best pair fit for the given algorithm."""
    def __init__(self, engine: VectorizedBacktesting, pairs: List[str], chunks: int, interval: str, age_days: int, strategy_func: Callable, params: Dict[str, List[Any]], metric: str = "Total Return"):
        self.engine = engine
        self.pairs = pairs
        self.chunks = chunks
        self.interval = interval
        self.age_days = age_days
        self.strategy_func = strategy_func
        self.params = params
        self.metric = metric
        self.results = {}

    def run(self) -> pd.DataFrame:
        console = Console()

        for pair in self.pairs:
            self.engine.fetch_data(
                symbol=pair,
                chunks=self.chunks,
                interval=self.interval,
                age_days=self.age_days
            )
            self.engine.run_strategy(self.strategy_func, **self.params)
            metrics = self.engine.get_performance_metrics()
            self.results[pair] = metrics
        table = Table(title="Algorithm Assignment Optimizer Results")
        table.add_column("Pair", style="cyan")
        table.add_column("Metric", style="green")
        self.results = dict(sorted(self.results.items(), key=lambda x: x[1][self.metric], reverse=True))
        for result in self.results:
            table.add_row(
                result,
                f"{self.results[result][self.metric]:.2%}"
            )
        console.print(table)
        return self.results
    
    def get_best_pair(self) -> str:
        return max(self.results, key=lambda x: self.results[x][self.metric])

    def plot_best_performance(self, show_graph: bool = True, advanced: bool = False):
        self.engine.fetch_data(
            symbol=self.get_best_pair(),
            chunks=self.chunks,
            interval=self.interval,
            age_days=self.age_days
        )
        self.engine.run_strategy(self.strategy_func, **self.params)
        return self.engine.plot_performance(show_graph=show_graph, advanced=advanced)


if __name__ == "__main__":
    vb = VectorizedBacktesting(
        initial_capital=10000.0,
    )
    AAO = AlgorithmAssignmentOptimizer(
        engine=vb,
        pairs=['BONK-USDT', 'WIF-USDT', 'PNUT-USDT', 'AI16Z-USDT', 'MEW-USDT', 'BABYDOGE-USDT'],
        chunks=50,
        interval="5min",
        age_days=0,
        strategy_func=vb.ema_cross_strategy,
        params={"fast_period": 9, "slow_period": 26},
        metric="Active_Returns",
    )
    AAO.run()
    AAO.plot_best_performance(advanced=True)

