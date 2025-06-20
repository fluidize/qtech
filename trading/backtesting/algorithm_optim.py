import itertools
import json
import pandas as pd
from typing import Dict, List, Any, Callable
from rich import print
from rich.table import Table
from rich.console import Console
from tqdm import tqdm
import numpy as np
import sys

from backtesting import VectorizedBacktesting
import strategy
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING) #disable optuna printing

class GridSearch:
    def __init__(
        self,
        engine: VectorizedBacktesting,
        strategy_func: Callable,
        param_grid: Dict[str, List[Any]],
        metric: str = "Total_Return",
    ):
        """
        Initialize the grid search framework.
        
        Args:
            backtest: VectorizedBacktesting instance with data loaded
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
        
        
        with tqdm(total=total_combinations, desc="Grid Search Progress") as progress_bar:
            for count, params in enumerate(param_combinations):
                result = self._run_single_combination(params)
                self.results.append(result)
                progress_bar.update(1)

        sorted_results = sorted(self.results, key=lambda x: x["metrics"][self.metric], reverse=True)
        table_limit = 50
        for i, result in enumerate(sorted_results, 1):
            table.add_row(
                str(result["parameters"]),
                f"{result['metrics'][self.metric]:.2%}",
                f"{result['metrics']['Alpha']:.2%}",
                f"{result['metrics']['Active_Returns']:.2%}"
            )
            if i >= table_limit-1:
                break
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

    def plot_best_performance(self, show_graph: bool = True, extended: bool = False):
        """Plot the performance of the best parameter combination."""
        best_params = self.get_best_params()
        self.engine.run_strategy(self.strategy_func, **best_params)
        return self.engine.plot_performance(show_graph=show_graph, extended=extended)

class BayesianOptimizer:
    def __init__(self, engine: VectorizedBacktesting, strategy_func: Callable, param_space: Dict[str, tuple], metric: str = "Total_Return", n_trials: int = 30, direction: str = "maximize"):
        self.engine = engine
        self.strategy_func = strategy_func
        self.param_space = param_space  # e.g., {"fast_period": (1, 50), "slow_period": (1, 50)}
        self.metric = metric
        self.param_names = list(param_space.keys())
        self.best_params = None
        self.best_metrics = None
        self.n_trials = n_trials
        self.direction = direction

        invalid = float('-inf') if self.direction == "maximize" else float('inf')

        def save_callback(study, trial):
            if study.best_trial.number == trial.number:
                self.save_params(trial)
        self.save_callback = save_callback

        def objective(trial):
            param_dict = self._suggest_params(trial)

            self.engine.run_strategy(self.strategy_func, **param_dict)
            metrics = self.engine.get_performance_metrics()

            if metrics[self.metric] == np.nan:
                return -1000 if self.direction == "maximize" else 1000

            return metrics[self.metric]
        
        self.objective_function = objective

    def save_params(self, trial):
        with open(f"{self.engine.symbol}-{self.engine.interval}-{self.strategy_func.__name__}-{self.metric}-{self.n_trials}trials-params.json", "w") as f:
            json.dump(trial.params, f)

    def _suggest_params(self, trial):
        params = {}
        float_exceptions = [
            {"strategy": "custom_scalper_strategy", "params": ["wick_threshold", "adx_threshold", "momentum_threshold"]},
            {"strategy": "ETHBTC_trader", "params": ["lower_zscore_threshold", "upper_zscore_threshold"]},
            {"strategy": "sr_strategy", "params": ["threshold", "rejection_ratio_threshold"]},
            {"strategy": "ma_trend_strategy", "params": ["pct_band", "adx_ma_threshold"]},
        ] #float exceptions
        
        fixed_param_exceptions = [
            {"strategy": "ETHBTC_trader", "params": ["chunks", "interval", "age_days"]},
        ] #fixed parameter exceptions
        
        for k, v in self.param_space.items():
            # Check if this is a fixed parameter that should not be optimized
            is_fixed = any(exception["strategy"] == self.strategy_func.__name__ and k in exception["params"] 
                          for exception in fixed_param_exceptions)
            
            if is_fixed:
                # For fixed parameters, just use the first value if it's a tuple/list, otherwise use the value directly
                params[k] = v[0] if isinstance(v, (tuple, list)) else v
            elif any(exception["strategy"] == self.strategy_func.__name__ and k in exception["params"] for exception in float_exceptions):
                params[k] = trial.suggest_float(k, v[0], v[1])
            else:
                params[k] = trial.suggest_int(k, v[0], v[1])
        return params

    def run(self, save_params: bool = False):
        study = optuna.create_study(direction=self.direction, pruner=optuna.pruners.NopPruner())
        study.optimize(self.objective_function, n_trials=self.n_trials, show_progress_bar=True)
        self.best_params = study.best_params
        self.best_metrics = study.best_value
        if save_params:
            self.save_params(study.best_trial)
        return self.best_metrics

    def plot_best_performance(self, show_graph: bool = True, extended: bool = False):
        self.engine.run_strategy(self.strategy_func, **self.best_params)
        return self.engine.plot_performance(show_graph=show_graph, extended=extended)

    def get_best_params(self):
        return self.best_params

    def get_best_metrics(self):
        return self.best_metrics

class AssignmentOptimizer:
    """ Instead of optimizing the algorithm itself, find the best pair fit for the given algorithm."""
    def __init__(self, engine: VectorizedBacktesting, pairs: List[str], chunks: int, interval: str, age_days: int, strategy_func: Callable, params: Dict[str, List[Any]], metric: str = "Total_Return"):
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

    def plot_best_performance(self, show_graph: bool = True, extended: bool = False):
        self.engine.fetch_data(
            symbol=self.get_best_pair(),
            chunks=self.chunks,
            interval=self.interval,
            age_days=self.age_days
        )
        self.engine.run_strategy(self.strategy_func, **self.params)
        return self.engine.plot_performance(show_graph=show_graph, extended=extended)

class AlgorithmOptimizer:
    def __init__(self, symbol: str, chunks: int, age_days: int, data_source: str = "binance"):
        self.engine = VectorizedBacktesting(
            instance_name="AlgorithmOptimizer",
            initial_capital=10000,
            slippage_pct=0.005,
            commission_pct=0.00,
            reinvest=False
        )

        self.symbol = symbol
        self.chunks = chunks
        self.age_days = age_days
        self.results = {}
        self.strategy_func = None
        self.data_source = data_source

    def optimize(self, strategy_func: Callable, param_space: Dict[str, tuple], metric: str = "Total_Return", n_trials: int = 30, direction: str = "maximize", save_params: bool = False):
        self.strategy_func = strategy_func
        
        timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        console = Console()
        table = Table(title="Timeframe Optimization Results")
        table.add_column("Timeframe", style="cyan")
        table.add_column("Best Metric", style="green")
        table.add_column("Best Parameters", style="blue")

        for timeframe in timeframes:
            console.print(f"\n[bold blue]Optimizing for {timeframe} timeframe...[/bold blue]")
            
            self.engine.fetch_data(
                symbol=self.symbol,
                chunks=self.chunks,
                interval=timeframe,
                age_days=self.age_days,
                data_source=self.data_source
            )
            print(self.engine.data)

            bayesian_op = BayesianOptimizer(
                engine=self.engine,
                strategy_func=strategy_func,
                param_space=param_space,
                metric=metric,
                n_trials=n_trials,
                direction=direction
            )

            best_metric = bayesian_op.run(save_params=False)
            best_params = bayesian_op.get_best_params()
            
            self.results[timeframe] = {
                "best_metric": best_metric,
                "best_params": best_params
            }

            table.add_row(
                timeframe,
                f"{best_metric:.2%}",
                str(best_params)
            )

        console.print("\n")
        console.print(table)
        
        # Print sorted results by metric
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]["best_metric"],
            reverse=True
        )
        
        console.print("\n[bold green]Timeframe Performance Ranking:[/bold green]")
        for i, (tf, result) in enumerate(sorted_results, 1):
            console.print(f"{i}. {tf}: {result['best_metric']:.2%}")
            
        # Only save parameters for the best timeframe if save_params is True
        if save_params:
            best_timeframe = self.get_best_timeframe()
            best_params = self.results[best_timeframe]["best_params"]
            with open(f"{self.symbol}-{best_timeframe}-{strategy_func.__name__}-{metric}-{n_trials}trials-params.json", "w") as f:
                json.dump(best_params, f)
            
        return self.results

    def get_best_timeframe(self) -> str:
        if not self.results:
            raise ValueError("No optimization results available. Run optimize() first.")
        
        return max(self.results.items(), key=lambda x: x[1]["best_metric"])[0]

    def get_best_params_with_timeframe(self) -> tuple[Dict[str, Any], str]:
        if not self.results:
            raise ValueError("No optimization results available. Run optimize() first.")
        
        best_timeframe = self.get_best_timeframe()
        best_params = self.results[best_timeframe]["best_params"]
        
        return best_params, best_timeframe

    def plot_best_performance(self, show_graph: bool = True, extended: bool = False):
        if not self.results:
            raise ValueError("No optimization results available. Run optimize() first.")
        
        best_timeframe = self.get_best_timeframe()
        best_params = self.results[best_timeframe]["best_params"]
        
        self.engine.fetch_data(
            symbol="BTC-USDT",
            chunks=100,
            interval=best_timeframe,
            age_days=0,
            data_source=self.data_source
        )
        
        self.engine.run_strategy(self.strategy_func, **best_params)
        return self.engine.plot_performance(show_graph=show_graph, extended=extended)

if __name__ == "__main__":
    vb = VectorizedBacktesting(
        instance_name="AlgorithmOptimizer",
        initial_capital=10000,
        slippage_pct=0.01,
        commission_pct=0.00,
        reinvest=False
    )
    vb.fetch_data(
        symbol="SOL-USDT",
        chunks=365,
        interval="5m",
        age_days=0,
        data_source="binance"
    )
    BO = BayesianOptimizer(
        engine=vb,
        strategy_func=strategy.ma_trend_strategy,
        param_space={
            "band_period":(2,200),
            "pct_band":(0.001,0.05),
            "adx_ma_period":(2,40),
        },
        metric="Alpha",
        n_trials=1000,
        direction="maximize"
    )
    BO.run(save_params=False)
    BO.plot_best_performance(extended=False)
    print(BO.get_best_params())

# if __name__ == "__main__":
#     A = AlgorithmOptimizer(
#         symbol="BTC-USDT",
#         chunks=100,
#         age_days=0,
#         data_source="kucoin"
#     )
    
#     A.optimize(
#         strategy_func=Strategy.ma_reversal_strategy,
#         param_space={
#             "atr_period":(2,40),
#             "ma_period":(2,40),
#             "atr_multiplier":(0.1, 2)
#         },
#         metric="Total_Return",
#         n_trials=300,
#         direction="maximize",
#         save_params=False
#     )

#     A.plot_best_performance(extended=False)
#     A.plot_best_performance(extended=True)

#     best_params, best_tf = A.get_best_params_with_timeframe()
#     print(f"\nBest timeframe: {best_tf}")
#     print(f"Best parameters: {best_params}")

