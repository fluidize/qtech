import itertools
import json
import pandas as pd
from typing import Dict, List, Any, Callable

from rich import print
from rich.table import Table
from rich.console import Console

from tqdm import tqdm
import numpy as np

from backtesting import VectorizedBacktesting
import strategy
import optuna

optuna.logging.set_verbosity(optuna.logging.ERROR) #disable optuna printing

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
    def __init__(self, engine: VectorizedBacktesting, strategy_func: Callable, param_space: Dict[str, tuple], metric: str = "Total_Return", n_trials: int = 30, direction: str = "maximize", float_exceptions: List[str] = None, fixed_exceptions: List[str] = None):
        self.engine = engine
        self.strategy_func = strategy_func
        self.param_space = param_space  # e.g., {"fast_period": (1, 50), "slow_period": (1, 50)}
        self.metric = metric
        self.param_names = list(param_space.keys())
        self.best_params = None
        self.best_metrics = None
        self.n_trials = n_trials
        self.direction = direction
        self.float_exceptions = float_exceptions or []
        self.fixed_exceptions = fixed_exceptions or []

        invalid = float('-inf') if self.direction == "maximize" else float('inf')

        def objective(trial):
            try:
                param_dict = self._suggest_params(trial)

                self.engine.run_strategy(self.strategy_func, **param_dict)
                metrics = self.engine.get_performance_metrics()

                if metrics[self.metric] == np.nan or np.isnan(metrics[self.metric]):
                    return -1000 if self.direction == "maximize" else 1000

                return metrics[self.metric]
            except Exception as e:
                # Return a very bad score if the strategy fails, but ensure the trial completes
                print(f"Trial failed with error: {e}")
                return -1000 if self.direction == "maximize" else 1000
        
        self.objective_function = objective

    def _suggest_params(self, trial):
        params = {}
        
        for k, v in self.param_space.items():
            if k in self.fixed_exceptions:
                params[k] = v[0] if isinstance(v, (tuple, list)) else v # For fixed parameters, just use the first value if it's a tuple/list, otherwise use the value directly
            elif k in self.float_exceptions:
                params[k] = trial.suggest_float(k, v[0], v[1])
            else:
                params[k] = trial.suggest_int(k, v[0], v[1])
        return params

    def run(self, show_progress_bar: bool = True):
        study = optuna.create_study(direction=self.direction, pruner=optuna.pruners.NopPruner())
        study.optimize(self.objective_function, n_trials=self.n_trials, show_progress_bar=show_progress_bar)
        
        # Check if any trials were completed
        if len(study.trials) == 0 or all(trial.state != optuna.trial.TrialState.COMPLETE for trial in study.trials):
            raise ValueError("No trials were completed successfully. Check if the strategy function is working properly.")
        
        self.best_params = study.best_params
        self.best_metric = study.best_value
        return self.best_metric, self.best_params

    def get_best(self):
        return self.best_params, self.best_metric

class QuantitativeScreener:
    """
    Optimize on all symbols and intervals for a given strategy to find the highest performing pairs and timeframes.
    Float and fixed exceptions are done by using (float , float) or (n,) respectively.
    """
    def __init__(self,
                 symbols: List[str],
                 days: int,
                 intervals: List[str],
                 age_days: int,
                 data_source: str = "binance",
                 initial_capital: float = 10000,
                 slippage_pct: float = 0.005,
                 commission_fixed: float = 0.00):
        
        self.engine = VectorizedBacktesting(
            instance_name="QuantitativeScreener",
            initial_capital=initial_capital,
            slippage_pct=slippage_pct,
            commission_fixed=commission_fixed,
            reinvest=False
        )

        self.symbols = symbols
        self.days = days
        self.intervals = intervals
        self.age_days = age_days
        self.data_source = data_source
        self.slippage_pct = slippage_pct
        self.commission_fixed = commission_fixed
    
        self.results = pd.DataFrame(columns=["symbol", "interval", "metric", "params"])

        self.console = Console()

    def optimize(
        self,
        strategy_func: Callable, 
        param_space: Dict[str, tuple], 
        metric: str = "Total_Return", 
        n_trials: int = 100, 
        direction: str = "maximize", 
        save_params: bool = False
    ):
        self.strategy_func = strategy_func
        self.metric = metric
        self.direction = direction

        total = len(self.symbols) * len(self.intervals)
        progress_count = 1

        float_exceptions = [] #auto add exceptions
        fixed_exceptions = []
        for param in param_space.keys():
            print(param_space)
            if len(param_space[param]) == 1:
                fixed_exceptions.append(param)
                continue
            if isinstance(param_space[param][0], float):
                float_exceptions.append(param)

        for symbol in self.symbols:
            for interval in self.intervals:
                print(f"{symbol} {interval} {progress_count}/{total} ({progress_count/total:.2%})")

                self.engine.fetch_data(
                    symbol=symbol,
                    interval=interval,
                    days=self.days,
                    age_days=self.age_days,
                    data_source=self.data_source,
                    verbose=False
                )

                BO = BayesianOptimizer(
                    engine=self.engine,
                    strategy_func=strategy_func,
                    param_space=param_space,
                    metric=metric,
                    n_trials=n_trials,
                    direction=direction,
                    float_exceptions=float_exceptions,
                    fixed_exceptions=fixed_exceptions
                )

                best_metrics, best_params = BO.run()

                self.results.loc[len(self.results)] = {
                    "symbol": symbol,
                    "interval": interval,
                    "metric": best_metrics,
                    "params": best_params
                }

                progress_count += 1
        
        self._generate_chart(print_results=True)
        if save_params:
            best = self.get_best()
            with open(f"{self.strategy_func.__name__}-{best["symbol"]}-{best["interval"]}-{self.metric}", "w") as f:
                json.dump(best["params"], f)

    def get_best(self):
        """Get the best performing symbol, interval, and parameters based on the specified metric."""
        if self.results.empty:
            raise ValueError("No results available. Run the optimization first.")
        
        best_row = self.results.loc[self.results['metric'].idxmax() if self.direction == "maximize" else self.results['metric'].idxmin()]
        return {
            "strategy": self.strategy_func.__name__,
            "symbol": best_row["symbol"],
            "interval": best_row["interval"],
            "metric": best_row["metric"],
            "params": best_row["params"]
        }
    
    def get_best_metrics(self):
        best = self.get_best()
        self.engine.fetch_data(
            symbol=best["symbol"],
            interval=best["interval"],
            days=self.days,
            age_days=self.age_days,
            data_source=self.data_source
        )
        self.engine.run_strategy(strategy_func=self.strategy_func, **best["params"])

        return self.engine.get_performance_metrics()
    
    def plot_best_performance(self, show_graph: bool = True, extended: bool = False):
        """Plot the performance of the best parameter combination."""
        best = self.get_best()
        self.engine.fetch_data(
            symbol=best["symbol"],
            interval=best["interval"],
            days=self.days,
            age_days=self.age_days,
            data_source=self.data_source
        )
        self.engine.run_strategy(strategy_func=self.strategy_func, **best["params"])
        return self.engine.plot_performance(show_graph=show_graph, extended=extended)
    
    def _generate_chart(self, print_results: bool = True):
        """Show the performance chart of all strategies."""
        results_table = Table(title="Quantitative Screener Results")
        results_table.add_column("Symbol", style="cyan")
        results_table.add_column("Interval", style="cyan")
        results_table.add_column("Metric", style="green")
        results_table.add_column("Parameters", style="blue")

        sorted_results = self.results.sort_values(by="metric", ascending=False if self.direction == "maximize" else True)

        percent_metrics = ["Total_Return", "Alpha", "Active_Return", "Max_Drawdown", "Win_Rate", "PT_Ratio"]
        if self.metric in percent_metrics:
            sorted_results["metric"] = sorted_results["metric"].apply(lambda x: f"{x:.2%}")
        
        for idx, result in sorted_results.iterrows():
            results_table.add_row(
                result["symbol"],
                result["interval"],
                str(result["metric"]),
                str(result["params"])
            )
        
        if print_results:
            self.console.print(results_table)

        return results_table

if __name__ == "__main__":
    sol_onchains = ["SOL-USDT", "JTO-USDT", "RAY-USDT", "JUP-USDT", "BONK-USDT", "RENDER-USDT"]
    qs = QuantitativeScreener(
        symbols=sol_onchains,
        days=100,
        intervals=["1h"],
        age_days=0,
        data_source="binance",
        initial_capital=100,
        slippage_pct=0.00,
        commission_fixed=0.0
    )

    qs.optimize(
        strategy_func=strategy.wavetrend_strategy,
        param_space={
            "channel_length": (2, 100),
            "average_length": (2, 100),
        },
        metric="Alpha",
        n_trials=500,
        direction="maximize",
        save_params=False
    )

    qs.plot_best_performance(show_graph=True, extended=False)
    qs.plot_best_performance(show_graph=True, extended=True)
    print(qs.get_best_metrics())
    print(qs.get_best())



