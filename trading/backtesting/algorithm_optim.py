import itertools
import json
import pandas as pd
from typing import Dict, List, Any, Callable, Optional, Tuple

from rich import print
from rich.table import Table
from rich.console import Console

from tqdm import tqdm
import numpy as np

import optuna
import optunahub
optuna.logging.set_verbosity(optuna.logging.ERROR) #disable optuna printing

from trading.backtesting.backtesting import VectorizedBacktest

class GridSearch:
    def __init__(self, metric: str = "Total_Return") -> None:
        """
        Initialize the grid search framework.
        
        Args:
            metric: Performance metric to optimize (default: Total Return)
        """
        self.metric = metric
        self.results = []
        self.console = Console()

    def _generate_param_combinations(self, strategy_func: Callable, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all possible combinations of parameters. Function constraints are applied here."""
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
        
        if strategy_func.__name__ == "rsi_strategy":
            valid_combinations = [
                param for param in combinations 
                if param["oversold"] < param["overbought"]
            ]
        if strategy_func.__name__ == "macd_strategy":
            valid_combinations = [
                param for param in combinations 
                if param["fastperiod"] < param["slowperiod"]
            ]
        elif strategy_func.__name__ == "ema_cross_strategy":
            valid_combinations = [
                param for param in combinations 
                if param["fast_period"] < param["slow_period"]
            ]
        else:
            valid_combinations = combinations
        
        return valid_combinations

    def _run_single_combination(self, engine: VectorizedBacktest, strategy_func: Callable, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single parameter combination and return the results."""
        engine.run_strategy(strategy_func, **params) #previous data gets overwritten
        metrics = engine.get_performance_metrics()
        return {
            "parameters": params,
            "metrics": metrics
        }

    def run(
        self,
        engine: VectorizedBacktest,
        strategy_func: Callable,
        param_grid: Dict[str, List[Any]]
    ) -> pd.DataFrame:
        """Run the grid search sequentially and return results sorted by the specified metric."""
        param_combinations = self._generate_param_combinations(strategy_func, param_grid)
        total_combinations = len(param_combinations)
        
        table = Table(title="Grid Search Results")
        table.add_column("Parameters", style="cyan")
        table.add_column("Metric", style="green")
        table.add_column("Alpha", style="blue")
        table.add_column("Active Returns", style="magenta")

        self.results = []
        
        
        with tqdm(total=total_combinations, desc="Grid Search Progress") as progress_bar:
            for count, params in enumerate(param_combinations):
                result = self._run_single_combination(engine, strategy_func, params)
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

    def plot_best_performance(self, engine: VectorizedBacktest, strategy_func: Callable, show_graph: bool = True, extended: bool = False) -> Any:
        """Plot the performance of the best parameter combination."""
        best_params = self.get_best_params()
        engine.run_strategy(strategy_func, **best_params)
        return engine.plot_performance(show_graph=show_graph, extended=extended)

class BayesianOptimizer:
    def __init__(
        self,
        metric: str = "Total_Return",
        n_trials: int = 100,
        direction: str = "maximize",
        callbacks: Optional[List[Callable]] = None,
    ) -> None:
        self.metric = metric
        self.n_trials = n_trials
        self.direction = direction
        self.callbacks = callbacks

    def _suggest_params(self, trial: optuna.Trial, param_space: Dict[str, tuple], float_exceptions: List[str], fixed_exceptions: List[str]) -> Dict[str, Any]:
        params = {}
        
        for k, v in param_space.items():
            if k in fixed_exceptions:
                params[k] = v[0] if isinstance(v, (tuple, list)) else v # For fixed parameters, just use the first value if it's a tuple/list, otherwise use the value directly
            elif k in float_exceptions:
                params[k] = trial.suggest_float(k, v[0], v[1])
            else:
                params[k] = trial.suggest_int(k, v[0], v[1])
        return params

    def optimize(
        self,
        engine: VectorizedBacktest,
        strategy_func: Callable,
        param_space: Dict[str, tuple],
        show_progress_bar: bool = True
    ) -> None:
        float_exceptions = []
        fixed_exceptions = []
        for param in param_space.keys():
            if len(param_space[param]) == 1:
                fixed_exceptions.append(param)
                continue
            if isinstance(param_space[param][0], float):
                float_exceptions.append(param)
        
        invalid = float('-inf') if self.direction == "maximize" else float('inf')

        def objective(trial: optuna.Trial) -> float:
            try:
                param_dict = self._suggest_params(trial, param_space, float_exceptions, fixed_exceptions)

                engine.run_strategy(strategy_func, **param_dict)
                metrics = engine.get_performance_metrics()
                safe_dict = {"__builtins__": None, "abs": abs, "max": max, "min": min, "np": np}
                eval_metric = eval(self.metric, safe_dict, metrics)

                if eval_metric == np.nan or np.isnan(eval_metric):
                    return invalid

                return eval_metric
            except Exception as e:
                print(f"Trial failed with error: {e}")
                raise e
        
        self.objective_function = objective
        
        self.study = optuna.create_study(direction=self.direction, sampler=optuna.samplers.TPESampler())
        self.study.optimize(self.objective_function, n_trials=self.n_trials, show_progress_bar=show_progress_bar, callbacks=self.callbacks)
        
        if len(self.study.trials) == 0 or all(trial.state != optuna.trial.TrialState.COMPLETE for trial in self.study.trials):
            raise ValueError("No trials were completed successfully. Check if the strategy function is working properly.")
        
        self.best_params = self.study.best_params
        self.best_metric = self.study.best_value

    def get_best(self) -> Tuple[Dict[str, Any], float]:
        return self.best_params, self.best_metric

    def get_study(self) -> optuna.Study:
        return self.study

class QuantitativeScreener:
    """
    Optimize on all symbols and intervals for a given strategy to find the highest performing pairs and timeframes.
    """
    def __init__(
        self,
        symbols: List[str],
        days: int,
        intervals: List[str],
        age_days: int,
        metric: str = "Total_Return",
        n_trials: int = 100,
        direction: str = "maximize",
        data_source: str = "binance",
        initial_capital: float = 10000,
        slippage_pct: float = 0.005,
        commission_fixed: float = 0.00,
        cache_expiry_hours: int = 24
    ) -> None:
        self.symbols = symbols
        self.days = days
        self.intervals = intervals
        self.age_days = age_days
        self.data_source = data_source
        self.initial_capital = initial_capital
        self.slippage_pct = slippage_pct
        self.commission_fixed = commission_fixed
        self.cache_expiry_hours = cache_expiry_hours
        self.metric = metric
        self.n_trials = n_trials
        self.direction = direction
        
        self.results = pd.DataFrame(columns=["symbol", "interval", "metric", "params", "study"])
        self.console = Console()

    def optimize(
        self,
        strategy_func: Callable,
        param_space: Dict[str, tuple],
        save_params: bool = False
    ) -> None:
        total = len(self.symbols) * len(self.intervals)
        progress_count = 1

        for symbol in self.symbols:
            for interval in self.intervals:
                print(f"{symbol} {interval} {progress_count}/{total} ({progress_count/total:.2%})")

                engine = VectorizedBacktest(
                    instance_name="QuantitativeScreener",
                    initial_capital=self.initial_capital,
                    slippage_pct=self.slippage_pct,
                    commission_fixed=self.commission_fixed
                )
                
                engine.fetch_data(
                    symbol=symbol,
                    interval=interval,
                    days=self.days,
                    age_days=self.age_days,
                    data_source=self.data_source,
                    cache_expiry_hours=self.cache_expiry_hours,
                    verbose=False
                )

                BO = BayesianOptimizer(
                    metric=self.metric,
                    n_trials=self.n_trials,
                    direction=self.direction
                )

                BO.optimize(
                    engine=engine,
                    strategy_func=strategy_func,
                    param_space=param_space
                )
                best_params, best_metric = BO.get_best()
                study = BO.get_study()

                self.results.loc[len(self.results)] = {
                    "symbol": symbol,
                    "interval": interval,
                    "metric": best_metric,
                    "params": best_params,
                    "study": study
                }

                progress_count += 1
        
        self._generate_chart(print_results=True)
        if save_params:
            best = self.get_best()
            file_name = f"strategy-{best['symbol']}-{best['interval']}-{self.metric}"
            file_name = file_name.replace("**", "pow")
            file_name = file_name.replace("*", "x")
            file_name = file_name.replace("/", "div")
            
            with open(file_name, "w") as f:
                json.dump(best["params"], f)

    def get_best(self) -> Dict[str, Any]:
        """Get the best performing symbol, interval, and parameters based on the specified metric."""
        if self.results.empty:
            raise ValueError("No results available. Run the optimization first.")
        
        best_row = self.results.loc[self.results['metric'].idxmax() if self.direction == "maximize" else self.results['metric'].idxmin()]
        return {
            "symbol": best_row["symbol"],
            "interval": best_row["interval"],
            "metric": best_row["metric"],
            "params": best_row["params"],
            "study": best_row["study"]
        }
    
    def get_best_metrics(self, strategy_func: Callable) -> Dict[str, float]:
        best = self.get_best()
        
        engine = VectorizedBacktest(
            instance_name="QuantitativeScreener",
            initial_capital=self.initial_capital,
            slippage_pct=self.slippage_pct,
            commission_fixed=self.commission_fixed
        )
        
        engine.fetch_data(
            symbol=best["symbol"],
            interval=best["interval"],
            days=self.days,
            age_days=self.age_days,
            data_source=self.data_source,
            cache_expiry_hours=self.cache_expiry_hours
        )
        engine.run_strategy(strategy_func=strategy_func, **best["params"])

        return engine.get_performance_metrics()
    
    def get_best_study(self) -> optuna.Study:
        best = self.get_best()
        return best["study"]
    
    def plot_best_performance(self, strategy_func: Callable, mode: str = "basic") -> Any:
        """Plot the performance of the best parameter combination."""
        best = self.get_best()
        
        engine = VectorizedBacktest(
            instance_name="QuantitativeScreener",
            initial_capital=self.initial_capital,
            slippage_pct=self.slippage_pct,
            commission_fixed=self.commission_fixed
        )
        
        engine.fetch_data(
            symbol=best["symbol"],
            interval=best["interval"],
            days=self.days,
            age_days=self.age_days,
            data_source=self.data_source,
            cache_expiry_hours=self.cache_expiry_hours
        )
        engine.run_strategy(strategy_func=strategy_func, **best["params"])
        return engine.plot_performance(mode=mode)
    
    def _generate_chart(self, print_results: bool = True) -> Table:
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