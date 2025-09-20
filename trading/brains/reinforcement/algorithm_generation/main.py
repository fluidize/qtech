import sys
sys.path.append("")

from trading.backtesting import backtesting as bt
from skeleton import Skeleton, Logic
import optuna

backtest = bt.VectorizedBacktesting(
    instance_name="default",
    initial_capital=10000,
    slippage_pct=0.001,
    commission_fixed=0.0,
    reinvest=False,
    leverage=1.0
)

backtest.fetch_data(
    symbol="SOL-USDT",
    days=20,
    interval="5m",
    age_days=0,
    data_source="binance",
    use_cache=True
)

def objective(trial):
    try:
        indicator = trial.suggest_categorical("indicator", list(Skeleton.INDICATORS.keys()))
        # [longlog, shortlog, longthreshold, shortthreshold, longperiod, shortperiod]
        values = [trial.suggest_categorical("longlog", list(Logic.OPERATORS.keys())), trial.suggest_categorical("shortlog", list(Logic.OPERATORS.keys())), trial.suggest_int("longthreshold", 0, 100), trial.suggest_int("shortthreshold", 0, 100), trial.suggest_int("longperiod", 0, 100), trial.suggest_int("shortperiod", 0, 100)]
        strategy = Skeleton(indicator, values)
        backtest.run_strategy(strategy, verbose=False)
        return backtest.get_performance_metrics()["Total_Return"]
    except:
        return -1000

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10000)
print(study.best_trial.params)
print(study.best_trial.value)
print(study.best_params)
print(study.best_value)

backtest.plot_performance(extended=False)