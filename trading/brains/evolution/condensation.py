from trading.backtesting.backtesting import VectorizedBacktesting
from trading.backtesting.algorithm_optim import BayesianOptimizer

from evolution import generate_population
from genetics.ast_tools import display_ast, ast_to_function, unparsify

from tqdm import tqdm
from rich import print

import faulthandler
faulthandler.enable()
#large sims can cause C side errors, faulthandler to prevent silent crashes

vb = VectorizedBacktesting(instance_name="Condensation",
    initial_capital=10000,
    slippage_pct=0.05,
    commission_fixed=0.0,
    leverage=1.0
)
vb.fetch_data(symbol="ETH-USDT", days=100, interval="30m", age_days=0, data_source="binance", cache_expiry_hours=48)

def quickstop_callback(study, trial):
    bad_trials = sum(1 for t in study.trials if t.value is not None and t.value < 0)
    if bad_trials >= 3:
        study.stop()

log = {}
population = generate_population(size=100, min_indicators=2, max_indicators=6, min_logic=2, max_logic=6, allow_logic_composition=True, logic_composition_prob=0.5)

progress_bar = tqdm(total=len(population), desc="Evaluating population")

for i, individual in enumerate(population):
    bo = BayesianOptimizer(engine=vb)
    bo.optimize(
        strategy_func=individual.get_compiled_function(), 
        param_space=individual.get_param_space(), 
        metric="Alpha * np.clip(1 + Max_Drawdown, 0, None)",
        n_trials=15,
        direction="maximize",
        callbacks=[quickstop_callback],
        show_progress_bar=False
    )
    log[individual] = bo.get_best()
    progress_bar.update(1)

progress_bar.close()

best_individual = max(log, key=lambda x: log[x][1])
display_ast(best_individual.get_function_ast())
print(f"Params: {log[best_individual][0]}")

vb.run_strategy(best_individual.get_compiled_function(), **log[best_individual][0])
print(vb.get_performance_metrics())
vb.plot_performance(mode="standard")

with open("best.txt", "w") as f:
    f.write(unparsify(best_individual.get_function_ast()))
    f.write(f"\nParams: {log[best_individual][0]}")
    f.write(f"\nSearch Space: {log[best_individual][1]}")