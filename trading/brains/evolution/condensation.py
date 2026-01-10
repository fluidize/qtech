from trading.backtesting.backtesting import VectorizedBacktesting
from trading.backtesting.algorithm_optim import BayesianOptimizer

from evolution import generate_population
from genetics.gp_tools import display_ast, ast_to_function, unparsify

from tqdm import tqdm
from rich import print
from time import time

import faulthandler
faulthandler.enable()
#large sims can cause C side errors, faulthandler to prevent silent crashes

vb = VectorizedBacktesting(instance_name="Condensation",
    initial_capital=10000,
    slippage_pct=0.0,
    commission_fixed=0.0,
    leverage=1.0
)
vb.fetch_data(symbol="SOL-USDT", days=365, interval="30m", age_days=0, data_source="binance", cache_expiry_hours=48)

start_time = time()
log = {}
population = generate_population(size=10000, min_indicators=2, max_indicators=6, min_logic=2, max_logic=6, allow_logic_composition=True, logic_composition_prob=0)
for genome in population:
    genome.remove_unused_genes()
end_time = time()
print(f"Genomes prepared in {end_time - start_time} seconds.")

progress_bar = tqdm(total=len(population), desc="Evaluating population")
def quickstop_callback(study, trial):
    bad_trials = sum(1 for t in study.trials if t.value is not None and t.value < 0)
    if bad_trials >= 3:
        study.stop()
for genome in population:
    bo = BayesianOptimizer(engine=vb)
    bo.optimize(
        strategy_func=genome.get_compiled_function(), 
        param_space=genome.get_param_space(), 
        metric="Sharpe_Ratio * (Total_Trades ** (1/4))",
        n_trials=15,
        direction="maximize",
        callbacks=[quickstop_callback],
        show_progress_bar=False
    )
    log[genome] = bo.get_best()
    progress_bar.update(1)
progress_bar.close()

best_genome = max(log, key=lambda x: log[x][1])
display_ast(best_genome.get_function_ast())
print(f"Params: {log[best_genome][0]}")

vb.run_strategy(best_genome.get_compiled_function(), **log[best_genome][0])
print(vb.get_performance_metrics())
vb.plot_performance(mode="standard")

with open("best.txt", "w") as f:
    f.write(unparsify(best_genome.get_function_ast()))
    f.write(f"\nParams: {log[best_genome][0]}")
    f.write(f"\nSearch Space: {log[best_genome][1]}")