#condense a large population of individuals based on 
from individual import generate_population

import sys
sys.path.append("")
from trading.backtesting.backtesting import VectorizedBacktesting
from trading.backtesting.algorithm_optim import BayesianOptimizer
from genetics.ast_tools import unparsify, ast_to_function
from genetics.gp_tools import paramspecs_to_dict

from rich import print

vb = VectorizedBacktesting(instance_name="Condensation",
    initial_capital=10000,
    slippage_pct=0.00,
    commission_fixed=0.0,
    leverage=1.0
)
vb.fetch_data(symbol="SOL-USDT", days=1095, interval="4h", age_days=0, data_source="binance", cache_expiry_hours=48)

log = {} #index : fitness score
population = generate_population(size=100, num_indicators=2, num_logic=2)
compiled_population = []
for individual in population:
    base_ast, algorithm_parameter_specs = individual._construct_algorithm_base()
    function = ast_to_function(base_ast)
    function.param_space = paramspecs_to_dict(algorithm_parameter_specs)
    compiled_population.append(function)

def quickstop_callback(study, trial):
    bad_trials = sum(1 for t in study.trials if t.value < 1)
    if bad_trials == 5:
        study.stop()

for i, (individual, strategy_func) in enumerate(zip(population, compiled_population)):
    bo = BayesianOptimizer(engine=vb)
    bo.optimize(
        strategy_func=strategy_func, 
        param_space=strategy_func.param_space, 
        metric="Alpha * Total_Trades",
        n_trials=100,
        direction="maximize",
        callbacks=[],
        show_progress_bar=False
    )
    log[individual._construct_algorithm_base()[0]] = bo.get_best() #ast : params, metric
    print(f"{i+1}/{len(population)}")

best_individual = max(log, key=lambda x: log[x][1])
unparsify(best_individual)
vb.run_strategy(ast_to_function(best_individual), **log[best_individual][0])
print(vb.get_performance_metrics())
vb.plot_performance(mode="standard")