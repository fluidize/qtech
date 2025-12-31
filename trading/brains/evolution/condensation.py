#condense a large population of individuals based on 
from individual import generate_population

import sys
sys.path.append("")
from trading.backtesting.backtesting import VectorizedBacktesting
from trading.backtesting.algorithm_optim import BayesianOptimizer
from genetics.ast_tools import display_ast, ast_to_function
from genetics.gp_tools import paramspecs_to_dict
from tqdm import tqdm
from rich import print

vb = VectorizedBacktesting(instance_name="Condensation",
    initial_capital=10000,
    slippage_pct=0.05,
    commission_fixed=0.0,
    leverage=1.0
)
vb.fetch_data(symbol="SOL-USDT", days=365, interval="1h", age_days=0, data_source="binance", cache_expiry_hours=48)

def quickstop_callback(study, trial):
    bad_trials = sum(1 for t in study.trials if t.value is not None and t.value < 0)
    if bad_trials >= 3:
        study.stop()

log = {}
population = generate_population(size=100, min_indicators=2, max_indicators=6, min_logic=2, max_logic=6, allow_logic_composition=True, logic_composition_prob=0.5)
compiled_population = []
individual_asts = []

for individual in population:
    base_ast, algorithm_parameter_specs = individual._construct_algorithm_base()
    function = ast_to_function(base_ast)
    function.param_space = paramspecs_to_dict(algorithm_parameter_specs)
    compiled_population.append(function)
    individual_asts.append(base_ast)

progress_bar = tqdm(total=len(population), desc="Evaluating population")

for i, (individual, strategy_func, base_ast) in enumerate(zip(population, compiled_population, individual_asts)):
    bo = BayesianOptimizer(engine=vb)
    try:
        bo.optimize(
            strategy_func=strategy_func, 
            param_space=strategy_func.param_space, 
            metric="Sharpe_Ratio * np.clip(1 + Max_Drawdown, 0, None) * np.clip(Total_Trades, 0, 20)",
            n_trials=15,
            direction="maximize",
            callbacks=[quickstop_callback],
            show_progress_bar=False
        )
        log[base_ast] = bo.get_best()
        progress_bar.update(1)
    except Exception as e:
        print(f"Individual {i+1} failed: {e}")

progress_bar.close()


best_individual = max(log, key=lambda x: log[x][1])
display_ast(best_individual)
print(f"Params: {log[best_individual][0]}")

vb.run_strategy(ast_to_function(best_individual), **log[best_individual][0])
print(vb.get_performance_metrics())
vb.plot_performance(mode="standard")