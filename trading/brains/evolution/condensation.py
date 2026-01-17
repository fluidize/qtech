from trading.backtesting.backtesting import VectorizedBacktesting
from trading.backtesting.algorithm_optim import BayesianOptimizer

from evolution import generate_population
from genetics.gp_tools import display_ast, unparsify, ast_to_function
import ast

from tqdm import tqdm
from rich import print
from time import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import faulthandler
faulthandler.enable()
#large sims can cause C side errors, faulthandler to prevent silent crashes

def quickstop_callback(study, trial):
    bad_trials = sum(1 for t in study.trials if t.value is not None and t.value < 0)
    if bad_trials >= 3:
        study.stop()

def evaluate_genome(args):
    """Evaluate a single genome - must be at module level for multiprocessing"""
    genome_index, ast_str, param_space, vb_config, data_config = args
    
    from trading.backtesting.backtesting import VectorizedBacktesting
    from trading.backtesting.algorithm_optim import BayesianOptimizer
    from trading.brains.evolution.genetics.gp_tools import ast_to_function
    
    # Reconstruct function from AST string
    ast_node = ast.parse(ast_str, mode='exec')
    function_ast = ast_node.body[0]  # Extract the function definition
    strategy_func = ast_to_function(function_ast)
    strategy_func.param_space = param_space
    
    vb = VectorizedBacktesting(**vb_config)
    vb.fetch_data(**data_config)
    
    bo = BayesianOptimizer(engine=vb)
    bo.optimize(
        strategy_func=strategy_func, 
        param_space=param_space, 
        metric="Alpha * (Total_Trades ** (1/2)) * np.clip((1 + Max_Drawdown), 0, None)",
        n_trials=15,
        direction="maximize",
        callbacks=[quickstop_callback],
        show_progress_bar=False
    )
    return genome_index, bo.get_best()


if __name__ == "__main__":
    start_time = time()
    log = {}
    population = generate_population(size=1000, min_indicators=2, max_indicators=6, min_logic=2, max_logic=6, allow_logic_composition=True, logic_composition_prob=0.5)
    for genome in population:
        genome.remove_unused_genes()
    end_time = time()
    print(f"Genomes prepared in {end_time - start_time} seconds.")

    vb_config = {
        "instance_name": "Condensation",
        "initial_capital": 10000,
        "slippage_pct": 0.01,
        "commission_fixed": 0.0,
        "leverage": 1.0
    }

    data_config = {
        "symbol": "SOL-USDT",
        "days": 900,
        "interval": "4h",
        "age_days": 0,
        "data_source": "binance",
        "cache_expiry_hours": 48,
        "verbose": False
    }

    progress_bar = tqdm(total=len(population), desc="Evaluating population")
    genome_data = []
    for i, genome in enumerate(population):
        ast_str = unparsify(genome.get_function_ast())
        param_space = genome.get_param_space()
        genome_data.append((i, ast_str, param_space))

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(evaluate_genome, (i, ast_str, param_space, vb_config, data_config)): i 
                   for i, ast_str, param_space in genome_data}

        for future in as_completed(futures):
            genome_index, result = future.result()
            log[population[genome_index]] = result
            progress_bar.update(1)
    progress_bar.close()

    sorted_log = sorted(log.items(), key=lambda x: x[1][1], reverse=True)
    top_5 = sorted_log[:5]  

    best_genome = top_5[0][0]
    display_ast(best_genome.get_function_ast())
    print(f"Params: {log[best_genome][0]}")

    vb = VectorizedBacktesting(**vb_config)
    vb.fetch_data(**data_config)

    vb.run_strategy(best_genome.get_compiled_function(), **log[best_genome][0])
    print(vb.get_performance_metrics())
    vb.plot_performance(mode="standard")

    with open("best.txt", "w") as f:
        for i, (genome, (params, metric)) in enumerate(top_5, 1):
            f.write(f"=== Algorithm {i} (Metric: {metric}) ===\n")
            f.write(unparsify(genome.get_function_ast()))
            f.write(f"\nparams={params}")
            f.write(f"\nsearch_space={genome.get_param_space()}")
            f.write(f"\n\n")