from trading.backtesting.backtesting import VectorizedBacktesting
import trading.backtesting.mc_analysis as mc
import trading.model_tools as mt

from evolution import generate_population
from genetics.gp_tools import display_ast, unparsify, ast_to_function
import ast

from tqdm import tqdm
from rich import print
from time import time
import matplotlib.pyplot as plt
import os

from concurrent.futures import ProcessPoolExecutor, as_completed

import faulthandler
faulthandler.enable()
#large sims can cause C side errors, faulthandler to prevent silent crashes

def quickstop_callback(study, trial):
    bad_trials = sum(1 for t in study.trials if t.value is not None and t.value < 0)
    if bad_trials >= 3:
        study.stop()

def evaluate_genome(args):
    genome_index, ast_str, param_space, vb_config, data_config, n_trials = args
    
    from trading.backtesting.backtesting import VectorizedBacktesting
    from trading.backtesting.algorithm_optim import BayesianOptimizer
    from trading.brains.evolution.genetics.gp_tools import ast_to_function
    from trading.backtesting.mc_analysis import MonteCarloAnalysis
    
    ast_node = ast.parse(ast_str, mode='exec')
    function_ast = ast_node.body[0]
    strategy_func = ast_to_function(function_ast)
    strategy_func.param_space = param_space
    
    vb = VectorizedBacktesting(**vb_config)
    vb.fetch_data(**data_config)
    
    bo = BayesianOptimizer(engine=vb)
    bo.optimize(
        strategy_func=strategy_func, 
        param_space=param_space, 
        metric="(Sharpe_Ratio)**3 * max(0, 1 + Max_Drawdown) * Total_Trades * R2",
        n_trials=n_trials,
        direction="maximize",
        callbacks=[quickstop_callback],
        show_progress_bar=False
    )
    return genome_index, bo.get_best()


if __name__ == "__main__":
    POPULATION_SIZE = 10000

    vb_config = {
        "instance_name": "Condensation",
        "initial_capital": 1,
        "slippage_pct": 0.05,
        "commission_fixed": 0.0,
        "leverage": 1.0
    }

    data_config = {
        "symbol": "SOL-USDT",
        "days": 800,
        "interval": "15m",
        "age_days": 0,
        "data_source": "binance",
        "cache_expiry_hours": 999,
        "verbose": False
    }
    mt.fetch_data(**data_config)

    passes = [
        {"n_trials": 1, "keep_top_n": 2000},
        {"n_trials": 5, "keep_top_n": 500},
        {"n_trials": 5, "keep_top_n": 100},
        # {"n_trials": 50, "keep_top_n": 10}
    ]

    start_time = time()
    population = generate_population(size=POPULATION_SIZE, min_indicators=2, max_indicators=8, min_logic=2, max_logic=8, allow_logic_composition=True, logic_composition_prob=0.8)
    for genome in population:
        genome.remove_unused_genes()
    end_time = time()
    print(f"Genomes prepared in {end_time - start_time} seconds.")

    with ProcessPoolExecutor(max_workers=os.cpu_count()-1) as executor:
        for pass_num, pass_config in enumerate(passes, 1):
            n_trials = pass_config["n_trials"]
            keep_top_n = pass_config["keep_top_n"]

            progress_bar = tqdm(total=len(population), desc=f"Pass {pass_num}")
            futures = {executor.submit(evaluate_genome, (i, unparsify(g.get_function_ast()), g.get_param_space(), vb_config, data_config, n_trials)): i
                       for i, g in enumerate(population)}

            for future in as_completed(futures):
                genome_index, (params, metric) = future.result()
                population[genome_index].set_best(params, metric)
                progress_bar.update(1)
            progress_bar.close()

            if pass_num < len(passes):
                population = sorted(population, key=lambda g: (m if (m := g.get_best_metric()) is not None else float("-inf")), reverse=True)[:keep_top_n]

    top_5 = sorted(population, key=lambda g: (m if (m := g.get_best_metric()) is not None else float("-inf")), reverse=True)[:5]
    best_genome = top_5[0]
    display_ast(best_genome.get_function_ast())

    vb = VectorizedBacktesting(**vb_config)
    vb.fetch_data(**data_config)
    vb.run_strategy(best_genome.get_compiled_function(), **best_genome.get_best_params())
    vb.plot_performance(mode="standard")

    num_trades = vb.get_performance_metrics()['Total_Trades']
    
    mc_analysis = mc.MonteCarloAnalysis(best_genome.get_compiled_function(), best_genome.get_best_params(), vb)
    mc_analysis.build_distribution()
    mc_analysis.spaghetti_plot(1000, num_trades)


    with open("best.txt", "w") as f:
        for i, genome in enumerate(top_5, 1):
            params, metric = genome.get_best_params(), genome.get_best_metric()
            f.write(f"=== Algorithm {i} (Metric: {metric}) ===\n")
            f.write(unparsify(genome.get_function_ast()))
            f.write(f"\nparams={params}")
            f.write(f"\nsearch_space={genome.get_param_space()}")
            f.write(f"\n\n")