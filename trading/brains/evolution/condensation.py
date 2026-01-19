from trading.backtesting.backtesting import VectorizedBacktesting
from trading.backtesting.algorithm_optim import BayesianOptimizer
import trading.backtesting.mc_analysis as mc
import trading.model_tools as mt

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
    genome_index, ast_str, param_space, vb_config, data_config, n_trials = args
    
    from trading.backtesting.backtesting import VectorizedBacktesting
    from trading.backtesting.algorithm_optim import BayesianOptimizer
    from trading.brains.evolution.genetics.gp_tools import ast_to_function
    
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
        metric="Sharpe_Ratio * min(1, Total_Trades/100)",
        n_trials=n_trials,
        direction="maximize",
        callbacks=[],
        show_progress_bar=False
    )
    return genome_index, bo.get_best()


if __name__ == "__main__":
    start_time = time()
    population = generate_population(size=10000, min_indicators=2, max_indicators=8, min_logic=2, max_logic=8, allow_logic_composition=True, logic_composition_prob=0.5)
    for genome in population:
        genome.remove_unused_genes()
    end_time = time()
    print(f"Genomes prepared in {end_time - start_time} seconds.")

    vb_config = {
        "instance_name": "Condensation",
        "initial_capital": 1,
        "slippage_pct": 0.04,
        "commission_fixed": 0.0,
        "leverage": 1.0
    }

    data_config = {
        "symbol": "SOL-USDT",
        "days": 800,
        "interval": "30m",
        "age_days": 0,
        "data_source": "binance",
        "cache_expiry_hours": 48,
        "verbose": False
    }
    mt.fetch_data(**data_config)

    passes = [
        {"n_trials": 5, "keep_top_n": 1000},
        {"n_trials": 10, "keep_top_n": 100},
        {"n_trials": 20, "keep_top_n": 10},
        {"n_trials": 50, "keep_top_n": 5}
    ]

    current_population = population
    log = {}

    for pass_num, pass_config in enumerate(passes, 1):
        n_trials = pass_config["n_trials"]
        keep_top_n = pass_config["keep_top_n"]
        
        print(f"\n[bold cyan]Pass {pass_num}/{len(passes)}: {n_trials} trials, keeping top {keep_top_n}[/bold cyan]")
        
        pass_log = {}
        progress_bar = tqdm(total=len(current_population), desc=f"Pass {pass_num}")
        
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(evaluate_genome, (i, unparsify(genome.get_function_ast()), genome.get_param_space(), vb_config, data_config, n_trials)): i 
                       for i, genome in enumerate(current_population)}

            for future in as_completed(futures):
                genome_index, result = future.result()
                pass_log[current_population[genome_index]] = result
                progress_bar.update(1)
        progress_bar.close()

        log.update(pass_log)
        
        if pass_num < len(passes):
            sorted_pass_log = sorted(pass_log.items(), key=lambda x: x[1][1], reverse=True)
            current_population = [genome for genome, _ in sorted_pass_log[:keep_top_n]]
            print(f"[green]Kept top {len(current_population)} genomes for next pass[/green]")

    sorted_log = sorted(log.items(), key=lambda x: x[1][1], reverse=True)
    top_5 = sorted_log[:5]

    best_genome = top_5[0][0]
    display_ast(best_genome.get_function_ast())

    vb = VectorizedBacktesting(**vb_config)
    vb.fetch_data(**data_config)

    mc_analysis = mc.MonteCarloAnalysis(best_genome.get_compiled_function(), log[best_genome][0], vb)
    mc_analysis.build_distribution()
    mc_analysis.spaghetti_plot(1000, 1000)

    vb.run_strategy(best_genome.get_compiled_function(), **log[best_genome][0])
    vb.plot_performance(mode="standard")

    with open("best.txt", "w") as f:
        for i, (genome, (params, metric)) in enumerate(top_5, 1):
            f.write(f"=== Algorithm {i} (Metric: {metric}) ===\n")
            f.write(unparsify(genome.get_function_ast()))
            f.write(f"\nparams={params}")
            f.write(f"\nsearch_space={genome.get_param_space()}")
            f.write(f"\n\n")