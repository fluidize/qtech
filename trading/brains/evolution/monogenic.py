### Evolve a single lineage

from typing import Dict, Any, Tuple
from trading.backtesting.backtesting import VectorizedBacktest
from trading.backtesting.algorithm_optim import BayesianOptimizer
import trading.backtesting.mc_analysis as mc
import trading.model_tools as mt

from genetics.algorithm_builder import generate_population, generate_genome
from genetics.tools import display_ast, unparsify

import ast
from tqdm import tqdm
from rich import print
from time import time
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os

from genetics.algorithm_builder import Genome

import faulthandler
faulthandler.enable()

###
EPOCHS = 10000

VB_CONFIG = {
    "instance_name": "Monogenic",
    "initial_capital": 1,
    "slippage_pct": 0.0,
    "commission_fixed": 0.0,
    "leverage": 1.0
}

DATA_CONFIG = {
    "symbols": ["SOL-USDT"],
    "days": 365,
    "interval": "1h",
    "age_days": 0,
    "data_source": "binance",
    "cache_expiry_hours": 999,
    "verbose": False
}

BO_CONFIG = {
    # "metric": "Sortino_Ratio * (Sortino_Ratio *Sharpe_Ratio)**2 * max(0, 1 + Max_Drawdown) * min(500,Total_Trades) * R2",
    "metric": "Total_Return + max(Sortino_Ratio, 0)",
    "n_trials": 3,
    "direction": "maximize",
    "callbacks": [
        # lambda trial: print(f"Trial {trial.number} completed with metric {trial.metric}")
    ]
}

def evaluate_genome(genome: Genome, vb: VectorizedBacktest, bo: BayesianOptimizer):
    bo.optimize(
        engine=vb,
        strategy_func=genome.get_compiled_function(),
        param_space=genome.param_space,
        show_progress_bar=False
    )
    best_params = bo.get_best_params()
    best_metric = bo.get_best_metric()

    genome.set_best(best_params, best_metric)

    return best_params, best_metric

vb = VectorizedBacktest(**VB_CONFIG)
vb.fetch_data(**DATA_CONFIG)

bo = BayesianOptimizer(
    metric=BO_CONFIG["metric"],
    n_trials=BO_CONFIG["n_trials"],
    direction=BO_CONFIG["direction"],
    callbacks=BO_CONFIG["callbacks"]
)

founder = generate_genome(
    num_indicators=2,
    num_logic=2,
    allow_logic_composition=False
) #the founder is very simple

metrics = []
previous_genome = founder
for epoch in tqdm(range(EPOCHS), desc="Epoch"):
    try:
        evaluate_genome(previous_genome, vb, bo)
        metrics.append(previous_genome.get_best_metric())
        previous_genome = previous_genome.mutate()
    except Exception as e:
        print(e)

print(metrics)
plt.plot(metrics)
plt.show()