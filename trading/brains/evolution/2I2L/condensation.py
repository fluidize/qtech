#condense a large population of individuals based on 
from individual import generate_population, generate_individual

import sys
sys.path.append("")
from trading.backtesting.backtesting import VectorizedBacktesting
from trading.backtesting.algorithm_optim import BayesianOptimizer
from trading.brains.evolution.ast_tools import unparsify

vb = VectorizedBacktesting(instance_name="Condensation",
    initial_capital=10000,
    slippage_pct=0.00,
    commission_fixed=0.0,
    reinvest=False,
    leverage=1.0
)
vb.fetch_data(symbol="BTCUSDT", days=10, interval="1d", age_days=30, data_source="binance", cache_expiry_hours=24)

log = {} #index : fitness score
population = generate_population(size=10, num_indicators=2, num_logic=2)
for i, individual in enumerate(population):
    bo = BayesianOptimizer(engine=vb)
    
    bo.optimize(
        strategy_func=individual, 
        param_space=individual.param_space, 
        metric="Sharpe_Ratio * np.clip((1 + Max_Drawdown), 0, None)",
        n_trials=100,
        direction="maximize",
    )
    log[individual] = bo.get_best() #params, metric

best_individual = max(log, key=lambda x: log[x][1])
print(log[best_individual][0])
vb.run_strategy(best_individual, **log[best_individual][0])
vb.plot_performance(mode="standard")