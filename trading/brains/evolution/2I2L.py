#two indicators two logic genes
import random
import ast

from ast_builder import Builder
from ast_tools import unparsify
from genes import IndicatorGene, IndicatorToConstant, IndicatorToPrice, IndicatorToIndicator
from gp_tools import get_indicators, random_operator, paramspecs_to_dict

from rich import print

def generate_individual(num_indicators=2, num_logic=2):
    """Generate one random individual."""
    available = get_indicators()
    
    # Random indicators
    indicator_genes = [
        IndicatorGene(function=random.choice(available))
        for _ in range(num_indicators)
    ]
    
    # Random logic
    logic_genes = []
    for _ in range(num_logic):
        logic_type = random.choice([IndicatorToPrice, IndicatorToConstant])
        
        if logic_type == IndicatorToPrice:
            logic_genes.append(IndicatorToPrice(
                left_index=random.randint(0, num_indicators - 1),
                column_index=random.randint(0, 3),
                operator=random_operator()
            ))
        else:
            logic_genes.append(IndicatorToConstant(
                left_index=random.randint(0, num_indicators - 1),
                constant=random.randint(0, 100),
                operator=random_operator()
            ))
    
    return Builder(indicator_genes=indicator_genes, logic_genes=logic_genes)

def generate_population(size=100, num_indicators=2, num_logic=2):
    """Generate population of random individuals."""
    return [generate_individual(num_indicators, num_logic) for _ in range(size)]

# Example usage
if __name__ == "__main__":
    import sys
    sys.path.append("")
    import trading.technical_analysis as ta
    from trading.backtesting.algorithm_optim import QuantitativeScreener
    import pandas as pd
    
    
    engine = QuantitativeScreener(
        symbols=["SOL-USDT"],
        days=365,
        intervals=["1d"],
        age_days=0,
        data_source="binance",
        initial_capital=10000,
        slippage_pct=0.005,
        commission_fixed=0.00,
        cache_expiry_hours=24
    )
    # Single individual
    builder = generate_individual(num_indicators=2, num_logic=2)
    base_ast, algorithm_parameter_specs = builder._construct_algorithm_base()
    unparsify(base_ast)
    
    base_func = compile(ast.fix_missing_locations(ast.Module(body=[base_ast])), "<string>", "exec")
    namespace = {"ta": ta, "pd": pd}
    exec(base_func, namespace)

    base_func = namespace["strategy"]
    base_param_space = paramspecs_to_dict(algorithm_parameter_specs)

    engine.optimize(
        strategy_func=base_func,
        param_space=base_param_space,
        metric="Sharpe_Ratio",
        n_trials=100,
        direction="maximize",
        save_params=False
    )

    engine.plot_best_performance(mode="standard")