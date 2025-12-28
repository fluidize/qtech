#2Indicators 2Logic
import random

import sys

sys.path.append("trading/brains/evolution")

from ast_builder import Builder
from ast_tools import ast_to_function
from genes import IndicatorGene, IndicatorToConstant, IndicatorToPrice, IndicatorToIndicator
from gp_tools import get_indicators, paramspecs_to_dict, random_operator

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
    population = []
    for _ in range(size):
        individual = generate_individual(num_indicators, num_logic)
        base_ast, algorithm_parameter_specs = individual._construct_algorithm_base()
        function = ast_to_function(base_ast)
        function.param_space = paramspecs_to_dict(algorithm_parameter_specs)
        population.append(function)
    return population