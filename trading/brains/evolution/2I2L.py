#two indicators two logic genes
import random

from ast_builder import Builder
from ast_tools import unparsify
from genes import IndicatorGene, IndicatorToConstant, IndicatorToPrice, IndicatorToIndicator
from gp_tools import get_available_indicators, random_operator

def generate_individual(num_indicators=2, num_logic=2):
    """Generate one random individual."""
    available = get_available_indicators()
    
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
    # Single individual
    builder = generate_individual(num_indicators=2, num_logic=2)
    base_ast, algorithm_parameter_specs = builder._construct_algorithm_base()
    unparsify(base_ast)
    
    population = generate_population(size=50, num_indicators=2, num_logic=2)