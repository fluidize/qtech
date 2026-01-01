import random

from genetics.ast_builder import Genome, IndicatorGene, IndicatorToConstant, IndicatorToPrice, IndicatorToIndicator, LogicToLogic, SignalGene
from genetics.ast_tools import ast_to_function
from genetics.gp_tools import get_indicators, paramspecs_to_dict, random_operator

EXCLUDED_INDICATORS = ["hma", "percent_rank", "ichimoku"]
#either noncausal or expensive
def generate_individual(
    num_indicators=None, 
    num_logic=None,
    min_indicators=2,
    max_indicators=6,
    min_logic=2,
    max_logic=6,
    allow_logic_composition=True,
    logic_composition_prob=0.5
):
    """Generate one random individual with configurable complexity.
    
    Args:
        num_indicators: Fixed number of indicators (if None, random between min/max)
        num_logic: Fixed number of logic genes (if None, random between min/max)
        min_indicators: Minimum number of indicators
        max_indicators: Maximum number of indicators
        min_logic: Minimum number of logic genes
        max_logic: Maximum number of logic genes
        allow_logic_composition: Whether to allow LogicToLogic genes
        logic_composition_prob: Probability of creating a LogicToLogic gene
    """
    available = get_indicators(exclude=EXCLUDED_INDICATORS)
    
    if num_indicators is None:
        num_indicators = random.randint(min_indicators, max_indicators)
    if num_logic is None:
        num_logic = random.randint(min_logic, max_logic)
    
    indicator_genes = [
        IndicatorGene(function=random.choice(available))
        for _ in range(num_indicators)
    ]
    
    logic_genes = []
    for i in range(num_logic):
        if (allow_logic_composition and 
            len(logic_genes) >= 2 and 
            random.random() < logic_composition_prob):
            left_idx = random.randint(0, len(logic_genes) - 1)
            right_idx = random.randint(0, len(logic_genes) - 1)
            while right_idx == left_idx and len(logic_genes) > 1:
                right_idx = random.randint(0, len(logic_genes) - 1)
            combine_type = random.choice(['and', 'or'])
            logic_genes.append(LogicToLogic(
                left_logic_index=left_idx,
                right_logic_index=right_idx,
                combine_type=combine_type
            ))
        else:
            logic_type = random.choice([IndicatorToPrice, IndicatorToConstant, IndicatorToIndicator])
            
            if logic_type == IndicatorToPrice:
                logic_genes.append(IndicatorToPrice(
                    left_index=random.randint(0, num_indicators - 1),
                    column_index=random.randint(0, 3),
                    operator=random_operator()
                ))
            elif logic_type == IndicatorToConstant:
                logic_genes.append(IndicatorToConstant(
                    left_index=random.randint(0, num_indicators - 1),
                    constant=random.uniform(0, 100),
                    operator=random_operator()
                ))
            else:
                left_idx = random.randint(0, num_indicators - 1)
                right_idx = random.randint(0, num_indicators - 1)
                while right_idx == left_idx and num_indicators > 1:
                    right_idx = random.randint(0, num_indicators - 1)
                logic_genes.append(IndicatorToIndicator(
                    left_index=left_idx,
                    right_index=right_idx,
                    operator=random_operator()
                ))
    long_logic_index, short_logic_index = random.sample(range(num_logic), 2)
    signal_gene = SignalGene(
        long_logic_index=long_logic_index,
        short_logic_index=short_logic_index
    )
    return Genome(indicator_genes=indicator_genes, logic_genes=logic_genes, signal_gene=signal_gene)

def generate_population(
    size=100, 
    num_indicators=None,
    num_logic=None,
    min_indicators=2,
    max_indicators=6,
    min_logic=2,
    max_logic=5,
    allow_logic_composition=True,
    logic_composition_prob=0.3
):
    """Generate population of random individuals with variable complexity."""
    population = []
    for _ in range(size):
        individual = generate_individual(
            num_indicators=num_indicators,
            num_logic=num_logic,
            min_indicators=min_indicators,
            max_indicators=max_indicators,
            min_logic=min_logic,
            max_logic=max_logic,
            allow_logic_composition=allow_logic_composition,
            logic_composition_prob=logic_composition_prob
        )
        population.append(individual)
    return population