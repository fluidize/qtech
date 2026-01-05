import random
import ast

from genetics.ast_builder import Genome, IndicatorGene, IndicatorToConstant, IndicatorToPrice, IndicatorToIndicator, LogicToLogic, SignalGene
from genetics.gp_tools import get_indicators, random_comparison_operator, random_composition_operator

EXCLUDED_INDICATORS = ["hma", "percent_rank", "ichimoku"]
#either noncausal or expensive

def generate_indicator_gene():
    return IndicatorGene(function=random.choice(get_indicators(exclude=EXCLUDED_INDICATORS)))

def generate_logic_gene_sequence(num_logic: int, num_indicators: int, allow_logic_composition: bool, logic_composition_prob: float):
    logic_genes = []
    for i in range(num_logic):
        if (allow_logic_composition and 
            len(logic_genes) >= 2 and 
            random.random() < logic_composition_prob):
            left_idx, right_idx = random.sample(range(len(logic_genes)), 2)
            logic_genes.append(LogicToLogic(
                left_logic_index=left_idx,
                right_logic_index=right_idx,
                operator=random_composition_operator()
            ))
        else:
            logic_type = random.choice([IndicatorToPrice, IndicatorToConstant, IndicatorToIndicator])
            
            if logic_type == IndicatorToPrice:
                logic_genes.append(IndicatorToPrice(
                    left_index=random.randint(0, num_indicators - 1),
                    column_index=random.randint(0, 3),
                    operator=random_comparison_operator()
                ))
            elif logic_type == IndicatorToConstant:
                logic_genes.append(IndicatorToConstant(
                    left_index=random.randint(0, num_indicators - 1),
                    operator=random_comparison_operator()
                ))
            else:
                left_idx = random.randint(0, num_indicators - 1)
                right_idx = random.randint(0, num_indicators - 1)
                while right_idx == left_idx and num_indicators > 1:
                    right_idx = random.randint(0, num_indicators - 1)
                logic_genes.append(IndicatorToIndicator(
                    left_index=left_idx,
                    right_index=right_idx,
                    operator=random_comparison_operator()
                ))
    return logic_genes

def generate_signal_gene(num_logic: int):
    long_logic_index, short_logic_index = random.sample(range(num_logic), 2)
    return SignalGene(long_logic_index=long_logic_index, short_logic_index=short_logic_index)

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
    if num_indicators is None:
        num_indicators = random.randint(min_indicators, max_indicators)
    if num_logic is None:
        num_logic = random.randint(min_logic, max_logic)
    
    indicator_genes = [generate_indicator_gene() for _ in range(num_indicators)]
    
    logic_genes = generate_logic_gene_sequence(num_logic, num_indicators, allow_logic_composition, logic_composition_prob)
    signal_genes = [generate_signal_gene(num_logic)]
    return Genome(indicator_genes=indicator_genes, logic_genes=logic_genes, signal_genes=signal_genes)

def generate_population(
    size=100, 
    num_indicators=None,
    num_logic=None,
    min_indicators=2,
    max_indicators=6,
    min_logic=2,
    max_logic=6,
    allow_logic_composition=True,
    logic_composition_prob=0.5
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