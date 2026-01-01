import random
from typing import List

from genetics.ast_builder import Genome, IndicatorGene, LogicGene, SignalGene, IndicatorToConstant, IndicatorToPrice, IndicatorToIndicator, LogicToLogic
from genetics.gp_tools import get_indicators, random_comparison_operator, random_composition_operator

EXCLUDED_INDICATORS = ["hma", "percent_rank", "ichimoku"]


def generate_indicator_gene():
    available = get_indicators(exclude=EXCLUDED_INDICATORS)
    return IndicatorGene(function=random.choice(available))

def generate_i2p_gene(num_indicators: int):
    return IndicatorToPrice(
        left_index=random.randint(0, num_indicators - 1) if num_indicators > 0 else 0,
        column_index=random.randint(0, 3),
        operator=random_comparison_operator()
    )

def generate_i2c_gene(num_indicators: int):
    return IndicatorToConstant(
        left_index=random.randint(0, num_indicators - 1) if num_indicators > 0 else 0,
        operator=random_comparison_operator()
    )

def generate_i2i_gene(num_indicators: int):
    if num_indicators < 2:
        raise ValueError("Need at least 2 indicators to create IndicatorToIndicator gene")
    left_idx, right_idx = random.sample(range(num_indicators), 2)
    return IndicatorToIndicator(
        left_index=left_idx,
        right_index=right_idx,
        operator=random_comparison_operator()
    )

def generate_l2l_gene(num_logic: int):
    if num_logic < 2:
        raise ValueError("Need at least 2 logic genes to create LogicToLogic gene")
    left_idx, right_idx = random.sample(range(num_logic), 2)
    return LogicToLogic(
        left_logic_index=left_idx,
        right_logic_index=right_idx,
        operator=random_composition_operator()
    )

def generate_simple_logic_gene(num_indicators: int):
    logic_type = random.choice([IndicatorToPrice, IndicatorToConstant, IndicatorToIndicator])
    
    if logic_type == IndicatorToPrice:
        return generate_i2p_gene(num_indicators)
    elif logic_type == IndicatorToConstant:
        return generate_i2c_gene(num_indicators)
    else:
        return generate_i2i_gene(num_indicators)

def generate_logic_gene(num_indicators: int, num_logic: int, allow_composition: bool = True, composition_prob: float = 0.5):
    if allow_composition and num_logic >= 2 and random.random() < composition_prob:
        return generate_l2l_gene(num_logic)
    else:
        return generate_simple_logic_gene(num_indicators)

def generate_signal_gene(num_logic: int):
    if num_logic < 2:
        raise ValueError("Need at least 2 logic genes to create SignalGene")
    long_idx, short_idx = random.sample(range(num_logic), 2)
    return SignalGene(
        long_logic_index=long_idx,
        short_logic_index=short_idx
    )

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
    if num_indicators is None:
        num_indicators = random.randint(min_indicators, max_indicators)
    if num_logic is None:
        num_logic = random.randint(min_logic, max_logic)
    
    indicator_genes = [generate_indicator_gene() for _ in range(num_indicators)]
    
    logic_genes = []
    for i in range(num_logic):
        logic_gene = generate_logic_gene(
            num_indicators=num_indicators,
            num_logic=len(logic_genes),
            allow_composition=allow_logic_composition,
            composition_prob=logic_composition_prob
        )
        logic_genes.append(logic_gene)
    
    signal_genes = [generate_signal_gene(num_logic=num_logic)]
    return Genome(indicator_genes=indicator_genes, logic_genes=logic_genes, signal_genes=signal_genes)

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

def crossover(parent1: Genome, parent2: Genome):
    pass

def mutate_genome(individual: Genome, p_mutate: float = 0.1):
    if random.random() > p_mutate:
        return individual
    
    indicator_genes, logic_genes, signal_genes = individual.get_genes()
    p_ind = 1/len(indicator_genes)
    p_log = 1/len(logic_genes)
    p_sig = 1/len(signal_genes)
    
    new_indicator_genes = [
        mutate_gene(gene) if random.random() < p_ind else gene for gene in indicator_genes
    ]
    new_num_indicators = len(new_indicator_genes)

    num_simple_logic = sum(1 for gene in logic_genes if not isinstance(gene, LogicToLogic))
    new_logic_genes = [
        mutate_gene(gene, num_indicators=new_num_indicators, num_logic=num_simple_logic) if (random.random() < p_log) else gene for gene in logic_genes
    ]
    new_num_logic = len(new_logic_genes)

    new_signal_genes = [
        mutate_gene(gene, num_indicators=new_num_indicators, num_logic=new_num_logic) if random.random() < p_sig else gene for gene in signal_genes
    ]
    return Genome(indicator_genes=new_indicator_genes, logic_genes=new_logic_genes, signal_genes=new_signal_genes)

def mutate_gene(gene: IndicatorGene | LogicGene | SignalGene, num_indicators: int = None, num_logic: int = None):
    if isinstance(gene, IndicatorGene):
        return generate_indicator_gene()
    elif isinstance(gene, LogicToLogic):
        return generate_l2l_gene(num_logic=num_logic)
    elif isinstance(gene, LogicGene):
        return generate_simple_logic_gene(num_indicators=num_indicators)
    elif isinstance(gene, SignalGene):
        return generate_signal_gene(num_logic=num_logic)

