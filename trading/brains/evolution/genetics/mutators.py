from .ast_builder import Genome, IndicatorGene, IndicatorToConstant, IndicatorToPrice, IndicatorToIndicator, LogicToLogic, SignalGene

def mutate(genome: Genome, mutation_rate: float) -> Genome:
    current_genes = genome.get_genes()