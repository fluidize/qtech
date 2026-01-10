import evolution
from genetics.gp_tools import display_ast

for x in range(1000):
    genome = evolution.generate_genome(logic_composition_prob=0.5)
    print(len(genome.indicator_genes) + len(genome.logic_genes) + len(genome.signal_genes))
    genome.remove_unused_genes()
    print(len(genome.indicator_genes) + len(genome.logic_genes) + len(genome.signal_genes), "\n")