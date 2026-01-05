import evolution
from genetics.gp_tools import display_ast

for x in range(1000):
    genome = evolution.generate_genome(logic_composition_prob=0)
    display_ast(genome.get_function_ast())
    genome.remove_unused_genes()
    display_ast(genome.get_function_ast())