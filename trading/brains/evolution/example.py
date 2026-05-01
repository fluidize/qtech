import evolution
from genetics.gp_tools import display_ast

for x in range(1000):
    genome = evolution.generate_genome(logic_composition_prob=0.5)
    display_ast(genome.get_function_ast())